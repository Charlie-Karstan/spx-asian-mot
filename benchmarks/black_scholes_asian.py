"""Constant-volatility benchmark for scripts.run_spx_mot's Asian contract.

Run from the repository root: python -m benchmarks.black_scholes_asian
Only calibration uses mids; the MOT bid/ask constraints are untouched.
"""
from __future__ import annotations

import argparse
import json
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import ndtr

from robust_pricing.asian.payoff import asian_payoff_from_integral
from robust_pricing.asian.types import AsianOption, Market
from scripts.run_spx_mot import (
    ASIAN_KIND, EXPIRATIONS, OBSERVATION_DATE, RESULTS_DIR, RESULTS_PATH,
    build_market, load_processed_market_data,
)

N_PATHS = 500_000
SEED = 42
BATCH_SIZE = 50_000
SIGMA_BOUNDS = (0.01, 2.0)
CALIBRATION_PRICE_FLOOR = 1.0  # SPX points in the relative-error denominator.


def validate_contract(market: Market, option: AsianOption) -> None:
    """Check finite inputs beyond the checks already made by project types."""
    if not np.isfinite(market.spot) or market.spot <= 0:
        raise ValueError("Market spot must be finite and positive.")

    for name in ("times", "carry", "discount"):
        if not np.all(np.isfinite(getattr(market, name))):
            raise ValueError(f"Market {name} must be finite.")

    if market.times[0] != 0.0 or not np.all(np.diff(market.times) > 0):
        raise ValueError("Monitoring times must start at zero and increase strictly.")
    if np.any(market.carry <= 0) or np.any(market.discount <= 0):
        raise ValueError("Carry and discount factors must be positive.")
    if not np.isfinite(option.strike) or option.strike < 0:
        raise ValueError("Asian strike must be finite and nonnegative.")


def load_contract() -> tuple[Market, AsianOption]:
    """Use the runner's spot, term structure, quote types and ATM convention."""
    chain, term = load_processed_market_data()
    if "quote_date" in chain:
        dates = pd.to_datetime(chain["quote_date"]).dt.date
        if not (dates == OBSERVATION_DATE).all():
            raise ValueError("Processed chain does not match the runner's observation date.")

    market = build_market(chain, term)
    option = AsianOption(strike=market.spot, kind=ASIAN_KIND)
    validate_contract(market, option)

    # Fail on missing/extra observations instead of silently changing monitoring.
    ordered = term.sort_values("maturity")
    if tuple(ordered["expiration"]) != EXPIRATIONS:
        raise ValueError("Term structure does not match the MOT observation dates.")

    return market, option


def calibration_quotes(market: Market) -> dict[str, np.ndarray]:
    """One valid side per (T,K), preferring OTM to avoid redundant parity data.

    Both calls and puts enter where available. If the OTM side is absent, keep
    the valid ITM side. No extra moneyness, volume or spread filters are added.
    Residuals are divided by max(mid, 1 SPX point): relative price errors with
    a floor, so expensive ITM contracts cannot dominate the price-space fit.
    """
    selected = {}
    for quote in market.quotes:
        if not np.all(np.isfinite([
            quote.maturity, quote.strike, quote.bid, quote.ask,
        ])):
            continue
        if (quote.maturity <= 0 or quote.strike <= 0 or quote.bid <= 0
                or quote.ask < quote.bid):
            continue

        t = market.time_index(quote.maturity)
        forward = market.spot / market.carry[0] * market.carry[t]
        preferred = "call" if quote.strike >= forward else "put"
        rank = (quote.kind != preferred, quote.ask - quote.bid)
        key = (quote.maturity, quote.strike)

        if key not in selected or rank < selected[key][0]:
            selected[key] = (rank, quote, forward, market.discount[t])

    if not selected:
        raise ValueError("No finite positive bid/ask quotes available for calibration.")

    rows = [selected[key] for key in sorted(selected)]
    _, selected_quotes, forwards, discounts = zip(*rows)
    quotes = {
        "times": np.array([quote.maturity for quote in selected_quotes]),
        "strikes": np.array([quote.strike for quote in selected_quotes]),
        "forwards": np.array(forwards),
        "discounts": np.array(discounts),
        "is_call": np.array([quote.kind == "call" for quote in selected_quotes]),
        "bid": np.array([quote.bid for quote in selected_quotes]),
        "ask": np.array([quote.ask for quote in selected_quotes]),
    }

    if set(quotes["times"]) != set(market.times[1:]):
        raise ValueError("Each MOT expiry must have at least one calibration quote.")

    quotes["mid"] = 0.5 * (quotes["bid"] + quotes["ask"])
    quotes["scale"] = np.maximum(quotes["mid"], CALIBRATION_PRICE_FLOOR)

    return quotes


def black_scholes_prices(
    sigma: float, times, strikes, forwards, discounts, is_call,
) -> np.ndarray:
    """Black's forward formula, with the actual market D(T) and F(T)."""
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Volatility must be finite and nonnegative.")

    times, strikes, forwards, discounts, is_call = np.broadcast_arrays(
        times, strikes, forwards, discounts, is_call,
    )
    if (not all(np.all(np.isfinite(a)) for a in (times, strikes, forwards, discounts))
            or np.any(times <= 0) or np.any(strikes <= 0)
            or np.any(forwards <= 0) or np.any(discounts <= 0)):
        raise ValueError("Vanilla times, strikes, forwards and discounts must be positive.")

    sign = np.where(is_call, 1.0, -1.0)
    if sigma == 0:
        return discounts * np.maximum(sign * (forwards - strikes), 0.0)

    vol_t = sigma * np.sqrt(times)
    d1 = np.log(forwards / strikes) / vol_t + 0.5 * vol_t
    d2 = d1 - vol_t

    return discounts * sign * (
        forwards * ndtr(sign * d1) - strikes * ndtr(sign * d2)
    )


def calibration_diagnostics(quotes: dict, prices: np.ndarray) -> dict:
    """Check vanilla bounds and disclose fit error and bid/ask compatibility."""
    prices = np.asarray(prices)
    sign = np.where(quotes["is_call"], 1.0, -1.0)
    lower = quotes["discounts"] * np.maximum(
        sign * (quotes["forwards"] - quotes["strikes"]), 0.0,
    )
    upper = quotes["discounts"] * np.where(
        quotes["is_call"], quotes["forwards"], quotes["strikes"],
    )

    if (prices.shape != quotes["mid"].shape or not np.all(np.isfinite(prices))
            or np.any(prices < lower - 1e-6) or np.any(prices > upper + 1e-6)):
        raise RuntimeError("Vanilla model prices are nonfinite or violate no-arbitrage bounds.")

    error = prices - quotes["mid"]

    return {
        "n_quotes": int(len(prices)),
        "n_calls": int(np.sum(quotes["is_call"])),
        "n_puts": int(np.sum(~quotes["is_call"])),
        "quote_selection": "One side per maturity/strike, OTM preferred",
        "weighting": "Residual / max(mid, 1 SPX point)",
        "rmse": float(np.sqrt(np.mean(error**2))),
        "weighted_rmse": float(np.sqrt(np.mean((error / quotes["scale"])**2))),
        "max_absolute_error": float(np.max(np.abs(error))),
        "quotes_outside_bid_ask": int(np.sum(
            (prices < quotes["bid"] - 1e-6) | (prices > quotes["ask"] + 1e-6),
        )),
    }


def calibrate_black_scholes(market: Market) -> tuple[float, dict]:
    """Fit one volatility using scaled mid-price errors across all expirations."""
    quotes = calibration_quotes(market)

    def prices(sigma):
        return black_scholes_prices(
            sigma=sigma, times=quotes["times"], strikes=quotes["strikes"],
            forwards=quotes["forwards"], discounts=quotes["discounts"],
            is_call=quotes["is_call"],
        )

    def objective(sigma):
        residual = (prices(sigma) - quotes["mid"]) / quotes["scale"]
        if not np.all(np.isfinite(residual)):
            raise RuntimeError("Nonfinite Black-Scholes calibration residuals.")

        return float(residual @ residual)

    fit = minimize_scalar(
        objective, bounds=SIGMA_BOUNDS, method="bounded", options={"xatol": 1e-10},
    )
    if not fit.success or not np.isfinite(fit.fun):
        raise RuntimeError(f"Black-Scholes calibration failed: {fit.message}")

    sigma = float(fit.x)
    diagnostics = calibration_diagnostics(quotes, prices(sigma))
    diagnostics.update(success=True, message=str(fit.message), nfev=int(fit.nfev))

    return sigma, diagnostics


def validate_simulation(n_paths: int, batch_size: int) -> None:
    # Independent sampling units are antithetic pairs, so require >= 2 pairs.
    if not isinstance(n_paths, (int, np.integer)) or n_paths < 4 or n_paths % 2:
        raise ValueError("n_paths must be an even integer of at least four.")
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 2 or batch_size % 2:
        raise ValueError("batch_size must be an even integer of at least two.")


def monte_carlo_summary(pair_payoffs: np.ndarray, discount: float) -> dict:
    """SE uses independent pair averages, never the correlated individual paths."""
    discounted_payoffs = discount * pair_payoffs
    if not np.all(np.isfinite(discounted_payoffs)) or np.any(discounted_payoffs < 0):
        raise RuntimeError("Monte Carlo produced nonfinite or negative discounted payoffs.")

    price = float(discounted_payoffs.mean())
    se = float(discounted_payoffs.std(ddof=1) / np.sqrt(len(discounted_payoffs)))

    return {
        "asian_price": price,
        "standard_error": se,
        "confidence_interval_95": [price - 1.96 * se, price + 1.96 * se],
        "independent_antithetic_pairs": int(len(discounted_payoffs)),
    }


def price_asian(
    market: Market, option: AsianOption, sigma: float,
    n_paths: int = N_PATHS, seed: int = SEED, batch_size: int = BATCH_SIZE,
) -> dict:
    """Exact GBM transitions at MOT dates; vectorized, batched antithetic paths."""
    validate_contract(market, option)
    validate_simulation(n_paths, batch_size)

    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Volatility must be finite and nonnegative.")

    rng = np.random.default_rng(seed)
    pair_payoffs = np.empty(n_paths // 2)

    for start in range(0, len(pair_payoffs), batch_size // 2):
        size = min(batch_size // 2, len(pair_payoffs) - start)
        x = np.full((2, size), market.spot / market.carry[0])
        previous_spot = np.full_like(x, market.spot)
        integral = np.zeros_like(x)

        for t, dt in enumerate(np.diff(market.times), start=1):
            z = rng.standard_normal(size)
            shocks = np.stack((z, -z))
            x *= np.exp(-0.5 * sigma**2 * dt + sigma * np.sqrt(dt) * shocks)
            spot = market.carry[t] * x
            integral += 0.5 * (previous_spot + spot) * dt
            previous_spot = spot

        payoff = asian_payoff_from_integral(
            integral, market.times[-1] - market.times[0], option,
        )
        pair_payoffs[start:start + size] = payoff.mean(axis=0)

    return monte_carlo_summary(pair_payoffs, float(market.discount[-1]))


def report_result(
    model: str, filename: str, market: Market, option: AsianOption,
    parameters: dict, calibration: dict, simulation: dict,
    n_paths: int, seed: int, batch_size: int, runtime: float,
) -> dict:
    """Shared by the two benchmarks to keep contract reporting identical."""
    result = {
        "model": model, "observation_date": OBSERVATION_DATE.isoformat(),
        "expirations": [expiry.isoformat() for expiry in EXPIRATIONS],
        "spot": market.spot, "asian_strike": option.strike, "asian_kind": option.kind,
        "times": market.times.tolist(), "carry": market.carry.tolist(),
        "discount": market.discount.tolist(), "terminal_discount": float(market.discount[-1]),
        "average_definition": "sum(0.5*(S[k]+S[k+1])*dt[k]) / (times[-1]-times[0])",
        "parameters": parameters, "calibration": calibration,
        "n_paths": n_paths, "seed": seed, "batch_size": batch_size,
        "antithetic": True, **simulation, "runtime_seconds": runtime,
        "warnings": [],
    }
    if RESULTS_PATH.exists():
        saved = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        matches = all(saved.get(key) == result[key] for key in (
            "observation_date", "expirations", "spot", "asian_strike", "asian_kind",
        ))

        if matches:
            bounds = saved.get("solver_result", {})
            lower, upper = bounds.get("lower"), bounds.get("upper")

            if (lower is not None and upper is not None
                    and np.all(np.isfinite([lower, upper])) and lower <= upper):
                result["mot_interval"] = [lower, upper]
                lo, hi = simulation["confidence_interval_95"]

                if hi < lower or lo > upper:
                    result["warnings"].append(
                        "Monte Carlo interval is outside the saved MOT bounds."
                    )
            else:
                result["warnings"].append("Saved MOT result has no valid bound interval.")
        else:
            result["warnings"].append("Saved MOT contract differs; bound comparison skipped.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{model}: {simulation['asian_price']:.4f}, SE {simulation['standard_error']:.4f}")

    for warning in result["warnings"]:
        print(warning)

    print(path)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=N_PATHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    validate_simulation(args.paths, args.batch_size)
    start = perf_counter()
    market, option = load_contract()
    print("Calibrating Black-Scholes to processed SPX quotes...", flush=True)
    sigma, calibration = calibrate_black_scholes(market)
    simulation = price_asian(
        market=market, option=option, sigma=sigma,
        n_paths=args.paths, seed=args.seed, batch_size=args.batch_size,
    )
    report_result(
        model="Black-Scholes", filename="black_scholes_asian.json",
        market=market, option=option, parameters={"sigma": sigma},
        calibration=calibration, simulation=simulation,
        n_paths=args.paths, seed=args.seed, batch_size=args.batch_size,
        runtime=perf_counter() - start,
    )


if __name__ == "__main__":
    main()
