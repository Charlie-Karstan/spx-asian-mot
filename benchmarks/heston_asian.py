"""Calibrated Heston benchmark for the identical SPX MOT Asian contract.

Run from the repository root: python -m benchmarks.heston_asian
The sibling benchmark supplies the common market/quote/report conventions;
the characteristic function, calibration and simulation are explicit below.
"""
from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares
from scipy.special import roots_legendre

from benchmarks.black_scholes_asian import (
    BATCH_SIZE, N_PATHS, SEED, black_scholes_prices, calibrate_black_scholes,
    calibration_diagnostics, calibration_quotes, load_contract,
    monte_carlo_summary, report_result, validate_contract, validate_simulation,
)
from robust_pricing.asian.payoff import asian_payoff_from_integral
from robust_pricing.asian.types import AsianOption, Market

PARAMETER_NAMES = ("v0", "kappa", "theta", "xi", "rho")
LOWER_BOUNDS = np.array([1e-4, 0.05, 1e-4, 0.01, -0.999])
UPPER_BOUNDS = np.array([1.0, 20.0, 1.0, 5.0, 0.999])
FOURIER_NODES, FOURIER_CUTOFF = 256, 512.0
CHECK_NODES, CHECK_CUTOFF = 512, 1024.0
FOURIER_PRICE_TOLERANCE = 0.01  # Maximum SPX price change at the finer resolution.

# Four substeps per trading day reduce Euler bias in the fast mean-reversion
# fit; this changes model resolution, never the Asian monitoring convention.
MAX_STEP = 1.0 / 1008.0


def validate_parameters(parameters: dict) -> tuple[float, ...]:
    values = tuple(float(parameters[key]) for key in PARAMETER_NAMES)
    v0, kappa, theta, xi, rho = values

    if (not np.all(np.isfinite(values)) or v0 < 0 or kappa <= 0
            or theta < 0 or xi < 0 or abs(rho) > 1):
        raise ValueError("Heston needs finite v0/theta/xi >= 0, kappa > 0, |rho| <= 1.")

    return values


def heston_characteristic_function(u, maturity: float, parameters: dict) -> np.ndarray:
    """E[exp(i*u*log(X_T/X_0))], without rates or deterministic carry.

    The decaying-exponential (little Heston trap) form uses Re(d) >= 0.
    Rationalizing b-d avoids cancellation as xi approaches zero; log1p and
    expm1 stabilize the complex logarithm and short-maturity differences.
    """
    v0, kappa, theta, xi, rho = validate_parameters(parameters)
    if not np.isfinite(maturity) or maturity < 0:
        raise ValueError("Characteristic-function maturity must be finite and nonnegative.")

    u = np.asarray(u, dtype=complex)
    q = u*u + 1j*u

    if xi == 0:
        integrated_variance = (
            theta * maturity + (v0 - theta) * (-np.expm1(-kappa*maturity)) / kappa
        )
        return np.exp(-0.5 * q * integrated_variance)

    b = kappa - rho * xi * 1j*u
    d = np.sqrt(b*b + xi*xi*q)
    d = np.where(d.real < 0, -d, d)
    b_minus_d_over_xi2 = -q / (b + d)
    g = xi*xi * b_minus_d_over_xi2 / (b + d)
    one_minus_exp = -np.expm1(-d*maturity)
    ratio = g * one_minus_exp / (1.0 - g)

    # NumPy's complex log1p can lose the tiny real part as xi -> 0; the
    # subsequent division by xi^2 magnifies that loss. Use its local series.
    series = ratio * (
        1.0 + ratio * (-0.5 + ratio * (1.0/3.0 + ratio * (-0.25 + ratio/5.0)))
    )
    log_ratio = np.where(np.abs(ratio) < 1e-4, series, np.log1p(ratio))
    log_cf_constant = kappa*theta * (b_minus_d_over_xi2*maturity - 2.0*log_ratio / (xi*xi))
    variance_coefficient = b_minus_d_over_xi2 * one_minus_exp / (1.0 - g*np.exp(-d*maturity))

    return np.exp(log_cf_constant + variance_coefficient*v0)


def fourier_quadrature(
    n_nodes: int = FOURIER_NODES, cutoff: float = FOURIER_CUTOFF,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed composite Gauss-Legendre nodes, concentrated near the half-shift pole.

    A single panel on [0,512] undersamples 1/(u^2+1/4) near zero even with
    hundreds of nodes. Geometric panels resolve that peak and the distant tail.
    """
    if n_nodes < 16 or not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Fourier integration needs >= 16 nodes and positive finite cutoff.")

    edges = [0.0, *[edge for edge in (1.0, 4.0, 16.0, 64.0, 256.0) if edge < cutoff], cutoff]
    nodes, weights = roots_legendre(max(16, int(np.ceil(n_nodes / (len(edges) - 1)))))
    panels = [(left, right) for left, right in zip(edges[:-1], edges[1:])]

    return (
        np.concatenate([left + 0.5*(right-left)*(nodes + 1.0) for left, right in panels]),
        np.concatenate([0.5*(right-left)*weights for left, right in panels]),
    )


def heston_prices(parameters: dict, quotes: dict, quadrature=None) -> np.ndarray:
    """Lewis half-shift inversion in forward coordinates, with put-call parity.

    C = D * [F - sqrt(F*K)/pi * integral Re(exp(i*u*log(F/K))
         * phi(u-i/2)) / (u^2+1/4) du].

    phi describes log(X_T/X_0); F = carry(T)*X_0 and D are market inputs.
    Only four characteristic-function evaluations are needed per parameter
    vector; strikes are broadcast along the Fourier nodes. Never clip prices.
    """
    validate_parameters(parameters)

    if quadrature is None:
        quadrature = fourier_quadrature()

    u, weights = quadrature
    prices = np.empty_like(quotes["times"], dtype=float)

    for maturity in np.unique(quotes["times"]):
        mask = quotes["times"] == maturity
        forward = quotes["forwards"][mask]
        strike = quotes["strikes"][mask]
        discount = quotes["discounts"][mask]

        if (maturity <= 0 or not np.isfinite(maturity)
                or not all(np.all(np.isfinite(a) & (a > 0)) for a in (forward, strike, discount))):
            raise ValueError("Heston vanilla times, strikes, forwards and discounts must be positive.")

        phi = heston_characteristic_function(u - 0.5j, float(maturity), parameters)
        phase = np.exp(1j * np.log(forward/strike)[:, None] * u[None, :])
        integral = (phase * phi[None, :]).real @ (weights / (u*u + 0.25))
        call = discount * (forward - np.sqrt(forward*strike) * integral / np.pi)
        put = call - discount*(forward-strike)
        prices[mask] = np.where(quotes["is_call"][mask], call, put)

    if not np.all(np.isfinite(prices)):
        raise RuntimeError("Heston Fourier integration produced nonfinite prices.")

    return prices


def check_black_scholes_limit(quotes: dict) -> float:
    """Check the formula/sign/discount convention before running calibration."""
    sigma = 0.2  # Test parameter only, never a calibrated production result.
    parameters = dict(v0=sigma**2, kappa=2.0, theta=sigma**2, xi=1e-6, rho=0.0)
    heston = heston_prices(parameters, quotes, fourier_quadrature(CHECK_NODES, CHECK_CUTOFF))
    bs = black_scholes_prices(
        sigma=sigma, times=quotes["times"], strikes=quotes["strikes"],
        forwards=quotes["forwards"], discounts=quotes["discounts"],
        is_call=quotes["is_call"],
    )
    error = float(np.max(np.abs(heston - bs)))

    if error > 1e-4:
        raise RuntimeError(f"Heston/Black-Scholes limit check failed: max error {error:g}.")

    calibration_diagnostics(quotes, heston)

    return error


def calibrate_heston(market: Market) -> tuple[dict, dict]:
    """Bounded price-space least squares from three deterministic starting points.

    No Feller constraint is imposed: SPX fits can violate it, and the simulation
    explicitly handles variance at zero. Positive rho is allowed. A successful
    optimizer is necessary but the quote errors and integration check are also
    reported; a mid fit does not establish MOT bid/ask feasibility.
    """
    quotes = calibration_quotes(market)
    limit_error = check_black_scholes_limit(quotes)
    sigma, _ = calibrate_black_scholes(market)
    variance = float(np.clip(sigma**2, 2e-4, 0.5))
    starts = (
        [variance, 1.0, variance, 0.3, -0.5],
        [variance, 5.0, variance, 0.8, -0.8],
        [variance, 0.5, variance, 0.15, 0.2],
    )
    quadrature = fourier_quadrature()

    def residuals(values):
        parameters = dict(zip(PARAMETER_NAMES, values))
        residual = (heston_prices(parameters, quotes, quadrature) - quotes["mid"]) / quotes["scale"]

        if not np.all(np.isfinite(residual)):
            raise RuntimeError("Nonfinite Heston calibration residuals.")

        return residual

    attempts, successful = [], []
    for index, initial in enumerate(starts, start=1):
        fit = least_squares(
            residuals, initial, bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
            x_scale="jac", max_nfev=600, ftol=1e-9, xtol=1e-9, gtol=1e-9,
        )
        attempts.append({
            "start": index, "success": bool(fit.success), "message": str(fit.message),
            "nfev": int(fit.nfev), "cost": float(fit.cost),
        })
        print(f"Heston start {index}/{len(starts)}: success={fit.success}, cost={fit.cost:.6g}", flush=True)

        if fit.success and np.all(np.isfinite(fit.fun)) and np.isfinite(fit.cost):
            successful.append(fit)

    if not successful:
        raise RuntimeError(f"All Heston calibration starts failed: {attempts}")

    best = min(successful, key=lambda fit: fit.cost)
    parameters = dict(zip(PARAMETER_NAMES, map(float, best.x)))
    prices = heston_prices(parameters, quotes, quadrature)

    # Double both cutoff and node count to detect truncation/quadrature errors
    # in the fitted short-dated model before accepting its parameters.
    fine_prices = heston_prices(parameters, quotes, fourier_quadrature(CHECK_NODES, CHECK_CUTOFF))
    integration_error = float(np.max(np.abs(prices - fine_prices)))

    if integration_error > FOURIER_PRICE_TOLERANCE:
        raise RuntimeError(
            f"Heston Fourier integration did not converge (max price change {integration_error:g}); "
            "increase calibration quadrature resolution."
        )

    diagnostics = calibration_diagnostics(quotes, fine_prices)
    diagnostics.update(
        success=True, message=str(best.message), attempts=attempts,
        black_scholes_limit_max_error=limit_error,
        fourier_convergence_max_error=integration_error,
        fourier_nodes=len(quadrature[0]), fourier_cutoff=FOURIER_CUTOFF,
        feller_satisfied=bool(2*parameters["kappa"]*parameters["theta"] >= parameters["xi"]**2),
        parameters_at_bounds=[
            name for i, name in enumerate(PARAMETER_NAMES) if best.active_mask[i] != 0
        ],
    )

    return parameters, diagnostics


def price_asian(
    market: Market, option: AsianOption, parameters: dict,
    n_paths: int = N_PATHS, seed: int = SEED, batch_size: int = BATCH_SIZE,
    max_step: float = MAX_STEP,
) -> dict:
    """Clipped full-truncation variance Euler and log-Euler martingale asset.

    Internal steps end exactly on each monitoring date. Intermediate prices
    never enter the payoff: only MOT observation endpoints form trapezoids.
    Antithetic pairs negate both independent shocks; SE uses pair averages.
    MC uncertainty does not include discretization or calibration error.
    """
    validate_contract(market, option)
    validate_simulation(n_paths, batch_size)
    v0, kappa, theta, xi, rho = validate_parameters(parameters)

    if not np.isfinite(max_step) or max_step <= 0:
        raise ValueError("max_step must be finite and positive.")

    rng = np.random.default_rng(seed)
    pair_payoffs = np.empty(n_paths // 2)
    intervals = np.diff(market.times)
    counts = np.ceil(intervals / max_step).astype(int)
    negative_variances = 0

    for start in range(0, len(pair_payoffs), batch_size // 2):
        size = min(batch_size // 2, len(pair_payoffs) - start)
        x = np.full((2, size), market.spot / market.carry[0])
        variance = np.full_like(x, v0)
        previous_spot = np.full_like(x, market.spot)
        integral = np.zeros_like(x)

        for t, (interval, count) in enumerate(zip(intervals, counts), start=1):
            dt = interval / count
            for _ in range(int(count)):
                z1, z2 = rng.standard_normal((2, size))
                asset_shock = np.stack((z1, -z1))
                correlated_shock = rho*z1 + np.sqrt(1-rho*rho)*z2
                variance_shock = np.stack((correlated_shock, -correlated_shock))
                positive_variance = np.maximum(variance, 0.0)
                root_v_dt = np.sqrt(positive_variance*dt)
                x *= np.exp(-0.5*positive_variance*dt + root_v_dt*asset_shock)
                variance += kappa*(theta-positive_variance)*dt + xi*root_v_dt*variance_shock
                negative_variances += int(np.count_nonzero(variance < 0))
                variance = np.maximum(variance, 0.0)

            spot = market.carry[t]*x
            integral += 0.5*(previous_spot + spot)*interval
            previous_spot = spot

        payoff = asian_payoff_from_integral(
            integral, market.times[-1] - market.times[0], option,
        )
        pair_payoffs[start:start + size] = payoff.mean(axis=0)

    result = monte_carlo_summary(pair_payoffs, float(market.discount[-1]))
    result.update(
        scheme="Clipped full-truncation variance Euler; log-Euler normalized asset",
        max_internal_step=max_step, internal_steps_per_interval=counts.tolist(),
        variance_clipping_fraction=negative_variances / (n_paths * int(counts.sum())),
        confidence_interval_scope="MC sampling only; excludes Euler bias and calibration uncertainty",
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=N_PATHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-step", type=float, default=MAX_STEP, help="Maximum internal step in years")
    args = parser.parse_args()
    validate_simulation(args.paths, args.batch_size)

    if not np.isfinite(args.max_step) or args.max_step <= 0:
        parser.error("--max-step must be finite and positive")

    start = perf_counter()
    market, option = load_contract()
    print("Calibrating Heston to processed SPX quotes...", flush=True)
    parameters, calibration = calibrate_heston(market)
    simulation = price_asian(
        market=market, option=option, parameters=parameters,
        n_paths=args.paths, seed=args.seed, batch_size=args.batch_size,
        max_step=args.max_step,
    )
    report_result(
        model="Heston", filename="heston_asian.json", market=market, option=option,
        parameters=parameters, calibration=calibration, simulation=simulation,
        n_paths=args.paths, seed=args.seed, batch_size=args.batch_size,
        runtime=perf_counter() - start,
    )


if __name__ == "__main__":
    main()
