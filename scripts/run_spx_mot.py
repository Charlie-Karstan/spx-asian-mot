from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd

from robust_pricing.asian.solver import solve_asian_bounds
from robust_pricing.asian.types import AsianOption, GridSpec, Market, SolveResult, VanillaQuote

OBSERVATION_DATE = date(2026, 9, 25)
EXPIRATIONS = (
    date(2026, 10, 16),
    date(2026, 10, 30),
    date(2026, 11, 20),
    date(2026, 12, 18),
)
DATA_DIR = Path("data/processed/marketdata") / OBSERVATION_DATE.isoformat()
CHAIN_PATH = DATA_DIR / "SPXW_clean.csv"
TERM_STRUCTURE_PATH = DATA_DIR / "term_structure.csv"
RESULTS_DIR = Path("results/spx_mot") / OBSERVATION_DATE.isoformat()
RESULTS_PATH = RESULTS_DIR / "mot_bounds.json"
N_X = 50
N_INTEGRAL = 21
MAX_VARIABLES = 500_000
X_BUFFER_FRACTION_OF_SPOT = 0.08
ASIAN_KIND = "call"


def load_processed_market_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not CHAIN_PATH.exists():
        raise FileNotFoundError(f"Missing processed option chain: {CHAIN_PATH}")
    if not TERM_STRUCTURE_PATH.exists():
        raise FileNotFoundError(f"Missing term structure: {TERM_STRUCTURE_PATH}")

    chain = pd.read_csv(CHAIN_PATH)
    term = pd.read_csv(TERM_STRUCTURE_PATH)
    chain["expiration"] = pd.to_datetime(chain["expiration"]).dt.date
    term["expiration"] = pd.to_datetime(term["expiration"]).dt.date
    chain = chain[chain["expiration"].isin(EXPIRATIONS)].copy()
    term = term[term["expiration"].isin(EXPIRATIONS)].copy()

    if chain.empty:
        raise ValueError("No option quotes remain for the requested expirations.")

    missing_expirations = set(EXPIRATIONS) - set(term["expiration"])
    if missing_expirations:
        missing = ", ".join(sorted(d.isoformat() for d in missing_expirations))
        raise ValueError(f"Missing term-structure rows for: {missing}")

    return chain, term


def build_market(chain: pd.DataFrame, term: pd.DataFrame) -> Market:
    required_chain = {"expiration", "strike", "bid", "ask", "underlying_price"}
    side_candidates = [
        "side",
        "option_type",
        "type",
        "right",
        "cp_flag",
    ]
    side_column = next(
        (col for col in side_candidates if col in chain.columns),
        None,
    )

    if side_column is None:
        raise ValueError(
            f"Could not find call/put column. Available columns: {list(chain.columns)}"
        )

    required_term = {"expiration", "maturity", "carry", "discount"}
    missing_chain = required_chain - set(chain.columns)
    missing_term = required_term - set(term.columns)

    if missing_chain:
        raise ValueError(f"SPXW_clean.csv is missing columns: {sorted(missing_chain)}")
    if missing_term:
        raise ValueError(f"term_structure.csv is missing columns: {sorted(missing_term)}")

    term = term.sort_values("maturity").reset_index(drop=True)
    spot = float(chain["underlying_price"].median())
    maturity_by_expiration = {
        row.expiration: float(row.maturity)
        for row in term.itertuples(index=False)
    }
    quotes = tuple(
        VanillaQuote(
            maturity=maturity_by_expiration[row.expiration],
            strike=float(row.strike),
            bid=float(row.bid),
            ask=float(row.ask),
            kind=str(getattr(row, side_column)).lower(),
        )
        for row in chain.itertuples(index=False)
    )
    times = np.array(
        [0.0, *term["maturity"].astype(float).to_numpy()],
        dtype=float,
    )
    carry = np.array(
        [1.0, *term["carry"].astype(float).to_numpy()],
        dtype=float,
    )
    discount = np.array(
        [1.0, *term["discount"].astype(float).to_numpy()],
        dtype=float,
    )

    return Market(
        spot=spot,
        times=times,
        carry=carry,
        discount=discount,
        quotes=quotes,
    )


def build_grid_spec(market: Market) -> GridSpec:
    normalized_strikes = []
    for quote in market.quotes:
        t_idx = market.time_index(quote.maturity)
        normalized_strikes.append(quote.strike / market.carry[t_idx])

    normalized_spot = market.spot / market.carry[0]
    x_min_market = min(normalized_strikes)
    x_max_market = max(normalized_strikes)
    buffer = X_BUFFER_FRACTION_OF_SPOT * normalized_spot
    x_min = max(
        0.0,
        min(normalized_spot, x_min_market) - buffer,
    )
    x_max = (
        max(normalized_spot, x_max_market)
        + buffer
    )
    x_grid = np.linspace(
        x_min,
        x_max,
        N_X,
    )

    return GridSpec(
        x_grid=x_grid,
        n_integral=N_INTEGRAL,
        max_variables=MAX_VARIABLES,
    )


def _small_result_summary(result: SolveResult) -> dict:
    """Save bounds and diagnostics without the large optimizer flow arrays."""
    return {
        "lower": result.lower,
        "upper": result.upper,
        "n_variables": result.n_variables,
        "n_equalities": result.n_equalities,
        "n_inequalities": result.n_inequalities,
        "lower_diagnostics": result.lower_diagnostics,
        "upper_diagnostics": result.upper_diagnostics,
        "timings": result.timings,
    }


def main() -> None:
    chain, term = load_processed_market_data()
    market = build_market(chain, term)
    option = AsianOption(strike=market.spot, kind=ASIAN_KIND)
    spec = build_grid_spec(market)

    print(f"SPX Asian {option.kind}, {OBSERVATION_DATE} to {EXPIRATIONS[-1]}")
    print(f"Spot/strike {market.spot:.4f}, {len(market.quotes)} quotes, "
          f"grid {len(spec.x_grid)} x {spec.n_integral}", flush=True)
    result = solve_asian_bounds(market=market, option=option, spec=spec)
    summary = _small_result_summary(result)
    width = result.upper - result.lower
    residual = max(result.lower_diagnostics["max_equality_residual"],
                   result.upper_diagnostics["max_equality_residual"])
    violation = max(result.lower_diagnostics["max_inequality_violation"],
                    result.upper_diagnostics["max_inequality_violation"])
    print(f"Lower {result.lower:.6f}, upper {result.upper:.6f}, width {width:.6f}")
    print(f"Residual {residual:.2e}, violation {violation:.2e}, "
          f"runtime {result.timings['total']:.1f}s")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "observation_date": OBSERVATION_DATE.isoformat(),
        "expirations": [expiry.isoformat() for expiry in EXPIRATIONS],
        "spot": market.spot,
        "asian_strike": option.strike,
        "asian_kind": option.kind,
        "n_quotes": len(market.quotes),
        "n_x": len(spec.x_grid),
        "n_integral": spec.n_integral,
        "x_min": float(spec.x_grid[0]),
        "x_max": float(spec.x_grid[-1]),
        "solver_result": summary,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(RESULTS_PATH)


if __name__ == "__main__":
    main()
