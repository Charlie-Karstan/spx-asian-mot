from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from robust_pricing.asian.solver import solve_asian_bounds
from robust_pricing.asian.types import AsianOption, GridSpec, Market
from scripts.run_spx_mot import (
    OBSERVATION_DATE,
    load_processed_market_data,
    build_market,
)

RESULTS_DIR = Path("results/spx_mot") / OBSERVATION_DATE.isoformat()
OUTPUT_CSV = RESULTS_DIR / "convergence.csv"
BASELINE_JSON = RESULTS_DIR / "mot_bounds.json"
MAX_VARIABLES = 500_000
ASIAN_KIND = "call"

# Baseline, finer integral grid, and finer X grid on the same support.
CONFIGS = [
    {"name": "baseline", "n_x": 50, "n_a": 21, "buffer": 0.08},
    {"name": "a25",      "n_x": 50, "n_a": 25, "buffer": 0.08},
    {"name": "x52",      "n_x": 52, "n_a": 21, "buffer": 0.08},
]


def build_parameterized_grid_spec(
    market: Market,
    n_x: int,
    n_a: int,
    buffer_fraction: float,
) -> GridSpec:
    """Build X support from normalized strikes and a spot-relative buffer."""
    normalized_strikes = []
    for quote in market.quotes:
        t_idx = market.time_index(quote.maturity)
        normalized_strikes.append(quote.strike / market.carry[t_idx])

    normalized_spot = market.spot / market.carry[0]
    x_min_market = min(normalized_strikes)
    x_max_market = max(normalized_strikes)
    buffer = buffer_fraction * normalized_spot
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
        n_x,
    )

    return GridSpec(
        x_grid=x_grid,
        n_integral=n_a,
        max_variables=MAX_VARIABLES,
    )


def result_to_row(
    name: str,
    n_x: int,
    n_a: int,
    buffer: float,
    x_min: float,
    x_max: float,
    result,
    source: str,
) -> dict:
    lower_diag = result.lower_diagnostics
    upper_diag = result.upper_diagnostics
    timings = result.timings

    return {
        "name": name,
        "n_x": n_x,
        "n_integral": n_a,
        "buffer": buffer,
        "x_min": x_min,
        "x_max": x_max,
        "x_spacing": (x_max - x_min) / (n_x - 1),
        "lower": result.lower,
        "upper": result.upper,
        "width": result.upper - result.lower,
        "n_variables": result.n_variables,
        "n_equalities": result.n_equalities,
        "n_inequalities": result.n_inequalities,
        "compile_time": timings.get("compile_total", np.nan),
        "lower_solve_time": timings.get("lower_solve", np.nan),
        "upper_solve_time": timings.get("upper_solve", np.nan),
        "total_time": timings.get("total", np.nan),
        "max_eq_residual": max(
            abs(lower_diag["max_equality_residual"]),
            abs(upper_diag["max_equality_residual"]),
        ),
        "max_ineq_violation": max(
            abs(lower_diag["max_inequality_violation"]),
            abs(upper_diag["max_inequality_violation"]),
        ),
        "source": source,
        "status": "success",
        "error": "",
    }


def existing_baseline_row() -> dict | None:
    """Reuse the saved 50 x 21 baseline with an 8% support buffer."""
    if not BASELINE_JSON.exists():
        return None

    try:
        payload = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
        result = payload.get("solver_result", {})
        required = {"lower", "upper", "n_variables", "n_equalities", "n_inequalities",
                    "lower_diagnostics", "upper_diagnostics", "timings"}

        if not required.issubset(result):
            return None

        timings = result["timings"]
        lower_diag = result["lower_diagnostics"]
        upper_diag = result["upper_diagnostics"]
        width = result["upper"] - result["lower"]

        return {
            "name": "baseline", "n_x": 50, "n_integral": 21, "buffer": 0.08,
            "x_min": payload["x_min"], "x_max": payload["x_max"],
            "x_spacing": (payload["x_max"] - payload["x_min"]) / 49,
            "lower": result["lower"], "upper": result["upper"], "width": width,
            "n_variables": result["n_variables"], "n_equalities": result["n_equalities"],
            "n_inequalities": result["n_inequalities"],
            "compile_time": timings.get("compile_total", np.nan),
            "lower_solve_time": timings.get("lower_solve", np.nan),
            "upper_solve_time": timings.get("upper_solve", np.nan),
            "total_time": timings.get("total", np.nan),
            "max_eq_residual": max(abs(lower_diag["max_equality_residual"]),
                                   abs(upper_diag["max_equality_residual"])),
            "max_ineq_violation": max(abs(lower_diag["max_inequality_violation"]),
                                      abs(upper_diag["max_inequality_violation"])),
            "source": "existing_baseline", "status": "success", "error": "",
        }
    except (OSError, ValueError, KeyError, TypeError):
        return None


def load_existing_rows() -> dict[str, dict]:
    """Resume configurations recorded as successful in the convergence CSV."""
    if not OUTPUT_CSV.exists():
        return {}

    frame = pd.read_csv(OUTPUT_CSV)
    if "name" not in frame.columns:
        return {}

    rows = {}
    for _, row in frame.iterrows():
        if row.get("status") == "success":
            rows[str(row["name"])] = row.to_dict()

    return rows


def save_rows(rows: dict[str, dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ordered = [rows[config["name"]] for config in CONFIGS if config["name"] in rows]
    pd.DataFrame(ordered).to_csv(OUTPUT_CSV, index=False)


def main() -> None:
    chain, term = load_processed_market_data()
    market = build_market(chain, term)
    option = AsianOption(strike=market.spot, kind=ASIAN_KIND)
    rows = load_existing_rows()

    if "baseline" not in rows:
        baseline = existing_baseline_row()
        if baseline is not None:
            rows["baseline"] = baseline
            save_rows(rows)

    print(f"SPX convergence, {OBSERVATION_DATE}, spot/strike {market.spot:.4f}")

    for config in CONFIGS:
        name = config["name"]
        if name in rows:
            print(f"{name}: using saved result")
            continue

        spec = build_parameterized_grid_spec(
            market=market, n_x=config["n_x"], n_a=config["n_a"],
            buffer_fraction=config["buffer"],
        )
        print(f"{name}: grid {config['n_x']} x {config['n_a']}, "
              f"buffer {config['buffer']:.0%}", flush=True)

        try:
            result = solve_asian_bounds(
                market=market, option=option, spec=spec, solver_options={"disp": True},
            )
            row = result_to_row(
                name=name, n_x=config["n_x"], n_a=config["n_a"], buffer=config["buffer"],
                x_min=float(spec.x_grid[0]), x_max=float(spec.x_grid[-1]),
                result=result, source="fresh_solve",
            )
            rows[name] = row
            save_rows(rows)
            print(f"{name}: lower {row['lower']:.6f}, upper {row['upper']:.6f}, "
                  f"width {row['width']:.6f}, runtime {row['total_time']:.1f}s")
        except Exception as error:
            # Record a failed configuration and continue the remaining grid sweep.
            rows[name] = {
                "name": name, "n_x": config["n_x"], "n_integral": config["n_a"],
                "buffer": config["buffer"],
                "x_min": float(spec.x_grid[0]), "x_max": float(spec.x_grid[-1]),
                "x_spacing": (spec.x_grid[-1] - spec.x_grid[0]) / (len(spec.x_grid) - 1),
                "lower": np.nan, "upper": np.nan, "width": np.nan,
                "n_variables": np.nan, "n_equalities": np.nan, "n_inequalities": np.nan,
                "compile_time": np.nan, "lower_solve_time": np.nan,
                "upper_solve_time": np.nan, "total_time": np.nan,
                "max_eq_residual": np.nan, "max_ineq_violation": np.nan,
                "source": "fresh_solve", "status": "failed", "error": str(error),
            }
            save_rows(rows)
            print(f"{name}: {error}")

    frame = pd.read_csv(OUTPUT_CSV)
    columns = ["name", "n_x", "n_integral", "buffer", "lower", "upper", "width",
               "total_time", "status"]
    print(frame[columns].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()
