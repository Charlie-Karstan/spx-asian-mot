"""Compare nested vanilla information sets on the unchanged SPX Asian grid.

Run from the repository root with python -m scripts.run_information_experiment.
Only vanilla constraints are removed; monitoring dates, carry, discounts and
the spatial/integral grids remain fixed across all three solves.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from robust_pricing.asian.solver import compile_asian_lp, _diagnostics
from robust_pricing.asian.types import AsianOption, GridSpec, Market, SolveResult
from scripts.run_spx_mot import (
    ASIAN_KIND, CHAIN_PATH, EXPIRATIONS, OBSERVATION_DATE, RESULTS_DIR,
    RESULTS_PATH, TERM_STRUCTURE_PATH, build_grid_spec, build_market,
    load_processed_market_data,
)


def solve_experiment(market: Market, option: AsianOption, spec: GridSpec) -> SolveResult:
    """Solve the same LP in scaled units, then check feasibility in price units."""
    problem = compile_asian_lp(option, market, spec)

    # Scale rows and the objective to reduce the coefficient range.
    eq_scale = np.maximum(abs(problem.equality_matrix).max(axis=1).toarray().ravel(), 1)
    ub_scale = np.maximum(abs(problem.inequality_matrix).max(axis=1).toarray().ravel(), 1)
    scaled = replace(
        problem, objective=problem.objective / market.spot,
        equality_matrix=problem.equality_matrix.multiply(1 / eq_scale[:, None]).tocsr(),
        equality_targets=problem.equality_targets / eq_scale,
        inequality_matrix=problem.inequality_matrix.multiply(1 / ub_scale[:, None]).tocsr(),
        inequality_targets=problem.inequality_targets / ub_scale,
    )
    started = perf_counter()
    common = dict(A_eq=scaled.equality_matrix, b_eq=scaled.equality_targets,
                  A_ub=scaled.inequality_matrix, b_ub=scaled.inequality_targets,
                  bounds=(0, None), method="highs-ds",
                  options={"disp": False, "time_limit": 180})

    # Check feasibility against the original, unscaled matrices.
    lower = linprog(scaled.objective, **common)
    if not lower.success:
        raise RuntimeError(f"Lower-bound optimization failed: {lower.message}")

    upper = linprog(-scaled.objective, **common)
    if not upper.success:
        raise RuntimeError(f"Upper-bound optimization failed: {upper.message}")

    return SolveResult(
        lower=float(problem.objective @ lower.x), upper=float(problem.objective @ upper.x),
        lower_flows=lower.x[:problem.grid.n_variables], upper_flows=upper.x[:problem.grid.n_variables],
        n_variables=len(problem.objective), n_equalities=len(problem.equality_targets),
        n_inequalities=len(problem.inequality_targets),
        lower_diagnostics=_diagnostics(problem, lower.x), upper_diagnostics=_diagnostics(problem, upper.x),
        timings={"total": problem.timings["compile_total"] + perf_counter() - started},
    )


def main() -> None:
    chain, term = load_processed_market_data()
    market = build_market(chain, term)
    option = AsianOption(strike=market.spot, kind=ASIAN_KIND)
    spec = build_grid_spec(market)
    saved = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    expected = {
        "observation_date": OBSERVATION_DATE.isoformat(),
        "expirations": [d.isoformat() for d in EXPIRATIONS],
        "spot": market.spot, "asian_strike": option.strike, "asian_kind": option.kind,
        "n_x": len(spec.x_grid), "n_integral": spec.n_integral,
    }
    matching_contract = all(saved[key] == value for key, value in expected.items())
    matching_support = np.allclose(
        [saved["x_min"], saved["x_max"]], spec.x_grid[[0, -1]], rtol=0, atol=1e-9,
    )
    if not matching_contract or not matching_support:
        raise ValueError("Saved MOT baseline does not match the current contract/grid.")

    selections = {"full": (1, 2, 3, 4), "reduced": (2, 4), "final_only": (4,)}
    contract = {
        "observation_date": OBSERVATION_DATE.isoformat(),
        "expirations": [d.isoformat() for d in EXPIRATIONS],
        "spot": market.spot, "asian_strike": option.strike, "asian_kind": option.kind,
        "times": market.times.tolist(), "carry": market.carry.tolist(),
        "discount": market.discount.tolist(), "terminal_discount": float(market.discount[-1]),
        "average_definition": "sum(0.5*(S[k]+S[k+1])*dt[k]) / (times[-1]-times[0])",
    }
    payload = {
        "contract": contract,
        "grid": {"n_x": len(spec.x_grid), "n_integral": spec.n_integral,
                 "x_min": float(spec.x_grid[0]), "x_max": float(spec.x_grid[-1])},
        "input_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (CHAIN_PATH, TERM_STRUCTURE_PATH, RESULTS_PATH)},
        "experiments": [],
        "numerical_scaling": "Positive row normalization; objective in spot units; HiGHS dual simplex; identical LP",
    }
    rows = []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, indices in selections.items():
        maturities = market.times[list(indices)]
        selected = tuple(q for q in market.quotes if q.maturity in maturities)
        experiment_market = replace(market, quotes=selected)
        print(f"Solving {name}", flush=True)
        result = solve_experiment(experiment_market, option, spec)
        residual = max(d["max_equality_residual"] for d in
                       (result.lower_diagnostics, result.upper_diagnostics))
        violation = max(d["max_inequality_violation"] for d in
                        (result.lower_diagnostics, result.upper_diagnostics))

        valid_bounds = (np.isfinite([result.lower, result.upper]).all()
                        and result.lower >= -1e-7 and result.upper >= result.lower - 1e-7)
        if not valid_bounds or residual > 1e-7 or violation > 1e-7:
            raise RuntimeError(f"{name}: invalid bounds or infeasible optimizer output.")
        if name == "full":
            original = saved["solver_result"]
            if not np.allclose([result.lower, result.upper], [original["lower"], original["upper"]],
                               rtol=0, atol=1e-6):
                raise RuntimeError("Full-information solve did not reproduce the saved MOT bounds.")

            payload["baseline_reproduced"] = True

        row = {
            "name": name, "n_maturities": len(indices), "n_quotes": len(selected),
            "vanilla_maturities": json.dumps(maturities.tolist()),
            "lower": result.lower, "upper": result.upper,
            "width": result.upper - result.lower, "status": "success",
            "n_x": len(spec.x_grid), "n_integral": spec.n_integral,
            "x_min": float(spec.x_grid[0]), "x_max": float(spec.x_grid[-1]),
            "total_time": result.timings["total"],
            "max_eq_residual": residual, "max_ineq_violation": violation,
        }
        rows.append(row)
        payload["experiments"].append(row)
        pd.DataFrame(rows).to_csv(RESULTS_DIR / "information_experiment.csv", index=False)
        (RESULTS_DIR / "information_experiment.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    widths = [row["width"] for row in rows]
    if np.any(np.diff(widths) < -0.01):
        raise RuntimeError(f"Information widths violate nesting: {widths}")

    print(pd.DataFrame(rows)[["name", "lower", "upper", "width"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
