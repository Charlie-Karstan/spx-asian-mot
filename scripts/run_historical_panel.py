"""Monthly ATM Asian bounds from the offline SPX archive."""

import hashlib
import json
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning

from market_data.cleaning import clean_spxw_chain
from market_data.transform import build_market_from_snapshot
from robust_pricing.asian.solver import solve_asian_bounds
from robust_pricing.asian.types import AsianOption
from scripts.run_convergence import build_parameterized_grid_spec

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/archive/marketdata"
RESULTS = ROOT / "results/spx_mot"
TARGET_DTES = (30, 60, 90, 120)
MONTHS = tuple(str(m) for m in pd.period_range("2026-01", "2026-09", freq="M"))
N_X, N_INTEGRAL, BUFFER = 52, 21, 0.08
RESIDUAL_TOLERANCE = 1e-6
SOLVER_OPTIONS = {
    "time_limit": 180.0, "run_crossover": "choose",
    "primal_feasibility_tolerance": 1e-8,
    "dual_feasibility_tolerance": 1e-8, "ipm_optimality_tolerance": 1e-9,
}
COLUMNS = [
    "month", "observation_date", "spot", "strike",
    *[f"expiration_{d}" for d in TARGET_DTES],
    *[f"actual_dte_{d}" for d in TARGET_DTES],
    "n_quotes", "lower", "upper", "width", "lower_over_spot", "upper_over_spot",
    "width_over_spot", "width_pct_spot", "n_x", "n_integral", "runtime_seconds",
    "max_equality_residual_lower", "max_equality_residual_upper",
    "max_inequality_violation_lower", "max_inequality_violation_upper",
    "terminal_discount", "terminal_carry", "status",
]


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".pending.json")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def input_hashes(day: Path) -> dict:
    hashes = {}
    for dte in TARGET_DTES:
        path = day / f"dte_{dte:03d}.csv"
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None

    return hashes


def load_snapshot(day: Path):
    """Build one market from four archived chains with distinct, ordered expiries."""
    frames, expirations = [], {}
    for dte in TARGET_DTES:
        frame = pd.read_csv(day / f"dte_{dte:03d}.csv")
        dates = pd.to_datetime(frame["observationDate"]).dt.strftime("%Y-%m-%d")

        if frame.empty or not frame["requestedDTE"].eq(dte).all() or not dates.eq(day.name).all():
            raise ValueError(f"Invalid archived chain for DTE {dte}")

        expiry = pd.to_datetime(frame["expiration"], utc=True).dt.tz_convert(None).dt.normalize()
        if expiry.isna().any() or expiry.nunique() != 1:
            raise ValueError("Expected one expiration per bucket")

        expirations[dte] = expiry.iloc[0]
        frames.append(frame)

    if list(expirations.values()) != sorted(set(expirations.values())):
        raise ValueError("Expected four ordered, distinct expirations")

    chain = clean_spxw_chain(pd.concat(frames, ignore_index=True))
    market, _ = build_market_from_snapshot(chain)

    if len(market.times) != 5 or set(chain.expiration) != set(expirations.values()):
        raise ValueError("A maturity was lost during cleaning")
    if not np.isfinite(np.r_[market.spot, market.carry, market.discount]).all():
        raise ValueError("Nonfinite market inputs")

    return market, expirations


def validate_row(row):
    spot, lower, upper = row["spot"], row["lower"], row["upper"]
    if not np.isfinite([spot, lower, upper]).all() or not spot > 0 or not 0 <= lower <= upper:
        raise ValueError("Invalid bounds")
    contract = (row["status"], row["strike"], row["n_x"], row["n_integral"])
    if contract != ("success", spot, N_X, N_INTEGRAL):
        raise ValueError("Wrong contract or grid")

    width = upper - lower
    np.testing.assert_allclose(
        [row["width"], row["lower_over_spot"], row["upper_over_spot"],
         row["width_over_spot"], row["width_pct_spot"]],
        [width, lower / spot, upper / spot, width / spot, 100 * width / spot],
        rtol=1e-10, atol=1e-10,
    )
    residuals = [row[c] for c in COLUMNS if "residual" in c or "violation" in c]

    if not np.isfinite(residuals).all() or min(residuals) < 0 or max(residuals) > RESIDUAL_TOLERANCE:
        raise ValueError("Solver residual exceeds tolerance")

    date = pd.Timestamp(row["observation_date"])
    days = [(pd.Timestamp(row[f"expiration_{d}"]) - date).days for d in TARGET_DTES]

    if str(date.to_period("M")) != row["month"] or days != sorted(set(days)) or min(days) <= 0:
        raise ValueError("Invalid monitoring dates")
    if days != [row[f"actual_dte_{d}"] for d in TARGET_DTES]:
        raise ValueError("Wrong actual DTEs")


def cached_row(path, hashes, identity, date):
    """Reuse only a validated row with matching input files and solver settings."""
    try:
        saved = json.loads(path.read_text())
        if saved["settings_sha256"] != identity or saved["input_sha256"] != hashes:
            return None

        row = saved["row"]
        validate_row(row)

        return row if row["observation_date"] == date else None
    except (OSError, ValueError, KeyError, TypeError, AssertionError):
        return None


def solve_snapshot(day: Path) -> dict:
    started = perf_counter()
    market, expirations = load_snapshot(day)
    spec = build_parameterized_grid_spec(
        market=market, n_x=N_X, n_a=N_INTEGRAL, buffer_fraction=BUFFER,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=OptimizeWarning,
                                message="Unrecognized options detected:.*run_crossover.*")
        result = solve_asian_bounds(
            market=market, option=AsianOption(strike=market.spot),
            spec=spec, solver_options=SOLVER_OPTIONS,
        )

    lower_diag, upper_diag = result.lower_diagnostics, result.upper_diagnostics
    for diagnostic in (lower_diag, upper_diag):
        minimum_flow = diagnostic["minimum_flow"]
        initial_outflow = diagnostic["total_initial_outflow"]

        if minimum_flow < -RESIDUAL_TOLERANCE or abs(initial_outflow - 1) > RESIDUAL_TOLERANCE:
            raise ValueError("Invalid probability flow")

    width = result.upper - result.lower
    row = {
        "month": day.name[:7], "observation_date": day.name,
        "spot": market.spot, "strike": market.spot,
    }

    for dte, expiration in expirations.items():
        row[f"expiration_{dte}"] = expiration.strftime("%Y-%m-%d")
        row[f"actual_dte_{dte}"] = (expiration - pd.Timestamp(day.name)).days

    row.update({
        "n_quotes": len(market.quotes), "lower": result.lower, "upper": result.upper, "width": width,
        "lower_over_spot": result.lower / market.spot, "upper_over_spot": result.upper / market.spot,
        "width_over_spot": width / market.spot, "width_pct_spot": 100 * width / market.spot,
        "n_x": N_X, "n_integral": N_INTEGRAL, "runtime_seconds": perf_counter() - started,
        "max_equality_residual_lower": lower_diag["max_equality_residual"],
        "max_equality_residual_upper": upper_diag["max_equality_residual"],
        "max_inequality_violation_lower": lower_diag["max_inequality_violation"],
        "max_inequality_violation_upper": upper_diag["max_inequality_violation"],
        "terminal_discount": float(market.discount[-1]), "terminal_carry": float(market.carry[-1]),
        "status": "success",
    })
    validate_row(row)

    return row


def save_progress(rows, metadata):
    frame = pd.DataFrame(rows, columns=COLUMNS)
    widths = frame["width_over_spot"]
    metadata["statistics"] = {"successful_months": len(frame)}

    for name in ("mean", "median", "min", "max", "std"):
        value = widths.agg(name)
        metadata["statistics"][f"{name}_width_over_spot"] = float(value) if pd.notna(value) else None

    metadata["updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    write_json(RESULTS / "historical_panel_metadata.json", metadata)
    temporary = RESULTS / "historical_panel.pending.csv"
    frame.to_csv(temporary, index=False)
    temporary.replace(RESULTS / "historical_panel.csv")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    cache_dir = RESULTS / "historical"
    cache_dir.mkdir(exist_ok=True)
    path = RESULTS / "historical_panel_metadata.json"
    metadata = json.loads(path.read_text()) if path.exists() else {}
    settings = {
        "months": list(MONTHS), "target_dtes": list(TARGET_DTES),
        "n_x": N_X, "n_integral": N_INTEGRAL,
        "x_buffer_fraction_of_spot": BUFFER, "max_variables": 500_000,
        "solver": "Existing solve_asian_bounds / HiGHS IPM",
        "solver_options": SOLVER_OPTIONS,
        "residual_tolerance": RESIDUAL_TOLERANCE,
        "asian_kind": "call", "strike": "spot",
        "average": "Trapezoidal arithmetic average on observation date and four actual expirations",
    }
    sources = list((ROOT / "robust_pricing/asian").glob("*.py"))
    sources += [ROOT / "market_data" / name for name in ("cleaning.py", "parity.py", "transform.py")]
    sources.append(ROOT / "scripts/run_convergence.py")
    sources.append(Path(__file__))
    source_hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sources
    }

    same_sources = metadata.get("source_sha256", {}) == source_hashes
    if metadata.get("settings") != settings or not same_sources:
        identity = hashlib.sha256(json.dumps([settings, source_hashes], sort_keys=True).encode()).hexdigest()
        metadata.update(settings=settings, source_sha256=source_hashes, settings_sha256=identity)

    identity = metadata["settings_sha256"]
    metadata.update(selected_dates={}, unusable_months={}, offline=True,
                    archive_root=str(ARCHIVE.relative_to(ROOT)),
                    selection_rule=("Latest archived date in each month with valid data "
                                    "and a successful fixed-grid solve."))
    attempts = metadata.setdefault("attempts", [])
    failed = {a["observation_date"]: a for a in attempts
              if a["status"] == "failed" and a["settings_sha256"] == identity}
    rows = []

    for month in MONTHS:
        for day in sorted((p for p in ARCHIVE.glob(month + "-*") if p.is_dir()), reverse=True):
            hashes = input_hashes(day)
            path = cache_dir / f"{day.name}.json"
            row = cached_row(path, hashes, identity, day.name)
            if row is None and day.name in failed and failed[day.name]["input_sha256"] == hashes:
                continue

            attempt = {"month": month, "observation_date": day.name,
                       "input_sha256": hashes, "settings_sha256": identity}
            try:
                if None in hashes.values():
                    raise ValueError("Missing archived chains")

                reused = row is not None

                if not reused:
                    print(f"Solving {day.name}", flush=True)
                    row = solve_snapshot(day)
                    write_json(path, {"settings_sha256": identity, "input_sha256": hashes,
                                      "settings": settings, "row": row})

                rows.append(row)
                metadata["selected_dates"][month] = day.name
                attempt.update(status="success", reused=reused)
            except (ValueError, RuntimeError, KeyError, TypeError, OSError, AssertionError) as error:
                attempt.update(status="failed", reason=str(error))
                print(f"{day.name}: {error}", flush=True)

            attempts.append(attempt)
            save_progress(rows, metadata)

            if attempt["status"] == "success":
                break
        else:
            metadata["unusable_months"][month] = "No usable archived snapshot"
            print(f"{month}: no usable snapshot", flush=True)
            save_progress(rows, metadata)

    save_progress(rows, metadata)
    summary = pd.DataFrame(rows, columns=COLUMNS)[["observation_date", "width_pct_spot"]]
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
