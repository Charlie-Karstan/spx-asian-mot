from __future__ import annotations

from pathlib import Path

import pandas as pd

from market_data import (
    build_market_from_snapshot,
    clean_spxw_chain,
)
from market_data.marketdata_app import download_spxw_snapshot

OBSERVATION_DATE = "2026-09-25"
EXPIRATIONS = [
    "2026-10-16",
    "2026-10-30",
    "2026-11-20",
    "2026-12-18",
]
STRIKE_LIMIT = 40


def main() -> None:
    raw = download_spxw_snapshot(
        observation_date=OBSERVATION_DATE,
        expirations=EXPIRATIONS,
        strike_limit=STRIKE_LIMIT,
    )
    clean = clean_spxw_chain(raw)
    market, parity_points = build_market_from_snapshot(clean)
    output_dir = Path("data/processed/marketdata") / OBSERVATION_DATE
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_csv = output_dir / "SPXW_clean.csv"
    clean.to_csv(clean_csv, index=False)

    try:
        clean.to_parquet(output_dir / "SPXW_clean.parquet", index=False)
    except ImportError:
        pass

    term_structure = pd.DataFrame(
        [
            {
                "maturity": point.maturity,
                "expiration": point.expiration,
                "discount": point.discount,
                "forward": point.forward,
                "carry": point.carry,
                "n_pairs": point.n_pairs,
                "parity_rmse": point.rmse,
            }
            for point in parity_points
        ]
    )

    term_structure.to_csv(output_dir / "term_structure.csv", index=False)

    print(f"SPXW {OBSERVATION_DATE}: spot {market.spot:.4f}, "
          f"{len(EXPIRATIONS)} expirations, {len(market.quotes)} quotes")
    print(term_structure.to_string(index=False))
    print(clean_csv)


if __name__ == "__main__":
    main()
