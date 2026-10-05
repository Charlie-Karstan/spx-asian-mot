from __future__ import annotations

from pathlib import Path

import pandas as pd
from marketdata import MarketDataClient


def download_spxw_snapshot(
    observation_date: str,
    expirations: list[str] | tuple[str, ...],
    *,
    strike_limit: int = 40,
    raw_root: str | Path = "data/raw/marketdata",
) -> pd.DataFrame:
    """
    Download historical PM-settled SPX option chains from MarketData.app.

    One request is made per expiration. The returned DataFrames are saved
    separately and combined into one raw snapshot.
    """
    client = MarketDataClient()
    snapshot_dir = Path(raw_root) / observation_date
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []

    for expiration in expirations:
        chain = client.options.chain(
            "SPX",
            date=observation_date,
            expiration=expiration,
            pm=True,
            strike_limit=strike_limit,
        )
        if not isinstance(chain, pd.DataFrame):
            raise RuntimeError(
                f"MarketData request failed for expiration {expiration}: {chain}"
            )
        if chain.empty:
            raise RuntimeError(
                f"MarketData returned an empty chain for expiration {expiration}."
            )

        chain = chain.reset_index()
        chain["observationDate"] = observation_date
        raw_path = snapshot_dir / f"SPXW_{expiration}.csv"
        chain.to_csv(raw_path, index=False)

        print(f"{expiration}: {len(chain)} rows saved to {raw_path}")

        frames.append(chain)

    combined = pd.concat(frames, ignore_index=True)
    combined_path = snapshot_dir / "SPXW_snapshot.csv"
    combined.to_csv(combined_path, index=False)

    return combined
