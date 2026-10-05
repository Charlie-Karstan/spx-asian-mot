import argparse
from pathlib import Path
import time

import pandas as pd
from marketdata import MarketDataClient

START_DATE = "2026-09-01"
END_DATE = "2026-09-25"
TARGET_DTES = [14, 30, 45, 60, 90, 120, 180]
STRIKE_LIMIT = 60
ARCHIVE_ROOT = Path("data/archive/marketdata")


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive historical PM-settled SPX chains.")
    parser.add_argument("--start-date", default=START_DATE, help="First observation date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=END_DATE, help="Last observation date (YYYY-MM-DD)")
    args = parser.parse_args()

    client = MarketDataClient()
    dates = pd.bdate_range(
        args.start_date,
        args.end_date,
    )
    log = []

    for date in dates:
        observation_date = date.strftime("%Y-%m-%d")
        date_dir = ARCHIVE_ROOT / observation_date
        date_dir.mkdir(parents=True, exist_ok=True)

        for dte in TARGET_DTES:
            output_path = date_dir / f"dte_{dte:03d}.csv"
            if output_path.exists():
                continue

            try:
                chain = client.options.chain(
                    "SPX",
                    date=observation_date,
                    dte=dte,
                    pm=True,
                    strike_limit=STRIKE_LIMIT,
                )
                if not isinstance(chain, pd.DataFrame):
                    raise RuntimeError(str(chain))
                if chain.empty:
                    raise RuntimeError("Empty chain")

                chain = chain.reset_index()
                chain["observationDate"] = observation_date
                chain["requestedDTE"] = dte

                chain.to_csv(output_path, index=False)

                expiration = chain["expiration"].iloc[0]
                spot = chain["underlyingPrice"].median()

                print(
                    f"{observation_date}, DTE {dte}: {len(chain)} rows | "
                    f"expiry={expiration} | "
                    f"spot={spot:.2f}"
                )

                log.append({
                    "date": observation_date,
                    "requested_dte": dte,
                    "expiration": expiration,
                    "rows": len(chain),
                    "spot": spot,
                    "status": "success",
                })

            except Exception as error:
                print(f"{observation_date}, DTE {dte}: {error}")

                log.append({
                    "date": observation_date,
                    "requested_dte": dte,
                    "expiration": None,
                    "rows": 0,
                    "spot": None,
                    "status": f"failed: {error}",
                })

            time.sleep(0.15)

    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(log).to_csv(
        ARCHIVE_ROOT / "archive_log.csv",
        index=False,
    )

    print("\nArchive complete.")


if __name__ == "__main__":
    main()
