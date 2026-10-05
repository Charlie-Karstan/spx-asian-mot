from __future__ import annotations

import numpy as np
import pandas as pd

_REQUIRED = {
    "observationDate",
    "optionSymbol",
    "expiration",
    "side",
    "strike",
    "bid",
    "ask",
    "underlyingPrice",
}
ACTUAL_DAYS_PER_YEAR = 365.0


def clean_spxw_chain(
    df: pd.DataFrame,
    *,
    max_relative_spread: float | None = None,
    min_open_interest: int | None = None,
) -> pd.DataFrame:
    """
    Convert a raw MarketData.app SPX chain into the canonical market schema.

    Zero bids are retained because they are legitimate lower bounds.
    Crossed/invalid markets are removed.
    """
    missing = _REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(f"Missing MarketData columns: {sorted(missing)}")

    out = df.copy()
    out["quote_date"] = pd.to_datetime(out["observationDate"]).dt.normalize()
    out["expiration"] = pd.to_datetime(out["expiration"], utc=True).dt.tz_convert(None).dt.normalize()

    if out["quote_date"].nunique() != 1:
        raise ValueError("A research snapshot must contain exactly one observation date.")

    numeric_columns = [
        "strike",
        "bid",
        "ask",
        "mid",
        "volume",
        "openInterest",
        "underlyingPrice",
    ]
    for column in numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    out["option_type"] = out["side"].astype(str).str.lower()
    out = out[out["option_type"].isin(["call", "put"])].copy()
    out = out[out["expiration"] > out["quote_date"]].copy()
    out = out[np.isfinite(out["strike"]) & (out["strike"] > 0)].copy()
    out = out[np.isfinite(out["bid"]) & np.isfinite(out["ask"])].copy()
    valid_bid_ask = (out["bid"] >= 0) & (out["ask"] > 0) & (out["bid"] <= out["ask"])
    out = out.loc[valid_bid_ask].copy()
    out = out[np.isfinite(out["underlyingPrice"]) & (out["underlyingPrice"] > 0)].copy()
    out["mid"] = 0.5 * (out["bid"] + out["ask"])
    out["spread"] = out["ask"] - out["bid"]
    out["relative_spread"] = out["spread"] / out["mid"].replace(0.0, np.nan)

    if max_relative_spread is not None:
        out = out[
            out["relative_spread"].isna()
            | (out["relative_spread"] <= max_relative_spread)
        ].copy()
    if min_open_interest is not None and "openInterest" in out.columns:
        out = out[out["openInterest"].fillna(0) >= min_open_interest].copy()

    duplicate_keys = ["expiration", "strike", "option_type"]
    duplicated = out.duplicated(duplicate_keys, keep=False)

    if duplicated.any():
        out = (
            out.sort_values("spread")
            .drop_duplicates(duplicate_keys, keep="first")
            .copy()
        )
    if out.empty:
        raise ValueError("No option quotes remain after cleaning.")

    quote_date = out["quote_date"].iloc[0]
    out["maturity"] = (out["expiration"] - quote_date).dt.days / ACTUAL_DAYS_PER_YEAR
    out = out.rename(
        columns={
            "optionSymbol": "option_symbol",
            "openInterest": "open_interest",
            "underlyingPrice": "underlying_price",
        }
    )
    preferred = [
        "quote_date",
        "expiration",
        "maturity",
        "option_symbol",
        "option_type",
        "strike",
        "bid",
        "ask",
        "mid",
        "spread",
        "relative_spread",
        "volume",
        "open_interest",
        "underlying_price",
    ]
    existing = [column for column in preferred if column in out.columns]

    return (
        out[existing]
        .sort_values(["expiration", "strike", "option_type"])
        .reset_index(drop=True)
    )
