from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_COMBINED_SPREAD = 0.05  # SPX points; caps the weight of very tight pairs.


@dataclass(frozen=True)
class ParityPoint:
    maturity: float
    expiration: pd.Timestamp
    discount: float
    forward: float
    carry: float
    n_pairs: int
    rmse: float


def _paired_quotes(expiry_df: pd.DataFrame) -> pd.DataFrame:
    calls = (
        expiry_df.loc[expiry_df["option_type"] == "call", ["strike", "mid", "spread"]]
        .rename(columns={"mid": "call_mid", "spread": "call_spread"})
    )
    puts = (
        expiry_df.loc[expiry_df["option_type"] == "put", ["strike", "mid", "spread"]]
        .rename(columns={"mid": "put_mid", "spread": "put_spread"})
    )
    pairs = calls.merge(puts, on="strike", how="inner")
    pairs["parity_value"] = pairs["call_mid"] - pairs["put_mid"]
    combined_spread = pairs["call_spread"] + pairs["put_spread"]
    pairs["weight"] = 1.0 / np.maximum(combined_spread.to_numpy(float), MIN_COMBINED_SPREAD)

    return pairs


def _weighted_line_fit(
    strike: np.ndarray,
    parity_value: np.ndarray,
    weight: np.ndarray,
) -> tuple[float, float, float]:
    # C - P = D*F - D*K = intercept + slope*K
    X = np.column_stack([np.ones_like(strike), strike])
    sqrt_w = np.sqrt(weight)
    beta, *_ = np.linalg.lstsq(
        X * sqrt_w[:, None],
        parity_value * sqrt_w,
        rcond=None,
    )
    intercept, slope = beta
    fitted = intercept + slope * strike
    rmse = float(
        np.sqrt(np.average((parity_value - fitted) ** 2, weights=weight))
    )

    return float(intercept), float(slope), rmse


def infer_term_structure(
    chain: pd.DataFrame,
    *,
    min_pairs: int = 8,
) -> tuple[float, list[ParityPoint]]:
    """
    Infer one spot level plus discount/forward/carry at each vanilla maturity.

    Spot is taken from the median historical underlyingPrice supplied with the
    option-chain snapshot.

    Put-call parity:
        C - P = D(T) * (F(T) - K)
    """
    spot = float(chain["underlying_price"].median())
    if not np.isfinite(spot) or spot <= 0:
        raise ValueError("Could not infer a valid SPX spot from underlyingPrice.")

    points: list[ParityPoint] = []
    for expiration, expiry_df in chain.groupby("expiration", sort=True):
        maturity_values = expiry_df["maturity"].unique()
        if len(maturity_values) != 1:
            raise ValueError("Each expiration must map to exactly one maturity.")

        maturity = float(maturity_values[0])
        pairs = _paired_quotes(expiry_df)

        if len(pairs) < min_pairs:
            raise ValueError(
                f"Only {len(pairs)} call/put strike pairs available for {expiration.date()}."
            )

        intercept, slope, rmse = _weighted_line_fit(
            pairs["strike"].to_numpy(float),
            pairs["parity_value"].to_numpy(float),
            pairs["weight"].to_numpy(float),
        )
        discount = -slope

        if not np.isfinite(discount) or discount <= 0:
            raise ValueError(
                f"Invalid discount factor inferred for {expiration.date()}: {discount}"
            )

        forward = intercept / discount
        if not np.isfinite(forward) or forward <= 0:
            raise ValueError(
                f"Invalid forward inferred for {expiration.date()}: {forward}"
            )

        points.append(
            ParityPoint(
                maturity=maturity,
                expiration=pd.Timestamp(expiration),
                discount=float(discount),
                forward=float(forward),
                carry=float(forward / spot),
                n_pairs=len(pairs),
                rmse=rmse,
            )
        )

    return spot, points
