from __future__ import annotations

import numpy as np
import pandas as pd

from robust_pricing.asian.types import Market, VanillaQuote
from .parity import ParityPoint, infer_term_structure


def dataframe_to_vanilla_quotes(chain: pd.DataFrame) -> tuple[VanillaQuote, ...]:
    return tuple(
        VanillaQuote(
            maturity=float(row.maturity),
            strike=float(row.strike),
            bid=float(row.bid),
            ask=float(row.ask),
            kind=str(row.option_type),
        )
        for row in chain.itertuples(index=False)
    )


def build_market_from_snapshot(
    chain: pd.DataFrame,
) -> tuple[Market, list[ParityPoint]]:
    spot, parity_points = infer_term_structure(chain)
    parity_by_maturity = {
        point.maturity: point
        for point in parity_points
    }
    maturities = sorted(chain["maturity"].unique())
    missing = [
        maturity
        for maturity in maturities
        if maturity not in parity_by_maturity
    ]

    if missing:
        raise ValueError(f"Missing term-structure estimates for maturities: {missing}")

    times = np.asarray([0.0, *maturities], dtype=float)
    carry = np.ones_like(times)
    discount = np.ones_like(times)

    for idx, maturity in enumerate(maturities, start=1):
        point = parity_by_maturity[maturity]
        carry[idx] = point.carry
        discount[idx] = point.discount

    market = Market(
        spot=spot,
        times=times,
        carry=carry,
        discount=discount,
        quotes=dataframe_to_vanilla_quotes(chain),
    )

    return market, parity_points
