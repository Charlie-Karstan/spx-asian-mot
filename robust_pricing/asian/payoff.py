from __future__ import annotations

import numpy as np

from .types import AsianOption, OptionKind


def vanilla_payoff(spot: np.ndarray | float, strike: float, kind: OptionKind):
    spot = np.asarray(spot, dtype=float)
    if kind == "call":
        return np.maximum(spot - strike, 0.0)
    if kind == "put":
        return np.maximum(strike - spot, 0.0)

    raise ValueError("kind must be 'call' or 'put'.")


def asian_payoff_from_integral(
    integral: np.ndarray | float,
    horizon: float,
    option: AsianOption,
):
    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    average = np.asarray(integral, dtype=float) / horizon
    if option.kind == "call":
        return np.maximum(average - option.strike, 0.0)
    if option.kind == "put":
        return np.maximum(option.strike - average, 0.0)

    raise ValueError("option kind must be 'call' or 'put'.")
