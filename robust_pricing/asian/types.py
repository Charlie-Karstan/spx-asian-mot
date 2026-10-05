from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

OptionKind = Literal["call", "put"]


@dataclass(frozen=True)
class AsianOption:
    """Fixed-strike arithmetic Asian on the market's trapezoidal monitoring grid."""

    strike: float
    kind: OptionKind = "call"

    def __post_init__(self) -> None:
        if not np.isfinite(self.strike) or self.strike < 0:
            raise ValueError("strike must be finite and non-negative.")
        if self.kind not in ("call", "put"):
            raise ValueError("kind must be 'call' or 'put'.")


@dataclass(frozen=True)
class VanillaQuote:
    """European vanilla bid/ask quote used as a calibration inequality."""

    maturity: float
    strike: float
    bid: float
    ask: float
    kind: OptionKind = "call"

    def __post_init__(self) -> None:
        if not np.isfinite(self.maturity) or self.maturity < 0:
            raise ValueError("maturity must be finite and non-negative.")
        if not np.isfinite(self.strike) or self.strike < 0:
            raise ValueError("strike must be finite and non-negative.")
        if not np.isfinite([self.bid, self.ask]).all() or self.bid < 0 or self.ask < 0:
            raise ValueError("bid and ask must be finite and non-negative.")
        if self.bid > self.ask:
            raise ValueError("bid cannot exceed ask.")
        if self.kind not in ("call", "put"):
            raise ValueError("kind must be 'call' or 'put'.")


@dataclass
class Market:
    """
    Deterministic carry/discount market inputs.

    The optimisation state X_t is a martingale. Actual spot is represented as

        S_t = carry[t] * X_t.

    This allows deterministic rates/dividends while keeping the martingale
    constraint linear in X.
    """

    spot: float
    times: np.ndarray
    carry: np.ndarray | None = None
    discount: np.ndarray | None = None
    quotes: tuple[VanillaQuote, ...] = ()
    fixed_marginals: dict[float, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=float)
        if self.times.ndim != 1 or len(self.times) < 2:
            raise ValueError("times must be a one-dimensional array with at least two entries.")
        if not np.isfinite(self.times).all():
            raise ValueError("times must be finite.")
        if not np.all(np.diff(self.times) > 0):
            raise ValueError("times must be strictly increasing.")
        if not np.isclose(self.times[0], 0.0):
            raise ValueError("times must currently start at 0.")
        if not np.isfinite(self.spot) or self.spot <= 0:
            raise ValueError("spot must be finite and positive.")

        n_times = len(self.times)
        if self.carry is None:
            self.carry = np.ones(n_times, dtype=float)
        else:
            self.carry = np.asarray(self.carry, dtype=float)

        if self.discount is None:
            self.discount = np.ones(n_times, dtype=float)
        else:
            self.discount = np.asarray(self.discount, dtype=float)

        if self.carry.shape != (n_times,):
            raise ValueError("carry must have the same length as times.")
        if self.discount.shape != (n_times,):
            raise ValueError("discount must have the same length as times.")
        if not np.isfinite(self.carry).all() or np.any(self.carry <= 0):
            raise ValueError("carry factors must be finite and positive.")
        if not np.isfinite(self.discount).all() or np.any(self.discount <= 0):
            raise ValueError("discount factors must be finite and positive.")

        for quote in self.quotes:
            self.time_index(quote.maturity)

        cleaned: dict[float, np.ndarray] = {}
        for maturity, probabilities in self.fixed_marginals.items():
            self.time_index(maturity)
            p = np.asarray(probabilities, dtype=float)

            if p.ndim != 1:
                raise ValueError("each fixed marginal must be one-dimensional.")
            if np.any(p < -1e-14):
                raise ValueError("fixed marginal probabilities cannot be negative.")
            if not np.isclose(p.sum(), 1.0, atol=1e-10):
                raise ValueError("each fixed marginal must sum to 1.")

            cleaned[float(maturity)] = p

        self.fixed_marginals = cleaned

    def time_index(self, maturity: float, atol: float = 1e-12) -> int:
        matches = np.flatnonzero(np.isclose(self.times, maturity, atol=atol, rtol=0.0))
        if len(matches) != 1:
            raise ValueError(
                f"maturity {maturity} is not represented exactly once in the numerical time grid."
            )

        return int(matches[0])


@dataclass(frozen=True)
class GridSpec:
    """Finite price and accumulated-integral grids for the monitored Asian LP."""

    x_grid: np.ndarray  # Martingale-normalized prices: X_t = S_t / carry[t].
    n_integral: int = 81  # Nodes for the accumulated integral at each non-initial date.
    max_variables: int = 500_000

    def __post_init__(self) -> None:
        x = np.asarray(self.x_grid, dtype=float)
        object.__setattr__(self, "x_grid", x)

        if x.ndim != 1 or len(x) < 2:
            raise ValueError("x_grid must be a one-dimensional array with at least two points.")
        if not np.isfinite(x).all():
            raise ValueError("x_grid must be finite.")
        if not np.all(np.diff(x) > 0):
            raise ValueError("x_grid must be strictly increasing.")
        if x[0] < 0:
            raise ValueError("x_grid cannot contain negative prices.")
        if self.n_integral < 2:
            raise ValueError("n_integral must be at least 2.")
        if self.max_variables < 1:
            raise ValueError("max_variables must be positive.")


@dataclass
class SolveResult:
    """Bounds and flow-only optimisers; n_variables counts all LP columns."""

    lower: float
    upper: float
    lower_flows: np.ndarray
    upper_flows: np.ndarray
    n_variables: int
    n_equalities: int
    n_inequalities: int
    lower_diagnostics: dict[str, float]
    upper_diagnostics: dict[str, float]
    timings: dict[str, float] = field(default_factory=dict)
