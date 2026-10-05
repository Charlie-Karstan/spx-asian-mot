from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import GridSpec, Market


@dataclass
class ProjectionLayer:
    """Projection of exact next integral values onto the next A-grid."""

    lower_index: np.ndarray
    upper_index: np.ndarray
    lower_weight: np.ndarray
    upper_weight: np.ndarray
    exact_integral: np.ndarray


@dataclass
class AugmentedGrid:
    times: np.ndarray
    x_grid: np.ndarray
    spot_grids: list[np.ndarray]
    integral_grids: list[np.ndarray]
    projections: list[ProjectionLayer]
    variable_offsets: np.ndarray
    n_variables: int
    start_mass: np.ndarray

    @property
    def n_times(self) -> int:
        return len(self.times)

    @property
    def n_x(self) -> int:
        return len(self.x_grid)

    def n_a(self, t: int) -> int:
        return len(self.integral_grids[t])

    def edge_index(self, t: int, a_idx: int, x_idx: int, next_x_idx: int) -> int:
        """Flatten a transition's (A, X, next X) indices into an LP column."""
        local = (a_idx * self.n_x + x_idx) * self.n_x + next_x_idx

        return int(self.variable_offsets[t] + local)

    def transition_slice(self, t: int) -> slice:
        return slice(int(self.variable_offsets[t]), int(self.variable_offsets[t + 1]))


def barycentric_projection(value: float, grid: np.ndarray, atol: float = 1e-12):
    """
    Represent value by weights on its two neighbouring grid points.

    Returns (lower_index, upper_index, lower_weight, upper_weight).
    Weights sum to one and preserve the value linearly.
    """
    if len(grid) == 1:
        if not np.isclose(value, grid[0], atol=atol, rtol=0.0):
            raise ValueError("value lies outside singleton grid.")

        return 0, 0, 1.0, 0.0
    if value < grid[0] - atol or value > grid[-1] + atol:
        raise ValueError(
            f"value {value:.12g} lies outside grid [{grid[0]:.12g}, {grid[-1]:.12g}]."
        )
    if value <= grid[0] + atol:
        return 0, 0, 1.0, 0.0
    if value >= grid[-1] - atol:
        last = len(grid) - 1
        return last, last, 1.0, 0.0

    upper = int(np.searchsorted(grid, value, side="right"))
    lower = upper - 1
    lower_value = grid[lower]
    upper_value = grid[upper]

    if np.isclose(value, lower_value, atol=atol, rtol=0.0):
        return lower, lower, 1.0, 0.0
    if np.isclose(value, upper_value, atol=atol, rtol=0.0):
        return upper, upper, 1.0, 0.0

    upper_weight = (value - lower_value) / (upper_value - lower_value)
    lower_weight = 1.0 - upper_weight

    return lower, upper, lower_weight, upper_weight


def _start_mass(spot: float, carry0: float, x_grid: np.ndarray) -> np.ndarray:
    normalized_spot = spot / carry0
    lower, upper, w_lower, w_upper = barycentric_projection(normalized_spot, x_grid)
    mass = np.zeros(len(x_grid), dtype=float)
    mass[lower] += w_lower
    mass[upper] += w_upper

    return mass


def _integral_bounds(market: Market, x_grid: np.ndarray):
    n_times = len(market.times)
    min_integral = np.zeros(n_times, dtype=float)
    max_integral = np.zeros(n_times, dtype=float)
    spot_grids = [market.carry[t] * x_grid for t in range(n_times)]

    for t in range(n_times - 1):
        dt = market.times[t + 1] - market.times[t]
        min_increment = 0.5 * (spot_grids[t][0] + spot_grids[t + 1][0]) * dt
        max_increment = 0.5 * (spot_grids[t][-1] + spot_grids[t + 1][-1]) * dt
        min_integral[t + 1] = min_integral[t] + min_increment
        max_integral[t + 1] = max_integral[t] + max_increment

    return spot_grids, min_integral, max_integral


def build_augmented_grid(market: Market, spec: GridSpec) -> AugmentedGrid:
    """Construct the finite (X, A) grid and conservative one-step projections.

    X is the normalized martingale price; A accumulates trapezoids of actual S.
    Projection onto neighboring A nodes preserves mass and the first moment.
    """
    x_grid = spec.x_grid
    n_x = len(x_grid)
    n_times = len(market.times)

    for maturity, marginal in market.fixed_marginals.items():
        if len(marginal) != n_x:
            raise ValueError(
                f"fixed marginal at maturity {maturity} has length {len(marginal)}, "
                f"but x_grid has length {n_x}."
            )

    spot_grids, min_integral, max_integral = _integral_bounds(market, x_grid)
    integral_grids: list[np.ndarray] = [np.array([0.0])]

    for t in range(1, n_times):
        integral_grids.append(
            np.linspace(min_integral[t], max_integral[t], spec.n_integral)
        )

    variable_offsets = [0]
    for t in range(n_times - 1):
        n_edges = len(integral_grids[t]) * n_x * n_x
        variable_offsets.append(variable_offsets[-1] + n_edges)

    n_variables = variable_offsets[-1]
    if n_variables > spec.max_variables:
        raise ValueError(
            f"Run interrupted: {n_variables:,} flow variables exceed "
            f"the limit of {spec.max_variables:,}."
        )

    projections: list[ProjectionLayer] = []
    for t in range(n_times - 1):
        current_a = integral_grids[t]
        next_a = integral_grids[t + 1]
        dt = market.times[t + 1] - market.times[t]
        shape = (len(current_a), n_x, n_x)
        lower_index = np.empty(shape, dtype=np.int32)
        upper_index = np.empty(shape, dtype=np.int32)
        lower_weight = np.empty(shape, dtype=float)
        upper_weight = np.empty(shape, dtype=float)
        exact_integral = np.empty(shape, dtype=float)
        current_spot = spot_grids[t]
        next_spot = spot_grids[t + 1]

        for a_idx, accumulated in enumerate(current_a):
            for x_idx, spot_now in enumerate(current_spot):
                for next_x_idx, spot_next in enumerate(next_spot):
                    exact = accumulated + 0.5 * (spot_now + spot_next) * dt
                    lo, hi, w_lo, w_hi = barycentric_projection(exact, next_a)
                    lower_index[a_idx, x_idx, next_x_idx] = lo
                    upper_index[a_idx, x_idx, next_x_idx] = hi
                    lower_weight[a_idx, x_idx, next_x_idx] = w_lo
                    upper_weight[a_idx, x_idx, next_x_idx] = w_hi
                    exact_integral[a_idx, x_idx, next_x_idx] = exact

        projections.append(
            ProjectionLayer(
                lower_index=lower_index,
                upper_index=upper_index,
                lower_weight=lower_weight,
                upper_weight=upper_weight,
                exact_integral=exact_integral,
            )
        )

    return AugmentedGrid(
        times=market.times.copy(),
        x_grid=x_grid.copy(),
        spot_grids=spot_grids,
        integral_grids=integral_grids,
        projections=projections,
        variable_offsets=np.asarray(variable_offsets, dtype=np.int64),
        n_variables=n_variables,
        start_mass=_start_mass(market.spot, market.carry[0], x_grid),
    )
