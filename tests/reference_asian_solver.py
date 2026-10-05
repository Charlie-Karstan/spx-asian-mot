"""Frozen pre-refactor compiler, used only as a regression oracle."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from robust_pricing.asian.grid import AugmentedGrid, build_augmented_grid
from robust_pricing.asian.payoff import asian_payoff_from_integral, vanilla_payoff
from robust_pricing.asian.types import AsianOption, GridSpec, Market


@dataclass
class CompiledLP:
    objective: np.ndarray
    equality_matrix: csr_matrix
    equality_targets: np.ndarray
    inequality_matrix: csr_matrix | None
    inequality_targets: np.ndarray | None
    grid: AugmentedGrid
    quote_labels: list[str]


def _node_number(a_idx: int, x_idx: int, n_x: int) -> int:
    return a_idx * n_x + x_idx


def _conservation_offsets(grid: AugmentedGrid) -> np.ndarray:
    offsets = [0]
    for t in range(grid.n_times - 1):
        offsets.append(offsets[-1] + grid.n_a(t) * grid.n_x)

    return np.asarray(offsets, dtype=np.int64)


def _transition_edge_indices(grid: AugmentedGrid, t: int):
    """Yield (edge_index, a_idx, x_idx, next_x_idx) for transition t -> t+1."""
    for a_idx in range(grid.n_a(t)):
        for x_idx in range(grid.n_x):
            base = grid.edge_index(t, a_idx, x_idx, 0)
            for next_x_idx in range(grid.n_x):
                yield base + next_x_idx, a_idx, x_idx, next_x_idx


def compile_asian_lp(
    option: AsianOption,
    market: Market,
    spec: GridSpec,
) -> CompiledLP:
    """
    Compile the continuous Asian robust-pricing target into one sparse occupancy LP.

    Continuous target:

        inf/sup_Q E_Q[ payoff((1/T) int_0^T S_t dt) ]

    over calibrated martingale measures Q.

    Numerical state:

        (t_k, X_k, A_k),

    where X is the martingale-normalised price and A approximates the running
    integral of actual spot. The A-dynamics use conservative linear projection
    onto the next integral grid rather than nearest-neighbour rounding.
    """
    grid = build_augmented_grid(market, spec)
    n_vars = grid.n_variables
    n_x = grid.n_x
    n_times = grid.n_times

    # --------------------------------------------------
    # Equality rows:
    #   1) probability conservation at every non-final node
    #   2) martingale condition at every non-final node
    #   3) optional exact marginal constraints
    # --------------------------------------------------
    conservation_offsets = _conservation_offsets(grid)
    n_conservation = int(conservation_offsets[-1])
    martingale_base = n_conservation
    fixed_marginal_rows: dict[tuple[int, int], int] = {}
    next_row = 2 * n_conservation

    for maturity in sorted(market.fixed_marginals):
        t = market.time_index(maturity)
        if t == 0:
            continue

        for x_idx in range(n_x):
            fixed_marginal_rows[(t, x_idx)] = next_row
            next_row += 1

    n_equalities = next_row
    b_eq = np.zeros(n_equalities, dtype=float)
    eq_rows: list[int] = []
    eq_cols: list[int] = []
    eq_data: list[float] = []

    # Initial node masses on A_0 = 0.
    for x_idx, mass in enumerate(grid.start_mass):
        row = int(conservation_offsets[0] + _node_number(0, x_idx, n_x))
        b_eq[row] = mass

    # Add each transition's contribution to conservation and martingale rows.
    for t in range(n_times - 1):
        projection = grid.projections[t]
        current_cons_offset = int(conservation_offsets[t])
        current_mart_offset = martingale_base + current_cons_offset

        for edge_idx, a_idx, x_idx, next_x_idx in _transition_edge_indices(grid, t):
            current_node = _node_number(a_idx, x_idx, n_x)
            cons_row = current_cons_offset + current_node
            mart_row = current_mart_offset + current_node

            # Outgoing mass from current node.
            eq_rows.append(cons_row)
            eq_cols.append(edge_idx)
            eq_data.append(1.0)

            # Martingale condition in X-space.
            dx = grid.x_grid[next_x_idx] - grid.x_grid[x_idx]
            if dx != 0.0:
                eq_rows.append(mart_row)
                eq_cols.append(edge_idx)
                eq_data.append(dx)

            # Incoming mass to the next node(s) after conservative A projection.
            if t + 1 < n_times - 1:
                next_cons_offset = int(conservation_offsets[t + 1])
                lo = int(projection.lower_index[a_idx, x_idx, next_x_idx])
                hi = int(projection.upper_index[a_idx, x_idx, next_x_idx])
                w_lo = float(projection.lower_weight[a_idx, x_idx, next_x_idx])
                w_hi = float(projection.upper_weight[a_idx, x_idx, next_x_idx])
                lo_row = next_cons_offset + _node_number(lo, next_x_idx, n_x)
                eq_rows.append(lo_row)
                eq_cols.append(edge_idx)
                eq_data.append(-w_lo)

                if hi != lo and w_hi != 0.0:
                    hi_row = next_cons_offset + _node_number(hi, next_x_idx, n_x)
                    eq_rows.append(hi_row)
                    eq_cols.append(edge_idx)
                    eq_data.append(-w_hi)

            # Exact x-marginal constraints, if requested.
            target_time = t + 1
            fixed_row = fixed_marginal_rows.get((target_time, next_x_idx))

            if fixed_row is not None:
                eq_rows.append(fixed_row)
                eq_cols.append(edge_idx)
                eq_data.append(1.0)

    # Fill exact marginal RHS values.
    for maturity, probabilities in market.fixed_marginals.items():
        t = market.time_index(maturity)
        if t == 0:
            if not np.allclose(probabilities, grid.start_mass, atol=1e-10, rtol=0.0):
                raise ValueError("fixed marginal at t=0 conflicts with the projected spot mass.")

            continue

        for x_idx, probability in enumerate(probabilities):
            b_eq[fixed_marginal_rows[(t, x_idx)]] = probability

    A_eq = coo_matrix(
        (eq_data, (eq_rows, eq_cols)),
        shape=(n_equalities, n_vars),
        dtype=float,
    ).tocsr()

    # --------------------------------------------------
    # Vanilla bid/ask inequalities.
    # Every quote creates:
    #
    #     model_price <= ask
    #    -model_price <= -bid
    # --------------------------------------------------
    ub_rows: list[int] = []
    ub_cols: list[int] = []
    ub_data: list[float] = []
    b_ub: list[float] = []
    quote_labels: list[str] = []

    for quote in market.quotes:
        t = market.time_index(quote.maturity)
        if t == 0:
            observed = market.discount[0] * float(
                vanilla_payoff(market.spot, quote.strike, quote.kind)
            )
            if observed < quote.bid - 1e-10 or observed > quote.ask + 1e-10:
                raise ValueError(
                    f"t=0 {quote.kind} quote K={quote.strike} conflicts with spot payoff."
                )

            continue

        upper_row = len(b_ub)
        lower_row = upper_row + 1

        b_ub.extend([quote.ask, -quote.bid])
        quote_labels.extend([
            f"{quote.kind} T={quote.maturity} K={quote.strike} ask",
            f"{quote.kind} T={quote.maturity} K={quote.strike} bid",
        ])

        transition_t = t - 1
        discount = market.discount[t]
        spot_grid = grid.spot_grids[t]
        payoff_by_x = discount * vanilla_payoff(spot_grid, quote.strike, quote.kind)

        for edge_idx, _, _, next_x_idx in _transition_edge_indices(grid, transition_t):
            coefficient = float(payoff_by_x[next_x_idx])
            if coefficient == 0.0:
                continue

            ub_rows.extend([upper_row, lower_row])
            ub_cols.extend([edge_idx, edge_idx])
            ub_data.extend([coefficient, -coefficient])

    if b_ub:
        A_ub = coo_matrix(
            (ub_data, (ub_rows, ub_cols)),
            shape=(len(b_ub), n_vars),
            dtype=float,
        ).tocsr()
        b_ub_array = np.asarray(b_ub, dtype=float)
    else:
        A_ub = None
        b_ub_array = None

    # --------------------------------------------------
    # Objective: discounted Asian payoff on the final transition.
    # We use the exact pre-projection terminal integral for the objective,
    # avoiding unnecessary terminal A-grid interpolation error.
    # --------------------------------------------------
    objective = np.zeros(n_vars, dtype=float)
    final_t = n_times - 2
    final_projection = grid.projections[final_t]
    horizon = market.times[-1] - market.times[0]
    discount_T = market.discount[-1]

    for edge_idx, a_idx, x_idx, next_x_idx in _transition_edge_indices(grid, final_t):
        exact_integral = final_projection.exact_integral[a_idx, x_idx, next_x_idx]
        objective[edge_idx] = discount_T * float(
            asian_payoff_from_integral(exact_integral, horizon, option)
        )

    return CompiledLP(
        objective=objective,
        equality_matrix=A_eq,
        equality_targets=b_eq,
        inequality_matrix=A_ub,
        inequality_targets=b_ub_array,
        grid=grid,
        quote_labels=quote_labels,
    )
