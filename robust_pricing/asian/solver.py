from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix

from .grid import AugmentedGrid, build_augmented_grid
from .payoff import asian_payoff_from_integral, vanilla_payoff
from .types import AsianOption, GridSpec, Market, SolveResult


@dataclass
class CompiledLP:
    objective: np.ndarray
    equality_matrix: csr_matrix
    equality_targets: np.ndarray
    inequality_matrix: csr_matrix | None
    inequality_targets: np.ndarray | None
    grid: AugmentedGrid
    quote_labels: list[str]
    timings: dict[str, float] = field(default_factory=dict)

    def marginal_index(self, t: int, x_idx: int) -> int:
        """LP column for P(X_t = x[x_idx]); flow columns precede marginals."""
        if not 1 <= t < self.grid.n_times or not 0 <= x_idx < self.grid.n_x:
            raise IndexError("marginal index outside non-initial time / X grid")

        return self.grid.n_variables + (t - 1) * self.grid.n_x + x_idx

    def solution_from_flows(self, flows: np.ndarray) -> np.ndarray:
        """Extend a flow-only vector with its uniquely determined X marginals."""
        flows = np.asarray(flows, dtype=float)
        if flows.shape != (self.grid.n_variables,):
            raise ValueError("flow vector has the wrong shape.")

        marginals = []
        for t in range(self.grid.n_times - 1):
            transition_flow = flows[self.grid.transition_slice(t)].reshape(-1, self.grid.n_x)
            marginals.append(transition_flow.sum(axis=0))

        return np.concatenate([flows, *marginals])


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
    """Compile Asian bounds into a sparse occupancy LP on (time, X, A).

    X is the normalized martingale price; A is the running integral of spot.
    Flows use conservative A projection; vanilla quotes constrain X marginals.
    """
    compile_start = perf_counter()
    grid = build_augmented_grid(market, spec)
    timings = {"grid": perf_counter() - compile_start}
    equality_start = perf_counter()
    n_vars = grid.n_variables + (grid.n_times - 1) * grid.n_x
    n_x = grid.n_x
    n_times = grid.n_times

    # Row order: conservation, martingale, flow-to-marginal links, fixed marginals.
    conservation_offsets = _conservation_offsets(grid)
    n_conservation = int(conservation_offsets[-1])
    martingale_base = n_conservation
    fixed_marginal_rows: dict[tuple[int, int], int] = {}
    linking_base = 2 * n_conservation
    next_row = linking_base + (n_times - 1) * n_x

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

    # p[t,j] equals the incoming flow at X index j, independently of A.
    for t in range(1, n_times):
        edge_slice = grid.transition_slice(t - 1)
        edges = np.arange(edge_slice.start, edge_slice.stop)
        next_x_indices = np.arange(len(edges)) % n_x
        linking_rows = linking_base + (t - 1) * n_x
        marginal_columns = grid.n_variables + (t - 1) * n_x
        eq_rows.extend(linking_rows + next_x_indices)
        eq_cols.extend(edges)
        eq_data.extend(-np.ones(len(edges)))
        eq_rows.extend(linking_rows + np.arange(n_x))
        eq_cols.extend(marginal_columns + np.arange(n_x))
        eq_data.extend(np.ones(n_x))

    # Fill exact marginal RHS values.
    for maturity, probabilities in market.fixed_marginals.items():
        t = market.time_index(maturity)
        if t == 0:
            if not np.allclose(probabilities, grid.start_mass, atol=1e-10, rtol=0.0):
                raise ValueError("fixed marginal at t=0 conflicts with the projected spot mass.")

            continue

        for x_idx, probability in enumerate(probabilities):
            row = fixed_marginal_rows[(t, x_idx)]
            b_eq[row] = probability
            eq_rows.append(row)
            eq_cols.append(grid.n_variables + (t - 1) * n_x + x_idx)
            eq_data.append(1.0)

    A_eq = coo_matrix(
        (eq_data, (eq_rows, eq_cols)),
        shape=(n_equalities, n_vars),
        dtype=float,
    ).tocsr()

    # Bid/ask rows: model_price <= ask and -model_price <= -bid.
    timings["equalities"] = perf_counter() - equality_start
    inequality_start = perf_counter()
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

        discount = market.discount[t]
        spot_grid = grid.spot_grids[t]
        payoff_by_x = discount * vanilla_payoff(spot_grid, quote.strike, quote.kind)

        for x_idx, payoff in enumerate(payoff_by_x):
            coefficient = float(payoff)
            if coefficient == 0.0:
                continue

            ub_rows.extend([upper_row, lower_row])
            column = grid.n_variables + (t - 1) * n_x + x_idx
            ub_cols.extend([column, column])
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

    # Use the pre-projection terminal integral to avoid terminal interpolation error.
    timings["inequalities"] = perf_counter() - inequality_start
    objective_start = perf_counter()
    objective = np.zeros(n_vars, dtype=float)
    final_t = n_times - 2
    final_projection = grid.projections[final_t]
    horizon = market.times[-1] - market.times[0]
    terminal_discount = market.discount[-1]

    for edge_idx, a_idx, x_idx, next_x_idx in _transition_edge_indices(grid, final_t):
        exact_integral = final_projection.exact_integral[a_idx, x_idx, next_x_idx]
        objective[edge_idx] = terminal_discount * float(
            asian_payoff_from_integral(exact_integral, horizon, option)
        )

    timings["objective"] = perf_counter() - objective_start
    timings["compile_total"] = perf_counter() - compile_start

    return CompiledLP(
        objective=objective,
        equality_matrix=A_eq,
        equality_targets=b_eq,
        inequality_matrix=A_ub,
        inequality_targets=b_ub_array,
        grid=grid,
        quote_labels=quote_labels,
        timings=timings,
    )


def _diagnostics(problem: CompiledLP, solution: np.ndarray) -> dict[str, float]:
    flows = solution[:problem.grid.n_variables]
    eq_residual = problem.equality_matrix @ solution - problem.equality_targets
    diagnostics = {
        "max_equality_residual": float(np.max(np.abs(eq_residual))),
        "minimum_flow": float(np.min(flows)),
        "total_initial_outflow": float(np.sum(flows[problem.grid.transition_slice(0)])),
    }

    if problem.inequality_matrix is not None:
        slack = problem.inequality_targets - problem.inequality_matrix @ solution
        diagnostics["max_inequality_violation"] = float(max(0.0, -np.min(slack)))
        diagnostics["minimum_inequality_slack"] = float(np.min(slack))
    else:
        diagnostics["max_inequality_violation"] = 0.0
        diagnostics["minimum_inequality_slack"] = float("inf")

    return diagnostics


def solve_compiled(problem: CompiledLP, solver_options: dict | None = None) -> SolveResult:
    """Solve both bounds with HiGHS IPM and report feasibility in original units."""
    solve_start = perf_counter()
    timings = dict(problem.timings)
    bounds = (0.0, None)
    options = {} if solver_options is None else dict(solver_options)
    options.setdefault("run_crossover", "choose")

    common = dict(
        A_ub=problem.inequality_matrix,
        b_ub=problem.inequality_targets,
        A_eq=problem.equality_matrix,
        b_eq=problem.equality_targets,
        bounds=bounds,
        options=options,
    )
    lower_start = perf_counter()
    lower = linprog(
        c=problem.objective,
        method="highs-ipm",
        **common,
    )
    timings["lower_solve"] = perf_counter() - lower_start

    if not lower.success:
        raise RuntimeError(
            "Lower-bound optimisation failed: "
            + lower.message
        )

    upper_start = perf_counter()
    upper = linprog(
        c=-problem.objective,
        method="highs-ipm",
        **common,
    )
    timings["upper_solve"] = perf_counter() - upper_start

    if not upper.success:
        raise RuntimeError(
            "Upper-bound optimisation failed: "
            + upper.message
        )

    timings["solve_total"] = perf_counter() - solve_start
    timings["total"] = timings.get("compile_total", 0.0) + timings["solve_total"]

    return SolveResult(
        lower=float(lower.fun),
        upper=float(-upper.fun),
        lower_flows=lower.x[:problem.grid.n_variables].copy(),
        upper_flows=upper.x[:problem.grid.n_variables].copy(),
        n_variables=len(problem.objective),
        n_equalities=problem.equality_matrix.shape[0],
        n_inequalities=0 if problem.inequality_matrix is None else problem.inequality_matrix.shape[0],
        timings=timings,
        lower_diagnostics=_diagnostics(problem, lower.x),
        upper_diagnostics=_diagnostics(problem, upper.x),
    )


def solve_asian_bounds(
    market: Market,
    option: AsianOption,
    spec: GridSpec,
    solver_options: dict | None = None,
) -> SolveResult:
    """Compile and solve the lower and upper prices for one market and contract."""
    problem = compile_asian_lp(
        option=option,
        market=market,
        spec=spec,
    )

    return solve_compiled(problem, solver_options=solver_options)
