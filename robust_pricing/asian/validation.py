from __future__ import annotations

import numpy as np

from .solver import CompiledLP


def validate_flow(problem: CompiledLP, flows: np.ndarray, tolerance: float = 1e-8):
    """Independent residual checks for a solved occupancy-measure LP."""
    flows = np.asarray(flows, dtype=float)
    if flows.shape != (problem.grid.n_variables,):
        raise ValueError("flow vector has the wrong shape.")

    solution = problem.solution_from_flows(flows)
    equality_residual = problem.equality_matrix @ solution - problem.equality_targets
    max_equality_residual = float(np.max(np.abs(equality_residual)))

    if problem.inequality_matrix is None:
        max_inequality_violation = 0.0
    else:
        inequality_residual = problem.inequality_matrix @ solution - problem.inequality_targets
        max_inequality_violation = float(max(0.0, np.max(inequality_residual)))

    minimum_flow = float(np.min(flows))
    total_initial_outflow = float(np.sum(flows[problem.grid.transition_slice(0)]))
    passed = (
        max_equality_residual <= tolerance
        and max_inequality_violation <= tolerance
        and minimum_flow >= -tolerance
        and abs(total_initial_outflow - 1.0) <= tolerance
    )

    return {
        "passed": passed,
        "max_equality_residual": max_equality_residual,
        "max_inequality_violation": max_inequality_violation,
        "minimum_flow": minimum_flow,
        "total_initial_outflow": total_initial_outflow,
    }


def compare_bounds(
    reference_lower: float,
    reference_upper: float,
    candidate_lower: float,
    candidate_upper: float,
    tolerance: float = 1e-7,
):
    lower_error = abs(reference_lower - candidate_lower)
    upper_error = abs(reference_upper - candidate_upper)

    return {
        "passed": lower_error <= tolerance and upper_error <= tolerance,
        "lower_error": lower_error,
        "upper_error": upper_error,
    }
