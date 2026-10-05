import unittest

import numpy as np
from scipy.optimize import linprog

from robust_pricing.asian.solver import compile_asian_lp, solve_compiled, _diagnostics
from robust_pricing.asian.types import AsianOption, Market, GridSpec, VanillaQuote
from robust_pricing.asian.payoff import vanilla_payoff
from robust_pricing.asian.validation import validate_flow
from tests.reference_asian_solver import compile_asian_lp as reference_compile


def optimise(problem, sign=1):
    return linprog(
        c=sign * problem.objective,
        A_eq=problem.equality_matrix, b_eq=problem.equality_targets,
        A_ub=problem.inequality_matrix, b_ub=problem.inequality_targets,
        bounds=(0, None), method='highs',
    )


class MarginalRegression(unittest.TestCase):
    def test_equivalence(self):
        spec = GridSpec(np.array([80., 100., 120.]), n_integral=5)
        for fixed in (False, True):
            for kind in ('call', 'put'):
                with self.subTest(fixed=fixed, kind=kind):
                    probabilities = np.array([.25, .5, .25])
                    carry = np.array([1., 1.02, 1.04])
                    discount = np.array([1., .98, .96])
                    quotes = []

                    for t in (1, 2):
                        for side in ('call', 'put'):
                            for strike in (80., 100., 120., 200.):
                                payoff = vanilla_payoff(carry[t] * spec.x_grid, strike, side)
                                price = discount[t] * (payoff @ probabilities)
                                quotes.append(VanillaQuote(
                                    maturity=t, strike=strike, bid=max(0, price - .1),
                                    ask=price + .1, kind=side,
                                ))

                    fixed_marginals = {}
                    if fixed:
                        fixed_marginals = {
                            0.: np.array([0., 1., 0.]), 1.: probabilities, 2.: probabilities,
                        }

                    market = Market(
                        spot=100., times=np.array([0., 1., 2.]), carry=carry,
                        discount=discount, quotes=tuple(quotes), fixed_marginals=fixed_marginals,
                    )
                    option = AsianOption(strike=101., kind=kind)
                    old_problem = reference_compile(option, market, spec)
                    problem = compile_asian_lp(option, market, spec)
                    flow_count = problem.grid.n_variables
                    n_conservation = sum(
                        problem.grid.n_a(t) * problem.grid.n_x
                        for t in range(problem.grid.n_times - 1)
                    )
                    original_rows = 2 * n_conservation
                    old_constraints = old_problem.equality_matrix[:original_rows]
                    constraints = problem.equality_matrix[:original_rows, :flow_count]
                    self.assertEqual((old_constraints != constraints).nnz, 0)
                    np.testing.assert_array_equal(
                        old_problem.equality_targets[:original_rows],
                        problem.equality_targets[:original_rows],
                    )
                    np.testing.assert_array_equal(old_problem.objective, problem.objective[:flow_count])
                    self.assertTrue(np.all(problem.objective[flow_count:] == 0))
                    self.assertEqual(problem.inequality_matrix[:, :flow_count].nnz, 0)
                    self.assertLessEqual(problem.inequality_matrix.nnz, 2*len(quotes)*problem.grid.n_x)

                    for sign in (1, -1):
                        old_solution, solution = optimise(old_problem, sign), optimise(problem, sign)
                        self.assertEqual(old_solution.status, solution.status)
                        self.assertTrue(solution.success)
                        self.assertAlmostEqual(old_solution.fun, solution.fun, delta=1e-8)
                        np.testing.assert_allclose(
                            problem.solution_from_flows(solution.x[:flow_count]),
                            solution.x, atol=1e-8, rtol=0,
                        )
                        diagnostics = _diagnostics(problem, solution.x)
                        self.assertLess(diagnostics['max_equality_residual'], 1e-8)
                        self.assertLess(diagnostics['max_inequality_violation'], 1e-8)
                        self.assertAlmostEqual(diagnostics['total_initial_outflow'], 1., delta=1e-8)
                        self.assertTrue(validate_flow(problem, solution.x[:flow_count])['passed'])

                        for quote in quotes:
                            t = market.time_index(quote.maturity)
                            start = problem.marginal_index(t, 0)
                            marginal = solution.x[start:start + 3]
                            payoff = vanilla_payoff(problem.grid.spot_grids[t], quote.strike, quote.kind)
                            price = discount[t] * (payoff @ marginal)
                            self.assertGreaterEqual(price, quote.bid-1e-8)
                            self.assertLessEqual(price, quote.ask+1e-8)

                    result = solve_compiled(problem)
                    self.assertEqual(result.lower_flows.shape, (flow_count,))
                    self.assertEqual(result.upper_flows.shape, (flow_count,))
                    self.assertEqual(result.n_variables, flow_count+6)

    def test_no_quotes_and_infeasible(self):
        spec = GridSpec(np.array([80.,100.,120.]), n_integral=3)
        for quotes in ((), (VanillaQuote(1.,100.,30.,31.),)):
            market = Market(100., np.array([0.,1.]), quotes=quotes)
            for sign in (1, -1):
                option = AsianOption(strike=100.)
                old_solution = optimise(reference_compile(option, market, spec), sign)
                solution = optimise(compile_asian_lp(option, market, spec), sign)
                self.assertEqual(old_solution.status, solution.status)
                self.assertEqual(solution.status, 2 if quotes else 0)

    def test_projected_start_and_initial_quote(self):
        spec = GridSpec(np.array([80.,100.,120.]), n_integral=4)
        market = Market(105., np.array([0., .4, 1.]),
                        quotes=(VanillaQuote(0., 100., 5., 5.),))
        old_problem = reference_compile(AsianOption(100.), market, spec)
        problem = compile_asian_lp(AsianOption(100.), market, spec)
        self.assertIsNone(problem.inequality_matrix)

        for sign in (1, -1):
            old_solution, solution = optimise(old_problem, sign), optimise(problem, sign)
            self.assertTrue(old_solution.success and solution.success)
            self.assertAlmostEqual(old_solution.fun, solution.fun, delta=1e-8)
            self.assertTrue(validate_flow(problem, solution.x[:problem.grid.n_variables])['passed'])

        for t, j in ((0, 0), (3, 0), (1, -1), (1, 3)):
            with self.assertRaises(IndexError):
                problem.marginal_index(t, j)

    def test_initial_consistency(self):
        spec = GridSpec(np.array([80.,100.,120.]), n_integral=3)
        for compiler in (reference_compile, compile_asian_lp):
            for kwargs in ({'fixed_marginals': {0.: np.array([.5,0.,.5])}},
                           {'quotes': (VanillaQuote(0.,90.,0.,1.),)}):
                with self.assertRaises(ValueError):
                    compiler(AsianOption(100.), Market(100., np.array([0., 1.]), **kwargs), spec)


if __name__ == '__main__':
    unittest.main()
