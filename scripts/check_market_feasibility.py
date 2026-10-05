"""Check parity bands and individual-maturity feasibility on the SPX X support."""

import numpy as np
from scipy.optimize import linprog

from scripts.run_spx_mot import load_processed_market_data, build_market, build_grid_spec

N_X_TESTS = [15, 20, 25, 30, 35, 40, 50, 60]


def main() -> None:
    chain, term = load_processed_market_data()
    market = build_market(chain, term)
    spec = build_grid_spec(market)
    normalized_spot = market.spot / market.carry[0]
    by_contract = {}

    for quote in market.quotes:
        key = (quote.maturity, quote.strike)
        by_contract.setdefault(key, {})[quote.kind] = quote

    violations = []
    paired_strikes = 0

    for (maturity, strike), sides in by_contract.items():
        if "call" not in sides or "put" not in sides:
            continue

        paired_strikes += 1
        call, put = sides["call"], sides["put"]
        t = market.time_index(maturity)
        discount = market.discount[t]
        forward = market.spot * market.carry[t]
        parity_value = discount * (forward - strike)
        lower = call.bid - put.ask
        upper = call.ask - put.bid

        if parity_value < lower:
            gap = lower - parity_value
            violations.append((maturity, strike, parity_value, lower, upper, gap))
        elif parity_value > upper:
            gap = parity_value - upper
            violations.append((maturity, strike, parity_value, lower, upper, gap))

    print(f"Parity: {paired_strikes} paired strikes, {len(violations)} violations")

    for maturity, strike, parity, lower, upper, gap in violations[:10]:
        print(f"T={maturity:.6f}, K={strike:.1f}, required={parity:.4f}, "
              f"allowed=[{lower:.4f}, {upper:.4f}], gap={gap:.4f}")

    # Change resolution while keeping the MOT support endpoints fixed.
    x_min, x_max = spec.x_grid[0], spec.x_grid[-1]
    for n_x in N_X_TESTS:
        x_grid = np.linspace(x_min, x_max, n_x)
        all_feasible = True

        for maturity in market.times[1:]:
            t = market.time_index(maturity)
            spot_grid = market.carry[t] * x_grid
            discount = market.discount[t]
            quotes = [q for q in market.quotes if np.isclose(q.maturity, maturity)]

            # A marginal has unit mass and the same normalized mean as X_0.
            A_eq = np.vstack([np.ones(n_x), x_grid])
            b_eq = np.array([1.0, normalized_spot])
            A_ub, b_ub = [], []

            for quote in quotes:
                if quote.kind == "call":
                    payoff = np.maximum(spot_grid - quote.strike, 0.0)
                elif quote.kind == "put":
                    payoff = np.maximum(quote.strike - spot_grid, 0.0)
                else:
                    raise ValueError(f"Unknown option kind: {quote.kind}")

                price_vector = discount * payoff
                A_ub.append(price_vector)
                b_ub.append(quote.ask)
                A_ub.append(-price_vector)
                b_ub.append(-quote.bid)

            result = linprog(
                c=np.zeros(n_x), A_ub=np.asarray(A_ub), b_ub=np.asarray(b_ub),
                A_eq=A_eq, b_eq=b_eq, bounds=(0.0, None), method="highs",
            )
            status = "PASS" if result.success else "FAIL"

            if not result.success:
                all_feasible = False

            print(f"N_X={n_x:2d}, T={maturity:.6f}, quotes={len(quotes):3d}: {status}")

        overall = "ALL PASS" if all_feasible else "INFEASIBLE"
        spacing = (x_max - x_min) / (n_x - 1)
        print(f"N_X={n_x}: {overall}, X spacing={spacing:.2f}")


if __name__ == "__main__":
    main()
