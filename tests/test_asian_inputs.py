import unittest

import numpy as np

from robust_pricing.asian.types import AsianOption, GridSpec, Market, VanillaQuote


class FiniteInputChecks(unittest.TestCase):
    def test_contract_and_quote_inputs(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(contract_strike=value):
                with self.assertRaises(ValueError):
                    AsianOption(strike=value)

            for field in ("maturity", "strike", "bid", "ask"):
                quote = {"maturity": 1., "strike": 100., "bid": 1., "ask": 2.}
                quote[field] = value
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        VanillaQuote(**quote)

    def test_market_and_grid_inputs(self):
        for value in (np.nan, np.inf, -np.inf):
            for field in ("spot", "times", "carry", "discount"):
                inputs = {"spot": 100., "times": np.array([0., 1.])}
                if field == "spot":
                    inputs[field] = value
                else:
                    initial = 0. if field == "times" else 1.
                    inputs[field] = np.array([initial, value])
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        Market(**inputs)

            with self.subTest(grid=value):
                with self.assertRaises(ValueError):
                    GridSpec(x_grid=np.array([0., value]))


if __name__ == "__main__":
    unittest.main()
