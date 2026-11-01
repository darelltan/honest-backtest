"""Tests for honest_backtest.scorecard.

Expected values are worked out by hand in the comments. All data is synthetic: it shows
the code works, not that any strategy works. Every random draw uses a fixed seed.
"""

import math
import unittest

from honest_backtest import overfit, scorecard as sc
from honest_backtest.scorecard import MarketRule


class RiskFree(unittest.TestCase):
    def test_latest_yield_on_or_before_each_date_simple(self):
        # 3.65% a year over 365 periods is 0.0001 per period; 7.3% is 0.0002.
        out = sc.rates_from_yields([1, 2, 3, 4], [(1, 0.0365), (3, 0.073)], 365)
        for got, want in zip(out, [0.0001, 0.0001, 0.0002, 0.0002]):
            self.assertAlmostEqual(got, want, places=15)

    def test_geometric(self):
        out = sc.rates_from_yields([5], [(1, 0.0365)], 365, compounding="geometric")
        self.assertAlmostEqual(out[0], 1.0365 ** (1 / 365) - 1, places=15)
        self.assertAlmostEqual((1 + out[0]) ** 365, 1.0365, places=12)

    def test_no_guessing_before_the_first_yield(self):
        with self.assertRaises(ValueError):
            sc.rates_from_yields([1, 2], [(2, 0.03)], 365)

    def test_a_percent_typed_as_a_number_is_refused(self):
        # 3.65 instead of 0.0365 would be a 365% yield
        with self.assertRaises(ValueError):
            sc.rates_from_yields([1], [(1, 3.65)], 365)

    def test_periods_per_year_is_required_and_checked(self):
        with self.assertRaises(TypeError):
            sc.rates_from_yields([1], [(1, 0.03)])  # pylint: disable=no-value-for-parameter
        for bad in (0, -1, float("nan"), True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                sc.rates_from_yields([1], [(1, 0.03)], bad)


class Benchmarks(unittest.TestCase):
    def test_hold_pays_cost_and_fee_once(self):
        # qty = 1000 / (100 * 1.001) / 1.002, so equity = 1000 * p / (100 * 1.001 * 1.002).
        eq = sc.hold_equity([100, 110, 121], 1000, fee=0.002, cost=0.001)
        base = 1000 / (1.001 * 1.002)
        for got, want in zip(eq, [base, base * 1.1, base * 1.21]):
            self.assertAlmostEqual(got, want, places=9)
        self.assertLess(eq[0], 1000)  # day 0 is below the budget by exactly the costs

    def test_dca_spends_equal_parts_without_costs_or_interest(self):
        # buys at periods 0 and 2: 500 at 10 (50 units), then 500 at 20 (25 units).
        eq = sc.dca_equity([10, 10, 20, 20], 1000, fee=0, cost=0, every=2, risk_free=[0] * 4)
        self.assertEqual(eq, [1000, 1000, 1500, 1500])

    def test_dca_cash_earns_the_risk_free_rate(self):
        # price 10, buys at 0 and 2, 1% per period on cash; the first rate (0.5) is not used.
        # t0: spend 500, cash 500. t1: cash 505 -> 1005. t2: cash 510.05, all spent -> 1010.05.
        eq = sc.dca_equity([10, 10, 10], 1000, fee=0, cost=0, every=2, risk_free=[0.5, 0.01, 0.01])
        for got, want in zip(eq, [1000, 1005, 1010.05]):
            self.assertAlmostEqual(got, want, places=9)

    def test_dca_pays_the_fee_on_every_buy(self):
        # two buys of 500 at price 10, each 500 / 10 / 1.01 units: equity 1000 / 1.01.
        eq = sc.dca_equity([10, 10], 1000, fee=0.01, cost=0, every=1, risk_free=[0, 0])
        self.assertAlmostEqual(eq[-1], 1000 / 1.01, places=9)

    def test_dca_with_one_buy_is_hold(self):
        ps = [100, 90, 120]
        self.assertEqual(sc.dca_equity(ps, 1000, 0.002, 0.001, every=10, risk_free=[0] * 3),
                         sc.hold_equity(ps, 1000, 0.002, 0.001))

    def test_bad_inputs(self):
        for kw in ({"budget": 0}, {"fee": -0.01}, {"cost": 1.0}, {"fee": float("nan")}):
            args = {"budget": 1000, "fee": 0.001, "cost": 0.001, **kw}
            with self.assertRaises(ValueError, msg=repr(kw)):
                sc.hold_equity([1, 2], **args)
        for prices in ([], [1, 0], [1, -2], [1, float("inf")], [1, True]):
            with self.assertRaises(ValueError, msg=repr(prices)):
                sc.hold_equity(prices, 1000, 0, 0)
        with self.assertRaises(ValueError):
            sc.dca_equity([1, 2], 1000, 0, 0, every=0, risk_free=[0, 0])
        with self.assertRaises(ValueError):
            sc.dca_equity([1, 2], 1000, 0, 0, every=1, risk_free=[0])


class Metrics(unittest.TestCase):
    def test_returns_from_equity(self):
        rs, wiped = sc.returns_from_equity([1100, 550], 1000)
        self.assertEqual(wiped, False)
        for got, want in zip(rs, [0.1, -0.5]):
            self.assertAlmostEqual(got, want, places=12)

    def test_a_wipe_out_is_minus_100_percent_and_said(self):
        # Equity at 0 must not give returns of 0, which would hide the wipe-out.
        rs, wiped = sc.returns_from_equity([500, 0, 0], 1000)
        self.assertEqual((rs, wiped), ([-0.5, -1.0, 0.0], True))
        self.assertEqual(sc.max_drawdown(rs), 1.0)
        with self.assertRaises(ValueError):
            sc.returns_from_equity([0, 5], 1000)   # back from nothing: not a return
        with self.assertRaises(ValueError):
            sc.returns_from_equity([500, -1], 1000)

    def test_compound_round_trips(self):
        self.assertAlmostEqual(sc.compound([0.1, -0.1]), -0.01, places=15)   # 1.1 * 0.9 = 0.99
        rs, _ = sc.returns_from_equity([1030, 990, 1210], 1000)
        self.assertAlmostEqual(sc.compound(rs), 1210 / 1000 - 1, places=12)

    def test_max_drawdown(self):
        # equity 100, 120, 60, 90 from 100: the fall from 120 to 60 is 50%.
        rs, _ = sc.returns_from_equity([100, 120, 60, 90], 100)
        self.assertAlmostEqual(sc.max_drawdown(rs), 0.5, places=12)
        self.assertEqual(sc.max_drawdown([0.01, 0.02, 0.0]), 0.0)
        self.assertAlmostEqual(sc.max_drawdown([0.1, -0.5, 0.2]), 0.5, places=12)

    def test_sharpe_known_value(self):
        # excess returns 0.02, 0, 0.02, 0: mean 0.01, sample variance 0.0004/3,
        # so the per-period Sharpe is 0.01 / sqrt(0.0004/3) = sqrt(3)/2.
        rs = [0.02, 0.0, 0.02, 0.0]
        self.assertAlmostEqual(sc.annualised_sharpe(rs, [0.0] * 4, 1), math.sqrt(3) / 2, places=12)
        self.assertAlmostEqual(sc.annualised_sharpe(rs, [0.0] * 4, 365), math.sqrt(3) / 2 * math.sqrt(365),
                               places=10)

    def test_sharpe_matches_overfit_and_subtracts_the_risk_free_rate(self):
        rs = [0.012, -0.004, 0.009, 0.001, -0.002, 0.007]
        rf = [0.0001] * 6
        per_period = overfit.sharpe_and_moments([r - f for r, f in zip(rs, rf)])["sr"]
        self.assertAlmostEqual(sc.annualised_sharpe(rs, rf, 252), per_period * math.sqrt(252), places=12)

    def test_no_sharpe_without_variation(self):
        self.assertIsNone(sc.annualised_sharpe([0.01, 0.01], [0.0, 0.0], 365))
        self.assertIsNone(sc.annualised_sharpe([0.01], [0.0], 365))

    def test_annualised_return(self):
        self.assertAlmostEqual(sc.annualised_return(0.1, 365, 365), 0.1, places=12)
        self.assertAlmostEqual(sc.annualised_return(0.21, 730, 365), 0.1, places=12)   # 1.1^2 = 1.21
        self.assertEqual(sc.annualised_return(-1.0, 10, 365), -1.0)
        self.assertIsNone(sc.annualised_return(0.1, 0, 365))

    def test_quantile_known_values(self):
        # Linear interpolation: for [1, 2, 3, 4], position (4 - 1) * q.
        xs = [1, 2, 3, 4]
        self.assertAlmostEqual(sc.quantile(xs, 0.05), 1.15, places=12)
        self.assertAlmostEqual(sc.quantile(xs, 0.95), 3.85, places=12)
        self.assertEqual((sc.quantile(xs, 0), sc.quantile(xs, 1), sc.quantile([7], 0.3)), (1, 4, 7))
        with self.assertRaises(ValueError):
            sc.quantile([], 0.5)


RULE = MarketRule(average=3, slope=2, crash_fall=0.2, crash_within=3)


class MarketLabels(unittest.TestCase):
    def test_trend_labels_after_the_warm_up(self):
        # The average exists from period 2 (average 3) and its slope from period 4 (2 more):
        # average + slope - 1 = 4 unlabelled periods.
        up = sc.market_labels([100 + i for i in range(10)], RULE)
        self.assertEqual(up, ["unlabelled"] * 4 + ["bull"] * 6)
        down = sc.market_labels([100 - i for i in range(10)], RULE)   # 1% a period: no crash
        self.assertEqual(down, ["unlabelled"] * 4 + ["bear"] * 6)
        flat = sc.market_labels([100] * 10, RULE)
        self.assertEqual(flat, ["unlabelled"] * 4 + ["sideways"] * 6)

    def test_a_crash_runs_from_the_peak_to_the_low(self):
        # flat at 100 for periods 0-7, then 90, 80, 75. 80 is exactly 20% below 100 within 3
        # periods (a fall of exactly crash_fall counts), so periods 7 (the last 100) to 10 are crash.
        labels = sc.market_labels([100] * 8 + [90, 80, 75], RULE)
        self.assertEqual(labels[:7], ["unlabelled"] * 4 + ["sideways"] * 3)
        self.assertEqual(labels[7:], ["crash"] * 4)

    def test_a_crash_overrides_unlabelled(self):
        # Every period here would be unlabelled (warm-up). 70 at periods 1 and 2 is 30% below
        # the peak at period 0, within 3 periods: so periods 0 to 2 are all crash.
        self.assertEqual(sc.market_labels([100, 70, 70], RULE), ["crash", "crash", "crash"])
        # Once the peak is more than crash_within periods back, the trend label returns:
        # period 4 (70) looks back over periods 1-3, all at 70, so no crash. Its average (70)
        # is falling (80 two periods earlier) but the price equals the average rather than
        # being below it, so it is sideways, not bear.
        self.assertEqual(sc.market_labels([100, 70, 70, 70, 70], RULE),
                         ["crash"] * 4 + ["sideways"])

    def test_a_slow_fall_is_not_a_crash(self):
        # 1% a period for 30 periods (26% in all) never falls 20% within 3 periods.
        labels = sc.market_labels([100 * 0.99 ** i for i in range(31)], RULE)
        self.assertNotIn("crash", labels)

    def test_the_rule_needs_every_value(self):
        with self.assertRaises(TypeError):
            MarketRule(average=200, slope=20, crash_fall=0.2)  # pylint: disable=missing-kwoa
        for kw in ({"average": 0}, {"slope": 1.5}, {"crash_fall": 20}, {"crash_fall": 0}, {"crash_within": True}):
            with self.assertRaises(ValueError, msg=repr(kw)):
                MarketRule(**{"average": 3, "slope": 2, "crash_fall": 0.2, "crash_within": 3, **kw})


class Bootstrap(unittest.TestCase):
    def series(self):
        s = [0.01 * math.sin(i) + 0.001 for i in range(200)]
        h = [0.012 * math.cos(i * 1.3) for i in range(200)]
        return s, h

    def test_it_is_repeatable(self):
        s, h = self.series()
        a = sc.bootstrap({"strategy": s, "hold": h}, [0.0] * 200, 365, draws=300)
        b = sc.bootstrap({"strategy": s, "hold": h}, [0.0] * 200, 365, draws=300)
        self.assertEqual(a, b)

    def test_a_series_identical_to_hold_never_beats_it(self):
        s, _ = self.series()
        out = sc.bootstrap({"strategy": s, "hold": list(s)}, [0.0] * 200, 365, draws=200)
        self.assertEqual(out["strategy"]["sharpe_beats"]["hold"], 0.0)

    def test_a_constant_series_has_no_sharpe_and_a_zero_width_range(self):
        # An undefined Sharpe must not be counted as 0 in the ranges.
        out = sc.bootstrap({"strategy": [0.001] * 50}, [0.0] * 50, 365, draws=100)
        self.assertIsNone(out["strategy"]["sharpe_90"])
        self.assertEqual(out["strategy"]["undefined_sharpe_draws"], 100)
        lo, hi = out["strategy"]["annual_return_90"]
        self.assertAlmostEqual(lo, hi, places=12)
        self.assertAlmostEqual(lo, 1.001 ** 365 - 1, places=9)

    def test_the_observed_sharpe_is_inside_its_range(self):
        s, h = self.series()
        out = sc.bootstrap({"strategy": s, "hold": h}, [0.0] * 200, 365, draws=500)
        lo, hi = out["strategy"]["sharpe_90"]
        self.assertLess(lo, sc.annualised_sharpe(s, [0.0] * 200, 365))
        self.assertGreater(hi, sc.annualised_sharpe(s, [0.0] * 200, 365))


def synthetic(n=120):
    """A made-up price path and a made-up strategy equity curve. Shows the code works, not
    that any strategy works."""
    prices = [100 * (1 + 0.002 * i) * (1 + 0.03 * math.sin(i / 5)) for i in range(n)]
    equity = [1000 * (1 + 0.0015 * i) * (1 + 0.01 * math.sin(i / 3)) for i in range(n)]
    return list(range(n)), prices, equity


class Scorecard(unittest.TestCase):
    KW = {"budget": 1000, "fee": 0.002, "cost": 0.001, "periods_per_year": 365, "dca_every": 7, "draws": 200}

    def test_a_strategy_that_holds_scores_like_hold(self):
        dates, prices, _ = synthetic()
        eq = sc.hold_equity(prices, 1000, 0.002, 0.001)
        out = sc.scorecard(dates, prices, eq, [0.0001] * len(dates), **self.KW)
        self.assertEqual(out["metrics"]["strategy"], out["metrics"]["hold"])

    def test_the_risk_free_benchmark_compounds_after_the_first_date(self):
        # constant 0.001 per period over 4 dates: three periods pass, 1.001^3 - 1.
        out = sc.scorecard([1, 2, 3, 4], [10, 11, 12, 13], [1000, 1001, 1002, 1003], [0.001] * 4,
                           **dict(self.KW, dca_every=1))
        self.assertAlmostEqual(out["metrics"]["risk_free"]["return"], 1.001 ** 3 - 1, places=12)

    def test_the_first_dates_rate_is_not_used(self):
        dates, prices, eq = synthetic(40)
        a = sc.scorecard(dates, prices, eq, [0.0001] * 40, **self.KW)
        b = sc.scorecard(dates, prices, eq, [0.9] + [0.0001] * 39, **self.KW)
        self.assertEqual(a, b)

    def test_by_type_multiplies_back_to_the_total(self):
        # Every period has exactly one label, so the per-label totals compound to the whole.
        dates, prices, eq = synthetic()
        out = sc.scorecard(dates, prices, eq, [0.0001] * len(dates), market_rule=RULE, **self.KW)
        for k in ("strategy", "hold", "dca"):
            g = math.prod(1 + v[k] for v in out["by_type"].values())
            self.assertAlmostEqual(g - 1, out["metrics"][k]["return"], places=10, msg=k)
        self.assertEqual(sum(v["periods"] for v in out["by_type"].values()), len(dates))

    def test_mismatched_inputs_are_refused_not_trimmed(self):
        # Days missing from one series must be refused, not silently dropped.
        dates, prices, eq = synthetic(30)
        rf = [0.0] * 30
        for args in ((dates, prices[:-1], eq, rf), (dates, prices, eq[:-1], rf), (dates, prices, eq, rf[:-1]),
                     (dates[:1], prices[:1], eq[:1], rf[:1])):
            with self.assertRaises(ValueError):
                sc.scorecard(*args, **self.KW)
        with self.assertRaises(ValueError):
            sc.scorecard([0, 2, 1] + dates[3:], prices, eq, rf, **self.KW)    # not increasing
        with self.assertRaises(ValueError):
            sc.scorecard(dates, prices, [float("nan")] + eq[1:], rf, **self.KW)

    def test_periods_per_year_has_no_default(self):
        dates, prices, eq = synthetic(30)
        kw = {k: v for k, v in self.KW.items() if k != "periods_per_year"}
        with self.assertRaises(TypeError):
            sc.scorecard(dates, prices, eq, [0.0] * 30, **kw)

    def test_a_wiped_out_strategy_is_reported(self):
        dates, prices, _ = synthetic(10)
        out = sc.scorecard(dates, prices, [900, 500, 0] + [0] * 7, [0.0] * 10, **self.KW)
        self.assertTrue(out["wiped_out"])
        self.assertEqual(out["metrics"]["strategy"]["return"], -1.0)
        self.assertEqual(out["metrics"]["strategy"]["max_drawdown"], 1.0)
        self.assertTrue(any("wiped out" in n for n in out["notes"]))

    def test_it_says_benchmarks_pay_no_exit_costs(self):
        dates, prices, eq = synthetic(10)
        out = sc.scorecard(dates, prices, eq, [0.0] * 10, **self.KW)
        self.assertTrue(any("no exit costs" in n for n in out["notes"]))


if __name__ == "__main__":
    unittest.main()
