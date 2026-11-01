"""Tests for honest_backtest.checks.

All data is synthetic: it shows the code works, not that any strategy works. Every random
draw is seeded. Boundaries are tested on both sides.
"""

import datetime as dt
import json
import math
import os
import random
import shutil
import tempfile
import unittest
from unittest import mock

from honest_backtest import checks, overfit, scorecard as sc
from honest_backtest.checks import (CheckResult, Market, SealedPeriodError, Thresholds, Trades, run,
                                    trade_summary)
from honest_backtest.trials import TrialLog

START = dt.date(2020, 1, 1)


def days(n, start=START):
    return [start + dt.timedelta(days=i) for i in range(n)]


def market(n=300, drift=0.001, seed=3, start=START, equity=None):
    """Synthetic prices (a seeded random walk) and a synthetic, steadily rising equity curve."""
    rnd = random.Random(seed)
    prices, p = [], 100.0
    for _ in range(n):
        p *= 1 + rnd.gauss(0.0003, 0.02)
        prices.append(p)
    if equity is None:
        equity, e = [1000.0], 1000.0                     # day 0: the budget (no entry costs here)
        for _ in range(n - 1):
            e *= 1 + drift + rnd.gauss(0, 0.002)
            equity.append(e)
    return Market(dates=days(n, start), prices=prices, equity=equity, risk_free=[0.0001] * n, budget=1000,
                  fee=0.001, cost=0.0005, periods_per_year=365, dca_every=7)


def trades(n=40, gross=0.3, cost=0.1, **kw):
    return Trades(gross=[gross] * n, costs=[cost] * n, **kw)


def random_result(strategy=1.0, values=None, runs=None, too_few=0, failed=0, warnings=(), **extra):
    """Hand-made random_baseline output: by default 500 valid runs with values 0..499, and the
    p-value worked out as random_baseline does: (1 + runs >= strategy) / (1 + valid runs)."""
    values = list(range(500)) if values is None else values
    p_value = (1 + sum(1 for v in values if v >= strategy)) / (1 + len(values))
    out = {"strategy": strategy, "values": [float(v) for v in values], "runs": runs or len(values) + too_few + failed,
           "valid_runs": len(values), "too_few_trades": too_few, "failed": failed, "warnings": list(warnings),
           "percentile": 0.5, "p_value": p_value}
    out.update(extra)
    return out


class LogMixin:
    def make_log(self, sharpes=(0.01, 0.02, 0.03), dirty=False, periods=None, idea="idea"):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        log = TrialLog(os.path.join(tmp, "trials.jsonl"))
        for i, s in enumerate(sharpes):
            log.record(idea, {"i": i}, {"sharpe": s}, code={"commit": "abc", "dirty": dirty},
                       period=(periods[i] if periods else None))
        return log


# --------------------------------------------------------------------------- A

class CostsVsEdge(unittest.TestCase):
    def a(self, **kw):
        return run(trades=kw.pop("t"), thresholds=Thresholds(**kw)).check("A")

    def test_edge_left_after_doubled_costs_passes(self):
        r = self.a(t=trades(gross=0.3, cost=0.1))        # net 0.2; doubled costs 0.1 > 0
        self.assertEqual(r.status, "PASS")
        self.assertAlmostEqual(r.numbers["avg_net_stressed"], 0.1, places=12)

    def test_doubled_costs_exactly_eating_the_edge_fails(self):
        r = self.a(t=trades(gross=0.3, cost=0.15))       # doubled costs: 0.3 - 0.3 = 0, not above 0
        self.assertEqual(r.status, "FAIL")
        self.assertTrue(any("x2" in x for x in r.reasons))

    def test_cost_share_warns_only_above_half(self):
        self.assertEqual(self.a(t=trades(gross=0.4, cost=0.2), cost_multiplier=1.5).status, "PASS")   # exactly 50%
        r = self.a(t=trades(gross=0.4, cost=0.21), cost_multiplier=1.5)                                # 52.5%
        self.assertEqual(r.status, "WARN")

    def test_no_edge_before_costs_fails(self):
        self.assertEqual(self.a(t=trades(gross=-0.1, cost=0.0)).status, "FAIL")
        self.assertEqual(self.a(t=trades(n=0)).status, "FAIL")

    def test_bad_trades_are_refused(self):
        for t in (Trades([0.1], [0.1, 0.2]), Trades([0.1], [-0.01]), Trades([float("nan")], [0.0]),
                  Trades([0.1], [0.0], entries=[]), Trades([0.1], [0.0], slices={"y": []})):
            with self.assertRaises(ValueError):
                run(trades=t)


# --------------------------------------------------------------------------- B, B2

class RandomChecks(unittest.TestCase):
    def test_strictly_above_the_95th_percentile_with_a_small_p_value_passes(self):
        # values 0..499: the 95th percentile (linear) is 474.05.
        # 475.5: 24 runs at or above it, p = 25/501 = 0.0499 <= 0.05: PASS.
        self.assertEqual(run(random={"random_entry": random_result(475.5)}).check("B").status, "PASS")
        # 474.05: not strictly above the bar: FAIL.
        self.assertEqual(run(random={"random_entry": random_result(474.05)}).check("B").status, "FAIL")

    def test_above_the_bar_but_p_too_large_fails(self):
        # 474.06 is above the bar, but 25 runs are at or above it,
        # p = 26/501 = 0.0519 > 0.05, so it must not pass.
        r = run(random={"random_entry": random_result(474.06)}).check("B")
        self.assertEqual(r.status, "FAIL")
        self.assertTrue(any("p = 0.0519 is above 0.05" in x for x in r.reasons), r.reasons)

    def test_fewer_than_400_valid_runs_fail(self):
        # A handful of random runs can't place a strategy.
        r = run(random={"random_entry": random_result(1000.0, values=range(399))}).check("B")
        self.assertEqual(r.status, "FAIL")
        self.assertTrue(any("fewer than 400" in x for x in r.reasons), r.reasons)
        self.assertEqual(run(random={"random_entry": random_result(1000.0, values=range(400))}).check("B").status,
                         "PASS")
        th = Thresholds(min_random_runs=100)
        self.assertEqual(run(random={"random_entry": random_result(1000.0, values=range(100))},
                             thresholds=th).check("B").status, "PASS")

    def test_too_few_valid_runs_fail(self):
        r = run(random={"random_entry": random_result(1000.0, too_few=56)}).check("B")   # 500 of 556 = 89.9%
        self.assertEqual(r.status, "FAIL")
        r = run(random={"random_entry": random_result(1000.0, too_few=55)}).check("B")   # 500 of 555 = 90.1%
        self.assertEqual(r.status, "WARN")                                            # passes, with a warning

    def test_baseline_mismatches_are_fails(self):
        # A baseline that doesn't match the strategy is a FAIL, not a WARN.
        r = run(random={"random_entry": random_result(1000.0, warnings=["exit_fn may not match"])}).check("B")
        self.assertEqual(r.status, "FAIL")
        self.assertIn("exit_fn may not match", r.reasons)

    def test_random_direction_warns_by_default_fails_when_strict_and_says_which(self):
        res = {"random_direction": random_result(10.0, direction="coin", direction_p={"A": 0.5})}
        r = run(random=res).check("B2")
        self.assertEqual(r.status, "WARN")
        self.assertTrue(any("mode: WARN" in x and "coin" in x for x in r.reasons))
        r = run(random=res, thresholds=Thresholds(direction_strict=True)).check("B2")
        self.assertEqual(r.status, "FAIL")
        self.assertTrue(any("mode: strict (FAIL on failure)" in x for x in r.reasons))
        ok = {"random_direction": random_result(1000.0, direction="long_share", direction_p={"A": 0.7})}
        self.assertEqual(run(random=ok).check("B2").status, "PASS")

    def test_a_different_percentile_is_used_and_shown(self):
        # bar 449.1; 451: 49 runs at or above, p = 50/501 = 0.0998 <= 0.10.
        rep = run(random={"random_entry": random_result(451.0)}, thresholds=Thresholds(random_percentile=0.9))
        self.assertEqual(rep.check("B").status, "PASS")
        t = {x["name"]: x for x in rep.thresholds}["random_percentile"]
        self.assertEqual((t["value"], t["source"]), (0.9, "set by you"))


# --------------------------------------------------------------------------- C, C2

class DeflatedSharpe(LogMixin, unittest.TestCase):
    def test_it_uses_the_logs_trial_count_and_spread(self):
        log = self.make_log(sharpes=(0.01, 0.02, 0.06, 0.03))
        m = market()
        r = run(market=m, trial_log=log).check("C")
        strat, _ = sc.returns_from_equity(m.equity, m.budget)
        ex = sc.excess(strat, [0.0] + m.risk_free[1:])
        want = overfit.deflated_sharpe(ex, 4, overfit.sharpe_variance([0.01, 0.02, 0.06, 0.03]))
        self.assertEqual(r.numbers["trials"], 4)
        self.assertAlmostEqual(r.numbers["dsr"], want["dsr"], places=12)
        self.assertTrue(r.numbers["var_measured"])

    def test_the_threshold_boundary(self):
        log, m = self.make_log(), market()
        dsr = run(market=m, trial_log=log).check("C").numbers["dsr"]
        at = run(market=m, trial_log=log, thresholds=Thresholds(dsr_min=dsr)).check("C")
        self.assertIn(at.status, ("PASS", "WARN"))                    # WARN: completeness not verified
        self.assertFalse(any("below" in x for x in at.reasons))
        self.assertEqual(run(market=m, trial_log=log,
                             thresholds=Thresholds(dsr_min=min(1.0, dsr + 1e-9))).check("C").status, "FAIL")

    def test_a_breakdown_of_the_formula_is_a_fail(self):
        fake = {"dsr": 0.0, "sr": 2.0, "sr0": 0.1, "trials": 3, "sharpe_var": 0.01, "var_measured": True,
                "psr_valid": False, "skew": 1.0, "kurt": 1.0, "periods": 300}
        with mock.patch.object(checks.overfit, "deflated_sharpe", return_value=fake):
            r = run(market=market(), trial_log=self.make_log()).check("C")
        self.assertEqual(r.status, "FAIL")
        self.assertIn("broke down", r.reasons[0])

    def test_a_broken_log_is_a_fail(self):
        log = self.make_log()
        with open(log.path, encoding="utf-8") as fh:
            text = fh.read()
        with open(log.path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text.replace('"sharpe":0.02', '"sharpe":0.9'))
        rep = run(market=market(), trial_log=log)
        self.assertEqual(rep.check("C").status, "FAIL")
        self.assertEqual(rep.check("H").status, "FAIL")
        self.assertEqual(rep.verdict, "FAIL")

    def test_an_empty_log_fails(self):
        # An empty log must not be counted as 1 trial with only a warning.
        r = run(market=market(), trial_log=self.make_log(sharpes=())).check("C")
        self.assertEqual(r.status, "FAIL")
        self.assertEqual(r.reasons, ["no trials logged; the luck correction can't be applied"])

    def test_an_unmeasured_spread_warns(self):
        r = run(market=market(), trial_log=self.make_log(sharpes=(0.01,))).check("C")
        self.assertIn(r.status, ("WARN", "FAIL"))
        self.assertTrue(any("not measured" in x for x in r.reasons))


class SharpeTest(LogMixin, unittest.TestCase):
    FAKE = {"delta": 0.01, "se": 0.02, "p": 0.3, "p_se": 0.01, "level": 0.05, "trials": 3, "block": 10,
            "draws": 1000, "too_many_trials": False, "passed": False}

    def c2(self, fake, **th):
        with mock.patch.object(checks.overfit, "sharpe_diff_test", return_value=fake):
            return run(market=market(), trial_log=self.make_log(), thresholds=Thresholds(**th)).check("C2")

    def test_warn_by_default_fail_when_strict_and_the_mode_is_stated(self):
        r = self.c2(self.FAKE)
        self.assertEqual(r.status, "WARN")
        self.assertIn("mode: WARN on failure", r.reasons)
        r = self.c2(self.FAKE, sharpe_test_strict=True)
        self.assertEqual(r.status, "FAIL")
        self.assertIn("mode: strict (FAIL on failure)", r.reasons)
        self.assertEqual(self.c2(dict(self.FAKE, passed=True, p=0.01)).status, "PASS")

    def test_too_many_trials_says_why(self):
        r = self.c2(dict(self.FAKE, too_many_trials=True, p=1.0), sharpe_test_strict=True)
        self.assertEqual(r.status, "FAIL")
        self.assertIn("too many trials", r.reasons[0])

    def test_it_really_runs_on_the_market_inputs(self):
        r = run(market=market(), trial_log=self.make_log()).check("C2")
        self.assertIn(r.status, ("PASS", "WARN"))
        self.assertEqual(r.numbers["trials"], 3)


# --------------------------------------------------------------------------- D

class SealedGuard(LogMixin, unittest.TestCase):
    SEALED = (dt.date(2021, 1, 1), dt.date(2021, 12, 31))

    def test_data_on_either_edge_is_refused(self):
        # 2020-12-28 .. 2021-01-01: the last date is the sealed start.
        # 2021-12-31 .. 2022-01-04: the first date is the sealed end.
        for start in (dt.date(2020, 12, 28), dt.date(2021, 12, 31)):
            with self.assertRaises(SealedPeriodError, msg=str(start)):
                run(market=market(n=5, start=start), sealed=self.SEALED)

    def test_one_day_outside_is_accepted(self):
        log = self.make_log(periods=[("2020-01-01", "2020-06-30")] * 3)
        m = market(n=10, start=dt.date(2020, 12, 22))                   # ends 2020-12-31
        self.assertEqual(run(market=m, trial_log=log, sealed=self.SEALED).check("D").status, "PASS")
        m = market(n=10, start=dt.date(2022, 1, 1))
        self.assertEqual(run(market=m, trial_log=log, sealed=self.SEALED).check("D").status, "PASS")

    def test_a_date_only_end_includes_the_whole_day_and_trade_entries_are_checked(self):
        late = Trades([0.1], [0.0], entries=[dt.datetime(2021, 12, 31, 23, 0)])
        with self.assertRaises(SealedPeriodError):
            run(trades=late, sealed=self.SEALED)
        with self.assertRaises(SealedPeriodError):
            run(trades=Trades([0.1], [0.0], entries=["2021-06-01"]), sealed=self.SEALED)

    def test_trial_log_records_in_the_sealed_period_fail(self):
        log = self.make_log(sharpes=(0.01, 0.02), periods=[("2020-01-01", "2020-06-30"), ("2020-07-01", "2021-01-05")])
        r = run(trial_log=log, sealed=self.SEALED).check("D")
        self.assertEqual(r.status, "FAIL")
        self.assertIn("T0002", r.reasons[0])

    def test_opening_it_once_warns_twice_fails(self):
        once = self.make_log(sharpes=(0.01,), periods=[("2021-01-01", "2021-12-31")])
        m = market(n=30, start=dt.date(2021, 3, 1))
        r = run(market=m, trial_log=once, sealed=self.SEALED, open_sealed=True).check("D")
        self.assertEqual(r.status, "WARN")
        twice = self.make_log(sharpes=(0.01, 0.02), periods=[("2021-01-01", "2021-12-31")] * 2)
        r = run(market=m, trial_log=twice, sealed=self.SEALED, open_sealed=True).check("D")
        self.assertEqual(r.status, "FAIL")

    def test_no_sealed_period_warns(self):
        self.assertEqual(run().check("D").status, "WARN")
        with self.assertRaises(ValueError):
            run(open_sealed=True)

    def test_numbers_as_dates_are_refused_by_the_guard(self):
        m = market(n=5)
        m.dates = [1, 2, 3, 4, 5]
        with self.assertRaises(ValueError):
            run(market=m, sealed=self.SEALED)


# --------------------------------------------------------------------------- E, F, G, H

class CrossMarket(unittest.TestCase):
    def test_the_threshold_is_required(self):
        with self.assertRaises(ValueError):
            run(markets={"x": 1.0, "y": 2.0})

    def test_the_boundary_passes_and_losers_warn(self):
        th = Thresholds(min_markets_positive=2 / 3)
        r = run(markets={"x": 1.0, "y": 0.5, "z": -0.1}, thresholds=th).check("E")
        self.assertEqual(r.status, "WARN")                               # 2/3 passes; z warns
        self.assertTrue(any("z" in x for x in r.reasons))
        r = run(markets={"x": 1.0, "y": 0.0, "z": -0.1}, thresholds=th).check("E")
        self.assertEqual(r.status, "FAIL")                               # 1/3; 0 is not positive

    def test_one_market_is_not_run(self):
        self.assertEqual(run(markets={"x": 1.0}).check("E").status, "NOT RUN")


class Concentration(unittest.TestCase):
    def f(self, nets, labels):
        return run(trades=Trades(gross=nets, costs=[0.0] * len(nets), slices={"year": labels})).check("F")

    def test_more_than_half_from_one_slice_fails(self):
        self.assertEqual(self.f([60, 40], ["2020", "2021"]).status, "FAIL")
        self.assertEqual(self.f([50, 50], ["2020", "2021"]).status, "PASS")   # exactly half passes
        self.assertEqual(self.f([30, 30, 40], ["a", "a", "b"]).status, "FAIL")  # a: 60%

    def test_no_net_profit_fails(self):
        r = self.f([10, -10], ["2020", "2021"])
        self.assertEqual(r.status, "FAIL")
        self.assertIn("no net profit", r.reasons[0])

    def test_costs_are_taken_off_first(self):
        t = Trades(gross=[60, 60], costs=[10, 30], slices={"y": ["a", "b"]})   # net 50, 30: a 62.5%
        self.assertEqual(run(trades=t).check("F").status, "FAIL")


class TradeSummary(unittest.TestCase):
    def test_known_values(self):
        # results 1, -1, 2, -1, -1, 3: running sum 1, 0, 2, 1, 0, 3 -> worst fall from 2 to 0 = 2;
        # wins 6, losses 3 -> profit factor 2; longest run of losses 2.
        s = trade_summary([1, -1, 2, -1, -1, 3])
        self.assertEqual(s, {"trades": 6, "expectancy": 0.5, "profit_factor": 2.0, "win_rate": 0.5, "total": 3.0,
                             "max_drawdown": 2.0, "longest_losing_streak": 2})

    def test_edge_cases(self):
        self.assertEqual(trade_summary([1, 2])["profit_factor"], math.inf)
        self.assertIsNone(trade_summary([0, 0])["profit_factor"])
        self.assertEqual(trade_summary([])["trades"], 0)
        self.assertEqual(trade_summary([-1, -2])["max_drawdown"], 3.0)          # from 0

    def test_the_floor(self):
        self.assertEqual(run(trades=trades(n=30)).check("G").status, "PASS")
        self.assertEqual(run(trades=trades(n=29)).check("G").status, "FAIL")
        r = run(trades=trades(n=30)).check("G")
        self.assertTrue(any("minimum, not enough" in x for x in r.reasons))


class CodeState(LogMixin, unittest.TestCase):
    def test_dirty_records_warn_with_their_ids(self):
        r = run(trial_log=self.make_log(dirty=True)).check("H")
        self.assertEqual(r.status, "WARN")
        self.assertIn("T0001, T0002, T0003", r.reasons[0])
        self.assertEqual(run(trial_log=self.make_log()).check("H").status, "PASS")

    def test_the_idea_filter(self):
        log = self.make_log(dirty=True, idea="other")
        r = run(trial_log=log, idea="mine").check("H")
        self.assertTrue(any("no records" in x for x in r.reasons))


# --------------------------------------------------------------------------- I

class ScorecardCheck(unittest.TestCase):
    def test_a_steady_strategy_beats_the_benchmarks(self):
        r = run(market=market()).check("I")
        self.assertIn(r.status, ("PASS", "WARN"))
        self.assertTrue(any("no exit costs" in x for x in r.reasons))

    def test_holding_is_not_above_hold(self):
        m = market()
        m.equity = sc.hold_equity(m.prices, m.budget, m.fee, m.cost)
        r = run(market=m).check("I")
        self.assertEqual(r.status, "FAIL")
        self.assertTrue(any("not above hold's" in x for x in r.reasons))

    def test_a_wipe_out_fails(self):
        m = market(n=20)
        m.equity = [900.0, 500.0] + [0.0] * 18
        r = run(market=m).check("I")
        self.assertTrue(any("reached 0" in x for x in r.reasons))


# --------------------------------------------------------------------------- verdict and report

class Verdict(unittest.TestCase):
    def results(self, **status):
        return [CheckResult(k, k, status.get(k, "PASS")) for k in ("A", "B", "B2", "C", "C2", "D", "E", "F", "G",
                                                                    "H", "I")]

    def test_any_fail_is_a_fail(self):
        self.assertEqual(checks._verdict(self.results(E="FAIL")), "FAIL")
        self.assertEqual(checks._verdict(self.results(A="NOT RUN", F="FAIL")), "FAIL")

    def test_a_missing_core_check_is_incomplete(self):
        for k in checks.CORE:
            self.assertEqual(checks._verdict(self.results(**{k: "NOT RUN"})), "INCOMPLETE", k)
        self.assertIn("G", checks.CORE)
        self.assertEqual(checks._verdict(self.results(E="NOT RUN", D="NOT RUN")), "PASS")

    def test_warnings_dont_fail(self):
        self.assertEqual(checks._verdict(self.results(B2="WARN", C2="WARN", H="WARN")), "PASS")

    def test_nothing_given_is_incomplete(self):
        rep = run()
        self.assertEqual(rep.verdict, "INCOMPLETE")
        self.assertTrue(any("core check(s) not run" in n for n in rep.notes))


class ReportFormats(LogMixin, unittest.TestCase):
    def full(self):
        return run(trades=trades(n=40, slices={"half": ["a", "b"] * 20}), market=market(),
                   trial_log=self.make_log(), random={"random_entry": random_result(1000.0)},
                   thresholds=Thresholds(dsr_min=0.5))

    def test_every_used_threshold_is_printed_with_its_source(self):
        rep = self.full()
        names = {t["name"]: t for t in rep.thresholds}
        for want in ("cost_multiplier", "max_cost_share", "random_percentile", "min_valid_runs", "min_random_runs",
                     "dsr_min", "alpha",
                     "sharpe_test_strict", "max_slice_share", "min_trades"):
            self.assertIn(want, names)
            self.assertIn(want, rep.text())
        self.assertEqual(names["dsr_min"]["source"], "set by you")
        self.assertEqual(names["min_trades"]["source"], "standard")
        self.assertNotIn("min_markets_positive", names)          # E wasn't run: its threshold isn't listed
        self.assertNotIn("direction_strict", names)

    def test_json_round_trips_with_infinities_as_text(self):
        rep = run(trades=Trades(gross=[1.0] * 30, costs=[0.0] * 30))   # no losses: profit factor inf
        d = json.loads(rep.json())
        self.assertEqual(d["verdict"], rep.verdict)
        g = next(c for c in d["checks"] if c["key"] == "G")
        self.assertEqual(g["numbers"]["profit_factor"], "inf")
        self.assertEqual(d["disclaimer"], checks.DISCLAIMER)

    def test_the_text_ends_with_the_disclaimer(self):
        text = self.full().text()
        self.assertTrue(text.rstrip().endswith(checks.DISCLAIMER))
        self.assertIn("VERDICT:", text)

    def test_unknown_or_bad_thresholds_are_refused(self):
        for kw in ({"dsr": 0.9}, {"dsr_min": 1.5}, {"min_trades": 0}, {"cost_multiplier": 0.5},
                   {"direction_strict": 1}, {"min_markets_positive": 0}):
            with self.assertRaises(ValueError, msg=repr(kw)):
                Thresholds(**kw)


if __name__ == "__main__":
    unittest.main()
