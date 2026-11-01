"""Inputs and report contents: random direction from the actual trades, git history of an
uncommitted log, random trade exits in the sealed period, trades against the market,
negligible costs, the settings and thresholds a report prints, and the README's statement of
what the tool can't do. Each test fails without the fix it guards. All data is synthetic: it shows the code works, not that any strategy works.
"""

import datetime as dt
import json
import os
import shutil
import subprocess
import tempfile
import unittest

from honest_backtest import scorecard as sc, trials
from honest_backtest.checks import SealedPeriodError, Thresholds, Trades, run
from honest_backtest.random_baseline import Config, random_baseline

try:
    from test_checks import market, trades
except ImportError:                                  # run as tests.test_checks_inputs_and_reports
    from tests.test_checks import market, trades

def side_exit(entry, side, stop):
    return 0.01 * side, entry + 1


class DirectionFromActualTrades(unittest.TestCase):
    def b2(self, sides, long_share=None):
        cfg = Config(eligible=range(500), actual=[(i * 10, s, 1.0) for i, s in enumerate(sides)],
                     long_share=long_share)
        return run(random=random_baseline({"A": cfg}, side_exit, one_at_a_time=False, runs=400)).check("B2")

    def test_all_long_actual_trades_are_one_sided_whatever_long_share_says(self):
        r = self.b2([1] * 20, long_share=0.5)
        self.assertEqual(r.status, "NOT RUN")
        self.assertIn("actual trades are one-sided", r.reasons[0])

    def test_mixed_actual_trades_run_even_if_long_share_is_set_to_1(self):
        self.assertNotEqual(self.b2([1, -1] * 10, long_share=1.0).status, "NOT RUN")


class HistoryNeedsACommittedLog(unittest.TestCase):
    def test_a_path_with_no_committed_version_raises(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        git = ["git", "-C", tmp, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
               "-c", "commit.gpgsign=false"]
        subprocess.run(git + ["init", "-q"], check=True)
        with open(os.path.join(tmp, "other.txt"), "w", encoding="utf-8") as fh:
            fh.write("x")
        subprocess.run(git + ["add", "other.txt"], check=True)
        subprocess.run(git + ["commit", "-q", "-m", "one"], check=True)
        with self.assertRaises(ValueError) as e:
            trials.history_problems(tmp, "trails.jsonl")                       # mistyped
        self.assertIn("no committed version", str(e.exception))


class RandomTradeExitsInTheSealedGuard(unittest.TestCase):
    SEALED = (dt.date(2021, 1, 1), dt.date(2021, 12, 31))

    def baseline(self, hold_days):
        dates = [dt.date(2020, 10, 1) + dt.timedelta(days=i) for i in range(200)]
        idx = {d: i for i, d in enumerate(dates)}
        eligible = [d for d in dates if d < dt.date(2020, 12, 1)]               # all before the sealed period
        cfg = Config(eligible=eligible, actual=[(eligible[i], 1 if i % 2 else -1, 1.0) for i in range(0, 60, 6)])
        return random_baseline({"A": cfg}, lambda e, s, st: (0.01 * s, dates[idx[e] + hold_days]),
                               one_at_a_time=False, runs=20)

    def test_random_trades_ending_in_the_sealed_period_are_refused(self):
        with self.assertRaises(SealedPeriodError) as e:
            run(random=self.baseline(hold_days=70), sealed=self.SEALED)         # some exits in 2021
        self.assertIn("random trade exits", str(e.exception))
        run(random=self.baseline(hold_days=1), sealed=self.SEALED)              # all exits in 2020

    def test_exits_that_are_not_dates_are_reported(self):
        cfg = Config(eligible=range(50), actual=[(i, 1 if i % 2 else -1, 1.0) for i in range(0, 50, 5)])
        rb = random_baseline({"A": cfg}, side_exit, one_at_a_time=False, runs=5)
        self.assertGreaterEqual(rb["random_entry"]["exit_span"]["A"][0], 1)
        r = run(random=rb, sealed=self.SEALED).check("D")
        self.assertTrue(any("random trade exits are not dates" in x for x in r.reasons), r.reasons)


class TradesAgainstTheMarket(unittest.TestCase):
    def trades(self, n=40, net=0.002, entries=None, unit="return"):
        return Trades(gross=[net] * n, costs=[0.0] * n, entries=entries, unit=unit)

    def test_trade_dates_outside_the_market_dates_warn(self):
        m = market(n=300)
        late = [m.dates[0] + dt.timedelta(days=5 * i) for i in range(40)]
        late[-1] = m.dates[-1] + dt.timedelta(days=3)
        r = run(trades=self.trades(entries=late, unit=None), market=m).check("G")
        self.assertEqual(r.status, "WARN")
        self.assertTrue(any("1 trade entries fall outside the market dates" in x for x in r.reasons), r.reasons)

    def test_a_total_unlike_the_equity_change_warns(self):
        m = market(n=300)
        growth = m.equity[-1] / m.budget - 1
        each = (1 + growth) ** (1 / 40) - 1                                   # compounds to the equity change
        self.assertEqual(run(trades=self.trades(net=each), market=m).check("G").status, "PASS")
        r = run(trades=self.trades(net=each * 2), market=m).check("G")
        self.assertEqual(r.status, "WARN")
        self.assertTrue(any("doesn't roughly match the equity curve's change" in x for x in r.reasons), r.reasons)

    def test_money_units_and_no_unit(self):
        m = market(n=300)
        change = m.equity[-1] - m.budget
        self.assertEqual(run(trades=self.trades(net=change / 40, unit="money"), market=m).check("G").status, "PASS")
        r = run(trades=self.trades(unit=None), market=m).check("G")
        self.assertTrue(any("was not compared with the equity curve" in x for x in r.reasons), r.reasons)
        with self.assertRaises(ValueError):
            run(trades=self.trades(unit="R"))


README = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "README.md")


class NegligibleCosts(unittest.TestCase):
    def test_costs_under_1_percent_of_the_edge_warn(self):
        r = run(trades=trades(gross=0.3, cost=0.002)).check("A")            # 0.67% of the edge
        self.assertEqual(r.status, "WARN")
        self.assertTrue(any("costs look negligible" in x and "check they're entered" in x for x in r.reasons))
        self.assertEqual(run(trades=trades(gross=0.3, cost=0.003)).check("A").status, "PASS")   # exactly 1%


class SettingsAreReported(unittest.TestCase):
    def test_seed_runs_names_sharpe_key_and_benchmark_costs_are_printed(self):
        cfg = Config(eligible=range(500), actual=[(i * 10, 1 if i % 2 else -1, 1.0) for i in range(40)])

        def my_score(rs):
            return sum(rs) / len(rs)
        rb = random_baseline({"A": cfg}, lambda e, s, st: (0.01 * s, e + 1), one_at_a_time=False, runs=400,
                             seed=42, score=my_score)
        rep = run(random=rb, market=market(), sharpe_key="sr_daily")
        d = json.loads(rep.json())["settings"]
        self.assertEqual(d["random_entry"]["runs"], 400)
        self.assertEqual(d["random_entry"]["seed"], 42)
        self.assertIn("my_score", d["random_entry"]["score"])
        self.assertEqual(d["random_entry"]["select"], "best_score() (default)")
        self.assertEqual((d["sharpe_key"], d["benchmark_fee"], d["benchmark_cost"]), ("sr_daily", 0.001, 0.0005))
        self.assertIs(d["expected_head"], False)
        text = rep.text()
        for want in ("Settings used:", "sr_daily", "seed 42", "my_score", "benchmark_fee", "expected_head"):
            self.assertIn(want, text)

    def test_a_fee_or_cost_that_looks_like_a_percentage_is_refused(self):
        for kw in ({"fee": 0.05}, {"cost": 0.1}):
            args = {"budget": 1000, "fee": 0.001, "cost": 0.0005, **kw}
            with self.assertRaises(ValueError) as e:
                sc.hold_equity([1, 2], **args)
            self.assertIn("looks like a percentage", str(e.exception))
        m = market()
        m.fee = 0.2
        with self.assertRaises(ValueError):
            run(market=m)
        sc.hold_equity([1, 2], budget=1000, fee=0.0499, cost=0.0)            # just under 5% is allowed


class LooserThresholds(unittest.TestCase):
    def test_a_looser_threshold_warns_and_names_it(self):
        rep = run(thresholds=Thresholds(dsr_min=0.5, max_slice_share=0.8, allow_overrides=True))
        t = rep.check("T")
        self.assertEqual(t.status, "WARN")
        self.assertIn("looser than standard: dsr_min 0.5 < 0.95", t.reasons)
        self.assertIn("looser than standard: max_slice_share 0.8 > 0.5", t.reasons)
        self.assertIn("looser than standard: allow_overrides True (standard False)", t.reasons)

    def test_stricter_or_unchanged_is_fine(self):
        rep = run(thresholds=Thresholds(dsr_min=0.99, min_trades=100, alpha=0.01, sharpe_test_strict=True,
                                        max_cost_share=0.5))
        self.assertEqual(rep.check("T").status, "PASS")


class NamedTolerance(unittest.TestCase):
    def test_the_trades_vs_equity_tolerance_is_a_threshold(self):
        m = market(n=300)
        growth = m.equity[-1] / m.budget - 1
        each = (1 + growth * 1.15) ** (1 / 40) - 1                            # 15% off the equity's change
        t = Trades(gross=[each] * 40, costs=[0.0] * 40, unit="return")
        self.assertTrue(any("within 10%" in x for x in run(trades=t, market=m).check("G").reasons))
        loose = run(trades=t, market=m, thresholds=Thresholds(trades_equity_tolerance=0.2))
        self.assertFalse(any("roughly match" in x for x in loose.check("G").reasons))
        self.assertIn("trades_equity_tolerance", {x["name"] for x in loose.thresholds})


class HonestMistakesNotCheating(unittest.TestCase):
    def test_the_readme_says_so_near_the_top(self):
        with open(README, encoding="utf-8") as fh:
            text = fh.read()
        sentence = ("This tool helps catch honest mistakes and self-deception. It cannot stop deliberate\n"
                    "  cheating: anyone can fabricate inputs.")
        self.assertIn(sentence, text)
        self.assertLess(text.index(sentence), text.index("## Install"))


if __name__ == "__main__":
    unittest.main()
