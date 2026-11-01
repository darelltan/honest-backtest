"""Ways a check could wrongly PASS, each closed: random baselines that don't match the
strategy, overridden defaults, day-0 equity above the budget, excluded settings, lines cut
off the end of the trial log, reruns with other results, unverified log completeness,
filled-in data and repeated trades. Each test fails without the fix it guards.
All data is synthetic: it shows the code works, not that any strategy works.
"""

import os
import shutil
import tempfile
import unittest

from honest_backtest import scorecard as sc, trials
from honest_backtest.checks import Thresholds, Trades, run
from honest_backtest.random_baseline import Config, random_baseline
from honest_backtest.trials import TrialLog

try:
    from test_checks import market
except ImportError:                                  # run as tests.test_checks_false_pass
    from tests.test_checks import market

def exit_fn(entry, side, stop):
    return 0.01 * side, entry + 1


def config(**kw):
    """Entries 0..999 eligible; 40 actual trades, 30 long and 10 short, stops 1 and 2."""
    actual = [(i * 20, 1 if i % 4 else -1, 1.0 + i % 2) for i in range(40)]
    return Config(eligible=range(1000), actual=actual, **kw)


def baseline(cfg=None, **kw):
    return random_baseline({"A": cfg or config()}, exit_fn, one_at_a_time=False, runs=400, **kw)


def given_trades(n=40, net=0.005):
    """Trades whose mean net matches the baseline's strategy value (30 longs +0.01, 10 shorts
    -0.01: mean 0.005), so only the mismatch under test is present."""
    return Trades(gross=[net] * n, costs=[0.0] * n)


class TradeCountAndOverrides(unittest.TestCase):
    def test_a_different_random_trade_count_fails(self):
        r = run(random=baseline(config(n_trades=25))).check("B")
        self.assertEqual(r.status, "FAIL")
        self.assertIn("config 'A' draws 25 random trades but the strategy has 40 actual trades", r.reasons)
        same = run(random=baseline(config(n_trades=40))).check("B")
        self.assertFalse(any("draws" in x for x in same.reasons))

    def test_overriding_long_share_or_stops_warns_and_names_it(self):
        r = run(random=baseline(config(long_share=0.9))).check("B")
        self.assertTrue(any("long_share overridden to 0.9 (its actual trades: 0.75)" in x for x in r.reasons),
                        r.reasons)
        r = run(random=baseline(config(stops=[5.0, 6.0, 7.0]))).check("B")
        self.assertTrue(any("stop distances overridden (3 values)" in x for x in r.reasons), r.reasons)
        plain = run(random=baseline()).check("B")
        self.assertFalse(any("overridden" in x for x in plain.reasons))


class MismatchesFail(unittest.TestCase):
    def test_real_results_disagreeing_with_exit_fn_fail(self):
        real = [0.01 if i % 4 else -0.01 for i in range(40)]
        real[3] = 0.5                                                 # one trade's result differs
        r = run(random=baseline(real_results={"A": real}), trades=given_trades()).check("B")
        self.assertEqual(r.status, "FAIL")
        self.assertTrue(any("different result from the real one for 1 of 40" in x for x in r.reasons), r.reasons)

    def test_real_results_without_actual_trades_to_check_fail(self):
        cfg = Config(eligible=range(1000), actual=[], n_trades=40, long_share=0.75, stops=[1.0])
        r = run(random=baseline(cfg, real_results={"A": [0.005] * 40})).check("B")
        self.assertEqual(r.status, "FAIL")
        self.assertTrue(any("no actual trades to check them against" in x for x in r.reasons), r.reasons)

    def test_a_strategy_value_or_count_unlike_the_given_trades_fails(self):
        # The actual entries (every 20th) return 0.05 whatever the side; other entries +-0.01.
        # So the strategy (mean 0.05) clearly beats random entries, and only a mismatch with
        # the given trades can make check B fail.
        def edge(entry, side, stop):
            return (0.05 if entry % 20 == 0 else 0.01 * side), entry + 1
        b = random_baseline({"A": config()}, edge, one_at_a_time=False, runs=400)
        self.assertEqual(run(random=b, trades=given_trades(net=0.05)).check("B").status, "PASS")
        r = run(random=b, trades=given_trades(net=0.04)).check("B")
        self.assertEqual(r.status, "FAIL")
        self.assertTrue(any("doesn't match the given trades' mean net" in x for x in r.reasons), r.reasons)
        r = run(random=b, trades=given_trades(n=39, net=0.05)).check("B")
        self.assertEqual(r.status, "FAIL")
        self.assertTrue(any("had 40 trade(s), but 39 trades were given" in x for x in r.reasons), r.reasons)


class DayZeroEquity(unittest.TestCase):
    def test_equity_above_the_budget_on_day_0_is_refused(self):
        kw = dict(budget=1000, fee=0.001, cost=0.0, periods_per_year=365, dca_every=1, draws=50)
        with self.assertRaises(ValueError) as e:
            sc.scorecard([1, 2, 3], [10, 11, 12], [1000.5, 1001, 1002], [0.0] * 3, **kw)
        self.assertIn("budget minus entry costs", str(e.exception))
        sc.scorecard([1, 2, 3], [10, 11, 12], [1000, 1001, 1002], [0.0] * 3, **kw)       # equal is fine
        sc.scorecard([1, 2, 3], [10, 11, 12], [999, 1001, 1002], [0.0] * 3, **kw)        # costs paid


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.path = os.path.join(self.tmp, "t.jsonl")


class ExcludedSettings(Tmp):
    def test_distinct_values_are_counted_per_idea_and_warned(self):
        log = TrialLog(self.path, exclude=("seed",))
        for seed in (1, 2, 3):
            log.record("a", {"hold": 5, "seed": seed}, {"sharpe": 0.01 * seed})
        log.record("b", {"hold": 5, "seed": 9}, {"sharpe": 0.02})
        r = run(market=market(), trial_log=log).check("C")
        self.assertEqual(r.numbers["excluded_values"], {"a": {"seed": 3}, "b": {"seed": 1}})
        self.assertTrue(any("idea 'a': setting 'seed'" in x and "3 distinct value(s)" in x for x in r.reasons),
                        r.reasons)
        self.assertIn(r.status, ("WARN", "FAIL"))

    def test_an_idea_with_every_setting_excluded_fails(self):
        log = TrialLog(self.path, exclude=("seed", "hold"))
        for seed in (1, 2):
            log.record("a", {"hold": seed, "seed": seed}, {"sharpe": 0.01 * seed})
        r = run(market=market(), trial_log=log).check("C")
        self.assertEqual(r.status, "FAIL")
        self.assertTrue(any("idea 'a': every setting is left out" in x for x in r.reasons), r.reasons)


class CutOffTheEnd(Tmp):
    def make(self, n=4):
        log = TrialLog(self.path)
        for i in range(n):
            log.record("a", {"i": i}, {"sharpe": 0.01 * (i + 1)})
        return log

    def test_verify_alone_doesnt_see_a_cut_end_but_a_stored_head_does(self):
        log = self.make()
        head = log.head()
        with open(self.path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        with open(self.path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines[:3]) + "\n")                         # the last trial cut off
        self.assertEqual(trials.verify(self.path), [])                    # documented limitation
        r = run(market=market(), trial_log=log, expected_head=head).check("C")
        self.assertEqual(r.status, "FAIL")
        self.assertIn("is not in the trial log", r.reasons[0])
        r = run(market=market(), trial_log=log, expected_min_records=4).check("C")
        self.assertEqual(r.status, "FAIL")
        self.assertIn("fewer than the 4 expected", r.reasons[0])

    def test_a_log_that_only_grew_still_passes_the_head_check(self):
        log = self.make(3)
        head = log.head()
        log.record("a", {"i": 9}, {"sharpe": 0.05})
        r = run(market=market(), trial_log=log, expected_head=head, expected_min_records=3).check("C")
        self.assertFalse(any("cut off" in x or "not in the trial log" in x for x in r.reasons), r.reasons)

    def test_the_docs_say_so(self):
        self.assertIn("cutting lines off the end", trials.__doc__)
        readme = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "README.md")
        with open(readme, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("Cutting lines off the end", text)
        self.assertIn("expected_head", text)


README = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "README.md")


class TmpLogs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def log(self, **kw):
        return TrialLog(os.path.join(self.tmp, "t%d.jsonl" % len(os.listdir(self.tmp))), **kw)


# --------------------------------------------------------------------------- 1

class RerunsWithOtherResults(TmpLogs):
    def test_the_same_settings_with_different_results_count_as_separate_trials(self):
        log = self.log()
        log.record("a", {"hold": 5}, {"sharpe": 0.01})
        log.record("a", {"hold": 5}, {"sharpe": 0.03})                 # same settings, a different result
        log.record("a", {"hold": 5}, {"sharpe": 0.03})                 # an exact rerun: no new trial
        log.record("a", {"hold": 6}, {"sharpe": 0.02})
        self.assertEqual(log.trial_count()["trials_total"], 2)        # what the log itself says
        r = run(market=market(), trial_log=log).check("C")
        self.assertEqual(r.numbers["trials"], 3)
        self.assertEqual(r.numbers["trials_split"], 1)
        self.assertTrue(any("1 trial(s) were split" in x and "3 trials, not the log's 2" in x for x in r.reasons),
                        r.reasons)

    def test_excluded_seeds_with_different_results_are_not_split(self):
        log = self.log(exclude=("seed",))
        for seed in (1, 2, 3):
            log.record("a", {"hold": 5, "seed": seed}, {"sharpe": 0.01 * seed})
        r = run(market=market(), trial_log=log).check("C")
        self.assertEqual((r.numbers["trials"], r.numbers["trials_split"]), (1, 0))


# --------------------------------------------------------------------------- 2

class Completeness(TmpLogs):
    def make(self):
        log = self.log()
        for i in range(3):
            log.record("a", {"i": i}, {"sharpe": 0.01 * (i + 1)})
        return log

    def test_unverified_completeness_warns(self):
        r = run(market=market(), trial_log=self.make()).check("C")
        self.assertIn("log completeness not verified (pass expected_head): lines cut off its end would not be seen",
                      r.reasons)
        self.assertNotEqual(r.status, "PASS")

    def test_what_was_compared_is_printed(self):
        log = self.make()
        r = run(market=market(), trial_log=log, expected_head=log.head(), expected_min_records=3).check("C")
        self.assertFalse(any("not verified" in x for x in r.reasons))
        self.assertTrue(any(x.startswith("log completeness: compared the stored head") and "record T0003 of 3" in x
                            for x in r.reasons), r.reasons)
        self.assertIn("log completeness: compared 3 records, at least the 3 expected", r.reasons)


# --------------------------------------------------------------------------- 3

class FilledInData(TmpLogs):
    def flat_log(self):
        log = self.log()
        log.record("a", {"i": 1}, {"sharpe": 0.01})
        return log

    def test_geometrically_interpolated_equity_warns_in_c_and_i(self):
        m = market(n=300)
        e = m.equity
        for t in range(101, 130):                                         # 30 periods of the same return
            e[t] = e[t - 1] * 1.001
        for t in range(130, 300):
            e[t] = e[t - 1] * (1 + 0.001 * ((t * 7919) % 5 - 1))
        rep = run(market=m, trial_log=self.flat_log())
        for key in ("C", "I"):
            self.assertTrue(any("identical consecutive returns" in x for x in rep.check(key).reasons), key)

    def test_linearly_interpolated_equity_warns(self):
        m = market(n=300)
        e = m.equity
        for t in range(101, 130):
            e[t] = e[t - 1] + 2.5                                         # equal steps
        r = run(market=m).check("I")
        self.assertNotEqual(r.status, "PASS")
        self.assertTrue(any("equal consecutive steps" in x for x in r.reasons), r.reasons)

    def test_forward_filled_prices_warn(self):
        m = market(n=300)
        for t in range(50, 70):
            m.prices[t] = m.prices[49]
        r = run(market=m).check("I")
        # flat prices are a warning (a price can't be out of the market), not just a note
        self.assertTrue(any(x.startswith("price returns: ") and "returns of exactly 0" in x and "forward-filled" in x
                            for x in r.reasons), r.reasons)
        self.assertNotEqual(r.status, "PASS")
        self.assertFalse(any(x.startswith("note: price returns") for x in r.reasons))

    def test_the_threshold_is_named_and_the_readme_explains_it(self):
        m = market(n=300)
        rep = run(market=m, thresholds=Thresholds(max_identical_run=3))
        self.assertIn("max_identical_run", {t["name"] for t in rep.thresholds})
        with open(README, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("Interpolated or filled-in data", text)
        self.assertIn("inflates", text)


# --------------------------------------------------------------------------- 4

class DuplicateTrades(unittest.TestCase):
    def test_a_repeated_actual_trade_is_refused(self):
        cfg = Config(eligible=range(100), actual=[(0, 1, 1.0), (10, 1, 1.0), (0, 1, 2.0)])
        with self.assertRaises(ValueError) as e:
            random_baseline({"A": cfg}, lambda e, s, st: (0.0, e + 1), one_at_a_time=False, runs=1)
        self.assertIn("repeat (entry 0, side +1)", str(e.exception))
        ok = Config(eligible=range(100), actual=[(0, 1, 1.0), (0, -1, 1.0)])   # same time, other side
        random_baseline({"A": ok}, lambda e, s, st: (0.0, e + 1), one_at_a_time=False, runs=1)

    def test_a_repeated_trade_in_trades_is_refused(self):
        with self.assertRaises(ValueError):
            run(trades=Trades(gross=[0.1, 0.1], costs=[0.0, 0.0], entries=["2020-01-02", "2020-01-02"]))
        with self.assertRaises(ValueError):
            run(trades=Trades(gross=[0.1, 0.1], costs=[0.0, 0.0], entries=[3, 3], sides=[1, 1]))
        run(trades=Trades(gross=[0.1, 0.1], costs=[0.0, 0.0], entries=[3, 3], sides=[1, -1]))


# --------------------------------------------------------------------------- 5

class OverridesFailUnlessAllowed(unittest.TestCase):
    def baseline(self, **kw):
        cfg = Config(eligible=range(1000), actual=[(i * 20, 1 if i % 4 else -1, 1.0) for i in range(40)], **kw)
        return random_baseline({"A": cfg}, lambda e, s, st: (0.01 * s, e + 1), one_at_a_time=False, runs=400)

    def test_an_override_fails_by_default(self):
        r = run(random=self.baseline(long_share=0.9)).check("B")
        self.assertEqual(r.status, "FAIL")
        self.assertTrue(any("long_share overridden" in x and "allow_overrides=True" in x for x in r.reasons))

    def test_allowed_overrides_only_warn_and_the_report_says_so(self):
        rep = run(random=self.baseline(stops=[3.0]), thresholds=Thresholds(allow_overrides=True))
        r = rep.check("B")
        self.assertTrue(any("stop distances overridden" in x and "allowed by allow_overrides=True" in x
                            for x in r.reasons), r.reasons)
        self.assertFalse(any("overridden" in x and "pass Thresholds" in x for x in r.reasons))
        t = {x["name"]: x for x in rep.thresholds}["allow_overrides"]
        self.assertEqual((t["value"], t["source"]), (True, "set by you"))
        self.assertIn("allow_overrides", rep.text())


if __name__ == "__main__":
    unittest.main()
