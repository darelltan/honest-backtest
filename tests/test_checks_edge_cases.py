"""Edge cases in the checks: zero costs, how far the sealed-period guard reaches, random
direction for one-sided strategies, canonical trial-log lines, excluded settings and
baselines run on other trades, and small input errors. Each test fails without the fix it
guards. All data is synthetic: it shows the code works, not that any strategy works.
"""

import datetime as dt
import json
import os
import shutil
import tempfile
import unittest

from honest_backtest import trials
from honest_backtest.checks import SealedPeriodError, Trades, run
from honest_backtest.random_baseline import Config, random_baseline
from honest_backtest.trials import TrialLog

try:
    from test_checks import market, random_result, trades
except ImportError:                                  # run as tests.test_checks_edge_cases
    from tests.test_checks import market, random_result, trades

SEALED = (dt.date(2021, 1, 1), dt.date(2021, 12, 31))


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def log(self, n=3, period=("2020-01-01", "2020-06-30"), **kw):
        log = TrialLog(os.path.join(self.tmp, "t%d.jsonl" % len(os.listdir(self.tmp))), **kw)
        for i in range(n):
            log.record("idea", {"i": i, "seed": i}, {"sharpe": 0.01 * (i + 1)}, period=period,
                       code={"commit": "abc", "dirty": False})
        return log


# --------------------------------------------------------------------------- 3

class ZeroCosts(unittest.TestCase):
    def test_all_zero_costs_warn_that_the_stress_test_does_nothing(self):
        r = run(trades=trades(gross=0.3, cost=0.0)).check("A")
        self.assertEqual(r.status, "WARN")
        self.assertIn("cost stress test has no effect: every trade's cost is 0", r.reasons)
        self.assertEqual(run(trades=trades(gross=0.3, cost=0.01)).check("A").status, "PASS")


# --------------------------------------------------------------------------- 4

class SealedGuardReach(Tmp):
    def dated_baseline(self, start):
        """A real random baseline whose eligible times are dates from ``start``."""
        dates = [start + dt.timedelta(days=i) for i in range(60)]
        idx = {d: i for i, d in enumerate(dates)}
        cfg = Config(eligible=dates[:50], actual=[(dates[i], 1 if i % 2 else -1, 1.0) for i in range(0, 50, 5)])
        return random_baseline({"A": cfg}, lambda e, s, st: (0.01 * s, dates[idx[e] + 1]), one_at_a_time=True,
                               runs=5)

    def test_it_reports_what_it_could_not_check(self):
        log = self.log(period=None)
        t = Trades(gross=[0.1] * 3, costs=[0.0] * 3)                     # no entry or exit dates
        r = run(trades=t, trial_log=log, sealed=SEALED, random={"random_entry": random_result(1000.0)}).check("D")
        self.assertEqual(r.status, "WARN")
        text = " | ".join(r.reasons)
        for want in ("trade entries were not given", "trade exits were not given",
                     "3 trial-log record(s) have no period: T0001, T0002, T0003",
                     "doesn't say which times were eligible"):
            self.assertIn(want, text)

    def test_index_eligible_times_cant_be_checked_and_say_so(self):
        cfg = Config(eligible=range(50), actual=[(i, 1 if i % 2 else -1, 1.0) for i in range(0, 50, 5)])
        rb = random_baseline({"A": cfg}, lambda e, s, st: (0.01 * s, e + 1), one_at_a_time=True, runs=5)
        r = run(random=rb, trial_log=self.log(), sealed=SEALED).check("D")
        self.assertTrue(any("config 'A''s eligible times are not dates" in x for x in r.reasons), r.reasons)

    def test_trade_exits_in_the_sealed_period_are_refused(self):
        t = Trades(gross=[0.1], costs=[0.0], entries=[dt.date(2020, 12, 30)], exits=[dt.date(2021, 1, 2)])
        with self.assertRaises(SealedPeriodError):
            run(trades=t, sealed=SEALED)

    def test_eligible_times_in_the_sealed_period_are_refused(self):
        with self.assertRaises(SealedPeriodError):
            run(random=self.dated_baseline(dt.date(2020, 12, 1)), sealed=SEALED)    # runs into 2021
        r = run(random=self.dated_baseline(dt.date(2020, 1, 1)), trial_log=self.log(), sealed=SEALED).check("D")
        self.assertEqual(r.status, "PASS")

    def test_a_date_is_its_whole_day_on_both_sides(self):
        # Sealed from noon on 2021-01-01: the date 2021-01-01 overlaps it, so it is refused.
        noon = (dt.datetime(2021, 1, 1, 12, 0), dt.datetime(2021, 12, 31, 12, 0))
        with self.assertRaises(SealedPeriodError):
            run(trades=Trades(gross=[0.1], costs=[0.0], entries=[dt.date(2021, 1, 1)], exits=[dt.date(2020, 6, 1)]),
                sealed=noon)
        with self.assertRaises(SealedPeriodError):
            run(market=market(n=5, start=dt.date(2020, 12, 28)), sealed=noon)          # ends 2021-01-01
        # A trial period ending on the date 2021-01-01 touched the sealed period too.
        log = self.log(n=1, period=("2020-06-01", "2021-01-01"))
        self.assertEqual(run(trial_log=log, sealed=noon).check("D").status, "FAIL")
        # The day before is fine.
        ok = self.log(n=1, period=("2020-06-01", "2020-12-31"))
        self.assertEqual(run(trial_log=ok, sealed=noon).check("D").status, "PASS")


# --------------------------------------------------------------------------- 5

class OneSidedDirection(unittest.TestCase):
    def baseline(self, sides):
        cfg = Config(eligible=range(100), actual=[(i * 5, s, 1.0) for i, s in enumerate(sides)])
        return random_baseline({"A": cfg}, lambda e, s, st: (0.01 * s, e + 1), one_at_a_time=False, runs=5)

    def test_random_direction_is_not_run_for_a_one_sided_strategy(self):
        for side in (1, -1):
            r = run(random=self.baseline([side] * 10)).check("B2")
            self.assertEqual(r.status, "NOT RUN")
            self.assertIn("one-sided", r.reasons[0])
        self.assertNotEqual(run(random=self.baseline([1, -1] * 5)).check("B2").status, "NOT RUN")

    def test_the_baseline_reports_long_shares_and_more(self):
        out = self.baseline([1, 1, 1, -1] * 5)["random_entry"]
        self.assertEqual(out["long_share"], {"A": 0.75})
        self.assertEqual(out["real_trades"], {"A": 20})
        self.assertEqual(out["eligible_span"], {"A": (0, 99)})
        self.assertTrue(out["score_is_mean"])


# --------------------------------------------------------------------------- 6

class CanonicalLines(Tmp):
    def test_a_repeated_key_is_caught(self):
        log = self.log(n=2)
        with open(log.path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        # Put a second, nicer "results" before the real one. JSON readers keep the last key,
        # so the record, its hash and the chain are all unchanged, but the line says two things.
        fake = lines[1].replace('"results":', '"results":{"sharpe":9.9},"results":', 1)
        self.assertEqual(json.loads(fake), json.loads(lines[1]))
        with open(log.path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(lines[0] + "\n" + fake + "\n")
        problems = trials.verify(log.path)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("line 2 is not in canonical form", problems[0])

    def test_extra_spaces_are_caught(self):
        log = self.log(n=1)
        with open(log.path, encoding="utf-8") as fh:
            line = fh.read().strip()
        with open(log.path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(line.replace('","', '", "', 1) + "\n")
        self.assertTrue(trials.verify(log.path))


# --------------------------------------------------------------------------- 8

class ExcludedAndMismatched(Tmp):
    def test_check_c_shows_excluded_settings(self):
        log = self.log(exclude=("seed",))
        r = run(market=market(), trial_log=log).check("C")
        self.assertEqual(r.numbers["excluded_settings"], ["seed"])
        self.assertTrue(any("left out of trial counting" in x and "seed" in x for x in r.reasons), r.reasons)

    def test_a_baseline_run_on_other_trades_is_reported(self):
        t = trades(n=40, gross=0.3, cost=0.1)                              # mean net 0.2
        same = random_result(0.2, values=[0.0] * 500, real_trades={"A": 40}, score_is_mean=True)
        self.assertEqual(run(trades=t, random={"random_entry": same}).check("B").reasons[1:], [])
        count = random_result(0.2, values=[0.0] * 500, real_trades={"A": 30}, score_is_mean=True)
        r = run(trades=t, random={"random_entry": count}).check("B")
        self.assertTrue(any("had 30 trade(s), but 40 trades were given" in x for x in r.reasons), r.reasons)
        value = random_result(0.25, values=[0.0] * 500, real_trades={"A": 40}, score_is_mean=True)
        r = run(trades=t, random={"random_entry": value}).check("B")
        self.assertTrue(any("doesn't match the given trades' mean net result" in x for x in r.reasons), r.reasons)

    def test_a_custom_score_is_not_compared_with_the_mean(self):
        t = trades(n=40, gross=0.3, cost=0.1)
        other = random_result(0.25, values=[0.0] * 500, real_trades={"A": 40}, score_is_mean=False)
        r = run(trades=t, random={"random_entry": other}).check("B")
        self.assertFalse(any("doesn't match" in x for x in r.reasons))


# --------------------------------------------------------------------------- 9

class SmallFixes(unittest.TestCase):
    def test_zero_runs_is_a_clear_error(self):
        bad = random_result(1.0, values=[], runs=0)
        with self.assertRaises(ValueError) as e:
            run(random={"random_entry": bad})
        self.assertIn("runs=0", str(e.exception))

    def test_the_docs_say_what_the_log_cant_do(self):
        self.assertIn("blocked", trials.__doc__)
        self.assertIn("unlock", trials.__doc__)
        self.assertIn("does not follow renames", trials.history_problems.__doc__)
        readme = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "README.md")
        with open(readme, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("Memmel, C. (2003)", text)
        self.assertIn("does not follow renames", text)
        self.assertIn("blocked", text)


if __name__ == "__main__":
    unittest.main()
