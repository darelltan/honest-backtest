"""Interpolated or stretched data: the share of repeated non-zero returns and steps, and
runs of zero returns as a softer note. Each test (except the false-alarm guard) fails
without the fix it guards. All data is synthetic: it shows the code works, not that any strategy works.
"""

import os
import shutil
import tempfile
import unittest

from honest_backtest.checks import Thresholds, run
from honest_backtest.trials import TrialLog

try:
    from test_checks import market
except ImportError:                                  # run as tests.test_checks_interpolation
    from tests.test_checks import market

def stretched(values, budget=None, parts=6):
    """Daily levels stretched to `parts` bars a day by geometric interpolation: each day's
    return is split into `parts` equal returns, as if 4-hour bars were made from daily data."""
    out = [values[0]]
    for a, b in zip(values, values[1:]):
        step = (b / a) ** (1 / parts)
        for _ in range(parts):
            out.append(out[-1] * step)
    return out


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.log = TrialLog(os.path.join(self.tmp, "t.jsonl"))
        self.log.record("a", {"i": 1}, {"sharpe": 0.01})

    def market(self, equity=None, prices=None):
        daily = market(n=120)
        eq = stretched(daily.equity) if equity is None else equity
        px = stretched(daily.prices) if prices is None else prices
        n = len(eq)
        m = market(n=n)
        m.equity, m.prices, m.risk_free, m.periods_per_year = eq, px, [0.0] * n, 365 * 6
        return m


class StretchedData(Base):
    def test_daily_data_stretched_to_4_hour_bars_warns_in_c_and_i(self):
        # Each day's move repeats 6 times: runs of 6 (under max_identical_run 10), but 5 of
        # every 6 non-zero returns equal the one before (83%, over max_repeated_share 20%).
        rep = run(market=self.market(), trial_log=self.log)
        for key in ("C", "I"):
            text = " | ".join(rep.check(key).reasons)
            self.assertIn("strategy returns: 83% of the", text, key)
            self.assertIn("equal the one before (more than 20%)", text, key)
        self.assertIn("price returns: 83% of the", " | ".join(rep.check("I").reasons))
        self.assertIn("max_repeated_share", {t["name"] for t in rep.thresholds})

    def test_linearly_stretched_equity_warns_by_its_steps(self):
        daily = market(n=120).equity
        eq = [daily[0]]
        for a, b in zip(daily, daily[1:]):
            for k in range(1, 7):
                eq.append(a + (b - a) * k / 6)                           # 6 equal steps a day
        r = run(market=self.market(equity=eq)).check("I")
        self.assertTrue(any("non-zero steps equal the one before" in x for x in r.reasons), r.reasons)

    def test_the_share_threshold_can_be_changed_and_looser_warns(self):
        rep = run(market=self.market(), trial_log=self.log, thresholds=Thresholds(max_repeated_share=0.9))
        self.assertFalse(any("equal the one before" in x for x in rep.check("C").reasons))
        self.assertIn("looser than standard: max_repeated_share 0.9 > 0.2", rep.check("T").reasons)


class ZeroRunsAreASofterNote(Base):
    def test_a_long_flat_stretch_of_strategy_equity_is_only_a_note(self):
        m = market(n=300)
        for t in range(100, 140):
            m.equity[t] = m.equity[99]                                  # 40 periods out of the market
        rep = run(market=m, trial_log=self.log)
        for key in ("C", "I"):
            reasons = rep.check(key).reasons
            notes = [x for x in reasons if x.startswith("note: strategy returns")]
            self.assertEqual(len(notes), 1, reasons)
            self.assertIn("may be time out of the market; ignore if so", notes[0])
            self.assertFalse(any("exactly 0" in x for x in reasons if not x.startswith("note:")), reasons)


class NoFalseAlarm(Base):
    def test_ordinary_random_data_raises_no_interpolation_warning(self):
        rep = run(market=market(n=500), trial_log=self.log)
        for key in ("C", "I"):
            self.assertFalse(any("equal the one before" in x or "identical consecutive" in x
                                 for x in rep.check(key).reasons), key)


if __name__ == "__main__":
    unittest.main()
