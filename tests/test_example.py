"""The synthetic example runs and gives the verdicts it promises (with fewer random runs,
for speed). Synthetic data: shows the code works, not that any strategy works."""

import importlib.util
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, os.pardir, "examples", "synthetic_example.py")


def load():
    spec = importlib.util.spec_from_file_location("synthetic_example", PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class SyntheticExample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load()
        cls.lines = []
        cls.reports = cls.mod.main(runs=400, draws=300, out=cls.lines.append)   # check B needs 400 runs

    def test_no_skill_fails_and_the_planted_edge_passes(self):
        self.assertEqual(self.reports["no skill"].verdict, "FAIL")
        self.assertEqual(self.reports["planted edge"].verdict, "PASS")

    def test_no_skill_fails_for_the_right_reasons(self):
        rep = self.reports["no skill"]
        for key in ("B", "C", "I"):
            self.assertEqual(rep.check(key).status, "FAIL", key)

    def test_every_check_ran_except_random_direction_for_these_one_sided_trades(self):
        for rep in self.reports.values():
            self.assertEqual([c.key for c in rep.checks if c.status == "NOT RUN"], ["B2"])
            self.assertIn("one-sided", rep.check("B2").reasons[0])

    def test_the_sealed_guard_checked_everything_it_could(self):
        for rep in self.reports.values():
            self.assertEqual(rep.check("D").status, "PASS")
            self.assertEqual(rep.check("D").numbers["not_checked"], [])

    def test_each_tried_setting_has_its_own_result(self):
        # Each of the five logged "settings" must have its own result.
        import tempfile
        from honest_backtest.trials import TrialLog
        for planted in (False, True):
            with tempfile.TemporaryDirectory() as folder:
                self.mod.scenario("x", planted, folder, runs=400, draws=50)
                recs = TrialLog(os.path.join(folder, "x.jsonl")).records()
            sharpes = [r["results"]["sharpe"] for r in recs]
            self.assertEqual([r["settings"]["hold_days"] for r in recs], [1, 2, 3, 4, 5])
            self.assertEqual(len(set(sharpes)), 5, sharpes)

    def test_the_synthetic_label_is_in_the_file_and_the_output(self):
        banner = "SYNTHETIC DATA. Shows the code works, not that any strategy works."
        self.assertEqual(self.mod.BANNER, banner)
        with open(PATH, encoding="utf-8") as fh:
            self.assertTrue(fh.read().startswith('"""SYNTHETIC DATA.'))
        output = "\n".join(self.lines)
        self.assertEqual(output.count(banner), 3)                    # each scenario, and the end
        self.assertTrue(output.rstrip().endswith(banner))


if __name__ == "__main__":
    unittest.main()
