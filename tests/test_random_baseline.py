"""Tests for honest_backtest.random_baseline.

All data is synthetic: it shows the code works, not that any strategy works. Every run is
seeded. Expected values are exact or worked out in the comments.
"""

import math
import random
import unittest

from honest_backtest.random_baseline import Config, best_score, mean_score, random_baseline


def const_exit(value):
    def exit_fn(entry, side, stop):
        return value, entry + 1
    return exit_fn


def side_exit(entry, side, stop):
    """+1 for a long, -1 for a short, whatever the entry."""
    return float(side), entry + 1


def binomial_quantile(n, q):
    """Exact q-quantile (smallest value with CDF >= q) of (2L - n) / n, L ~ Binomial(n, 1/2)."""
    cdf = 0.0
    for k in range(n + 1):
        cdf += math.comb(n, k) / 2 ** n
        if cdf >= q - 1e-12:
            return (2 * k - n) / n
    return 1.0


def cfg(n=20, pool=200, long_share=None, stops=(1.0, 2.0), sides=None):
    """A configuration with entries 0..pool-1 and n actual trades at entries 0, 10, 20, ..."""
    sides = sides or [1] * n
    actual = [(10 * i, sides[i], stops[i % len(stops)]) for i in range(n)]
    return Config(eligible=range(pool), actual=actual, long_share=long_share)


class KnownAnswers(unittest.TestCase):
    def test_every_trade_the_same_gives_every_run_the_same(self):
        out = random_baseline({"A": cfg()}, const_exit(1.0), one_at_a_time=False, runs=50)
        for bench in ("random_entry", "random_direction"):
            r = out[bench]
            self.assertEqual(r["values"], [1.0] * 50)
            self.assertEqual((r["strategy"], r["p95"], r["percentile"]), (1.0, 1.0, 0.5))  # all ties
            self.assertFalse(r["beats_p95"])                     # strictly above, not equal
            self.assertEqual(r["p_value"], 1.0)                   # (1 + 50) / (1 + 50)
            self.assertEqual((r["valid_runs"], r["too_few_trades"], r["failed"]), (50, 0, 0))

    def test_all_long_scores_one(self):
        out = random_baseline({"A": cfg(long_share=1.0)}, side_exit, one_at_a_time=False, runs=30,
                              benchmarks=("random_entry",))
        self.assertEqual(out["random_entry"]["values"], [1.0] * 30)

    def test_coin_flips_follow_the_binomial(self):
        # Each trade is +1 or -1 with p 1/2, so a run's mean is (2L - n)/n, L ~ Binomial(20, 1/2).
        # The 95th percentile of 2000 seeded runs is within one step (2/20) of the exact one.
        out = random_baseline({"A": cfg(long_share=0.5)}, side_exit, one_at_a_time=False, runs=2000,
                              benchmarks=("random_entry",))
        r = out["random_entry"]
        self.assertLessEqual(abs(r["p95"] - binomial_quantile(20, 0.95)), 0.1 + 1e-12)
        self.assertAlmostEqual(sum(r["values"]) / 2000, 0.0, delta=0.02)   # sd of the mean ~ 0.005
        self.assertTrue(all(round(v * 20) % 2 == 0 for v in r["values"]))  # only (2L - n)/n values

    def test_using_the_whole_pool_gives_the_exact_mean(self):
        vals = [((i * 37) % 11) / 10 for i in range(30)]
        c = Config(eligible=range(30), actual=[(i, 1, 1.0) for i in range(30)])
        out = random_baseline({"A": c}, lambda e, s, st: (vals[e], e), one_at_a_time=False, runs=20,
                              benchmarks=("random_entry",))
        self.assertTrue(all(abs(v - sum(vals) / 30) < 1e-12 for v in out["random_entry"]["values"]))

    def test_the_mean_score_and_percentile_maths(self):
        self.assertIsNone(mean_score([]))
        self.assertEqual(mean_score([1, 2, 6]), 3)
        # strategy 2 among values [1, 2, 3, 4]: one below, one tie -> (1 + 0.5) / 4.
        c = cfg(n=1, pool=4, stops=(1.0,))
        vals = {0: 2.0, 1: 1.0, 2: 3.0, 3: 4.0}
        out = random_baseline({"A": c}, lambda e, s, st: (vals[e], e), one_at_a_time=False, runs=400,
                              benchmarks=("random_entry",))
        r = out["random_entry"]
        below = sum(v < 2 for v in r["values"])
        ties = sum(v == 2 for v in r["values"])
        self.assertAlmostEqual(r["percentile"], (below + 0.5 * ties) / 400, places=12)
        self.assertAlmostEqual(r["p_value"], (1 + sum(v >= 2 for v in r["values"])) / 401, places=12)


class DrawnLikeTheStrategy(unittest.TestCase):
    def test_trade_count_long_share_and_stops_come_from_the_actual_trades(self):
        seen = []

        def exit_fn(entry, side, stop):
            seen.append((side, stop))
            return 0.0, entry
        sides = [1, 1, 1, -1] * 5                                  # 75% long
        c = cfg(n=20, sides=sides, stops=(0.5, 1.5, 4.0))
        random_baseline({"A": c}, exit_fn, one_at_a_time=False, runs=200, benchmarks=("random_entry",))
        random_seen = seen[20:]                                     # the first 20 are the real trades
        self.assertEqual({st for _, st in random_seen}, {0.5, 1.5, 4.0})
        share = sum(1 for s, _ in random_seen if s == 1) / len(random_seen)
        self.assertAlmostEqual(share, 0.75, delta=0.02)

    def test_one_at_a_time_never_overlaps_and_still_reaches_the_count(self):
        kept = []

        def exit_fn(entry, side, stop):
            return entry, entry + 5                                 # the result is the entry itself
        c = Config(eligible=range(300), actual=[(i * 10, 1, 1.0) for i in range(25)])
        out = random_baseline({"A": c}, exit_fn, one_at_a_time=True, runs=50, benchmarks=("random_entry",),
                              select=lambda s: kept.append(sorted(s["A"]["results"])) or 0.0)
        self.assertEqual(out["random_entry"]["valid_runs"], 50)
        for entries in kept[1:]:                                    # kept[0] is the real selection
            self.assertEqual(len(entries), 25)
            self.assertTrue(all(b >= a + 5 for a, b in zip(entries, entries[1:])))

    def test_a_pool_too_small_is_too_few_trades(self):
        c = Config(eligible=range(10), actual=[(i, 1, 1.0) for i in range(5)], n_trades=11)
        r = random_baseline({"A": c}, const_exit(1.0), one_at_a_time=False, runs=7,
                            benchmarks=("random_entry",))["random_entry"]
        self.assertEqual((r["too_few_trades"], r["valid_runs"]), (7, 0))
        self.assertIsNone(r["percentile"])
        self.assertFalse(r["beats_p95"])

    def test_trades_that_cant_be_taken_are_redrawn_or_counted(self):
        # Odd entries can't be taken: random entry redraws until it has 10 even ones.
        def exit_fn(entry, side, stop):
            return (1.0, entry) if entry % 2 == 0 else None
        c = Config(eligible=range(100), actual=[(2 * i, 1, 1.0) for i in range(10)])
        r = random_baseline({"A": c}, exit_fn, one_at_a_time=False, runs=20,
                            benchmarks=("random_entry",))["random_entry"]
        self.assertEqual(r["valid_runs"], 20)
        # Only 5 even entries in the pool, but 10 trades wanted: too few, after the redraws.
        c = Config(eligible=range(10), actual=[(0, 1, 1.0)] * 1, n_trades=6)
        r = random_baseline({"A": c}, exit_fn, one_at_a_time=False, runs=5,
                            benchmarks=("random_entry",))["random_entry"]
        self.assertEqual(r["too_few_trades"], 5)

    def test_the_redraw_limit_is_a_parameter(self):
        # One position at a time, every trade lasting 1000 periods: only one trade ever fits,
        # so 2 trades are never reached, however many rounds are allowed.
        c = Config(eligible=range(50), actual=[(0, 1, 1.0), (1, 1, 1.0)])
        for rounds in (1, 30):
            r = random_baseline({"A": c}, lambda e, s, st: (1.0, e + 1000), one_at_a_time=True, runs=3,
                                max_rounds=rounds, benchmarks=("random_entry",),
                                real_results={"A": [1.0, 1.0]})["random_entry"]
            self.assertEqual(r["too_few_trades"], 3)


class ConfigsAreNotChanged(unittest.TestCase):
    def test_the_users_config_is_left_as_it_was(self):
        c = cfg(n=20, sides=[1, -1] * 10)
        before = (c.eligible, list(c.actual), c.n_trades, c.long_share, c.stops)
        random_baseline({"A": c}, side_exit, one_at_a_time=True, runs=5)
        self.assertEqual((c.eligible, c.actual, c.n_trades, c.long_share, c.stops), before)
        self.assertIsNone(c.n_trades)                               # defaults were not written back


class RealResultsCheckedAgainstExitFn(unittest.TestCase):
    def run_with(self, real, exit_fn=side_exit):
        return random_baseline({"A": cfg(n=4, sides=[1, 1, -1, 1])}, exit_fn, one_at_a_time=False, runs=5,
                               real_results={"A": real})

    def test_matching_results_give_no_warning(self):
        out = self.run_with([1.0, 1.0, -1.0, 1.0 + 1e-13])          # within rounding
        self.assertEqual(out["random_entry"]["warnings"], [])
        self.assertEqual(out["random_direction"]["warnings"], [])

    def test_different_results_are_a_warning(self):
        out = self.run_with([1.0, 2.0, -1.0, 3.0])                  # two of four differ
        for bench in ("random_entry", "random_direction"):
            w = out[bench]["warnings"]
            self.assertEqual(len(w), 1)
            self.assertIn("2 of 4 trades", w[0])
            self.assertIn("may not match the strategy", w[0])

    def test_trades_exit_fn_cant_take_and_a_count_mismatch_are_warnings(self):
        def exit_fn(entry, side, stop):
            return None if entry == 10 else (float(side), entry)
        w = self.run_with([1.0, 1.0, -1.0, 1.0], exit_fn)["random_entry"]["warnings"]
        self.assertTrue(any("can't take 1" in x for x in w), w)
        w = self.run_with([1.0, 1.0, -1.0])["random_entry"]["warnings"]
        self.assertTrue(any("3 real results but 4 actual trades" in x for x in w), w)

    def test_tuple_results_are_compared_element_by_element(self):
        ex = lambda e, s, st: ((float(s), 2.0 * s), e)                 # noqa: E731
        first = lambda rs: rs[0][0]                                    # noqa: E731  score of tuples
        c = cfg(n=2)
        same = random_baseline({"A": c}, ex, one_at_a_time=False, runs=3, score=first,
                               real_results={"A": [(1.0, 2.0), (1.0, 2.0)]})
        self.assertEqual(same["random_entry"]["warnings"], [])
        other = random_baseline({"A": c}, ex, one_at_a_time=False, runs=3, score=first,
                                real_results={"A": [(1.0, 2.0), (1.0, 2.5)]})
        self.assertIn("1 of 2 trades", other["random_entry"]["warnings"][0])


class FairBestOfN(unittest.TestCase):
    def test_the_selection_rule_is_applied_to_random_versions_of_every_config(self):
        # Config A's entries (0-99) always return 1, config B's (100-199) always 2. With
        # select = max every random run is 2, as is the strategy; with select = min, 1.
        a = Config(eligible=range(0, 100), actual=[(0, 1, 1.0)])
        b = Config(eligible=range(100, 200), actual=[(100, 1, 1.0)])
        ex = lambda e, s, st: (1.0 if e < 100 else 2.0, e)        # noqa: E731
        top = random_baseline({"A": a, "B": b}, ex, one_at_a_time=False, runs=10,
                              benchmarks=("random_entry",))["random_entry"]
        self.assertEqual((top["strategy"], top["values"]), (2.0, [2.0] * 10))
        low = random_baseline({"A": a, "B": b}, ex, one_at_a_time=False, runs=10, benchmarks=("random_entry",),
                              select=lambda s: min(v["score"] for v in s.values()))["random_entry"]
        self.assertEqual((low["strategy"], low["values"]), (1.0, [1.0] * 10))

    def test_min_trades_in_the_selection_rule(self):
        sel = best_score(min_trades=3)
        self.assertEqual(sel({"A": {"score": 5.0, "trades": 2}, "B": {"score": 1.0, "trades": 3}}), 1.0)
        self.assertIsNone(sel({"A": {"score": 5.0, "trades": 2}}))
        self.assertIsNone(sel({"A": {"score": float("nan"), "trades": 9}}))

    def test_a_rule_that_picks_nothing_counts_as_failed(self):
        calls = []

        def select(summary):
            calls.append(1)
            return 1.0 if len(calls) == 1 else None                # real: 1.0; every random run: none
        r = random_baseline({"A": cfg()}, const_exit(1.0), one_at_a_time=False, runs=6, select=select,
                            benchmarks=("random_entry",))["random_entry"]
        self.assertEqual((r["failed"], r["valid_runs"]), (6, 0))
        with self.assertRaises(ValueError):
            random_baseline({"A": cfg()}, const_exit(1.0), one_at_a_time=False, runs=2, select=lambda s: None)

    def test_fair_comparison_rejects_about_5_percent_under_the_null(self):
        # Seeded null: each "real" configuration is itself 10 random entries from the same pool,
        # so no skill. Best of 2 against fair best-of-2 random runs should beat the 95th
        # percentile about 5% of the time; against single random runs about 1 - 0.95^2 = 9.75%.
        pool = 100
        fair = unfair = 0
        trials = 200
        for t in range(trials):
            rnd = random.Random(500 + t)
            vals = [rnd.gauss(0, 1) for _ in range(pool)]
            ex = lambda e, s, st, v=vals: (v[e], e)               # noqa: E731
            configs = {}
            for name in ("A", "B"):
                picks = rnd.sample(range(pool), 10)
                configs[name] = Config(eligible=range(pool), actual=[(e, 1, 1.0) for e in picks])
            r = random_baseline(configs, ex, one_at_a_time=False, runs=100, seed=t,
                                benchmarks=("random_entry",))["random_entry"]
            fair += r["beats_p95"]
            best = max(configs, key=lambda n: mean_score([vals[e] for e, _, _ in configs[n].actual]))
            single = random_baseline({"A": configs["A"]}, ex, one_at_a_time=False, runs=100, seed=t,
                                     benchmarks=("random_entry",),
                                     real_results={"A": [vals[e] for e, _, _ in configs[best].actual]})
            unfair += single["random_entry"]["beats_p95"]
        # Measured with these seeds: fair 7 of 200 (3.5%), unfair 18 of 200 (9%).
        self.assertGreaterEqual(fair, 1)
        self.assertLessEqual(fair, 0.10 * trials)                   # about 5%
        self.assertGreaterEqual(unfair, 2 * fair)                   # rigged in the strategy's favour


class RandomDirection(unittest.TestCase):
    def test_it_uses_exactly_the_actual_entries_and_stops(self):
        seen = []

        def exit_fn(entry, side, stop):
            seen.append((entry, stop))
            return float(side), entry
        c = cfg(n=8, stops=(0.5, 3.0))
        random_baseline({"A": c}, exit_fn, one_at_a_time=True, runs=25, benchmarks=("random_direction",))
        actual = [(e, st) for e, _, st in c.actual]
        self.assertEqual(seen, actual * 26)                          # the real trades, then 25 runs

    def test_coin_flip_follows_the_binomial_and_says_so(self):
        r = random_baseline({"A": cfg(n=20)}, side_exit, one_at_a_time=False, runs=2000,
                            benchmarks=("random_direction",))["random_direction"]
        self.assertLessEqual(abs(r["p95"] - binomial_quantile(20, 0.95)), 0.1 + 1e-12)
        self.assertEqual((r["direction"], r["direction_p"]), ("coin", {"A": 0.5}))
        self.assertEqual(r["strategy"], 1.0)                         # all 20 real trades long

    def test_the_strategys_own_long_share_as_an_option(self):
        c = cfg(n=20, sides=[1] * 15 + [-1] * 5)
        r = random_baseline({"A": c}, side_exit, one_at_a_time=False, runs=2000, direction="long_share",
                            benchmarks=("random_direction",))["random_direction"]
        self.assertEqual((r["direction"], r["direction_p"]), ("long_share", {"A": 0.75}))
        self.assertAlmostEqual(sum(r["values"]) / 2000, 0.5, delta=0.02)  # E = 2 * 0.75 - 1

    def test_a_flipped_trade_that_cant_be_taken_fails_the_run(self):
        def exit_fn(entry, side, stop):
            return (1.0, entry) if side == 1 else None
        r = random_baseline({"A": cfg(n=10)}, exit_fn, one_at_a_time=False, runs=30,
                            benchmarks=("random_direction",))["random_direction"]
        self.assertEqual(r["valid_runs"] + r["failed"], 30)
        self.assertGreater(r["failed"], 25)                          # all 10 long: (1/2)^10 chance


class Seeds(unittest.TestCase):
    def run_it(self, configs, seed=1):
        return random_baseline(configs, side_exit, one_at_a_time=False, runs=40, seed=seed)

    def test_same_seed_same_output_other_seed_other_output(self):
        a = self.run_it({"A": cfg(long_share=0.5)})
        b = self.run_it({"A": cfg(long_share=0.5)})
        c = self.run_it({"A": cfg(long_share=0.5)}, seed=2)
        self.assertEqual(a, b)
        self.assertNotEqual(a["random_entry"]["values"], c["random_entry"]["values"])

    def test_adding_a_config_doesnt_change_the_others_draws(self):
        seen = {"one": [], "two": []}

        def recorder(key):
            def exit_fn(entry, side, stop):
                if entry < 1000:
                    seen[key].append((entry, side, stop))
                return 0.0, entry
            return exit_fn
        a = cfg(long_share=0.5)
        b = Config(eligible=range(1000, 1200), actual=[(1000, 1, 1.0)])
        random_baseline({"A": a}, recorder("one"), one_at_a_time=False, runs=10, benchmarks=("random_entry",))
        random_baseline({"A": cfg(long_share=0.5), "B": b}, recorder("two"), one_at_a_time=False, runs=10,
                        benchmarks=("random_entry",))
        self.assertEqual(seen["one"], seen["two"])


class Inputs(unittest.TestCase):
    def test_one_at_a_time_is_required(self):
        with self.assertRaises(TypeError):
            random_baseline({"A": cfg()}, const_exit(1.0))           # pylint: disable=missing-kwoa
        with self.assertRaises(ValueError):
            random_baseline({"A": cfg()}, const_exit(1.0), one_at_a_time=1)

    def test_bad_configs_are_refused(self):
        bad = [
            Config(eligible=[], actual=[(0, 1, 1.0)]),
            Config(eligible=[1, 1, 2], actual=[(1, 1, 1.0)]),
            Config(eligible=[[1], [2]], actual=[(1, 1, 1.0)]),
            Config(eligible=range(5), actual=[(0, 2, 1.0)]),
            Config(eligible=range(5), actual=[(0, True, 1.0)]),
            Config(eligible=range(5), actual=[(0, 1, float("nan"))]),
            Config(eligible=range(5), actual=[(0, 1, 1.0)], long_share=1.5),
            Config(eligible=range(5), actual=[(0, 1, 1.0)], n_trades=0),
            Config(eligible=range(5), actual=[(0, 1, 1.0)], stops=[]),
            Config(eligible=range(5), actual=[]),
        ]
        for c in bad:
            with self.assertRaises(ValueError, msg=repr(c)):
                random_baseline({"A": c}, const_exit(1.0), one_at_a_time=False, runs=1)

    def test_bad_options_are_refused(self):
        for kw in ({"runs": 0}, {"max_rounds": 0}, {"direction": "up"}, {"benchmarks": ("x",)}, {"benchmarks": ()}):
            with self.assertRaises(ValueError, msg=repr(kw)):
                random_baseline({"A": cfg()}, const_exit(1.0), one_at_a_time=False, **kw)
        with self.assertRaises(ValueError):
            random_baseline({}, const_exit(1.0), one_at_a_time=False)

    def test_exit_fn_must_return_a_pair_and_its_errors_are_not_hidden(self):
        with self.assertRaises(TypeError):
            random_baseline({"A": cfg()}, lambda e, s, st: 1.0, one_at_a_time=False, runs=1)

        def broken(entry, side, stop):
            raise ZeroDivisionError("bug in the user's exit")
        with self.assertRaises(ZeroDivisionError):
            random_baseline({"A": cfg()}, broken, one_at_a_time=False, runs=1)

    def test_the_strategys_own_trades_must_be_takeable(self):
        with self.assertRaises(ValueError):
            random_baseline({"A": cfg()}, lambda e, s, st: None, one_at_a_time=False, runs=1)


if __name__ == "__main__":
    unittest.main()
