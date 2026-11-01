"""Tests for honest_backtest.overfit.

Every expected value says where it comes from. Every test that uses randomness uses a
fixed seed, so a run can't fail by chance. Source keys ([BLdP2014], [LW2008], ...) are
listed in the module docstring of honest_backtest.overfit.
"""

import math
import random
import unittest

from honest_backtest import overfit
from honest_backtest.overfit import NORMAL


def exact_expected_max(n, steps=20000, lo=-12.0, hi=12.0):
    """E[max of n iid standard normals] = integral of x * n * pdf(x) * cdf(x)^(n-1) dx,
    by Simpson's rule. Independent of the code under test (only uses NormalDist)."""
    h = (hi - lo) / steps
    total = 0.0
    for i in range(steps + 1):
        x = lo + i * h
        w = 1 if i in (0, steps) else (4 if i % 2 else 2)
        total += w * x * n * NORMAL.pdf(x) * NORMAL.cdf(x) ** (n - 1)
    return total * h / 3


class SharpeAndMoments(unittest.TestCase):
    def test_known_small_series(self):
        # Hand computation for [1, 2, 3, 4, 10]: mean 4, deviations -3, -2, -1, 0, 6;
        # sum of squares 50, so sample sd = sqrt(50/4) = sqrt(12.5);
        # population m2 = 50/5 = 10, m3 = (-27 - 8 - 1 + 216)/5 = 36, m4 = (81 + 16 + 1 + 1296)/5 = 278.8.
        m = overfit.sharpe_and_moments([1, 2, 3, 4, 10])
        self.assertAlmostEqual(m["mean"], 4.0, places=12)
        self.assertAlmostEqual(m["sd"], math.sqrt(12.5), places=12)
        self.assertAlmostEqual(m["sr"], 4 / math.sqrt(12.5), places=12)
        self.assertAlmostEqual(m["skew"], 36 / 10 ** 1.5, places=12)
        self.assertAlmostEqual(m["kurt"], 278.8 / 100, places=12)
        self.assertEqual(m["periods"], 5)

    def test_symmetric_series_has_no_skew(self):
        # By symmetry: deviations -1, 0, +1 give m3 = 0.
        self.assertAlmostEqual(overfit.sharpe_and_moments([1, 2, 3])["skew"], 0.0, places=12)

    def test_empty_input_is_a_clear_error_not_a_crash(self):
        # An empty series must be a clear ValueError, not a ZeroDivisionError.
        for bad in ([], [0.01]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                overfit.sharpe_and_moments(bad)

    def test_bad_values_are_refused(self):
        for bad in ([0.1, float("nan")], [0.1, float("inf")], [0.1, True], [0.1, "x"]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                overfit.sharpe_and_moments(bad)

    def test_a_flat_series_has_no_sharpe(self):
        with self.assertRaises(ValueError):
            overfit.sharpe_and_moments([0.01] * 10)


class ProbabilisticSharpe(unittest.TestCase):
    def test_normal_returns_known_value(self):
        # Hand computation from the PSR formula ([BLdP2014] Eq. (2)) with skew 0, kurt 3:
        # z = 0.1 * sqrt(364) / sqrt(1 + (3 - 1)/4 * 0.01) = 0.1 * sqrt(364) / sqrt(1.005).
        p, valid = overfit.probabilistic_sharpe(0.1, 365, skew=0.0, kurt=3.0, sr0=0.0)
        self.assertTrue(valid)
        self.assertAlmostEqual(p, NORMAL.cdf(0.1 * math.sqrt(364) / math.sqrt(1.005)), places=12)

    def test_skewed_fat_tailed_known_value(self):
        # Hand computation: sr 0.2, sr0 0.05, n 501, skew -1, kurt 6:
        # denom = 1 + 0.2 + 5/4 * 0.04 = 1.25; z = 0.15 * sqrt(500) / sqrt(1.25) = 0.15 * 20 = 3.
        p, valid = overfit.probabilistic_sharpe(0.2, 501, skew=-1.0, kurt=6.0, sr0=0.05)
        self.assertTrue(valid)
        self.assertAlmostEqual(p, NORMAL.cdf(3.0), places=12)

    def test_formula_breakdown_is_flagged_and_fails(self):
        # Hand computation: sr 2, skew 1, kurt 1 gives denom = 1 - 2 + 0 = -1 < 0.
        p, valid = overfit.probabilistic_sharpe(2.0, 100, skew=1.0, kurt=1.0)
        self.assertEqual((p, valid), (0.0, False))

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            overfit.probabilistic_sharpe(0.1, 1, 0.0, 3.0)
        with self.assertRaises(ValueError):
            overfit.probabilistic_sharpe(float("nan"), 100, 0.0, 3.0)


class ExpectedMaxSharpe(unittest.TestCase):
    def test_the_integrator_matches_closed_forms(self):
        # Known exact results: E[max of 2 std normals] = 1/sqrt(pi); of 3 = 3/(2 sqrt(pi)).
        self.assertAlmostEqual(exact_expected_max(2), 1 / math.sqrt(math.pi), places=8)
        self.assertAlmostEqual(exact_expected_max(3), 3 / (2 * math.sqrt(math.pi)), places=8)

    def test_the_module_computes_the_exact_value(self):
        # The deflated Sharpe must not use the paper's approximation (8% off at N = 2).
        # The module's value must match closed forms and this file's independent integrator.
        self.assertAlmostEqual(overfit.expected_max_sharpe(2, 1.0), 1 / math.sqrt(math.pi), places=8)
        self.assertAlmostEqual(overfit.expected_max_sharpe(3, 1.0), 3 / (2 * math.sqrt(math.pi)), places=8)
        for n in (5, 10, 100, 1000, 10000):
            self.assertAlmostEqual(overfit.expected_max_sharpe(n, 1.0), exact_expected_max(n), places=7, msg=n)

    def test_very_many_trials_stay_accurate(self):
        # E[max of 10^6 standard normals] = 4.8629 by both integrators (the module's works in
        # log space; this file's raises cdf to the power N - 1 directly, over a wider range).
        self.assertAlmostEqual(overfit.expected_max_sharpe(10 ** 6, 1.0), exact_expected_max(10 ** 6, hi=16.0),
                               places=6)

    def test_the_papers_approximation_is_kept_separately(self):
        # [BLdP2014] Appendix A.1 approximation vs the exact value by integration.
        # Tolerances were set from the measured errors (7.9% at N=2, 2.3% at N=10, under 1%
        # from N=100 on, shrinking as N grows), with a little headroom.
        for n, tol in ((2, 0.08), (10, 0.025), (100, 0.01), (1000, 0.005), (10000, 0.003)):
            exact = exact_expected_max(n)
            approx = overfit.expected_max_sharpe_approx(n, 1.0)
            self.assertLess(abs(approx - exact) / exact, tol, msg="N=%d" % n)
        self.assertEqual(overfit.expected_max_sharpe_approx(1, 1.0), 0.0)

    def test_it_matches_a_seeded_simulation(self):
        # Monte Carlo of the same quantity, fixed seed: mean of the max of 100 std normals.
        rnd = random.Random(7)
        sim = sum(max(rnd.gauss(0, 1) for _ in range(100)) for _ in range(3000)) / 3000
        self.assertAlmostEqual(overfit.expected_max_sharpe(100, 1.0), sim, delta=0.06)

    def test_scales_with_the_square_root_of_the_variance(self):
        # From the formula: SR0 is proportional to sqrt(V).
        self.assertAlmostEqual(overfit.expected_max_sharpe(50, 4.0),
                               2 * overfit.expected_max_sharpe(50, 1.0), places=12)

    def test_the_bar_rises_with_the_number_of_trials(self):
        self.assertEqual(overfit.expected_max_sharpe(1, 0.01), 0.0)
        values = [overfit.expected_max_sharpe(n, 0.01) for n in (2, 10, 100, 1000)]
        self.assertEqual(values, sorted(values))
        self.assertGreater(values[0], 0)

    def test_bad_trial_counts_are_refused_not_clamped(self):
        for bad in (0, -3, 2.5, True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                overfit.expected_max_sharpe(bad, 1.0)


class DeflatedSharpe(unittest.TestCase):
    # [BLdP2014] section "A Numerical Example": annualised Sharpe 2.5 from 5 years of daily
    # data (T = 1250, 250 observations per year), N = 100 independent trials, variance of
    # the trials' annualised Sharpes 1/2, skew -3, kurtosis 10. The paper reports
    # SR0 = 0.1132 (non-annualised) and a DSR of 0.9004 ("only a 90% chance"); with
    # N = 46 the DSR "would have been 0.9505"; with normal returns (skew 0, kurtosis 3)
    # the DSR is 0.9505 after N = 88 trials.
    # Note: the paper's maths is set in an embedded font. The inputs and the figures 0.1132
    # and 0.9004 were decoded from it; "90%", "N=46", "0.9505" and "N=88" are plain text.
    SR = 2.5 / math.sqrt(250)
    VAR = 0.5 / 250

    def paper(self, trials, skew=-3.0, kurt=10.0):
        """The paper's DSR: its SR0 approximation, then the PSR against it."""
        sr0 = overfit.expected_max_sharpe_approx(trials, self.VAR)
        dsr, valid = overfit.probabilistic_sharpe(self.SR, 1250, skew, kurt, sr0)
        return {"sr0": sr0, "dsr": dsr, "psr_valid": valid}

    def test_the_papers_numerical_example(self):
        out = self.paper(100)
        self.assertAlmostEqual(out["sr0"], 0.1132, delta=5e-5)
        self.assertAlmostEqual(out["dsr"], 0.9004, delta=5e-5)
        self.assertTrue(out["psr_valid"])

    def test_the_papers_fewer_trials_case(self):
        self.assertAlmostEqual(self.paper(46)["dsr"], 0.9505, delta=5e-5)

    def test_the_papers_normal_returns_case(self):
        self.assertAlmostEqual(self.paper(88, skew=0.0, kurt=3.0)["dsr"], 0.9505, delta=5e-5)

    def test_the_library_uses_the_exact_sr0_not_the_papers(self):
        # The exact SR0 (2.5076 * sqrt(V) at N = 100) is below the paper's
        # approximation (2.5306 * sqrt(V)), so the library's DSR is a little higher.
        out = overfit.deflated_sharpe_from_stats(self.SR, 1250, -3.0, 10.0, 100, self.VAR)
        self.assertAlmostEqual(out["sr0"], overfit.expected_max_sharpe(100, self.VAR), places=12)
        self.assertNotAlmostEqual(out["sr0"], overfit.expected_max_sharpe_approx(100, self.VAR), places=4)
        self.assertGreater(out["dsr"], self.paper(100)["dsr"])

    def test_the_floor_does_not_bind_in_the_paper_example(self):
        # V = 0.002 is above the default floor 1/1250 = 0.0008, so it is used as given.
        out = overfit.deflated_sharpe_from_stats(self.SR, 1250, -3.0, 10.0, 100, self.VAR)
        self.assertEqual(out["sharpe_var"], self.VAR)
        self.assertTrue(out["var_measured"])

    def test_one_trial_is_the_plain_psr_against_zero(self):
        # From the definitions: with N = 1, SR0 = 0, so DSR = PSR(sr0 = 0).
        rnd = random.Random(3)
        returns = [rnd.gauss(0.001, 0.02) for _ in range(365)]
        m = overfit.sharpe_and_moments(returns)
        psr, _ = overfit.probabilistic_sharpe(m["sr"], m["periods"], m["skew"], m["kurt"], 0.0)
        self.assertAlmostEqual(overfit.deflated_sharpe(returns, 1)["dsr"], psr, places=12)

    def test_it_is_the_psr_against_the_expected_best_of_n(self):
        rnd = random.Random(4)
        returns = [rnd.gauss(0.001, 0.02) for _ in range(500)]
        m = overfit.sharpe_and_moments(returns)
        out = overfit.deflated_sharpe(returns, 30, sharpe_var=0.004)
        sr0 = overfit.expected_max_sharpe(30, 0.004)
        psr, _ = overfit.probabilistic_sharpe(m["sr"], 500, m["skew"], m["kurt"], sr0)
        self.assertAlmostEqual(out["sr0"], sr0, places=12)
        self.assertAlmostEqual(out["dsr"], psr, places=12)

    def test_more_trials_deflate_the_same_result(self):
        rnd = random.Random(3)
        returns = [rnd.gauss(0.001, 0.02) for _ in range(365)]
        one = overfit.deflated_sharpe(returns, trials=1, sharpe_var=0.001)
        many = overfit.deflated_sharpe(returns, trials=200, sharpe_var=0.001)
        self.assertGreater(one["dsr"], many["dsr"])
        self.assertEqual(many["trials"], 200)

    def test_heavy_tails_count_against_it(self):
        calm = [0.002 + 0.01 * ((-1) ** i) for i in range(400)]
        wild = list(calm)
        for i in range(0, 400, 50):
            wild[i] = -0.15  # rare big losses
        self.assertLess(overfit.deflated_sharpe(wild, 5, 0.001)["dsr"],
                        overfit.deflated_sharpe(calm, 5, 0.001)["dsr"])

    def test_a_measured_spread_never_goes_below_the_sampling_variance(self):
        rs = [0.001 * ((-1) ** i) + 0.0001 for i in range(400)]
        low = overfit.deflated_sharpe(rs, 500, 1e-9)
        self.assertEqual(low["sharpe_var"], 1.0 / 400)
        off = overfit.deflated_sharpe(rs, 500, 1e-9, min_sharpe_var=0)
        self.assertEqual(off["sharpe_var"], 1e-9)

    def test_a_breakdown_is_reported(self):
        # Hand-picked stats where 1 - skew*sr + (kurt-1)/4*sr^2 = 1 - 2 + 0 < 0.
        out = overfit.deflated_sharpe_from_stats(2.0, 100, 1.0, 1.0, 5)
        self.assertFalse(out["psr_valid"])
        self.assertEqual(out["dsr"], 0.0)


class SharpeVariance(unittest.TestCase):
    def test_sample_variance(self):
        # Hand computation: (0.01, 0.03, 0.05) has mean 0.03, squared deviations
        # 0.0004, 0, 0.0004; sample variance 0.0008 / 2 = 0.0004.
        self.assertAlmostEqual(overfit.sharpe_variance([0.01, 0.03, 0.05]), 0.0004, places=15)
        self.assertIsNone(overfit.sharpe_variance([0.01]))

    def test_bad_values_are_refused(self):
        with self.assertRaises(ValueError):
            overfit.sharpe_variance([0.01, float("nan"), 0.03])


class Sidak(unittest.TestCase):
    def test_closed_form(self):
        # [Sidak1967]: level = 1 - (1 - alpha)^(1/N).
        self.assertAlmostEqual(overfit.sidak(0.05, 1), 0.05, places=15)
        self.assertAlmostEqual(overfit.sidak(0.05, 30), 1 - 0.95 ** (1 / 30), places=15)

    def test_round_trip(self):
        # By definition: N independent tests at the level have family-wise error alpha.
        for n in (2, 10, 1000):
            self.assertAlmostEqual(1 - (1 - overfit.sidak(0.05, n)) ** n, 0.05, places=12)

    def test_between_bonferroni_and_alpha(self):
        # Known bound: alpha/N <= Sidak level <= alpha.
        for n in (2, 10, 1000):
            level = overfit.sidak(0.05, n)
            self.assertLessEqual(0.05 / n, level)
            self.assertLessEqual(level, 0.05)

    def test_bad_input_is_refused(self):
        for alpha, n in ((0, 5), (1, 5), (0.05, 0), (0.05, 1.5)):
            with self.assertRaises(ValueError, msg=repr((alpha, n))):
                overfit.sidak(alpha, n)


def two_series(n=1500, seed=3, edge=0.0, ar=0.0):
    """Synthetic data: a market-like series b and a series a = half b + half a steadier
    stream. Shows the code works, not that any strategy works."""
    rnd = random.Random(seed)
    b, a, e = [], [], 0.0
    for _ in range(n):
        e = ar * e + rnd.gauss(0, 0.03)
        h = 0.0008 + e
        other = 0.0004 + edge + rnd.gauss(0, 0.004)
        b.append(h)
        a.append(0.5 * h + 0.5 * other)
    return a, b


def pop_sharpe(x):
    m = sum(x) / len(x)
    return m / math.sqrt(sum((v - m) ** 2 for v in x) / len(x))


class SharpeDifferenceTest(unittest.TestCase):
    def test_the_difference_is_the_plain_sharpe_difference(self):
        a, b = two_series()
        out = overfit.sharpe_diff_test(a, b, trials=1, draws=200)
        self.assertAlmostEqual(out["delta"], pop_sharpe(a) - pop_sharpe(b), places=12)
        self.assertGreater(out["se"], 0)

    def test_its_sharpe_convention_differs_by_the_documented_factor(self):
        # Module docstring: the test's Sharpes (divisor n) are sqrt(n/(n-1)) times those of
        # sharpe_and_moments (divisor n - 1). Checked with b against itself shifted to 0 Sharpe.
        a, b = two_series(n=500)
        zero = [x - sum(b) / len(b) for x in b]  # Sharpe exactly 0 under either convention
        out = overfit.sharpe_diff_test(a, zero, trials=1, draws=200)
        sr = overfit.sharpe_and_moments(a)["sr"]
        self.assertAlmostEqual(out["delta"], sr * math.sqrt(500 / 499), places=10)

    def test_swapping_the_series_flips_the_sign_only(self):
        a, b = two_series()
        ab = overfit.sharpe_diff_test(a, b, trials=1, draws=200)
        ba = overfit.sharpe_diff_test(b, a, trials=1, draws=200)
        self.assertAlmostEqual(ab["delta"], -ba["delta"], places=12)
        self.assertAlmostEqual(ab["se"], ba["se"], places=12)

    def test_iid_standard_error_matches_an_independent_delta_method(self):
        # With block = 1 the code's se is the delta method of [LW2008] Eq. (5) with the
        # plain iid covariance of y_t = (a, b, a^2, b^2). Recomputed here with explicit
        # matrices, independently of the code's block-sum shortcut.
        a, b = two_series(n=800, seed=11)
        n = len(a)
        ys = [(x, y, x * x, y * y) for x, y in zip(a, b)]
        m = [sum(v[k] for v in ys) / n for k in range(4)]
        cov = [[sum((v[i] - m[i]) * (v[j] - m[j]) for v in ys) / n for j in range(4)] for i in range(4)]
        v1, v2 = m[2] - m[0] ** 2, m[3] - m[1] ** 2
        g = [m[2] / v1 ** 1.5, -m[3] / v2 ** 1.5, -m[0] / (2 * v1 ** 1.5), m[1] / (2 * v2 ** 1.5)]
        se = math.sqrt(sum(g[i] * cov[i][j] * g[j] for i in range(4) for j in range(4)) / n)
        out = overfit.sharpe_diff_test(a, b, trials=1, block=1, draws=200)
        self.assertAlmostEqual(out["se"], se, places=10)

    def test_iid_normal_standard_error_matches_memmel(self):
        # [LW2008] Section 2 gives Memmel's (2003) covariance of (mu_a, mu_b, var_a, var_b)
        # for iid bivariate normal returns: [[s_a^2, s_ab, 0, 0], [s_ab, s_b^2, 0, 0],
        # [0, 0, 2 s_a^4, 2 s_ab^2], [0, 0, 2 s_ab^2, 2 s_b^4]]. With the true parameters of
        # a seeded normal sample, the code's se (block = 1) should be close to it.
        rnd = random.Random(21)
        n, mu_a, mu_b, s_a, s_b, rho = 20000, 0.05, 0.02, 1.0, 1.5, 0.6
        a, b = [], []
        for _ in range(n):
            z1, z2 = rnd.gauss(0, 1), rnd.gauss(0, 1)
            a.append(mu_a + s_a * z1)
            b.append(mu_b + s_b * (rho * z1 + math.sqrt(1 - rho * rho) * z2))
        s_ab = rho * s_a * s_b
        omega = [[s_a ** 2, s_ab, 0, 0], [s_ab, s_b ** 2, 0, 0],
                 [0, 0, 2 * s_a ** 4, 2 * s_ab ** 2], [0, 0, 2 * s_ab ** 2, 2 * s_b ** 4]]
        # gradient of mu_a/sqrt(var_a) - mu_b/sqrt(var_b) in (mu_a, mu_b, var_a, var_b)
        g = [1 / s_a, -1 / s_b, -mu_a / (2 * s_a ** 3), mu_b / (2 * s_b ** 3)]
        memmel = math.sqrt(sum(g[i] * omega[i][j] * g[j] for i in range(4) for j in range(4)) / n)
        out = overfit.sharpe_diff_test(a, b, trials=1, block=1, draws=200)
        # Measured ratio with this seed: 1.0008. 1% allows for the sampling error at n = 20000.
        self.assertAlmostEqual(out["se"] / memmel, 1.0, delta=0.01)

    def test_blocks_widen_the_error_when_periods_hang_together(self):
        a, b = two_series(ar=0.5, seed=7)
        iid = overfit.sharpe_diff_test(a, b, trials=1, block=1, draws=200)
        blk = overfit.sharpe_diff_test(a, b, trials=1, block=10, draws=200)
        self.assertGreater(blk["se"], iid["se"])

    def test_under_the_null_it_rejects_about_alpha_of_the_time(self):
        # Two independent iid normal series with the same mean and sd (true difference 0),
        # 100 fixed seeds. A correct one-sided 5% test rejects about 5 of 100 times
        # (binomial sd about 2.2). The bound catches gross miscalibration, e.g. a sign
        # error (about 95 rejections) or a test that never rejects.
        rejections = 0
        for seed in range(100):
            rnd = random.Random(1000 + seed)
            a = [rnd.gauss(0.05, 1.0) for _ in range(250)]
            b = [rnd.gauss(0.05, 1.0) for _ in range(250)]
            out = overfit.sharpe_diff_test(a, b, trials=1, block=5, draws=400, seed=seed)
            rejections += out["p"] <= 0.05
        self.assertGreaterEqual(rejections, 1)
        self.assertLessEqual(rejections, 12)

    def test_a_clearly_steadier_series_passes(self):
        a, b = two_series(edge=0.002)
        out = overfit.sharpe_diff_test(a, b, trials=1)
        self.assertLess(out["p"], 0.01)
        self.assertTrue(out["passed"])

    def test_no_difference_does_not_pass(self):
        b = two_series()[1]
        self.assertFalse(overfit.sharpe_diff_test(list(b), b, trials=1)["passed"])
        a, b = two_series(edge=-0.002)  # a worse series
        self.assertFalse(overfit.sharpe_diff_test(a, b, trials=1)["passed"])

    def test_more_trials_need_more_confidence(self):
        a, b = two_series(edge=0.0, seed=5)  # p about 0.004: between the two levels
        one = overfit.sharpe_diff_test(a, b, trials=1, draws=20000)
        many = overfit.sharpe_diff_test(a, b, trials=40, draws=20000)
        self.assertEqual(one["p"], many["p"])  # same evidence (same draws), a higher bar
        self.assertTrue(one["passed"], one)
        self.assertFalse(many["passed"], many)

    def test_enough_draws_for_the_trial_count(self):
        a, b = two_series(n=400, edge=0.004)
        out = overfit.sharpe_diff_test(a, b, trials=60, draws=2000)
        self.assertGreaterEqual(out["draws"], 20 / out["level"])
        self.assertTrue(out["passed"], out)
        self.assertFalse(out["too_many_trials"])

    def test_past_the_draw_cap_it_fails_and_says_why(self):
        a, b = two_series(n=400, edge=0.004)
        out = overfit.sharpe_diff_test(a, b, trials=10 ** 7, draws=200, max_draws=500)
        self.assertTrue(out["too_many_trials"])
        self.assertFalse(out["passed"])
        self.assertEqual((out["p"], out["p_se"], out["draws"]), (1.0, 0.0, 0))

    def test_it_reports_its_own_monte_carlo_error(self):
        a, b = two_series(edge=0.0006)
        out = overfit.sharpe_diff_test(a, b, trials=1, draws=2000)
        self.assertAlmostEqual(out["p_se"], math.sqrt(out["p"] * (1 - out["p"]) / out["draws"]))

    def test_it_is_repeatable_and_checks_its_input(self):
        a, b = two_series(edge=0.0006)
        self.assertEqual(overfit.sharpe_diff_test(a, b, trials=3), overfit.sharpe_diff_test(a, b, trials=3))
        bad_inputs = [
            (a, b[:-1], {}),                          # different lengths
            (a[:-1] + [float("nan")], b, {}),         # non-finite value
            (a[:-1] + [True], b, {}),                 # a boolean is not a return
            (a[:30], b[:30], {"block": 20}),          # too short for the blocks
            (a, b, {"trials": 0}),                    # no trials
        ]
        for x, y, kw in bad_inputs:
            with self.assertRaises(ValueError, msg=repr(kw)):
                overfit.sharpe_diff_test(x, y, **{"trials": 1, **kw})


class Neighbours(unittest.TestCase):
    def test_one_step_each_way_in_each_setting(self):
        grid = {"a": [1, 2, 3], "b": [10, 20]}
        self.assertEqual(overfit.neighbours({"a": 2, "b": 10}, grid),
                         [{"a": 1, "b": 10}, {"a": 3, "b": 10}, {"a": 2, "b": 20}])

    def test_a_setting_not_on_its_grid_is_refused(self):
        with self.assertRaises(ValueError):
            overfit.neighbours({"a": 5}, {"a": [1, 2, 3]})
        with self.assertRaises(ValueError):
            overfit.neighbours({}, {"a": [1, 2, 3]})

    def test_grids_must_be_distinct_values_in_order(self):
        # An unordered grid makes "one step" meaningless; a repeat walks the base again.
        for bad in ([10, 10], [3, 1, 2], [1, "x"], [1, float("nan")], [False, True]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                overfit.neighbours({"a": bad[0]}, {"a": bad})

    def test_a_grid_with_no_neighbours_is_refused(self):
        # [10] alone has nothing to compare with; "all of none passed" must not be a pass.
        with self.assertRaises(overfit.NoNeighbours):
            overfit.neighbours({"a": 10}, {"a": [10]})


class NearbySettingsCheck(unittest.TestCase):
    GRID = {"lookback": [10, 20, 30]}

    def test_passes_only_when_every_neighbour_passes(self):
        out = overfit.nearby_settings_check({"lookback": 20}, self.GRID, lambda s: True)
        self.assertTrue(out["passed"])
        self.assertEqual([r["settings"]["lookback"] for r in out["neighbours"]], [10, 30])
        out = overfit.nearby_settings_check({"lookback": 20}, self.GRID, lambda s: s["lookback"] != 30)
        self.assertFalse(out["passed"])
        self.assertEqual([r["passed"] for r in out["neighbours"]], [True, False])

    def test_the_base_setting_is_not_checked_again(self):
        seen = []
        overfit.nearby_settings_check({"lookback": 20}, self.GRID, lambda s: seen.append(s) or True)
        self.assertNotIn({"lookback": 20}, seen)

    def test_nothing_to_compare_is_an_error_not_a_pass(self):
        with self.assertRaises(overfit.NoNeighbours):
            overfit.nearby_settings_check({"lookback": 20}, {"lookback": [20]}, lambda s: True)

    def test_passes_must_return_a_boolean(self):
        with self.assertRaises(TypeError):
            overfit.nearby_settings_check({"lookback": 20}, self.GRID, lambda s: 1)


if __name__ == "__main__":
    unittest.main()
