"""Overfitting checks: is a backtest's Sharpe ratio more than the luck of trying many ideas?

All Sharpe ratios here are **per period** (per day, per hour, ...), never annualised,
because that is what the formulas expect. Pass returns *above the risk-free rate*
(excess returns) if you want the Sharpe to mean what it usually means.

Moment conventions (used by :func:`sharpe_and_moments` and the deflated Sharpe):

* standard deviation: sample, divisor n - 1 (as pandas and Excel's STDEV.S do). This
  gives a slightly lower, so slightly conservative, Sharpe ratio;
* skew and kurtosis: population moment ratios, ``skew = m3 / m2**1.5`` and
  ``kurt = m4 / m2**2``, where ``mk`` is the k-th central moment with divisor n.
  Kurtosis is non-excess: a normal distribution has kurt 3.

:func:`sharpe_diff_test` instead follows [LW2008] exactly: its Sharpe ratios are built
from the first and second moments (divisor n), so each is larger than the one above by
the factor sqrt(n / (n - 1)) (0.05% at n = 1000).

Sources (volume, issue, pages and DOI checked against Crossref)
-------
[BLdP2014] Bailey, D. H. and Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
    The Journal of Portfolio Management 40(5), 94-107. doi:10.3905/jpm.2014.40.5.094
[BLdP2012] Bailey, D. H. and Lopez de Prado, M. (2012). "The Sharpe Ratio Efficient
    Frontier." The Journal of Risk 15(2), 3-44. doi:10.21314/JOR.2012.255
    (Introduces the probabilistic Sharpe ratio.)
[LW2008] Ledoit, O. and Wolf, M. (2008). "Robust performance hypothesis testing with
    the Sharpe ratio." Journal of Empirical Finance 15(5), 850-859.
    doi:10.1016/j.jempfin.2008.03.002
[Sidak1967] Sidak, Z. (1967). "Rectangular Confidence Regions for the Means of
    Multivariate Normal Distributions." Journal of the American Statistical
    Association 62(318), 626-633. doi:10.1080/01621459.1967.10482935
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from statistics import NormalDist

__all__ = [
    "NORMAL",
    "EULER_GAMMA",
    "NoNeighbours",
    "sharpe_and_moments",
    "probabilistic_sharpe",
    "expected_max_sharpe",
    "expected_max_sharpe_approx",
    "sharpe_variance",
    "deflated_sharpe_from_stats",
    "deflated_sharpe",
    "sidak",
    "sharpe_diff_test",
    "neighbours",
    "nearby_settings_check",
]

NORMAL = NormalDist()
EULER_GAMMA = 0.5772156649015329  # Euler-Mascheroni constant


# --------------------------------------------------------------------------- helpers

def _is_number(x: object) -> bool:
    """A finite int or float. Booleans are refused: True is not a return of 1."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _check_returns(returns: Sequence[float], name: str = "returns") -> list[float]:
    xs = list(returns)
    if len(xs) < 2:
        raise ValueError("%s: need at least 2 values, got %d" % (name, len(xs)))
    bad = [x for x in xs if not _is_number(x)]
    if bad:
        raise ValueError("%s must be finite numbers (got %r)" % (name, bad[0]))
    return [float(x) for x in xs]


def _check_trials(trials: object) -> int:
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("trials must be a whole number of at least 1, got %r" % (trials,))
    return trials


# --------------------------------------------------------------------------- Sharpe and moments

def sharpe_and_moments(returns: Sequence[float]) -> dict:
    """Per-period Sharpe ratio, mean, sd, skew and (non-excess) kurtosis of a return series.

    Sample sd (divisor n - 1); population skew and kurtosis (see the module docstring).
    Raises ValueError for fewer than 2 values, non-finite values, or a series that
    doesn't vary (no Sharpe).
    """
    xs = _check_returns(returns)
    n = len(xs)
    mean = math.fsum(xs) / n
    dev = [x - mean for x in xs]
    ss = math.fsum(d * d for d in dev)
    if ss <= 0:
        raise ValueError("returns don't vary: the Sharpe ratio is undefined")
    m2 = ss / n
    m3 = math.fsum(d ** 3 for d in dev) / n
    m4 = math.fsum(d ** 4 for d in dev) / n
    sd = math.sqrt(ss / (n - 1))
    return {"sr": mean / sd, "mean": mean, "sd": sd, "skew": m3 / m2 ** 1.5,
            "kurt": m4 / (m2 * m2), "periods": n}


def probabilistic_sharpe(sr: float, n: int, skew: float, kurt: float, sr0: float = 0.0) -> tuple[float, bool]:
    """Probabilistic Sharpe ratio: P(true Sharpe > sr0), given an observed per-period Sharpe
    ``sr`` over ``n`` periods with this skew and non-excess kurtosis.

    PSR = Z[ (sr - sr0) * sqrt(n - 1) / sqrt(1 - skew*sr + (kurt - 1)/4 * sr^2) ]

    Source: [BLdP2012]; restated as the outer part of Eq. (2) of [BLdP2014],
    section "The Deflated Sharpe Ratio".

    Returns ``(psr, valid)``. When the term under the square root is not positive
    the formula breaks down; the result is then ``(0.0, False)``: treat it as a fail.
    """
    for name, v in (("sr", sr), ("skew", skew), ("kurt", kurt), ("sr0", sr0)):
        if not _is_number(v):
            raise ValueError("%s must be a finite number, got %r" % (name, v))
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError("n must be a whole number of at least 2, got %r" % (n,))
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    if denom <= 0:
        return 0.0, False
    return NORMAL.cdf((sr - sr0) * math.sqrt(n - 1) / math.sqrt(denom)), True


def _check_var(sharpe_var) -> float:
    if not _is_number(sharpe_var) or sharpe_var < 0:
        raise ValueError("sharpe_var must be a finite number >= 0, got %r" % (sharpe_var,))
    return float(sharpe_var)


def _log_cdf(x: float) -> float:
    """log of the standard normal cdf, accurate in both tails."""
    if x > 0:
        return math.log1p(-NORMAL.cdf(-x))
    c = NORMAL.cdf(x)
    return math.log(c) if c > 0 else -math.inf


def expected_max_sharpe(trials: int, sharpe_var: float, steps: int = 4000) -> float:
    """The Sharpe the best of ``trials`` independent tries would show by luck alone, when the
    true Sharpe of every try is 0 and their observed Sharpes are normal with variance
    ``sharpe_var``: sqrt(V) * E[max of N standard normals].

    Computed **exactly** (to about 1e-9) by numerical integration of
    E[max] = integral of x * N * pdf(x) * cdf(x)^(N-1) dx, with Simpson's rule over
    [-10, 10 + sqrt(2 ln N)]. This is the quantity [BLdP2014] approximate in the SR0 of
    Eq. (2); their closed-form approximation is :func:`expected_max_sharpe_approx`, kept to
    reproduce the paper. The deflated Sharpe here uses this exact value. Returns 0.0 for
    one trial.
    """
    trials = _check_trials(trials)
    sharpe_var = _check_var(sharpe_var)
    if trials == 1 or sharpe_var == 0:
        return 0.0
    lo, hi = -10.0, 10.0 + math.sqrt(2 * math.log(trials))
    h = (hi - lo) / steps
    total = 0.0
    for i in range(steps + 1):
        x = lo + i * h
        w = 1 if i in (0, steps) else (4 if i % 2 else 2)
        lc = (trials - 1) * _log_cdf(x)
        if lc > -745:                                    # below this exp() is 0
            total += w * x * trials * NORMAL.pdf(x) * math.exp(lc)
    return math.sqrt(sharpe_var) * total * h / 3


def expected_max_sharpe_approx(trials: int, sharpe_var: float) -> float:
    """[BLdP2014]'s closed-form approximation of :func:`expected_max_sharpe`:

    SR0 = sqrt(V) * ( (1 - g) * Zinv(1 - 1/N) + g * Zinv(1 - 1/(N e)) ),  g = Euler-Mascheroni

    Source: the definition of SR0 under Eq. (2), section "The Deflated Sharpe Ratio";
    derived in Appendix A.1. It improves as N grows (about 8% off at N = 2, 2% at N = 10,
    under 1% from N = 100). Kept only to reproduce the paper's numbers; the deflated
    Sharpe here uses the exact value. Returns 0.0 for one trial.
    """
    trials = _check_trials(trials)
    sharpe_var = _check_var(sharpe_var)
    if trials == 1 or sharpe_var == 0:
        return 0.0
    return math.sqrt(sharpe_var) * ((1 - EULER_GAMMA) * NORMAL.inv_cdf(1 - 1 / trials)
                                    + EULER_GAMMA * NORMAL.inv_cdf(1 - 1 / (trials * math.e)))


def sharpe_variance(sharpes: Sequence[float]) -> float | None:
    """Sample variance (divisor n - 1) of the per-period Sharpe ratios of all trials tried.
    None when there are fewer than 2 values. Non-finite values raise ValueError."""
    xs = list(sharpes)
    bad = [x for x in xs if not _is_number(x)]
    if bad:
        raise ValueError("sharpes must be finite numbers (got %r)" % (bad[0],))
    if len(xs) < 2:
        return None
    xs = [float(x) for x in xs]
    m = math.fsum(xs) / len(xs)
    return math.fsum((x - m) ** 2 for x in xs) / (len(xs) - 1)


# --------------------------------------------------------------------------- deflated Sharpe

def deflated_sharpe_from_stats(sr: float, n: int, skew: float, kurt: float, trials: int,
                               sharpe_var: float | None = None,
                               min_sharpe_var: float | None = None) -> dict:
    """Deflated Sharpe ratio from summary statistics: the PSR against the Sharpe that the
    best of ``trials`` lucky tries would show.

    Source: [BLdP2014] Eq. (2), section "The Deflated Sharpe Ratio", with SR0 computed
    exactly (:func:`expected_max_sharpe`) instead of by the paper's approximation, so
    results differ slightly from the paper's (most for small numbers of trials).

    ``sharpe_var`` is the measured variance of the trials' Sharpes (see
    :func:`sharpe_variance`). It is never allowed below ``min_sharpe_var``, which defaults
    to 1/n, roughly the sampling variance of a Sharpe over n periods. Without the floor,
    trials that happen to agree closely would lower the bar for luck to about zero.
    Pass ``min_sharpe_var=0`` to turn the floor off.
    """
    trials = _check_trials(trials)
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError("n must be a whole number of at least 2, got %r" % (n,))
    floor = 1.0 / n if min_sharpe_var is None else min_sharpe_var
    if not _is_number(floor) or floor < 0:
        raise ValueError("min_sharpe_var must be a finite number >= 0, got %r" % (min_sharpe_var,))
    measured = sharpe_var is not None and _is_number(sharpe_var) and sharpe_var > 0
    if sharpe_var is not None and not _is_number(sharpe_var):
        raise ValueError("sharpe_var must be a finite number or None, got %r" % (sharpe_var,))
    var = max(sharpe_var if measured else 0.0, floor)
    sr0 = expected_max_sharpe(trials, var)
    dsr, valid = probabilistic_sharpe(sr, n, skew, kurt, sr0)
    return {"sr": sr, "sr0": sr0, "skew": skew, "kurt": kurt, "periods": n, "trials": trials,
            "sharpe_var": var, "var_measured": measured, "psr_valid": valid, "dsr": dsr}


def deflated_sharpe(returns: Sequence[float], trials: int, sharpe_var: float | None = None,
                    min_sharpe_var: float | None = None) -> dict:
    """Deflated Sharpe ratio of a per-period return series after ``trials`` tries.

    ``trials`` is the total number of ideas and settings tried, not just the ones kept.
    See :func:`deflated_sharpe_from_stats` for ``sharpe_var`` and ``min_sharpe_var``.
    A result with ``psr_valid`` False means the formula broke down: treat it as a fail.
    """
    m = sharpe_and_moments(returns)
    return deflated_sharpe_from_stats(m["sr"], m["periods"], m["skew"], m["kurt"], trials,
                                      sharpe_var, min_sharpe_var)


# --------------------------------------------------------------------------- multiple testing

def sidak(alpha: float, trials: int) -> float:
    """The level one test must meet so that, across ``trials`` independent tries, the chance
    of any passing by luck is ``alpha``: 1 - (1 - alpha)^(1/N).

    Source: [Sidak1967]. With correlated tries it is stricter than needed.
    """
    if not _is_number(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1, got %r" % (alpha,))
    trials = _check_trials(trials)
    return 1 - (1 - alpha) ** (1.0 / trials)


def sharpe_diff_test(a: Sequence[float], b: Sequence[float], trials: int, block: int = 10,
                     draws: int = 10000, seed: int = 1, alpha: float = 0.05,
                     max_draws: int = 200000) -> dict:
    """One-sided test that series ``a`` has a higher Sharpe ratio than series ``b``, over the
    same periods.

    Method, following [LW2008]:
      * the difference of Sharpe ratios as f(a, b, c, d) of the first and second moments,
        Section 2, Eqs. (1)-(2);
      * its standard error by the delta method, Section 2, Eqs. (4)-(5), with the moments'
        covariance estimated from sums over blocks of ``block`` periods, so periods that
        hang together aren't counted as independent evidence (the estimator of
        Section 3.2.2, Eq. (8));
      * the p-value from a circular block bootstrap of the studentized statistic,
        centred on the observed difference, Section 3.2.2.

    Differences from [LW2008]: the test is one-sided; the observed standard error uses the
    same block estimator as the bootstrap, not the prewhitened QS kernel of Section 3.1;
    the block size is fixed by the caller, not chosen by calibration.

    Passes when p is at most the Sidak level for ``trials``. The number of draws grows with
    the trial count (at least 20 per unit of the level, so a pass is possible and its Monte
    Carlo error is small), up to ``max_draws``. Beyond that the test can't support so many
    trials: it fails and says so (``too_many_trials``), spending no draws. Seeded: the same
    input always gives the same answer.
    """
    xa, xb = _check_returns(a, "a"), _check_returns(b, "b")
    n = len(xa)
    if n != len(xb):
        raise ValueError("the two return series cover different periods (%d vs %d)" % (n, len(xb)))
    if isinstance(block, bool) or not isinstance(block, int) or block < 1 or n < 2 * block:
        raise ValueError("too few periods (%d) for blocks of %r" % (n, block))
    trials = _check_trials(trials)
    xs = [(x, y, x * x, y * y) for x, y in zip(xa, xb)]
    # sums over each circular block of `block` periods, one for every start period
    ring = xs + xs[:block]
    sums, run = [], [0.0] * 4
    for t in range(block):
        run = [r + v for r, v in zip(run, ring[t])]
    for s in range(n):
        sums.append(tuple(run))
        run = [r + v - w for r, v, w in zip(run, ring[s + block], ring[s])]

    def diff_and_grad(m):
        # [LW2008] Eq. (2) and the gradient under Eq. (4)
        v1, v2 = m[2] - m[0] ** 2, m[3] - m[1] ** 2
        if v1 <= 0 or v2 <= 0:
            return None, None
        d = m[0] / math.sqrt(v1) - m[1] / math.sqrt(v2)
        g = (m[2] / v1 ** 1.5, -m[3] / v2 ** 1.5, -m[0] / (2 * v1 ** 1.5), m[1] / (2 * v2 ** 1.5))
        return d, g

    def se_of(blocks, m, g, length):
        # delta method, [LW2008] Eq. (5), with the block covariance of Eq. (8):
        # psi = mean over blocks of (g . (S_j - block * m))^2 / block
        gm = block * sum(gi * mi for gi, mi in zip(g, m))
        psi = sum((sum(gi * si for gi, si in zip(g, s)) - gm) ** 2 for s in blocks) / (len(blocks) * block)
        return math.sqrt(psi / length) if psi > 0 else 0.0

    mean = [math.fsum(x[k] for x in xs) / n for k in range(4)]
    delta, grad = diff_and_grad(mean)
    if delta is None:
        raise ValueError("a series doesn't vary: no Sharpe")
    se = se_of(sums, mean, grad, n)
    t = delta / se if se > 0 else 0.0
    level = sidak(alpha, trials)
    need = int(math.ceil(20 / level))
    too_many = need > max_draws
    if too_many:
        # it can't pass anyway: don't spend the draws
        return {"delta": delta, "se": se, "t": t, "p": 1.0, "p_se": 0.0, "level": level,
                "trials": trials, "block": block, "draws": 0, "periods": n,
                "too_many_trials": True, "passed": False}
    draws = min(max(draws, need), max_draws)
    rnd = random.Random(seed)
    k = n // block
    length = k * block
    hits = 0
    for _ in range(draws):
        picked = [sums[rnd.randrange(n)] for _ in range(k)]
        m = [sum(s[j] for s in picked) / length for j in range(4)]
        d, g = diff_and_grad(m)
        if d is None:
            hits += 1  # a draw with no Sharpe counts against passing
            continue
        s_ = se_of(picked, m, g, length)
        if s_ <= 0 or (d - delta) / s_ >= t:
            hits += 1
    p = (hits + 1) / (draws + 1)
    return {"delta": delta, "se": se, "t": t, "p": p, "p_se": math.sqrt(p * (1 - p) / draws),
            "level": level, "trials": trials, "block": block, "draws": draws, "periods": n,
            "too_many_trials": False, "passed": p <= level and delta > 0}


# --------------------------------------------------------------------------- nearby settings

class NoNeighbours(ValueError):
    """The grid gives no setting to compare with."""


def neighbours(settings: Mapping[str, object], grid: Mapping[str, Sequence[float]]) -> list[dict]:
    """Each gridded setting moved one step either way on its grid, the others kept.

    A grid must be distinct finite numbers in increasing order, so "one step" means
    something. Raises :class:`NoNeighbours` if the grid gives nothing to compare with.
    """
    out = []
    for key in sorted(grid):
        values = list(grid[key])
        if not all(_is_number(v) for v in values):
            raise ValueError("the grid for %s must be finite numbers: %r" % (key, values))
        if any(y <= x for x, y in zip(values, values[1:])):
            raise ValueError("the grid for %s must be distinct values in increasing order: %r" % (key, values))
        if key not in settings or settings[key] not in values:
            raise ValueError("setting %s = %r is not on its grid %r" % (key, settings.get(key), values))
        i = values.index(settings[key])
        for j in (i - 1, i + 1):
            if 0 <= j < len(values):
                out.append({**settings, key: values[j]})
    if not out:
        raise NoNeighbours("the grid %r gives no nearby setting to compare with" % (dict(grid),))
    return out


def nearby_settings_check(settings: Mapping[str, object], grid: Mapping[str, Sequence[float]],
                          passes: Callable[[dict], bool]) -> dict:
    """"A plateau, not a spike": every neighbouring setting (see :func:`neighbours`) must also
    pass. ``passes`` is your own function: it takes a settings dict and returns True if your
    backtest with those settings passes your checks. This library does not run it for you
    in any other way.

    Returns ``{"neighbours": [{"settings": ..., "passed": ...}, ...], "passed": all passed}``.
    """
    results = []
    for s in neighbours(settings, grid):
        ok = passes(s)
        if not isinstance(ok, bool):
            raise TypeError("passes() must return True or False, got %r" % (ok,))
        results.append({"settings": s, "passed": ok})
    return {"neighbours": results, "passed": all(r["passed"] for r in results)}
