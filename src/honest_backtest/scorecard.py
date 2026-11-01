"""Scorecard: does a strategy beat simply holding, buying a little every so often (DCA),
or leaving the money in a risk-free asset (e.g. T-bills), after costs?

Inputs are plain sequences: ``dates`` (any strictly increasing values: dates, datetimes or
numbers; they are labels only, no calendar maths is done on them), ``prices`` of the asset
the strategy trades, and the strategy's ``equity`` (account value) on the same dates.

Conventions
-----------
* All rates, fees, costs and yields are **fractions**: 0.002 means 0.2%.
* The budget is invested at ``dates[0]``. Returns are measured from the budget:
  ``r[0] = equity[0] / budget - 1`` (the entry: no time passes, so it carries only costs)
  and ``r[t] = equity[t] / equity[t-1] - 1``. There is one return per date.
* ``risk_free[t]`` is the rate for the period that *ends* at ``dates[t]``. The first date's
  rate is not used, because no time passes before it.
* ``periods_per_year`` is required everywhere (365 for daily crypto, about 252 for daily
  stocks, 52 for weekly data): it decides how Sharpe ratios and returns are annualised.
* Sharpe ratios are of returns above the risk-free rate, use the sample standard deviation
  (divisor n - 1, as in :mod:`honest_backtest.overfit`), and are annualised by
  ``sqrt(periods_per_year)``. A series that doesn't vary has no Sharpe (None).

Benchmarks and costs
--------------------
``fee`` and ``cost`` (slippage plus half the spread) are required: use the same values your
backtest assumed. Every benchmark buy pays both. A fee or cost of 0.05 (5%) or more is
refused: it almost always means a percentage was typed instead of a fraction.

* **hold**: buys with the whole budget at the first price, then holds.
* **dca**: buys every ``dca_every`` periods, starting at the first, spending
  ``cash / buys left`` each time. Cash waiting to be invested earns the risk-free rate.
* **risk_free**: the budget compounding at the risk-free rate, no costs.

Benchmarks pay **no exit costs**: they are valued at the last price as if still held. That
makes them slightly better, so the bar for the strategy is slightly higher. For a fair
comparison **your strategy's equity must include its own exit costs** (value any open
position net of the costs of closing it).
"""

from __future__ import annotations

import bisect
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

__all__ = [
    "LABELS",
    "MarketRule",
    "rates_from_yields",
    "hold_equity",
    "dca_equity",
    "returns_from_equity",
    "compound",
    "excess",
    "max_drawdown",
    "annualised_sharpe",
    "annualised_return",
    "quantile",
    "market_labels",
    "bootstrap",
    "scorecard",
]

LABELS = ("bull", "bear", "sideways", "crash", "unlabelled")


# --------------------------------------------------------------------------- validation

def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _whole(x, name: str, minimum: int = 1) -> int:
    if isinstance(x, bool) or not isinstance(x, int) or x < minimum:
        raise ValueError("%s must be a whole number of at least %d, got %r" % (name, minimum, x))
    return x


def _ppy(periods_per_year) -> float:
    if not _is_number(periods_per_year) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive number, got %r" % (periods_per_year,))
    return float(periods_per_year)


def _fraction(x, name: str) -> float:
    if not _is_number(x) or not 0 <= x < 1:
        raise ValueError("%s must be a fraction from 0 to below 1 (0.002 = 0.2%%), got %r" % (name, x))
    return float(x)


def _cost(x, name: str) -> float:
    """A fee or cost as a fraction. 0.05 (5%) or more per trade looks like a percentage."""
    x = _fraction(x, name)
    if x >= 0.05:
        raise ValueError("%s %r looks like a percentage: use fractions (0.002 = 0.2%%); a %s of 5%% or more "
                         "per trade is refused" % (name, x, name))
    return x


def _positive(x, name: str) -> float:
    if not _is_number(x) or x <= 0:
        raise ValueError("%s must be a positive number, got %r" % (name, x))
    return float(x)


def _prices(prices: Sequence[float]) -> list[float]:
    ps = list(prices)
    if not ps:
        raise ValueError("prices: need at least one")
    for i, p in enumerate(ps):
        if not _is_number(p) or p <= 0:
            raise ValueError("prices must be positive finite numbers (position %d: %r)" % (i, p))
    return [float(p) for p in ps]


def _rates(rates: Sequence[float], n: int, name: str = "risk_free") -> list[float]:
    rs = list(rates)
    if len(rs) != n:
        raise ValueError("%s has %d values for %d dates" % (name, len(rs), n))
    for i, r in enumerate(rs):
        if not _is_number(r) or r <= -1:
            raise ValueError("%s must be finite rates above -1 (position %d: %r)" % (name, i, r))
    return [float(r) for r in rs]


def _check_dates(dates: Sequence) -> list:
    ds = list(dates)
    try:
        bad = next((i for i in range(1, len(ds)) if not ds[i - 1] < ds[i]), None)
    except TypeError:
        raise ValueError("dates must be comparable values of one kind") from None
    if bad is not None:
        raise ValueError("dates must be strictly increasing (position %d: %r after %r)" % (bad, ds[bad], ds[bad - 1]))
    return ds


# --------------------------------------------------------------------------- risk-free

def rates_from_yields(dates: Sequence, annual_yields: Sequence[tuple], periods_per_year: float, *,
                      compounding: str = "simple") -> list[float]:
    """Per-period risk-free rates for ``dates`` from dated annual yields that you supply.

    ``annual_yields`` is ``[(date, yield), ...]`` with yields as fractions (0.0365 for
    3.65%), dates strictly increasing and of the same kind as ``dates``. Each date uses the
    latest yield on or before it; a date before the first yield is an error (no guessing).

    ``compounding="simple"``: rate = yield / periods_per_year.
    ``compounding="geometric"``: rate = (1 + yield) ** (1 / periods_per_year) - 1.
    """
    ppy = _ppy(periods_per_year)
    if compounding not in ("simple", "geometric"):
        raise ValueError("compounding must be 'simple' or 'geometric', got %r" % (compounding,))
    ds = _check_dates(dates)
    ys = list(annual_yields)
    if not ys:
        raise ValueError("annual_yields: need at least one (date, yield)")
    ydates = _check_dates([d for d, _ in ys])
    for d, y in ys:
        if not _is_number(y) or not -0.5 < y < 1:
            raise ValueError("yield on %r must be a fraction (0.0365 = 3.65%%), got %r" % (d, y))
    out = []
    for d in ds:
        try:
            i = bisect.bisect_right(ydates, d) - 1
        except TypeError:
            raise ValueError("dates and yield dates must be of the same kind") from None
        if i < 0:
            raise ValueError("no yield on or before %r (the first is on %r)" % (d, ydates[0]))
        y = float(ys[i][1])
        out.append(y / ppy if compounding == "simple" else (1 + y) ** (1 / ppy) - 1)
    return out


# --------------------------------------------------------------------------- benchmarks

def hold_equity(prices: Sequence[float], budget: float, fee: float, cost: float) -> list[float]:
    """Buy with the whole budget at the first price, paying ``cost`` on the price and ``fee``
    on top, then hold: the value at each price. No exit costs (see the module docstring)."""
    ps = _prices(prices)
    budget, fee, cost = _positive(budget, "budget"), _cost(fee, "fee"), _cost(cost, "cost")
    qty = budget / (ps[0] * (1 + cost)) / (1 + fee)
    return [qty * p for p in ps]


def dca_equity(prices: Sequence[float], budget: float, fee: float, cost: float, every: int,
               risk_free: Sequence[float]) -> list[float]:
    """Buy every ``every`` periods, starting with the first, spending ``cash / buys left`` each
    time, so the last buy spends what is left. Cash waiting to be invested earns
    ``risk_free`` (the rate for the period ending at each date; the first is not used).
    Each buy pays ``cost`` on the price and ``fee`` on top. No exit costs."""
    ps = _prices(prices)
    budget, fee, cost = _positive(budget, "budget"), _cost(fee, "fee"), _cost(cost, "cost")
    every = _whole(every, "dca_every")
    rf = _rates(risk_free, len(ps))
    buys_left = math.ceil(len(ps) / every)
    cash, qty, out = budget, 0.0, []
    for t, p in enumerate(ps):
        if t > 0:
            cash *= 1 + rf[t]
        if t % every == 0 and buys_left > 0:
            spend = cash / buys_left
            qty += spend / (p * (1 + cost)) / (1 + fee)
            cash -= spend
            buys_left -= 1
        out.append(cash + qty * p)
    return out


# --------------------------------------------------------------------------- metrics

def returns_from_equity(equity: Sequence[float], budget: float) -> tuple[list[float], bool]:
    """Per-period returns of an equity curve started with ``budget`` (see the module docstring
    for the first one). Returns ``(returns, wiped_out)``.

    Equity of 0 counts as a -100% return and must stay 0 afterwards (``wiped_out`` True).
    Negative or non-finite equity is refused.
    """
    budget = _positive(budget, "budget")
    out, prev, wiped = [], budget, False
    for i, v in enumerate(equity):
        if not _is_number(v) or v < 0:
            raise ValueError("equity must be finite and not negative (position %d: %r)" % (i, v))
        if prev == 0:
            if v != 0:
                raise ValueError("equity came back from 0 at position %d: can't measure a return" % i)
            out.append(0.0)
            continue
        out.append(v / prev - 1)
        if v == 0:
            wiped = True
        prev = v
    return out, wiped


def compound(rs: Sequence[float]) -> float:
    """Total return of a sequence of per-period returns."""
    g = 1.0
    for r in rs:
        g *= 1 + r
    return g - 1


def excess(rs: Sequence[float], rf: Sequence[float]) -> list[float]:
    """Returns above the risk-free rate, period by period."""
    if len(rs) != len(rf):
        raise ValueError("%d returns but %d risk-free rates" % (len(rs), len(rf)))
    return [r - f for r, f in zip(rs, rf)]


def max_drawdown(rs: Sequence[float]) -> float:
    """The largest fall from a peak, as a fraction (0.5 = halved), starting from 1."""
    g, peak, worst = 1.0, 1.0, 0.0
    for r in rs:
        g *= 1 + r
        peak = max(peak, g)
        worst = max(worst, 1 - g / peak)
    return worst


def annualised_sharpe(rs: Sequence[float], rf: Sequence[float], periods_per_year: float) -> float | None:
    """Sharpe of returns above the risk-free rate (sample sd, divisor n - 1), times
    sqrt(periods_per_year). None for fewer than 2 returns or returns that don't vary."""
    ppy = _ppy(periods_per_year)
    ex = excess(rs, rf)
    if len(ex) < 2:
        return None
    mean = math.fsum(ex) / len(ex)
    var = math.fsum((x - mean) ** 2 for x in ex) / (len(ex) - 1)
    if var <= 0:
        return None
    return mean / math.sqrt(var) * math.sqrt(ppy)


def annualised_return(total: float, periods: float, periods_per_year: float) -> float | None:
    """(1 + total) ** (periods_per_year / periods) - 1; None when no time passed."""
    ppy = _ppy(periods_per_year)
    if periods <= 0:
        return None
    if total <= -1:
        return -1.0
    return (1 + total) ** (ppy / periods) - 1


def quantile(sorted_xs: Sequence[float], q: float) -> float:
    """The q-quantile of already-sorted values by linear interpolation between closest ranks
    (the method numpy uses by default, and Excel's PERCENTILE.INC)."""
    if not sorted_xs:
        raise ValueError("quantile of nothing")
    if not 0 <= q <= 1:
        raise ValueError("q must be between 0 and 1, got %r" % (q,))
    h = (len(sorted_xs) - 1) * q
    lo = math.floor(h)
    hi = min(lo + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (h - lo) * (sorted_xs[hi] - sorted_xs[lo])


# --------------------------------------------------------------------------- market types

@dataclass(frozen=True)
class MarketRule:
    """How periods are labelled bull, bear, sideways, crash or unlabelled. Every value is
    required: fix your own before looking at results.

    average: periods in the moving average of prices.
    slope: periods over which the moving average must have risen (bull) or fallen (bear).
    crash_fall: a fall of at least this fraction (0.20 = 20%) from a peak ...
    crash_within: ... completed within this many periods is a crash.
    """
    average: int
    slope: int
    crash_fall: float
    crash_within: int

    def __post_init__(self):
        _whole(self.average, "average")
        _whole(self.slope, "slope")
        _whole(self.crash_within, "crash_within")
        if not _is_number(self.crash_fall) or not 0 < self.crash_fall < 1:
            raise ValueError("crash_fall must be a fraction between 0 and 1 (0.20 = 20%%), got %r"
                             % (self.crash_fall,))


def market_labels(prices: Sequence[float], rule: MarketRule) -> list[str]:
    """A label for each period.

    Trend: bull = price above its moving average, and the average higher than ``slope``
    periods before; bear = below a falling average; sideways = anything else; unlabelled =
    before the average and its slope exist (the first ``average + slope - 1`` periods).

    Crash, which overrides all of the above: every period from a peak to the low point of a
    fall of at least ``crash_fall`` completed within ``crash_within`` periods. The peak is
    the last highest price in the ``crash_within`` periods before the fall's low point.

    These labels describe results after the fact. They are **not trading signals**: the
    crash label uses later prices (a peak is labelled crash because of the fall after it).
    """
    if not isinstance(rule, MarketRule):
        raise ValueError("rule must be a MarketRule")
    px = _prices(prices)
    n, s = rule.average, rule.slope
    ma: list[float | None] = [None] * len(px)
    run = 0.0
    for i, p in enumerate(px):
        run += p
        if i >= n:
            run -= px[i - n]
        if i >= n - 1:
            ma[i] = run / n
    labels = []
    for i, p in enumerate(px):
        if ma[i] is None or i < s or ma[i - s] is None:
            labels.append("unlabelled")
        elif p > ma[i] and ma[i] > ma[i - s]:
            labels.append("bull")
        elif p < ma[i] and ma[i] < ma[i - s]:
            labels.append("bear")
        else:
            labels.append("sideways")
    k = 1 - rule.crash_fall
    for j in range(1, len(px)):
        lo = max(0, j - rule.crash_within)
        top = max(px[lo:j])
        if px[j] <= top * k:
            peak = max(i for i in range(lo, j) if px[i] == top)
            for m in range(peak, j + 1):
                labels[m] = "crash"
    return labels


# --------------------------------------------------------------------------- confidence

def bootstrap(series: Mapping[str, Sequence[float]], rf: Sequence[float], periods_per_year: float, *,
              draws: int = 2000, block: int = 10, seed: int = 1) -> dict:
    """Moving-block bootstrap: resample the same periods for every series, in blocks of
    ``block`` periods (keeping some of the period-to-period dependence). For each series, a
    90% range (5% and 95% quantiles, see :func:`quantile`) of its annualised Sharpe and
    annual return, and for "strategy" how often its Sharpe beat each other series' on the
    same resampled periods. Draws where a Sharpe doesn't exist are counted, not treated as 0.
    Annual returns treat each resampled return as one period (len(rf) periods).
    Seeded: the same input gives the same output.
    """
    ppy = _ppy(periods_per_year)
    draws, block = _whole(draws, "draws"), _whole(block, "block")
    length = len(rf)
    if length < 2:
        raise ValueError("need at least 2 periods to bootstrap")
    for k, rs in series.items():
        if len(rs) != length:
            raise ValueError("series %r has %d periods, the risk-free rates %d" % (k, len(rs), length))
    rnd = random.Random(seed)
    starts = range(max(1, length - block + 1))
    sharpes: dict[str, list[float]] = {k: [] for k in series}
    annual: dict[str, list[float]] = {k: [] for k in series}
    undefined = {k: 0 for k in series}
    others = [k for k in series if k != "strategy"]
    wins = {k: 0 for k in others}
    compared = {k: 0 for k in others}
    for _ in range(draws):
        idx: list[int] = []
        while len(idx) < length:
            s0 = rnd.choice(starts)
            idx.extend(range(s0, min(s0 + block, length)))
        idx = idx[:length]
        f = [rf[i] for i in idx]
        sh = {}
        for k, rs in series.items():
            r = [rs[i] for i in idx]
            sh[k] = annualised_sharpe(r, f, ppy)
            if sh[k] is None:
                undefined[k] += 1
            else:
                sharpes[k].append(sh[k])
            annual[k].append(annualised_return(compound(r), length, ppy))
        if "strategy" in series and sh["strategy"] is not None:
            for k in others:
                if sh[k] is not None:
                    compared[k] += 1
                    wins[k] += sh["strategy"] > sh[k]
    out = {}
    for k in series:
        s, a = sorted(sharpes[k]), sorted(annual[k])
        out[k] = {"sharpe_90": (quantile(s, 0.05), quantile(s, 0.95)) if s else None,
                  "annual_return_90": (quantile(a, 0.05), quantile(a, 0.95)),
                  "undefined_sharpe_draws": undefined[k]}
    if "strategy" in series:
        out["strategy"]["sharpe_beats"] = {k: (wins[k] / compared[k] if compared[k] else None) for k in others}
    out["draws"], out["block"], out["seed"] = draws, block, seed
    return out


# --------------------------------------------------------------------------- the scorecard

def scorecard(dates: Sequence, prices: Sequence[float], equity: Sequence[float], risk_free: Sequence[float], *,
              budget: float, fee: float, cost: float, periods_per_year: float, dca_every: int,
              market_rule: MarketRule | None = None, draws: int = 2000, block: int = 10,
              seed: int = 1) -> dict:
    """The strategy against hold, DCA and the risk-free rate, all after costs (see the module
    docstring for every convention). Gives numbers only, no verdict.

    Returns ``{"periods", "first", "last", "metrics", "by_type", "confidence",
    "wiped_out", "notes"}``. ``metrics`` has "strategy", "hold", "dca" and "risk_free",
    each with "return", "annual_return", "max_drawdown" and "sharpe". ``by_type`` (when a
    ``market_rule`` is given) has, per label, the periods and each series' total return,
    plus the strategy's Sharpe.
    """
    ds = _check_dates(dates)
    ps = _prices(prices)
    eq = list(equity)
    n = len(ds)
    if len(ps) != n or len(eq) != n:
        raise ValueError("dates, prices and equity must line up: %d dates, %d prices, %d equity values"
                         % (n, len(ps), len(eq)))
    if n < 2:
        raise ValueError("need at least 2 dates")
    ppy = _ppy(periods_per_year)
    if _is_number(eq[0]) and _is_number(budget) and eq[0] > budget:
        raise ValueError("equity on the first date (%r) is above the budget (%r): the budget is invested at "
                         "the first date, so day-0 equity can only be the budget minus entry costs"
                         % (eq[0], budget))
    rf_in = _rates(risk_free, n)
    rf = [0.0] + rf_in[1:]                       # no time passes before the first date
    strat, wiped = returns_from_equity(eq, budget)
    hold, _ = returns_from_equity(hold_equity(ps, budget, fee, cost), budget)
    dca, _ = returns_from_equity(dca_equity(ps, budget, fee, cost, dca_every, rf_in), budget)
    elapsed = n - 1

    def summary(rs, with_sharpe=True):
        total = compound(rs)
        return {"return": total, "annual_return": annualised_return(total, elapsed, ppy),
                "max_drawdown": max_drawdown(rs),
                "sharpe": annualised_sharpe(rs, rf, ppy) if with_sharpe else None}

    metrics = {"strategy": summary(strat), "hold": summary(hold), "dca": summary(dca),
               "risk_free": summary(rf, with_sharpe=False)}
    by_type = None
    if market_rule is not None:
        labels = market_labels(ps, market_rule)
        groups: dict[str, list[int]] = {}
        for i, lab in enumerate(labels):
            groups.setdefault(lab, []).append(i)
        by_type = {lab: {"periods": len(ix),
                         "strategy": compound([strat[i] for i in ix]),
                         "hold": compound([hold[i] for i in ix]),
                         "dca": compound([dca[i] for i in ix]),
                         "sharpe": annualised_sharpe([strat[i] for i in ix], [rf[i] for i in ix], ppy)}
                   for lab, ix in groups.items()}
    confidence = bootstrap({"strategy": strat, "hold": hold, "dca": dca}, rf, ppy,
                           draws=draws, block=block, seed=seed)
    notes = ["benchmarks pay no exit costs; the strategy's equity must include its own"]
    if wiped:
        notes.append("the strategy's equity reached 0 (wiped out)")
    return {"periods": n, "first": ds[0], "last": ds[-1], "metrics": metrics, "by_type": by_type,
            "confidence": confidence, "wiped_out": wiped, "notes": notes}
