"""One report: does a backtest survive the usual ways of fooling yourself?

:func:`run` takes whatever evidence you have and runs every check it can:

====  ===========================  =====================================================
 A    costs vs edge                 is the edge left after costs, even doubled costs?
 B    beats random entries          above the random-entry runs' 95th percentile?
 B2   random direction              above the random-direction runs' 95th percentile?
 C    deflated Sharpe               still significant after every trial in the log?
 C2   Sharpe reliably above hold    Ledoit-Wolf test vs hold, at the log's trial count
 D    sealed-period guard           no look at data reserved for the final test
 E    cross-market                  does it work on other markets too?
 F    concentration                 is the profit spread out, not from one slice?
 G    trade summary                 enough trades? expectancy, profit factor, ...
 H    code state                    were the trials run on committed code?
 I    scorecard                     beats hold, DCA and the risk-free rate?
====  ===========================  =====================================================

Each check is PASS, FAIL, WARN (passes, with something you should know) or NOT RUN
(its inputs weren't given). A check that wasn't run is never counted as passed.

Verdict: **FAIL** if any check fails; otherwise **INCOMPLETE** if a core check (A, B, C,
G, I) wasn't run; otherwise **PASS**, with every warning listed. PASS means these checks
found no problem. It is not proof that the strategy works.

Every threshold a check used is printed in the report, marked "standard" or "set by you".

The sealed-period guard is different: if you declare a sealed period and any date you pass
falls inside it, :func:`run` refuses (raises :class:`SealedPeriodError`) before computing
anything, unless you deliberately open it for the one final test (``open_sealed=True``).
"""

from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from . import __version__, overfit, scorecard as sc
from .scorecard import MarketRule, quantile
from .trials import LogError, TrialLog

__all__ = ["Thresholds", "Trades", "Market", "CheckResult", "Report", "SealedPeriodError",
           "trade_summary", "run", "CORE"]

PASS, FAIL, WARN, NOT_RUN, INFO = "PASS", "FAIL", "WARN", "NOT RUN", "INFO"
CORE = ("A", "B", "C", "G", "I")
DISCLAIMER = "PASS means these checks found no problem. It is not proof that the strategy works."


class SealedPeriodError(Exception):
    """Data inside the declared sealed period was passed in without opening it."""


# --------------------------------------------------------------------------- thresholds

class Thresholds:
    """Every threshold the checks use. Pass only the ones you want to change:
    ``Thresholds(dsr_min=0.99)``. The report marks each one "standard" or "set by you".

    ``min_markets_positive`` has no standard value: it is required when check E runs.
    ``direction_strict`` / ``sharpe_test_strict``: make B2 / C2 FAIL instead of WARN.
    """

    STANDARD = {
        "dsr_min": 0.95,                # C: deflated Sharpe at least this
        "random_percentile": 0.95,      # B, B2: strictly above this quantile of the random runs
        "min_valid_runs": 0.9,          # B, B2: at least this share of random runs must be valid
        "min_random_runs": 400,         # B, B2: at least this many valid random runs
        "cost_multiplier": 2.0,         # A: net must stay above 0 with costs times this
        "max_cost_share": 0.5,          # A: WARN when costs take more than this share of the edge
        "max_slice_share": 0.5,         # F: FAIL when one slice gives more than this share of profit
        "min_trades": 30,               # G: FAIL below this many trades (a floor, not enough)
        "alpha": 0.05,                  # C2: family-wise error, Sidak-adjusted for the trial count
        "min_markets_positive": None,   # E: required, no standard
        "direction_strict": False,      # B2: FAIL instead of WARN
        "sharpe_test_strict": False,    # C2: FAIL instead of WARN
        "allow_overrides": False,       # B: overriding a Config's long share or stops only WARNs
        "max_identical_run": 10,        # C, I: WARN at this many identical consecutive returns
        "max_repeated_share": 0.2,      # C, I: WARN when more non-zero returns than this equal the one before
        "min_cost_share": 0.01,         # A: WARN when costs are under this share of the edge
        "trades_equity_tolerance": 0.1,  # G: WARN when the trades' total is further than this from the equity's
    }

    # which way is stricter, for the "looser than standard" warning
    STRICTER = {"dsr_min": "higher", "random_percentile": "higher", "min_valid_runs": "higher",
                "min_random_runs": "higher", "cost_multiplier": "higher", "min_trades": "higher",
                "min_cost_share": "higher", "max_cost_share": "lower", "max_slice_share": "lower",
                "alpha": "lower", "max_identical_run": "lower", "trades_equity_tolerance": "lower",
                "max_repeated_share": "lower",
                "direction_strict": True, "sharpe_test_strict": True, "allow_overrides": False}

    def __init__(self, **kw):
        unknown = set(kw) - set(self.STANDARD)
        if unknown:
            raise ValueError("unknown thresholds: %s" % ", ".join(sorted(unknown)))
        values = {**self.STANDARD, **kw}
        for k in ("dsr_min", "random_percentile", "min_valid_runs", "max_cost_share", "max_slice_share",
                  "min_cost_share", "trades_equity_tolerance", "max_repeated_share"):
            if not _finite(values[k]) or not 0 < values[k] <= 1:
                raise ValueError("%s must be a number above 0 and at most 1, got %r" % (k, values[k]))
        if not _finite(values["alpha"]) or not 0 < values["alpha"] < 1:
            raise ValueError("alpha must be between 0 and 1, got %r" % (values["alpha"],))
        if not _finite(values["cost_multiplier"]) or values["cost_multiplier"] < 1:
            raise ValueError("cost_multiplier must be at least 1, got %r" % (values["cost_multiplier"],))
        for k in ("min_trades", "min_random_runs", "max_identical_run"):
            if isinstance(values[k], bool) or not isinstance(values[k], int) or values[k] < 1:
                raise ValueError("%s must be a whole number of at least 1, got %r" % (k, values[k]))
        mp = values["min_markets_positive"]
        if mp is not None and (not _finite(mp) or not 0 < mp <= 1):
            raise ValueError("min_markets_positive must be above 0 and at most 1, got %r" % (mp,))
        for k in ("direction_strict", "sharpe_test_strict", "allow_overrides"):
            if not isinstance(values[k], bool):
                raise ValueError("%s must be True or False" % k)
        self._values = values
        self.set_by_you = frozenset(kw)

    def __getattr__(self, name):
        try:
            return self.__dict__["_values"][name]
        except KeyError:
            raise AttributeError(name) from None

    def looser(self) -> list[str]:
        """Each threshold you set looser than its standard value, as a sentence."""
        out = []
        for k, way in self.STRICTER.items():
            v, std = self._values[k], self.STANDARD[k]
            if k not in self.set_by_you or v == std:
                continue
            if isinstance(way, bool):
                if v != way:
                    out.append("looser than standard: %s %s (standard %s)" % (k, v, std))
            elif (way == "higher" and v < std) or (way == "lower" and v > std):
                out.append("looser than standard: %s %s %s %s" % (k, _fmt(v), "<" if v < std else ">", _fmt(std)))
        return out

    def describe(self, name: str) -> dict:
        return {"name": name, "value": self._values[name],
                "source": "set by you" if name in self.set_by_you else "standard"}


# --------------------------------------------------------------------------- inputs

@dataclass
class Trades:
    """The strategy's trades, in one unit (R multiples, returns, money: your choice).

    gross: each trade's result before costs.
    costs: each trade's costs (fees, spread, slippage), not negative. Net = gross - costs.
    entries, exits: optional entry and exit dates, checked by the sealed-period guard and
        against the market dates.
    sides: optional, each trade's side (+1 long, -1 short). With entries given, the same
        (entry, side) twice is refused: a trade counted twice.
    unit: optional, what the results are: "return" (each a fraction of the whole equity,
        compounding) or "money" (in the equity's currency). When set, check G compares the
        trades' total with the equity curve's change.
    slices: optional ways to slice the trades for check F, e.g.
        ``{"year": ["2021", "2021", "2022", ...]}``, one label per trade.
    """
    gross: Sequence[float]
    costs: Sequence[float]
    entries: Sequence | None = None
    exits: Sequence | None = None
    sides: Sequence[int] | None = None
    slices: Mapping[str, Sequence] | None = None
    unit: str | None = None

    def net(self) -> list[float]:
        return [g - c for g, c in zip(self.gross, self.costs)]

    def check(self) -> None:
        n = len(self.gross)
        if len(self.costs) != n:
            raise ValueError("trades: %d gross results but %d costs" % (n, len(self.costs)))
        if not all(_finite(x) for x in self.gross):
            raise ValueError("trades: gross results must be finite numbers")
        if not all(_finite(x) and x >= 0 for x in self.costs):
            raise ValueError("trades: costs must be finite numbers, not negative")
        for what, values in (("entries", self.entries), ("exits", self.exits), ("sides", self.sides)):
            if values is not None and len(values) != n:
                raise ValueError("trades: %d %s for %d trades" % (len(values), what, n))
        if self.sides is not None and not all(x in (1, -1) and not isinstance(x, bool) for x in self.sides):
            raise ValueError("trades: sides must be +1 or -1")
        if self.entries is not None:
            sides = self.sides if self.sides is not None else [None] * n
            seen = set()
            for e, sd in zip(self.entries, sides):
                try:
                    key = (e, sd)
                    if key in seen:
                        raise ValueError("trades: the same trade appears twice (entry %r%s)"
                                         % (e, ", side %+d" % sd if sd is not None else ""))
                    seen.add(key)
                except TypeError:
                    raise ValueError("trades: entries must be hashable to check for repeats") from None
        if self.unit not in (None, "return", "money"):
            raise ValueError("trades: unit must be 'return', 'money' or None, got %r" % (self.unit,))
        for name, labels in (self.slices or {}).items():
            if len(labels) != n:
                raise ValueError("trades: slicing %r has %d labels for %d trades" % (name, len(labels), n))


@dataclass
class Market:
    """What :func:`honest_backtest.scorecard.scorecard` needs (see its docstring): the
    strategy's equity and the traded asset's prices on the same dates, the risk-free rates,
    the budget, the benchmarks' costs and ``periods_per_year`` (required)."""
    dates: Sequence
    prices: Sequence[float]
    equity: Sequence[float]
    risk_free: Sequence[float]
    budget: float
    fee: float
    cost: float
    periods_per_year: float
    dca_every: int
    market_rule: MarketRule | None = None


# --------------------------------------------------------------------------- results

@dataclass
class CheckResult:
    key: str
    name: str
    status: str
    reasons: list[str] = field(default_factory=list)
    numbers: dict = field(default_factory=dict)
    thresholds: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"key": self.key, "name": self.name, "status": self.status, "reasons": list(self.reasons),
                "numbers": _json_safe(self.numbers), "thresholds": list(self.thresholds)}


def _status(fails: list, warns: list) -> str:
    return FAIL if fails else WARN if warns else PASS


@dataclass
class Report:
    verdict: str
    checks: list[CheckResult]
    thresholds: list[dict]
    notes: list[str]
    settings: dict = field(default_factory=dict)
    version: str = __version__

    def check(self, key: str) -> CheckResult:
        return next(c for c in self.checks if c.key == key)

    def counts(self) -> dict:
        out = {s: 0 for s in (PASS, WARN, FAIL, NOT_RUN, INFO)}
        for c in self.checks:
            out[c.status] += 1
        return out

    def to_dict(self) -> dict:
        return {"library": "honest-backtest", "version": self.version, "verdict": self.verdict,
                "counts": self.counts(), "checks": [c.to_dict() for c in self.checks],
                "thresholds": _json_safe(self.thresholds), "settings": _json_safe(self.settings),
                "notes": list(self.notes), "disclaimer": DISCLAIMER}

    def json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False, allow_nan=False)

    def text(self) -> str:
        c = self.counts()
        lines = ["honest-backtest checks report (version %s)" % self.version,
                 "VERDICT: %s  (%d pass, %d warn, %d fail, %d not run)"
                 % (self.verdict, c[PASS], c[WARN], c[FAIL], c[NOT_RUN]), ""]
        for ch in self.checks:
            lines.append("[%s] %s  %s" % (ch.status, ch.key, ch.name))
            for r in ch.reasons:
                lines.append("       - " + r)
        lines += ["", "Thresholds used:"]
        if not self.thresholds:
            lines.append("  (none)")
        for t in self.thresholds:
            lines.append("  %-22s %-8s (%s; used by %s)" % (t["name"], _fmt(t["value"]), t["source"],
                                                          ", ".join(t["used_by"])))
        lines += ["", "Settings used:"]
        for k, v in self.settings.items():
            if v is None:
                continue
            if isinstance(v, Mapping):
                v = ", ".join("%s %s" % (a, b) for a, b in v.items() if b is not None)
            lines.append("  %-22s %s" % (k, v))
        if self.notes:
            lines += ["", "Notes:"] + ["  - " + n for n in self.notes]
        lines += ["", DISCLAIMER]
        return "\n".join(lines)


def _nth(q: float) -> str:
    return "%gth" % (100 * q)


def _mode(strict: bool) -> str:
    return "strict (FAIL on failure)" if strict else "WARN on failure"


def _fmt(v) -> str:
    if isinstance(v, float):
        return "%.4g" % v
    return str(v)


def _json_safe(v):
    if isinstance(v, float) and not math.isfinite(v):
        return repr(v)
    if isinstance(v, Mapping):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# --------------------------------------------------------------------------- trade summary

def trade_summary(results: Sequence[float]) -> dict:
    """Trades, expectancy, profit factor, win rate, total, maximum drawdown and longest losing
    streak of a list of trade results (net of costs), in their own unit.

    Standard trade statistics: drawdown is measured on the running sum of results
    (from 0); profit factor is total wins / total losses (``inf`` with no losses, None with
    neither wins nor losses); a losing streak counts consecutive results below 0.
    """
    rs = [float(r) for r in results]
    if not all(math.isfinite(r) for r in rs):
        raise ValueError("trade results must be finite numbers")
    if not rs:
        return {"trades": 0, "expectancy": None, "profit_factor": None, "win_rate": None, "total": 0.0,
                "max_drawdown": 0.0, "longest_losing_streak": 0}
    wins = math.fsum(r for r in rs if r > 0)
    losses = -math.fsum(r for r in rs if r < 0)
    run_total, peak, worst, streak, best = 0.0, 0.0, 0.0, 0, 0
    for r in rs:
        run_total += r
        peak = max(peak, run_total)
        worst = max(worst, peak - run_total)
        streak = streak + 1 if r < 0 else 0
        best = max(best, streak)
    pf = wins / losses if losses > 0 else (math.inf if wins > 0 else None)
    return {"trades": len(rs), "expectancy": math.fsum(rs) / len(rs), "profit_factor": pf,
            "win_rate": sum(1 for r in rs if r > 0) / len(rs), "total": math.fsum(rs),
            "max_drawdown": worst, "longest_losing_streak": best}


# --------------------------------------------------------------------------- sealed period

_TICK = dt.timedelta(microseconds=1)


def _interval(v, what: str) -> tuple[dt.datetime, dt.datetime]:
    """The time a value covers, as a half-open [start, end): a date-only value (a date, or an
    ISO string without a time) is its whole day; a datetime is that instant."""
    if isinstance(v, str):
        try:
            v = dt.datetime.fromisoformat(v) if ("T" in v or " " in v) else dt.date.fromisoformat(v)
        except ValueError:
            raise ValueError("%s: %r is not an ISO date or date-time" % (what, v)) from None
    if isinstance(v, dt.datetime):
        return v, v + _TICK
    if isinstance(v, dt.date):
        day = dt.datetime(v.year, v.month, v.day)
        return day, day + dt.timedelta(days=1)
    raise ValueError("%s: the sealed-period guard needs dates or datetimes, got %r" % (what, v))


def _is_date_like(v) -> bool:
    try:
        _interval(v, "")
        return True
    except ValueError:
        return False


class _Sealed:
    """The sealed period, inclusive at both ends: a date-only end includes its whole day.
    Any value that overlaps it at any time (a date-only value is its whole day) is inside."""

    def __init__(self, period):
        try:
            start, end = period
        except (TypeError, ValueError):
            raise ValueError("sealed must be (start, end), got %r" % (period,)) from None
        self.lo = _interval(start, "sealed start")[0]
        self.hi = _interval(end, "sealed end")[1]
        if (self.lo.tzinfo is None) != (self.hi.tzinfo is None):
            raise ValueError("sealed start and end must both have a time zone, or both not")
        if self.hi <= self.lo:
            raise ValueError("the sealed period ends before it starts")
        self.text = "%s to %s" % (start, end)

    def _overlaps(self, lo: dt.datetime, hi: dt.datetime) -> bool:
        if (lo.tzinfo is None) != (self.lo.tzinfo is None):
            raise ValueError("dates and the sealed period must both have a time zone, or both not")
        return lo < self.hi and hi > self.lo

    def touches(self, v, what: str) -> bool:
        return self._overlaps(*_interval(v, what))

    def touches_span(self, first, last, what: str) -> bool:
        return self._overlaps(_interval(first, what)[0], _interval(last, what)[1])


def _guard(seal: _Sealed, open_sealed: bool, market, trades, random) -> list[str]:
    """Refuse (raise SealedPeriodError) any input that touches the sealed period, unless it is
    opened; return what couldn't be checked."""
    found, not_checked = [], []
    if market is not None:
        for d in market.dates:
            if seal.touches(d, "market dates"):
                found.append(("market dates", d))
                break
    if trades is not None:
        for what, values in (("trade entries", trades.entries), ("trade exits", trades.exits)):
            if values is None:
                not_checked.append("%s were not given" % what)
                continue
            for d in values:
                if seal.touches(d, what):
                    found.append((what, d))
                    break
    if random is not None:
        spans = {}
        for res in random.values():
            spans.update(res.get("eligible_span") or {})
        if not spans:
            not_checked.append("the random baseline doesn't say which times were eligible")
        for name, span in spans.items():
            if span is None or not all(_is_date_like(v) for v in span):
                not_checked.append("config %r's eligible times are not dates" % name)
            elif seal.touches_span(span[0], span[1], "eligible times"):
                found.append(("config %r's eligible times" % name, "%s to %s" % tuple(span)))
        exits: dict = {}
        for res in random.values():
            for name, span in (res.get("exit_span") or {}).items():
                exits.setdefault(name, []).append(span)
        if not exits:
            not_checked.append("the random baseline doesn't say when its random trades ended")
        for name, found_spans in exits.items():
            for span in found_spans:
                if span is None or not all(_is_date_like(v) for v in span):
                    not_checked.append("config %r's random trade exits are not dates" % name)
                    break
                if seal.touches_span(span[0], span[1], "random trade exits"):
                    found.append(("config %r's random trade exits" % name, "%s to %s" % tuple(span)))
                    break
    if found and not open_sealed:
        what, d = found[0]
        raise SealedPeriodError("%s include %r, inside the sealed period %s. Remove it, or pass open_sealed=True "
                                "for the one final test" % (what, d, seal.text))
    return not_checked


# --------------------------------------------------------------------------- the checks

def _check_costs(trades: Trades | None, th: Thresholds) -> CheckResult:
    r = CheckResult("A", "costs vs edge", NOT_RUN)
    if trades is None:
        r.reasons.append("no trades given")
        return r
    if not trades.gross:
        r.status, r.reasons = FAIL, ["no trades"]
        return r
    n = len(trades.gross)
    gross = math.fsum(trades.gross) / n
    cost = math.fsum(trades.costs) / n
    net = gross - cost
    k = th.cost_multiplier
    stressed = gross - k * cost
    share = cost / gross if gross > 0 else None
    r.numbers = {"avg_before_costs": gross, "avg_costs": cost, "avg_net": net, "avg_net_stressed": stressed,
                 "cost_share_of_edge": share}
    r.thresholds = ["cost_multiplier", "max_cost_share", "min_cost_share"]
    fails, warns = [], []
    if gross <= 0:
        fails.append("no edge before costs: average %.4g per trade" % gross)
    if net <= 0:
        fails.append("net of costs the average trade is %.4g (not above 0)" % net)
    if stressed <= 0:
        fails.append("with costs x%g the average trade is %.4g (not above 0)" % (k, stressed))
    if share is not None and share > th.max_cost_share:
        warns.append("costs take %.0f%% of the edge (more than %.0f%%)" % (100 * share, 100 * th.max_cost_share))
    if all(c == 0 for c in trades.costs):
        warns.append("cost stress test has no effect: every trade's cost is 0")
    elif share is not None and share < th.min_cost_share:
        warns.append("costs look negligible (%.2f%% of the edge, under %g%%): check they're entered"
                     % (100 * share, 100 * th.min_cost_share))
    r.status = _status(fails, warns)
    r.reasons = fails + warns or ["average %.4g before costs, %.4g costs, %.4g net; %.4g with costs x%g"
                                  % (gross, cost, net, stressed, k)]
    return r


def _check_random(key: str, name: str, res: Mapping | None, th: Thresholds, strict: bool,
                  trades: Trades | None = None) -> CheckResult:
    r = CheckResult(key, name, NOT_RUN)
    if res is None:
        r.reasons.append("no random baseline given")
        return r
    runs = res.get("runs")
    if isinstance(runs, bool) or not isinstance(runs, int) or runs < 1:
        raise ValueError("the random baseline result for %s has runs=%r: it needs at least 1 run" % (key, runs))
    shares = res.get("actual_long_share") or {}
    if key == "B2" and shares and all(v in (0, 1) for v in shares.values() if v is not None) \
            and any(v is not None for v in shares.values()):
        r.reasons.append("the strategy's actual trades are one-sided (long share %s): a random direction "
                         "doesn't apply" % ", ".join("%s %g" % (k, v) for k, v in shares.items() if v is not None))
        return r
    q = th.random_percentile
    r.thresholds = ["random_percentile", "min_valid_runs", "min_random_runs"]
    if key == "B2":
        r.thresholds.append("direction_strict")
    values = sorted(res["values"])
    valid, runs = res["valid_runs"], res["runs"]
    bar = quantile(values, q) if values else None
    strategy = res["strategy"]
    r.numbers = {"strategy": strategy, "bar": bar, "percentile": res.get("percentile"), "p_value": res.get("p_value"),
                 "runs": runs, "valid_runs": valid, "too_few_trades": res["too_few_trades"], "failed": res["failed"]}
    if key == "B2":
        r.numbers["direction"] = res.get("direction")
        r.numbers["direction_p"] = res.get("direction_p")
    fails, warns, info = [], [], []
    p_value = res.get("p_value")
    above = bar is not None and strategy > bar
    p_ok = p_value is not None and p_value <= 1 - q
    beats = above and p_ok
    if valid / runs < th.min_valid_runs:
        fails.append("only %d of %d random runs were valid (less than %.0f%%)" % (valid, runs, 100 * th.min_valid_runs))
    if valid < th.min_random_runs:
        fails.append("only %d valid random runs (fewer than %d): too few to place the strategy reliably"
                     % (valid, th.min_random_runs))
    if beats:
        info.append("%.4g is above the random runs' %s percentile (%.4g); it ranks at %.1f%%, p = %.4f"
                    % (strategy, _nth(q), bar, 100 * (res.get("percentile") or 0), p_value))
    else:
        if not above:
            msg = ("%.4g is not above the random runs' %s percentile (%s)"
                   % (strategy, _nth(q), "%.4g" % bar if bar is not None else "no valid runs"))
        else:
            msg = ("%.4g is above the %s percentile, but p = %s is above %.4g"
                   % (strategy, _nth(q), "%.4f" % p_value if p_value is not None else "unknown", 1 - q))
        (fails if key == "B" or strict else warns).append(msg)
    if res["too_few_trades"] or res["failed"]:
        warns.append("%d random runs had too few trades and %d failed" % (res["too_few_trades"], res["failed"]))
    # the baseline must have been run on the strategy's own trades, measured the same way
    problems = list(res.get("warnings") or []) + _mismatch(res, trades)
    (fails if key == "B" or strict else warns).extend(problems)
    if key == "B":
        r.thresholds.append("allow_overrides")
        f, w = _overridden(res, th.allow_overrides)
        fails += f
        warns += w
    if key == "B2":
        info.append("direction: %s, p(long) %s; mode: %s" % (res.get("direction"), res.get("direction_p"),
                                                           _mode(strict)))
    r.status = _status(fails, warns)
    r.reasons = info[:1] + fails + warns + info[1:] if beats else fails + warns + info
    return r


def _overridden(res: Mapping, allow: bool = False) -> tuple[list[str], list[str]]:
    """FAIL when a configuration drew a different number of random trades than it has actual
    trades, or replaced a default (long share, stop distances) unless ``allow`` (then WARN)."""
    fails, warns = [], []
    replaced = []
    n_trades, actual = res.get("n_trades") or {}, res.get("actual_trades") or {}
    for name, n in n_trades.items():
        if name in actual and n != actual[name]:
            fails.append("config %r draws %d random trades but the strategy has %d actual trades"
                         % (name, n, actual[name]))
    for name, ov in (res.get("overrides") or {}).items():
        if "long_share" in ov:
            share = (res.get("actual_long_share") or {}).get(name)
            replaced.append("config %r: long_share overridden to %g (its actual trades: %s)"
                            % (name, ov["long_share"], "%g" % share if share is not None else "none"))
        if "stops" in ov:
            replaced.append("config %r: stop distances overridden (%d values) instead of its actual trades' own"
                            % (name, ov["stops"]))
    if replaced and allow:
        warns += [x + "; allowed by allow_overrides=True" for x in replaced]
    else:
        fails += [x + " (pass Thresholds(allow_overrides=True) if this is deliberate)" for x in replaced]
    return fails, warns


def _mismatch(res: Mapping, trades: Trades | None) -> list[str]:
    """Problems when the random baseline was run on other trades than the ones given."""
    if trades is None:
        return []
    out = []
    n = len(trades.gross)
    counts = res.get("real_trades") or {}
    if counts and n not in counts.values():
        out.append("the random baseline's strategy had %s trade(s), but %d trades were given"
                   % (" / ".join(str(v) for v in counts.values()), n))
    if len(counts) == 1 and res.get("score_is_mean") and n:
        mean_net = math.fsum(trades.net()) / n
        if not math.isclose(res["strategy"], mean_net, rel_tol=1e-9, abs_tol=1e-12):
            out.append("the random baseline's strategy value %.6g doesn't match the given trades' mean net result "
                       "%.6g" % (res["strategy"], mean_net))
    return out


def _excluded(records: Sequence[Mapping]) -> tuple[dict, list[str]]:
    """For each idea and each setting left out of trial counting, how many distinct values it
    took; and the ideas whose every record had all its settings left out."""
    seen: dict = {}
    all_out: dict = {}
    for rec in records:
        idea, settings = rec["idea"], rec.get("settings") or {}
        out = set(rec.get("excluded_from_trial") or [])
        for k in out:
            seen.setdefault(idea, {}).setdefault(k, set()).add(json.dumps(settings.get(k), sort_keys=True))
        everything = bool(settings) and set(settings) <= out
        all_out[idea] = all_out.get(idea, True) and everything
    counts = {idea: {k: len(v) for k, v in keys.items()} for idea, keys in seen.items()}
    return counts, sorted(i for i, v in all_out.items() if v)


def _same(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))


def _longest_run(xs: Sequence[float], skip_zero: bool = False) -> tuple[int, int]:
    """(length, start) of the longest run of identical consecutive values."""
    best, best_at, run, at = 0, 0, 0, 0
    for i, x in enumerate(xs):
        if skip_zero and x == 0:
            run = 0
            continue
        if run and _same(x, xs[i - 1]):
            run += 1
        else:
            run, at = 1, i
        if run > best:
            best, best_at = run, at
    return best, best_at


def _longest_zero_run(xs: Sequence[float]) -> tuple[int, int]:
    """(length, start) of the longest run of returns that are exactly 0."""
    best, best_at, run = 0, 0, 0
    for i, x in enumerate(xs):
        run = run + 1 if x == 0 else 0
        if run > best:
            best, best_at = run, i - run + 1
    return best, best_at


def _repeated_share(xs: Sequence[float]) -> tuple[float | None, int]:
    """Share of the non-zero values (after the first) that equal the value before them, and
    how many non-zero values there were. None when there are too few to judge (under 10)."""
    nonzero = [i for i in range(1, len(xs)) if xs[i] != 0]
    if len(nonzero) < 10:
        return None, len(nonzero)
    repeats = sum(1 for i in nonzero if _same(xs[i], xs[i - 1]))
    return repeats / len(nonzero), len(nonzero)


def _flat_runs(name: str, rets: Sequence[float], levels: Sequence[float], th: "Thresholds",
               zero_is_soft: bool = True) -> tuple[list[str], list[str]]:
    """(warnings, notes) when a series looks filled in or interpolated.

    WARN for: ``max_identical_run`` or more identical consecutive non-zero returns, or equal
    consecutive non-zero steps in the level (linear interpolation); more than
    ``max_repeated_share`` of the non-zero returns, or of the non-zero steps, equal to the one
    before (e.g. daily data stretched to 4-hour bars repeats each move 6 times). A long run of
    returns that are exactly 0 is only a note when ``zero_is_soft`` (it may be time out of
    the market), and a warning otherwise (prices don't sit out of the market).
    """
    limit, share_limit = th.max_identical_run, th.max_repeated_share
    warns, notes = [], []
    why = "which hides variation and inflates Sharpe-based tests"
    n, at = _longest_run(rets, skip_zero=True)
    if n >= limit:
        warns.append("%s: %d identical consecutive returns from period %d (%.4g each); at %d or more, the data "
                     "may be filled in or interpolated, %s" % (name, n, at, rets[at], limit, why))
    n, at = _longest_zero_run(rets)
    if n >= limit:
        msg = "%s: %d consecutive returns of exactly 0 from period %d" % (name, n, at)
        if zero_is_soft:
            notes.append(msg + "; this may be time out of the market; ignore if so (otherwise the data may be "
                         "forward-filled)")
        else:
            warns.append(msg + "; at %d or more, the data may be forward-filled, %s" % (limit, why))
    share, k = _repeated_share(rets)
    if share is not None and share > share_limit:
        warns.append("%s: %.0f%% of the %d non-zero returns equal the one before (more than %.0f%%): the data may "
                     "be stretched to a finer time step or interpolated, %s" % (name, 100 * share, k, 100 * share_limit, why))
    steps = [b - a for a, b in zip(levels, levels[1:])]
    n, at = _longest_run(steps, skip_zero=True)
    if n >= limit:
        warns.append("%s: %d equal consecutive steps from period %d; at %d or more, the data may be linearly "
                     "interpolated, %s" % (name, n, at + 1, limit, why))
    share, k = _repeated_share(steps)
    if share is not None and share > share_limit:
        warns.append("%s: %.0f%% of the %d non-zero steps equal the one before (more than %.0f%%): the data may be "
                     "linearly interpolated, %s" % (name, 100 * share, k, 100 * share_limit, why))
    return warns, notes


def _count_trials(records: Sequence[Mapping], sharpe_key: str) -> tuple[int, list[float], int]:
    """Trials as check C counts them: one per (idea, settings), but a trial whose records with
    identical settings show different results counts once per distinct result (the code or
    the data must have changed). Returns (trials, one Sharpe per counted trial, trials split)."""
    groups: dict = {}
    for rec in records:
        g = groups.setdefault((rec["idea"], rec["trial_key"]), {})
        s = json.dumps(rec.get("settings"), sort_keys=True)
        res = rec.get("results") or {}
        g.setdefault(s, {})[json.dumps(res, sort_keys=True)] = res.get(sharpe_key)
    trials, sharpes, split = 0, [], 0
    for g in groups.values():
        widest = max(g.values(), key=len)
        trials += len(widest)
        if len(widest) > 1:
            split += 1
        sharpes += [v for v in widest.values() if _finite(v)]
    return trials, sharpes, split


def _check_dsr(market_series, log: TrialLog | None, sharpe_key: str, th: Thresholds,
               expected_head: str | None = None, expected_min_records: int | None = None):
    r = CheckResult("C", "deflated Sharpe", NOT_RUN)
    if market_series is None or log is None:
        r.reasons.append("needs both the market inputs (returns) and a trial log")
        return r, None
    try:
        records = log.records()
        counts = log.trial_count()
    except LogError as e:
        r.status, r.reasons = FAIL, ["the trial log can't be trusted: %s" % e]
        return r, None
    cut = []
    if expected_min_records is not None and len(records) < expected_min_records:
        cut.append("the trial log has %d records, fewer than the %d expected: lines may have been cut off "
                   "its end" % (len(records), expected_min_records))
    if expected_head is not None and expected_head not in [x["hash"] for x in records]:
        cut.append("the stored head %s is not in the trial log: lines were cut off its end, or the log "
                   "was rewritten" % expected_head)
    if cut:
        r.status, r.reasons = FAIL, cut
        return r, None
    if counts["trials_total"] < 1:
        r.status, r.reasons = FAIL, ["no trials logged; the luck correction can't be applied"]
        return r, None
    compared = []
    if expected_head is None and expected_min_records is None:
        completeness = ["log completeness not verified (pass expected_head): lines cut off its end "
                        "would not be seen"]
    else:
        completeness = []
        if expected_head is not None:
            at = next(x["id"] for x in records if x["hash"] == expected_head)
            compared.append("the stored head %s... is record %s of %d" % (expected_head[:12], at, len(records)))
        if expected_min_records is not None:
            compared.append("%d records, at least the %d expected" % (len(records), expected_min_records))
    excl_counts, all_excluded = _excluded(records)
    excluded = sorted({k for keys in excl_counts.values() for k in keys})
    trials, sharpes, split = _count_trials(records, sharpe_key)
    var = overfit.sharpe_variance(sharpes)
    ex = market_series["strategy_excess"]
    r.thresholds = ["dsr_min", "max_identical_run", "max_repeated_share"]
    try:
        d = overfit.deflated_sharpe(ex, trials, var)
    except ValueError as e:
        r.status, r.reasons = FAIL, ["no deflated Sharpe: %s" % e]
        return r, trials
    r.numbers = {"dsr": d["dsr"], "sharpe_per_period": d["sr"], "sr0": d["sr0"], "trials": trials,
                 "excluded_settings": excluded,
                 "sharpe_var": d["sharpe_var"], "var_measured": d["var_measured"], "psr_valid": d["psr_valid"],
                 "skew": d["skew"], "kurt": d["kurt"], "periods": d["periods"]}
    r.numbers["excluded_values"] = excl_counts
    r.numbers["log_trials"] = counts["trials_total"]
    r.numbers["trials_split"] = split
    r.numbers["completeness"] = compared or None
    fails, warns = [], list(completeness)
    if split:
        warns.append("%d trial(s) were split: the same idea and settings gave different results (the code or the "
                     "data changed), so each distinct result counts as a trial: %d trials, not the log's %d"
                     % (split, trials, counts["trials_total"]))
    flat_warns, flat_notes = _flat_runs("strategy returns", market_series["strategy"], market_series["equity"], th)
    warns += flat_warns
    for idea in all_excluded:
        fails.append("idea %r: every setting is left out of trial counting, so all its runs count as one "
                     "trial" % idea)
    for idea, keys in excl_counts.items():
        for k, n in sorted(keys.items()):
            warns.append("idea %r: setting %r is left out of trial counting; it took %d distinct value(s), "
                         "counted as one trial" % (idea, k, n))
    if not d["psr_valid"]:
        fails.append("the PSR formula broke down (1 - skew*SR + (kurt-1)/4*SR^2 <= 0): counted as a fail")
    elif d["dsr"] < th.dsr_min:
        fails.append("deflated Sharpe %.4f is below %.2f after %d trials" % (d["dsr"], th.dsr_min, trials))
    if not d["var_measured"]:
        warns.append("fewer than 2 trials in the log have a %r result: the Sharpe spread was not measured, "
                     "the sampling variance 1/n was used" % sharpe_key)
    r.status = _status(fails, warns)
    r.reasons = fails + warns
    if not fails:
        r.reasons.insert(0, "deflated Sharpe %.4f >= %.2f after %d trials" % (d["dsr"], th.dsr_min, trials))
    r.reasons += ["log completeness: compared %s" % x for x in compared]
    r.reasons += ["note: %s" % x for x in flat_notes]
    return r, trials


def _check_sharpe_test(market_series, trials, th: Thresholds, block: int) -> CheckResult:
    r = CheckResult("C2", "Sharpe reliably above hold's", NOT_RUN)
    if market_series is None or trials is None:
        r.reasons.append("needs the market inputs and a readable trial log")
        return r
    strict = th.sharpe_test_strict
    r.thresholds = ["alpha", "sharpe_test_strict"]
    mode = "mode: %s" % _mode(strict)
    try:
        t = overfit.sharpe_diff_test(market_series["strategy_excess"], market_series["hold_excess"], trials,
                                     block=block, alpha=th.alpha)
    except ValueError as e:
        r.status = FAIL if strict else WARN
        r.reasons = ["not testable: %s" % e, mode]
        return r
    r.numbers = {k: t[k] for k in ("delta", "se", "p", "p_se", "level", "trials", "block", "draws",
                                   "too_many_trials", "passed")}
    if t["passed"]:
        r.status = PASS
        r.reasons = ["one-sided p %.4f <= %.4f (Sidak, %d trials)" % (t["p"], t["level"], trials), mode]
    else:
        r.status = FAIL if strict else WARN
        why = ("too many trials (%d) for the bootstrap to test at level %.2g" % (trials, t["level"])
               if t["too_many_trials"] else
               "one-sided p %.4f is above %.4f (Sidak, %d trials)" % (t["p"], t["level"], trials))
        r.reasons = [why, mode]
    return r


def _check_sealed(sealed: _Sealed | None, open_sealed: bool, log: TrialLog | None,
                  not_checked: Sequence[str] = ()) -> CheckResult:
    r = CheckResult("D", "sealed-period guard", NOT_RUN)
    if sealed is None:
        r.status, r.reasons = WARN, ["no sealed period declared: nothing is kept back for a final test"]
        return r
    r.numbers = {"sealed": sealed.text, "opened": open_sealed}
    fails, warns = [], []
    touched, no_period = [], []
    not_checked = list(not_checked)
    if log is not None:
        try:
            for rec in log.records():
                p = rec.get("period")
                if not p:
                    no_period.append(rec["id"])
                elif sealed.touches_span(p["start"], p["end"], "trial period"):
                    touched.append(rec["id"])
        except LogError as e:
            fails.append("the trial log can't be trusted: %s" % e)
    else:
        not_checked.append("no trial log given: earlier looks at the sealed period can't be seen")
    if no_period:
        not_checked.append("%d trial-log record(s) have no period: %s" % (len(no_period), ", ".join(no_period[:10])))
    r.numbers["log_records_in_sealed"] = touched
    r.numbers["not_checked"] = not_checked
    warns += ["not checked: %s" % x for x in not_checked]
    if open_sealed:
        warns.append("the sealed period %s was opened: it can be used for one final test only" % sealed.text)
        if len(touched) > 1:
            fails.append("the trial log shows the sealed period used %d times (%s): it was meant to be used once"
                         % (len(touched), ", ".join(touched[:10])))
    elif touched:
        fails.append("trial-log records %s used data from the sealed period" % ", ".join(touched[:10]))
    r.status = _status(fails, warns)
    r.reasons = fails + warns or ["no data from the sealed period %s was used" % sealed.text]
    return r


def _check_markets(markets: Mapping[str, float] | None, th: Thresholds) -> CheckResult:
    r = CheckResult("E", "cross-market", NOT_RUN)
    if markets is None:
        r.reasons.append("no other markets given")
        return r
    if len(markets) < 2:
        r.reasons.append("needs results on at least 2 markets")
        return r
    if th.min_markets_positive is None:
        raise ValueError("check E needs Thresholds(min_markets_positive=...): it has no standard value")
    for k, v in markets.items():
        if not _finite(v):
            raise ValueError("market %r: result must be a finite number, got %r" % (k, v))
    r.thresholds = ["min_markets_positive"]
    positive = [k for k, v in markets.items() if v > 0]
    share = len(positive) / len(markets)
    r.numbers = {"markets": dict(markets), "positive": len(positive), "share_positive": share}
    fails = [] if share >= th.min_markets_positive else [
        "positive on %d of %d markets (%.0f%%, needs %.0f%%)"
        % (len(positive), len(markets), 100 * share, 100 * th.min_markets_positive)]
    warns = ["not positive on %s (%.4g)" % (k, v) for k, v in markets.items() if v <= 0]
    r.status = _status(fails, warns)
    r.reasons = fails + warns or ["positive on all %d markets" % len(markets)]
    return r


def _check_concentration(trades: Trades | None, th: Thresholds) -> CheckResult:
    r = CheckResult("F", "concentration", NOT_RUN)
    if trades is None or not trades.slices:
        r.reasons.append("no slicings of the trades given")
        return r
    r.thresholds = ["max_slice_share"]
    net = trades.net()
    total = math.fsum(net)
    fails = []
    r.numbers = {"total_net": total, "slicings": {}}
    if total <= 0:
        fails.append("no net profit to share (total %.4g)" % total)
    for name, labels in trades.slices.items():
        sums: dict = {}
        for lab, x in zip(labels, net):
            sums[lab] = sums.get(lab, 0.0) + x
        top = max(sums, key=lambda k: sums[k])
        share = sums[top] / total if total > 0 else None
        r.numbers["slicings"][name] = {"by_slice": {str(k): v for k, v in sums.items()}, "largest": str(top),
                                       "largest_share": share}
        if share is not None and share > th.max_slice_share:
            fails.append("by %s: %s gives %.0f%% of the net profit (more than %.0f%%)"
                         % (name, top, 100 * share, 100 * th.max_slice_share))
    r.status = _status(fails, [])
    r.reasons = fails or ["no slice gives more than %.0f%% of the net profit" % (100 * th.max_slice_share)]
    return r


def _trades_vs_market(trades: Trades, market: Market | None, tolerance: float = 0.1) -> tuple[list[str], list[str]]:
    """(warnings, notes): trade dates outside the market's dates, and a trades' total that
    doesn't roughly match (within ``tolerance``, a share) the equity curve's change."""
    warns, notes = [], []
    if market is None:
        return warns, notes
    first, last = market.dates[0], market.dates[-1]
    for what, values in (("entries", trades.entries), ("exits", trades.exits)):
        if not values:
            continue
        try:
            outside = [v for v in values if v < first or v > last]
        except TypeError:
            notes.append("trade %s and market dates can't be compared" % what)
            continue
        if outside:
            warns.append("%d trade %s fall outside the market dates (%s to %s), e.g. %s"
                         % (len(outside), what, first, last, outside[0]))
    net = trades.net()
    change = market.equity[-1] - market.budget
    if trades.unit is None:
        notes.append("the trades' total was not compared with the equity curve (set Trades.unit to "
                     "'return' or 'money')")
    elif net:
        if trades.unit == "money":
            total, want, what = math.fsum(net), change, "money"
        else:
            total, want, what = sc.compound(net), change / market.budget, "compounded return"
        if abs(total - want) > tolerance * max(abs(total), abs(want)) + 1e-9:
            warns.append("the trades' total (%s %.4g) doesn't roughly match the equity curve's change (%.4g), within %g%%: "
                         "are they the same trades and costs?" % (what, total, want, 100 * tolerance))
    return warns, notes


def _check_trades(trades: Trades | None, th: Thresholds, market: Market | None = None) -> CheckResult:
    r = CheckResult("G", "trade summary", NOT_RUN)
    if trades is None:
        r.reasons.append("no trades given")
        return r
    s = trade_summary(trades.net())
    r.numbers = s
    r.thresholds = ["min_trades"]
    pf = s["profit_factor"]
    line = ("%d trades, expectancy %s, profit factor %s, win rate %s, total %.4g, max drawdown %.4g, "
            "longest losing streak %d" % (
                s["trades"], _fmt(s["expectancy"]), _fmt(pf),
                "%.0f%%" % (100 * s["win_rate"]) if s["win_rate"] is not None else "-",
                s["total"], s["max_drawdown"], s["longest_losing_streak"]))
    warns, notes = _trades_vs_market(trades, market, th.trades_equity_tolerance)
    if market is not None and trades.unit is not None:
        r.thresholds.append("trades_equity_tolerance")
    if s["trades"] < th.min_trades:
        r.status = FAIL
        r.reasons = ["%d trades is below the floor of %d" % (s["trades"], th.min_trades), line] + warns + notes
    else:
        r.status = WARN if warns else PASS
        r.reasons = warns + [line, "%d trades is a minimum, not enough on its own" % th.min_trades] + notes
    return r


def _check_code(log: TrialLog | None, idea: str | None) -> CheckResult:
    r = CheckResult("H", "code state", NOT_RUN)
    if log is None:
        r.reasons.append("no trial log given")
        return r
    try:
        recs = log.records()
    except LogError as e:
        r.status, r.reasons = FAIL, ["the trial log can't be trusted: %s" % e]
        return r
    mine = [x for x in recs if idea is None or x["idea"] == idea]
    dirty = [x["id"] for x in mine if isinstance(x.get("code"), Mapping) and x["code"].get("dirty")]
    unknown = [x["id"] for x in mine if not isinstance(x.get("code"), Mapping) or not x["code"].get("commit")]
    r.numbers = {"records": len(mine), "dirty": dirty, "no_commit": unknown}
    warns = []
    if dirty:
        warns.append("%d record(s) ran on uncommitted code and can't be reproduced exactly: %s"
                     % (len(dirty), ", ".join(dirty[:10])))
    if unknown:
        warns.append("%d record(s) don't say which code ran them: %s" % (len(unknown), ", ".join(unknown[:10])))
    if not mine:
        warns.append("no records%s in the trial log" % (" for idea %r" % idea if idea else ""))
    r.status = _status([], warns)
    r.reasons = warns or ["all %d record(s) ran on committed code" % len(mine)]
    return r


def _check_scorecard(card: Mapping | None, series: Mapping | None = None,
                     th: Thresholds | None = None) -> CheckResult:
    r = CheckResult("I", "scorecard vs hold, DCA and the risk-free rate", NOT_RUN)
    if card is None:
        r.reasons.append("no market inputs given")
        return r
    m = card["metrics"]
    s, h, d, f = m["strategy"], m["hold"], m["dca"], m["risk_free"]
    r.numbers = {"metrics": m, "confidence": card["confidence"], "by_type": card["by_type"],
                 "periods": card["periods"], "wiped_out": card["wiped_out"]}

    def beats(a, b):
        return a is not None and (b is None or a > b)

    def sh(x):
        return "%.3f" % x if x is not None else "none"
    fails, warns = [], []
    for name, other in (("hold", h), ("DCA", d)):
        if not beats(s["sharpe"], other["sharpe"]):
            fails.append("Sharpe %s is not above %s's (%s)" % (sh(s["sharpe"]), name, sh(other["sharpe"])))
    if not s["return"] > f["return"]:
        fails.append("return %.2f%% is not above the risk-free return %.2f%%" % (100 * s["return"], 100 * f["return"]))
    if card["wiped_out"]:
        fails.append("the strategy's equity reached 0")
    if not (s["max_drawdown"] < h["max_drawdown"] and s["max_drawdown"] < d["max_drawdown"]):
        warns.append("max drawdown %.1f%% is not below both hold's (%.1f%%) and DCA's (%.1f%%)"
                     % (100 * s["max_drawdown"], 100 * h["max_drawdown"], 100 * d["max_drawdown"]))
    flat_notes = []
    if series is not None and th is not None:
        r.thresholds = ["max_identical_run", "max_repeated_share"]
        w, flat_notes = _flat_runs("strategy returns", series["strategy"], series["equity"], th)
        warns += w
        w, _ = _flat_runs("price returns", series["prices_returns"], series["prices"], th, zero_is_soft=False)
        warns += w
    r.status = _status(fails, warns)
    r.reasons = fails + warns
    beat = (card["confidence"].get("strategy") or {}).get("sharpe_beats") or {}
    r.reasons.append("Sharpe: strategy %s, hold %s, DCA %s; return: strategy %.2f%%, risk-free %.2f%%; "
                     "Sharpe above hold's in %s of resamples"
                     % (sh(s["sharpe"]), sh(h["sharpe"]), sh(d["sharpe"]), 100 * s["return"], 100 * f["return"],
                        "%.0f%%" % (100 * beat["hold"]) if beat.get("hold") is not None else "-"))
    r.reasons += list(card.get("notes") or [])
    r.reasons += ["note: %s" % x for x in flat_notes]
    return r


# --------------------------------------------------------------------------- run

def _verdict(checks: Sequence[CheckResult]) -> str:
    """FAIL if any check fails; INCOMPLETE if a core check wasn't run; else PASS."""
    if any(c.status == FAIL for c in checks):
        return "FAIL"
    if any(c.status == NOT_RUN for c in checks if c.key in CORE):
        return "INCOMPLETE"
    return "PASS"


def run(*, trades: Trades | None = None, random: Mapping | None = None, market: Market | None = None,
        trial_log: TrialLog | None = None, idea: str | None = None, sharpe_key: str = "sharpe",
        sealed=None, open_sealed: bool = False, markets: Mapping[str, float] | None = None,
        thresholds: Thresholds | None = None, sharpe_test_block: int = 10, bootstrap_draws: int = 2000,
        seed: int = 1, expected_head: str | None = None, expected_min_records: int | None = None) -> Report:
    """Run every check the inputs allow and return a :class:`Report`.

    trades: :class:`Trades` (checks A, F, G; entries and exits for D).
    random: the output of :func:`honest_backtest.random_baseline.random_baseline` (B, B2;
        its eligible times for D).
    market: :class:`Market` (C, C2, I; dates for D).
    trial_log: a :class:`honest_backtest.trials.TrialLog` (C's trial count and Sharpe spread,
        D, H). Its ``results[sharpe_key]`` must be per-period Sharpes at the same frequency
        as ``market``'s returns (not annualised), or the spread is wrong.
    idea: limits check H to this idea's records.
    expected_head, expected_min_records: what you stored about the trial log earlier (its
        :meth:`~honest_backtest.trials.TrialLog.head`, its number of records). Lines cut off
        the end of the log are not caught by verification alone; check C FAILs if the stored
        head is no longer in the log, or the log has fewer records.
    sealed: (start, end) of the sealed period, inclusive (D). open_sealed: True only for
        the one final test on it.
    markets: {market: net result} on other markets (E).
    """
    th = thresholds or Thresholds()
    if not isinstance(th, Thresholds):
        raise ValueError("thresholds must be a Thresholds")
    if not isinstance(open_sealed, bool):
        raise ValueError("open_sealed must be True or False")
    if trades is not None:
        trades.check()
    seal = _Sealed(sealed) if sealed is not None else None
    if open_sealed and seal is None:
        raise ValueError("open_sealed=True but no sealed period was declared")

    # the guard first: refuse before computing anything
    not_checked = _guard(seal, open_sealed, market, trades, random) if seal is not None else []

    series = card = None
    if market is not None:
        card = sc.scorecard(market.dates, market.prices, market.equity, market.risk_free, budget=market.budget,
                            fee=market.fee, cost=market.cost, periods_per_year=market.periods_per_year,
                            dca_every=market.dca_every, market_rule=market.market_rule, draws=bootstrap_draws,
                            seed=seed)
        rf = [0.0] + list(market.risk_free)[1:]
        strat, _ = sc.returns_from_equity(market.equity, market.budget)
        hold, _ = sc.returns_from_equity(sc.hold_equity(market.prices, market.budget, market.fee, market.cost),
                                         market.budget)
        prices = list(market.prices)
        series = {"strategy_excess": sc.excess(strat, rf), "hold_excess": sc.excess(hold, rf),
                  "strategy": strat, "equity": list(market.equity), "prices": prices,
                  "prices_returns": [b / a - 1 for a, b in zip(prices, prices[1:])]}

    checks = [_check_costs(trades, th)]
    checks.append(_check_random("B", "beats random entries", (random or {}).get("random_entry"), th, True,
                                trades))
    checks.append(_check_random("B2", "random direction", (random or {}).get("random_direction"), th,
                                th.direction_strict, trades))
    dsr, trials = _check_dsr(series, trial_log, sharpe_key, th, expected_head, expected_min_records)
    checks.append(dsr)
    checks.append(_check_sharpe_test(series, trials, th, sharpe_test_block))
    checks.append(_check_sealed(seal, open_sealed, trial_log, not_checked))
    checks.append(_check_markets(markets, th))
    checks.append(_check_concentration(trades, th))
    checks.append(_check_trades(trades, th, market))
    checks.append(_check_code(trial_log, idea))
    checks.append(_check_scorecard(card, series, th))

    looser = th.looser()
    checks.append(CheckResult("T", "thresholds", WARN if looser else PASS,
                              looser or ["no threshold is looser than its standard value"]))
    verdict = _verdict(checks)
    used: dict[str, list[str]] = {}
    for c in checks:
        if c.status != NOT_RUN:
            for t in c.thresholds:
                used.setdefault(t, []).append(c.key)
    table = [dict(th.describe(name), used_by=keys) for name, keys in used.items()]
    notes = []
    missing = [c.key for c in checks if c.key in CORE and c.status == NOT_RUN]
    if missing:
        notes.append("core check(s) not run: %s; a verdict needs all of %s" % (", ".join(missing), ", ".join(CORE)))
    settings = {"sharpe_key": sharpe_key, "expected_head": expected_head is not None,
                "expected_min_records": expected_min_records, "open_sealed": open_sealed,
                "scorecard_bootstrap": {"draws": bootstrap_draws, "seed": seed} if market is not None else None,
                "sharpe_test_block": sharpe_test_block if market is not None else None,
                "benchmark_fee": market.fee if market is not None else None,
                "benchmark_cost": market.cost if market is not None else None}
    for bench, res in (random or {}).items():
        settings[bench] = {"runs": res.get("runs"), "seed": res.get("seed"), "select": res.get("select_name"),
                           "score": res.get("score_name")}
    return Report(verdict=verdict, checks=checks, thresholds=table, notes=notes, settings=settings)
