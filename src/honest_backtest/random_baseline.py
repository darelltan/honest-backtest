"""Random baselines: would random trades, made the same way as the strategy's, have done as
well?

Two benchmarks, each run many times with fixed seeds:

* **random entry**: random entry times from the times the strategy *could* have entered,
  with the strategy's own trade count, long/short mix and stop distances, exited by *your*
  exit function. Tests whether the strategy's choice of *when* to trade adds anything.
* **random direction**: the strategy's *actual* entry times and stops, with each trade's
  direction chosen at random. Tests whether its choice of *which way* adds anything.

The strategy's value is then placed among the random runs' values: its percentile, the
95th-percentile bar, and a p-value.

Describing a strategy
---------------------
Each configuration is a :class:`Config`:

* ``eligible``: every time the strategy could have entered. **This must cover exactly
  the times the strategy could have entered: the same sessions, the same filters, the same
  warm-up.** If it includes times the strategy could never trade (e.g. outside its
  session), random trades get chances the strategy never had; if it leaves out times the
  strategy could have traded, they get fewer. Either way the comparison is unfair, in one
  direction or the other.
* ``actual``: the strategy's real trades, as ``(entry, side, stop)`` with side +1 (long)
  or -1 (short). By default the trade count, the long share and the pool of stop distances
  are all taken from these, so random trades are made like the strategy's own.

``exit_fn(entry, side, stop)`` is **your** exit logic. It returns ``(result, exit_at)``, or
None when the trade can't be taken (e.g. it would run past the end of the data; the library
then draws another entry). ``result`` is whatever you measure: an R multiple, a return, or
a tuple such as (normal-cost result, doubled-cost result). ``exit_at`` is when the trade
ended (comparable with entries), used when ``one_at_a_time`` is True. It must give the same
answer for the same input. An exception raised inside it stops everything: it is a bug,
not a failed run.

The strategy's own results are, by default, ``exit_fn`` applied to its actual trades, so
the strategy and the random trades are measured by exactly the same machinery. If you pass
your real results instead, they are checked against ``exit_fn`` on the actual trades, and
a difference is reported as a warning.

Fair best-of-N
--------------
If you picked the best of several configurations, the random runs must do the same. In
every run, **every** configuration gets its own random version, and the same ``select``
rule picks one value, the same rule that picks the strategy's value from its real
configurations. Comparing a best-of-N strategy against single random runs would be rigged
in the strategy's favour.

Seeds
-----
Each (benchmark, run, configuration) has its own random stream, derived from SHA-256 of
``"seed:benchmark:run:configuration"``. Results don't depend on the order of runs, adding a
configuration doesn't change the others' draws, and they are the same on every platform.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass

from .scorecard import quantile

__all__ = ["Config", "best_score", "mean_score", "random_baseline"]


@dataclass
class Config:
    """One configuration of the strategy. See the module docstring.

    eligible: every time the strategy could have entered (distinct, hashable values).
    actual: its real trades, [(entry, side, stop), ...], side +1 or -1.
    n_trades, long_share, stops: default to those of ``actual``.

    The library never changes a Config you pass in: it works on a resolved copy.
    """
    eligible: Sequence[Hashable]
    actual: Sequence[tuple]
    n_trades: int | None = None
    long_share: float | None = None
    stops: Sequence[float] | None = None


@dataclass(frozen=True)
class _Resolved:
    """A checked copy of a Config, with every default filled in."""
    eligible: list
    actual: list
    n_trades: int
    long_share: float
    stops: list


def _resolve(cfg: Config, name: str) -> _Resolved:
    eligible = list(cfg.eligible)
    if not eligible:
        raise ValueError("config %r: eligible is empty" % name)
    try:
        if len(set(eligible)) != len(eligible):
            raise ValueError("config %r: eligible has repeated times" % name)
    except TypeError:
        raise ValueError("config %r: eligible times must be hashable" % name) from None
    actual = [tuple(t) for t in cfg.actual]
    for t in actual:
        if len(t) != 3 or t[1] not in (1, -1) or isinstance(t[1], bool) or not _finite(t[2]):
            raise ValueError("config %r: each actual trade must be (entry, side +1/-1, finite stop), got %r"
                             % (name, t))
    try:
        seen, dup = set(), None
        for t in actual:
            if (t[0], t[1]) in seen:
                dup = t
                break
            seen.add((t[0], t[1]))
    except TypeError:
        raise ValueError("config %r: actual entries must be hashable" % name) from None
    if dup is not None:
        raise ValueError("config %r: the actual trades repeat (entry %r, side %+d): each trade must appear once"
                         % (name, dup[0], dup[1]))
    n_trades = len(actual) if cfg.n_trades is None else cfg.n_trades
    if isinstance(n_trades, bool) or not isinstance(n_trades, int) or n_trades < 1:
        raise ValueError("config %r: n_trades must be a whole number of at least 1, got %r" % (name, n_trades))
    long_share = cfg.long_share
    if long_share is None:
        if not actual:
            raise ValueError("config %r: long_share is needed when there are no actual trades" % name)
        long_share = sum(1 for t in actual if t[1] == 1) / len(actual)
    if not _finite(long_share) or not 0 <= long_share <= 1:
        raise ValueError("config %r: long_share must be between 0 and 1, got %r" % (name, long_share))
    stops = [t[2] for t in actual] if cfg.stops is None else list(cfg.stops)
    if not stops:
        raise ValueError("config %r: no stop distances to draw from" % name)
    if not all(_finite(s) for s in stops):
        raise ValueError("config %r: stop distances must be finite numbers" % name)
    return _Resolved(eligible, actual, n_trades, float(long_share), stops)


def _same_result(a, b) -> bool:
    """Equal results: numbers within a tiny tolerance, tuples and lists element by element."""
    if _finite(a) and _finite(b):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(_same_result(x, y) for x, y in zip(a, b))
    return a == b


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def mean_score(results: Sequence) -> float | None:
    """The default score: the mean result (expectancy). None for no trades."""
    if not results:
        return None
    return math.fsum(results) / len(results)


def best_score(min_trades: int = 0) -> Callable[[Mapping[str, dict]], float | None]:
    """A selection rule: the highest score among configurations with at least
    ``min_trades`` trades and a finite score; None when none qualifies."""
    def select(summary: Mapping[str, dict]) -> float | None:
        ok = [v["score"] for v in summary.values() if v["trades"] >= min_trades and _finite(v["score"])]
        return max(ok) if ok else None
    return select


def _rng(seed: int, bench: str, run: int, name: str) -> random.Random:
    key = ("%r:%s:%d:%s" % (seed, bench, run, name)).encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))


def _call_exit(exit_fn, entry, side, stop):
    out = exit_fn(entry, side, stop)
    if out is None:
        return None
    if not isinstance(out, tuple) or len(out) != 2:
        raise TypeError("exit_fn must return None or (result, exit_at), got %r" % (out,))
    return out


class _Span:
    """The earliest and latest of a stream of values; None once they can't be ordered."""

    def __init__(self):
        self.lo = self.hi = None
        self.ok = True

    def add(self, v) -> None:
        if not self.ok or v is None:
            return
        try:
            self.lo = v if self.lo is None or v < self.lo else self.lo
            self.hi = v if self.hi is None or v > self.hi else self.hi
        except TypeError:
            self.ok = False

    def get(self):
        return (self.lo, self.hi) if self.ok and self.lo is not None else None


def _random_entries(cfg: _Resolved, exit_fn, rng: random.Random, one_at_a_time: bool, max_rounds: int,
                    exits: _Span | None = None):
    """One random version of a configuration. Returns the accepted trades' results, or None
    if the trade count couldn't be reached within the pool and ``max_rounds``."""
    pool = list(cfg.eligible)
    rng.shuffle(pool)
    nxt = 0                                   # pool[nxt:] are not drawn yet
    cand: dict = {}                           # entry -> (side, stop, outcome or None)

    def draw(k):
        nonlocal nxt
        for e in pool[nxt:nxt + k]:
            side = 1 if rng.random() < cfg.long_share else -1
            stop = rng.choice(cfg.stops)
            cand[e] = (side, stop, _call_exit(exit_fn, e, side, stop))
        nxt += min(k, len(pool) - nxt)

    draw(cfg.n_trades)
    for _ in range(max_rounds):
        ok = [(e, v[2]) for e, v in cand.items() if v[2] is not None]
        if one_at_a_time:
            ok.sort(key=lambda x: x[0])
            kept, busy_until = [], None
            for e, (res, exit_at) in ok:
                if busy_until is None or not e < busy_until:
                    kept.append((e, res, exit_at))
                    busy_until = exit_at
        else:
            kept = [(e, res, exit_at) for e, (res, exit_at) in ok]
        diff = cfg.n_trades - len(kept)
        if diff == 0:
            if exits is not None:
                for _, _, exit_at in kept:
                    exits.add(exit_at)
            return [res for _, res, _ in kept]
        if diff > 0:
            if nxt >= len(pool):
                return None                   # the pool is used up
            draw(diff)
        else:
            for e in rng.sample([e for e, _, _ in kept], -diff):
                del cand[e]
    return None


def _random_direction(cfg: _Resolved, exit_fn, rng: random.Random, p_long: float,
                      exits: _Span | None = None):
    out, ends = [], []
    for entry, _, stop in cfg.actual:
        side = 1 if rng.random() < p_long else -1
        o = _call_exit(exit_fn, entry, side, stop)
        if o is None:
            return None
        out.append(o[0])
        ends.append(o[1])
    if exits is not None:
        for v in ends:
            exits.add(v)
    return out


def _compare_real(name: str, real: list, outs: list) -> tuple[dict, list[str]]:
    """Compare the strategy's real results with exit_fn applied to its actual trades.
    Returns (facts, messages): any difference means exit_fn may not reproduce the strategy,
    so the random trades aren't measured the same way."""
    facts = {"given": True, "actual_trades": len(outs), "count_mismatch": False, "untakeable": 0, "differ": 0}
    if not outs:
        return facts, ["config %r: real results were given but there are no actual trades to check them "
                       "against" % name]
    if len(real) != len(outs):
        facts["count_mismatch"] = True
        return facts, ["config %r: %d real results but %d actual trades; exit_fn can't be checked against them"
                       % (name, len(real), len(outs))]
    facts["untakeable"] = sum(1 for o in outs if o is None)
    facts["differ"] = sum(1 for r, o in zip(real, outs) if o is not None and not _same_result(r, o[0]))
    out = []
    if facts["untakeable"]:
        out.append("config %r: exit_fn can't take %d of the strategy's own trades" % (name, facts["untakeable"]))
    if facts["differ"]:
        out.append("config %r: exit_fn gives a different result from the real one for %d of %d trades; "
                   "it may not match the strategy, so the random trades may not be measured the same way"
                   % (name, facts["differ"], len(real)))
    return facts, out


def _name(fn) -> str:
    return getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or repr(fn)


def _overrides(cfg: Config) -> dict:
    """The defaults a Config overrides (instead of taking them from its actual trades)."""
    out = {}
    if cfg.n_trades is not None:
        out["n_trades"] = cfg.n_trades
    if cfg.long_share is not None:
        out["long_share"] = cfg.long_share
    if cfg.stops is not None:
        out["stops"] = len(list(cfg.stops))
    return out


def _span(values: list):
    """(earliest, latest) of the eligible times, or None when they can't be ordered."""
    try:
        return (min(values), max(values))
    except (TypeError, ValueError):
        return None


def _place(strategy: float, values: list[float]) -> dict:
    k = len(values)
    if k == 0:
        return {"percentile": None, "p95": None, "beats_p95": False, "p_value": None}
    below = sum(1 for v in values if v < strategy)
    ties = sum(1 for v in values if v == strategy)
    p95 = quantile(sorted(values), 0.95)
    return {"percentile": (below + 0.5 * ties) / k, "p95": p95, "beats_p95": strategy > p95,
            "p_value": (1 + sum(1 for v in values if v >= strategy)) / (1 + k)}


def random_baseline(configs: Mapping[str, Config], exit_fn: Callable, *, one_at_a_time: bool,
                    runs: int = 1000, seed: int = 1, select: Callable | None = None,
                    score: Callable[[Sequence], float | None] = mean_score, max_rounds: int = 30,
                    direction: str = "coin", real_results: Mapping[str, Sequence] | None = None,
                    benchmarks: Sequence[str] = ("random_entry", "random_direction")) -> dict:
    """Place the strategy among random runs. See the module docstring.

    configs: {name: Config}. One configuration is fine; several are compared fairly.
    one_at_a_time: required. True: a random entry before the previous trade's ``exit_at``
        is dropped and another drawn, as if one position can be open at a time. (The random
        direction benchmark keeps the actual entries as they are, even if a flipped trade
        would now end later.)
    runs: random runs per benchmark (1000 by default: the smallest p-value is about 0.001).
    select: the rule that picks one value from all configurations' summaries
        ``{name: {"score", "trades", "results"}}``; default :func:`best_score` ().
    score: how one configuration's trade results are scored; default the mean.
    max_rounds: redraw rounds for random entry to reach the trade count (30 by default).
        A run that doesn't reach it counts as ``too_few_trades``.
    direction: "coin" (p = 0.5 long) or "long_share" (each configuration's own long share).
    real_results: the strategy's own results per configuration; by default ``exit_fn``
        applied to its actual trades. When given, they are compared with ``exit_fn``
        applied to the actual trades; any difference, or no actual trades to compare with,
        is reported (``real_check``, ``warnings``), and makes check B fail.

    Returns ``{benchmark: {...}}`` with, for each benchmark: "strategy" (the selected real
    value), "percentile" (share of valid runs below it, ties counting half), "p95",
    "beats_p95" (strictly above), "p_value" ((1 + runs >= strategy) / (1 + valid runs)),
    "runs", "valid_runs", "too_few_trades", "failed", "values", "seed", "warnings", and per
    configuration "real_trades" (the strategy's trade count), "real_check" (how its real
    results compared with exit_fn), "n_trades" (random trades drawn), "actual_trades",
    "overrides" (defaults the Config replaced), "long_share" (used to draw),
    "actual_long_share" (of the actual trades), "eligible_span" (earliest and latest
    eligible time) and "exit_span" (earliest and latest exit of the random trades), plus
    "score_is_mean"; random direction
    also says which "direction" and, per configuration, which "direction_p" were used.
    """
    if not isinstance(one_at_a_time, bool):
        raise ValueError("one_at_a_time must be True or False")
    if isinstance(runs, bool) or not isinstance(runs, int) or runs < 1:
        raise ValueError("runs must be a whole number of at least 1, got %r" % (runs,))
    if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds < 1:
        raise ValueError("max_rounds must be a whole number of at least 1, got %r" % (max_rounds,))
    if direction not in ("coin", "long_share"):
        raise ValueError("direction must be 'coin' or 'long_share', got %r" % (direction,))
    unknown = set(benchmarks) - {"random_entry", "random_direction"}
    if unknown or not benchmarks:
        raise ValueError("benchmarks must be 'random_entry' and/or 'random_direction', got %r" % (benchmarks,))
    if not configs:
        raise ValueError("no configurations")
    resolved = {}
    for name, cfg in configs.items():
        if not isinstance(cfg, Config):
            raise ValueError("config %r must be a Config" % (name,))
        resolved[name] = _resolve(cfg, name)
    select_name = "best_score() (default)" if select is None else _name(select)
    score_name = "mean_score (default)" if score is mean_score else _name(score)
    select = select or best_score()

    def summarise(results_by_name):
        return {n: {"score": score(rs), "trades": len(rs), "results": rs} for n, rs in results_by_name.items()}

    # the strategy's own value, by the same score and selection rule
    real, warnings, real_check = {}, [], {}
    for name, cfg in resolved.items():
        outs = [_call_exit(exit_fn, e, s, st) for e, s, st in cfg.actual]
        if real_results is not None and name in real_results:
            real[name] = list(real_results[name])
            real_check[name], msgs = _compare_real(name, real[name], outs)
            warnings += msgs
        else:
            real_check[name] = {"given": False, "actual_trades": len(outs), "count_mismatch": False,
                                "untakeable": 0, "differ": 0}
            if any(o is None for o in outs):
                raise ValueError("config %r: exit_fn can't take one of the strategy's own trades" % name)
            real[name] = [o[0] for o in outs]
    strategy = select(summarise(real))
    if not _finite(strategy):
        raise ValueError("the selection rule picks nothing from the strategy's own results (got %r)" % (strategy,))

    out = {}
    for bench in benchmarks:
        values, too_few, failed = [], 0, 0
        exits = {n: _Span() for n in resolved}
        p_used = {n: (0.5 if direction == "coin" else c.long_share) for n, c in resolved.items()}
        for run in range(runs):
            results, short = {}, False
            for name, cfg in resolved.items():
                rng = _rng(seed, bench, run, name)
                if bench == "random_entry":
                    rs = _random_entries(cfg, exit_fn, rng, one_at_a_time, max_rounds, exits[name])
                    if rs is None:
                        short = True
                        break
                else:
                    rs = _random_direction(cfg, exit_fn, rng, p_used[name], exits[name])
                    if rs is None:
                        break
                results[name] = rs
            if short:
                too_few += 1
                continue
            if len(results) < len(configs):
                failed += 1
                continue
            v = select(summarise(results))
            if not _finite(v):
                failed += 1
                continue
            values.append(float(v))
        res = {"strategy": strategy, "runs": runs, "valid_runs": len(values), "too_few_trades": too_few,
               "failed": failed, "values": values, "seed": seed, "warnings": list(warnings),
               "real_trades": {n: len(rs) for n, rs in real.items()},
               "real_check": real_check,
               "n_trades": {n: c.n_trades for n, c in resolved.items()},
               "actual_trades": {n: len(c.actual) for n, c in resolved.items()},
               "overrides": {n: _overrides(configs[n]) for n in resolved},
               "long_share": {n: c.long_share for n, c in resolved.items()},
               "actual_long_share": {n: (sum(1 for t in c.actual if t[1] == 1) / len(c.actual) if c.actual else None)
                                     for n, c in resolved.items()},
               "eligible_span": {n: _span(c.eligible) for n, c in resolved.items()},
               "exit_span": {n: x.get() for n, x in exits.items()},
               "score_is_mean": score is mean_score, "select_name": select_name, "score_name": score_name}
        res.update(_place(strategy, values))
        if bench == "random_direction":
            res["direction"] = direction
            res["direction_p"] = p_used
        out[bench] = res
    return out
