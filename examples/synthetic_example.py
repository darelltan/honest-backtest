"""SYNTHETIC DATA. This example shows the code works, not that any strategy works.

Two made-up scenarios, run through every check in honest-backtest:

1. "no skill": trades at random days of a random price path. Expected verdict: FAIL.
2. "planted edge": the data generator writes an obviously artificial marker, PLANTED, into
   the data every 10th day and then *forces* the price up over the next two days. The
   "trades" simply buy on each PLANTED day. No real market has such a marker; it is there
   so the checks have something real to find. Expected verdict: PASS.

Both scenarios only buy, so the random-direction check (B2) is NOT RUN: it doesn't apply
to a one-sided strategy.

There is no trading rule here, and nothing in this file says anything about real markets.

Run it:  python examples/synthetic_example.py
"""

from __future__ import annotations

import datetime as dt
import os
import random
import sys
import tempfile

from honest_backtest import checks, scorecard
from honest_backtest.random_baseline import Config, random_baseline
from honest_backtest.trials import TrialLog

BANNER = "SYNTHETIC DATA. Shows the code works, not that any strategy works."
START = dt.date(2001, 1, 1)          # a made-up calendar, used only for labels and the sealed period
DAYS = 730                           # the part the checks may see
SEALED_DAYS = 90                     # generated, kept back as the sealed period, never passed in
HOLD = 2                             # days each synthetic trade lasts
COST = 0.001                         # cost per side, as a fraction (entry and exit each pay it)
EVERY = 10                           # PLANTED marker every 10th day
PLANT = 0.015                        # the forced daily rise after a PLANTED marker


def synthetic_market(seed: int, planted: bool):
    """A seeded random price path, with the artificial PLANTED marker if ``planted``."""
    rnd = random.Random(seed)
    n = DAYS + SEALED_DAYS
    markers = [planted and i % EVERY == 3 and i + HOLD < n for i in range(n)]
    prices, p, boost = [], 100.0, 0
    for i in range(n):
        p *= 1 + rnd.gauss(0.0, 0.02) + (PLANT if boost > 0 else 0.0)
        boost = HOLD if markers[i] else max(0, boost - 1)
        prices.append(p)
    return prices, markers


def trade_return(prices, entry, side, hold=HOLD):
    """Return of one trade, net of both sides' costs: enter at the close of ``entry``, exit
    at the close ``hold`` days later."""
    return side * (prices[entry + hold] / prices[entry] - 1) - 2 * COST


def equity_curve(prices, entries, hold=HOLD, budget=1000.0):
    """Daily equity: in the market only during trades; costs paid at entry and exit, so
    the equity includes the strategy's own exit costs."""
    eq, e, inside = [], budget, {}
    for t in entries:
        for k in range(1, hold + 1):
            inside[t + k] = k == hold
    entry_days = set(entries)
    for t in range(DAYS):
        if t in inside:
            e *= prices[t] / prices[t - 1]
            if inside[t]:
                e *= 1 - COST
        if t in entry_days:
            e *= 1 - COST
        eq.append(e)
    return eq


def no_overlap(days, hold=HOLD):
    out, busy = [], -1
    for d in sorted(days):
        if d > busy:
            out.append(d)
            busy = d + hold
    return out


def entries_for(planted, markers, candidates, hold):
    """The trades' entry days: every PLANTED marker, or (no skill) the random candidate days
    that don't overlap for this holding period."""
    if planted:
        return [i for i in candidates if markers[i]]
    return no_overlap(candidates, hold)


def scenario(name: str, planted: bool, log_folder: str, runs: int, draws: int, seed: int = 7):
    prices, markers = synthetic_market(seed, planted)
    last_entry = DAYS - 5 - 1                       # every tried holding period (1-5 days) fits
    days = range(1, last_entry + 1)
    candidates = list(days) if planted else sorted(random.Random(seed + 1).sample(days, 120))
    dates = [START + dt.timedelta(days=i) for i in range(DAYS)]

    # a trial log: five "settings" were tried (hold of 1-5 days), each its own trades and
    # result; all five are recorded, and HOLD (2 days) is the one checked below
    log = TrialLog(os.path.join(log_folder, "%s.jsonl" % name))
    rf = scorecard.rates_from_yields(dates, [(START, 0.02)], 365)
    for hold in range(1, 6):
        eq = equity_curve(prices, entries_for(planted, markers, candidates, hold), hold)
        rs, _ = scorecard.returns_from_equity(eq, 1000.0)
        sr = scorecard.annualised_sharpe(rs, [0.0] + rf[1:], 1)       # per-period Sharpe
        log.record(name, {"hold_days": hold}, {"sharpe": sr}, period=(dates[0], dates[-1]),
                   code={"commit": "synthetic-example", "dirty": False})

    # keep these somewhere else (e.g. a commit message): they show if lines are later cut off the log
    stored_head, stored_records = log.head(), len(log.records())

    entries = entries_for(planted, markers, candidates, HOLD)
    gross = [trade_return(prices, e, 1) + 2 * COST for e in entries]
    trades = checks.Trades(gross=gross, costs=[2 * COST] * len(entries), entries=[dates[e] for e in entries],
                           exits=[dates[e + HOLD] for e in entries], unit="return",
                           slices={"quarter": ["%d-Q%d" % (dates[e].year, (dates[e].month - 1) // 3 + 1)
                                               for e in entries]})

    index = {d: i for i, d in enumerate(dates)}

    def exit_fn(entry, side, stop):                 # entries are dates, so the sealed guard can check them
        i = index[entry]
        return trade_return(prices, i, side), dates[i + HOLD]

    cfg = Config(eligible=[dates[i] for i in days], actual=[(dates[e], 1, 1.0) for e in entries])
    rand = random_baseline({name: cfg}, exit_fn, one_at_a_time=True, runs=runs, seed=seed)

    other_markets = {}
    for k in range(3):
        p2, m2 = synthetic_market(seed + 100 + k, planted)
        c2 = list(days) if planted else sorted(random.Random(seed + 200 + k).sample(days, 120))
        e2 = entries_for(planted, m2, c2, HOLD)
        other_markets["synthetic market %d" % (k + 1)] = sum(trade_return(p2, e, 1) for e in e2) / len(e2)

    market = checks.Market(dates=dates, prices=prices[:DAYS], equity=equity_curve(prices, entries),
                           risk_free=rf, budget=1000.0, fee=COST, cost=0.0, periods_per_year=365, dca_every=7,
                           # AN EXAMPLE, NOT A RECOMMENDATION: fix your own values before looking at results.
                           market_rule=scorecard.MarketRule(average=50, slope=10, crash_fall=0.15, crash_within=5))
    sealed = (START + dt.timedelta(days=DAYS), START + dt.timedelta(days=DAYS + SEALED_DAYS - 1))
    return checks.run(trades=trades, random=rand, market=market, trial_log=log, idea=name, sealed=sealed,
                      markets=other_markets, thresholds=checks.Thresholds(min_markets_positive=2 / 3),
                      bootstrap_draws=draws, expected_head=stored_head, expected_min_records=stored_records)


def main(runs: int = 1000, draws: int = 2000, out=print) -> dict:
    with tempfile.TemporaryDirectory() as folder:
        reports = {"no skill": scenario("no-skill", False, folder, runs, draws),
                   "planted edge": scenario("planted-edge", True, folder, runs, draws)}
    for name, rep in reports.items():
        out("=" * 78)
        out(BANNER)
        out("Scenario: %s" % name)
        out("=" * 78)
        out(rep.text())
        out("")
    out(BANNER)
    return reports


if __name__ == "__main__":
    reports = main()
    sys.exit(0 if reports["no skill"].verdict == "FAIL" and reports["planted edge"].verdict == "PASS" else 1)
