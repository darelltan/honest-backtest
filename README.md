# honest-backtest

A small Python library that checks a trading strategy's backtest results for **luck,
costs and overfitting**.

It does **not** run strategies and it does **not** download data. You run your own
backtest; honest-backtest examines the results you give it, and tells you which of the
usual ways of fooling yourself it can find.

**Status:** pre-alpha (v0.1). Python 3.11+, standard library only (no numpy, no
dependencies).

## Disclaimer

* **This tool helps catch honest mistakes and self-deception. It cannot stop deliberate
  cheating: anyone can fabricate inputs.**
* **The checks can only find problems. Passing them is not proof that a strategy works**,
  and it says nothing about how it will do in future.
* This is not financial advice.
* Every example here uses **synthetic data. It shows the code works, not that any
  strategy works.**

## Install

From a clone of the repository:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
```

On Linux or macOS use `.venv/bin/python` instead.

## Quick start

The fastest way to see everything is the synthetic example:

```powershell
.venv\Scripts\python.exe examples\synthetic_example.py
```

It runs two made-up scenarios through every check:

* **no skill:** trades at random days of a random price path. Verdict: FAIL.
* **planted edge:** the data generator writes an obviously artificial `PLANTED` marker into
  the data and forces the price up after it, and the trades buy on that marker. Verdict:
  PASS.

Both only buy, so the random-direction check (B2) is NOT RUN for them.

The planted marker exists so the checks have something real to find; no real market has
one.

## The checks

`checks.run(...)` runs every check its inputs allow and returns one report, as text
(`report.text()`) or JSON (`report.json()`).

| | Check | Fails when | Threshold (standard) |
|---|---|---|---|
| A | costs vs edge | net per trade is not above 0, or not above 0 with costs doubled; WARN when costs take more than half the edge, are under 1% of it (they look negligible), or are all 0 | `cost_multiplier` 2, `max_cost_share` 0.5, `min_cost_share` 0.01 |
| B | beats random entries | the strategy is not strictly above the random runs' 95th percentile with p <= 0.05; fewer than 400 valid runs, or under 90% valid; or the baseline doesn't match the strategy (trade count, real results vs `exit_fn`, value vs the trades given). a configuration overriding its long share or stops also FAILs, unless `allow_overrides=True` (then WARN) | `random_percentile` 0.95, `min_random_runs` 400, `min_valid_runs` 0.9, `allow_overrides` False |
| B2 | random direction | WARN when not above the 95th percentile (FAIL with `direction_strict=True`); NOT RUN when the actual trades are one-sided (all long or all short) | as B |
| C | deflated Sharpe | below 0.95 after every trial in the trial log; also FAIL for an empty log, the formula breaking down, a stored `expected_head` missing from the log, or an idea whose every setting is excluded from counting. Reruns of the same settings with different results count as separate trials. WARN for each excluded setting, with how many values it took; when log completeness isn't verified (no `expected_head`); and for filled-in or interpolated returns | `dsr_min` 0.95, `max_identical_run` 10, `max_repeated_share` 0.2 |
| C2 | Sharpe reliably above hold's | WARN when the Ledoit-Wolf test doesn't pass at the trial count (FAIL with `sharpe_test_strict=True`); too many trials to test is said | `alpha` 0.05 |
| D | sealed-period guard | market dates, trade entries and exits, or the random baseline's eligible times and random trade exits inside your sealed period are **refused** (an error; a date counts as its whole day); FAIL if the trial log shows it was used; WARN if none declared, and for anything it could not check | none |
| E | cross-market | too few other markets positive | `min_markets_positive`: **required**, no standard |
| F | concentration | one slice (a year, a market, ...) gives more than half the net profit | `max_slice_share` 0.5 |
| G | trade summary | fewer than 30 trades; also shows expectancy, profit factor, win rate, drawdown, longest losing streak. WARN when trade dates fall outside the market dates, or (with `Trades.unit` set) the trades' total doesn't match the equity curve's change within 10% | `min_trades` 30, `trades_equity_tolerance` 0.1 |
| H | code state | WARN when trials ran on uncommitted code (they can't be reproduced exactly) | none |
| T | thresholds | WARN for every threshold set looser than its standard value (stricter is fine) | |
| I | scorecard | Sharpe not above both hold's and DCA's, return not above the risk-free rate, or wiped out; WARN when the drawdown isn't below both, or the strategy's or prices' returns look filled in or interpolated | `max_identical_run` 10, `max_repeated_share` 0.2 |

**Verdict:** FAIL if any check fails; otherwise INCOMPLETE if a core check (A, B, C, G or I)
wasn't run; otherwise PASS, with every warning listed. Every threshold the report used is
printed, marked "standard" or "set by you", and so are the settings behind it: random seeds
and runs, custom selection and score functions, the trial-log key, the benchmarks' fee and
cost, and whether a stored `expected_head` was compared. Fees and costs are fractions: 0.05
or more is refused as a likely percentage.

**About 30 trades:** it is a floor, not a target. 30 trades is a minimum and not enough on
its own; 100 or more is much better, and every other check gets more reliable with more.

## The modules

### `overfit`: is the Sharpe more than the luck of trying many ideas?

```python
import random
from honest_backtest import overfit

rnd = random.Random(1)                                    # synthetic returns
returns = [rnd.gauss(0.0005, 0.01) for _ in range(500)]
result = overfit.deflated_sharpe(returns, trials=20)     # 20 ideas and settings tried
print(round(result["dsr"], 3), result["psr_valid"])
```

Also: `expected_max_sharpe`, `sidak`, `sharpe_diff_test` (Ledoit-Wolf, block bootstrap) and
`nearby_settings_check` ("a plateau, not a spike").

### `trials`: write down every trial, so the bar can rise with them

```python
import os, tempfile
from honest_backtest.trials import TrialLog

log = TrialLog(os.path.join(tempfile.mkdtemp(), "trials.jsonl"))
log.record("breakout", {"lookback": 20}, {"sharpe": 0.03})
log.record("breakout", {"lookback": 30}, {"sharpe": 0.01})
print(log.trial_count()["trials_total"])                  # 2
```

An append-only JSON Lines file with a SHA-256 hash chain: editing, removing or reordering
any line is caught by `trials.verify(path)`.

### `scorecard`: does it beat holding, DCA and the risk-free rate, after costs?

```python
from honest_backtest import scorecard

dates = list(range(10))                                   # synthetic
prices = [100, 101, 99, 102, 104, 103, 105, 107, 106, 108]
equity = [1000, 1004, 1006, 1011, 1015, 1019, 1022, 1027, 1030, 1034]
card = scorecard.scorecard(dates, prices, equity, [0.0001] * 10, budget=1000, fee=0.001, cost=0.0005,
                           periods_per_year=365, dca_every=3, draws=200)
print({k: round(v["return"], 4) for k, v in card["metrics"].items()})
```

### `random_baseline`: would random trades made the same way have done as well?

```python
from honest_backtest.random_baseline import Config, random_baseline

moves = [((i * 7919) % 13 - 6) / 100 for i in range(300)]          # synthetic price moves
actual = [(e, 1, 1.0) for e in range(0, 300, 10)]                   # (entry, side, stop)

def exit_fn(entry, side, stop):                                     # your exit logic
    return side * moves[entry], entry + 1                           # (result, exit_at)

out = random_baseline({"main": Config(eligible=range(300), actual=actual)}, exit_fn,
                      one_at_a_time=True, runs=400)   # check B needs at least 400
print(out["random_entry"]["percentile"], out["random_entry"]["p95"])
```

`eligible` must cover **exactly** the times the strategy could have entered (same
sessions, same filters), or the comparison is unfair in one direction or the other. If you
picked the best of several configurations, pass them all: each random run applies the
same selection rule to random versions of every one of them.

### `checks`: one report

See the table above and `examples/synthetic_example.py` for a complete call.

## Conventions

* **Fractions everywhere:** 0.002 means 0.2%, for fees, costs, rates and yields.
* **`periods_per_year` is required** wherever annualising happens (365 for daily crypto,
  about 252 for daily stocks, 52 for weekly data). There is no default.
* **Sharpe ratios:** the deflated Sharpe works on per-period Sharpes (never annualised);
  the scorecard reports annualised ones. Both use the sample standard deviation (divisor
  n - 1) and population skew and kurtosis. The Ledoit-Wolf test follows the paper
  (divisor n).
* **Costs:** the benchmarks pay entry costs but **no exit costs**, which makes them slightly
  better: the harder bar. **Your strategy's equity must include its own exit costs.**
* **Trial-log Sharpes** used by check C must be per-period, at the same frequency as the
  returns you check.

## Market-type labels: an example rule

`scorecard` can split results by market type. The rule has no defaults: you choose
every value. Here is one worked example, for daily prices:

```python
from honest_backtest.scorecard import MarketRule

# AN EXAMPLE, NOT A RECOMMENDATION. Fix your own values before looking at results.
rule = MarketRule(
    average=200,       # 200-period moving average of prices
    slope=20,          # bull/bear needs the average higher/lower than 20 periods before
    crash_fall=0.20,   # a fall of 20% or more ...
    crash_within=7,    # ... completed within 7 periods is a crash
)
```

With this rule, the first 219 periods (200 + 20 - 1) are "unlabelled", because the
average and its slope don't exist yet. The labels describe results after the fact. They
are not trading signals: the crash label uses later prices.

## Interpolated or filled-in data

Checks C and I warn when the strategy's returns or the prices look filled in:

* 10 or more (`max_identical_run`) identical consecutive non-zero returns, or equal
  consecutive non-zero steps in the equity or price level;
* more than 20% (`max_repeated_share`) of the non-zero returns, or of the non-zero steps,
  equal to the one before. Daily data stretched to 4-hour bars repeats each day's move 6
  times: runs too short for the first rule, but 5 of every 6 moves repeat.

That is what forward-filled, stretched, geometrically interpolated or linearly
interpolated data looks like. A long run of strategy returns of exactly 0 is reported
separately, as a note only: it may simply be time out of the market. For prices, a flat
run is a warning, since a price doesn't sit out of the market.

Why it matters: filling in or interpolating an equity curve replaces real day-to-day
variation with smooth, artificial steps. The returns then look less volatile and more
consistent than they were, which **inflates** the Sharpe ratio and every test built on it:
the deflated Sharpe, the Ledoit-Wolf test and the scorecard's comparisons. Value your
equity at real prices on every date.

## Keeping a trial log honest

The hash chain is **tamper-evident, not tamper-proof**: someone could rewrite the whole
chain and it would still verify. **Cutting lines off the end of the log is not caught by
`trials.verify()` alone either**: the lines that remain still form a valid chain. Only a
comparison with something stored elsewhere shows it. Two defences:

* keep `log.head()` (the last hash) and the number of records somewhere else, such as a
  commit message, and pass them to `checks.run(..., expected_head=..., expected_min_records=...)`:
  check C fails if the stored head is no longer in the log or records are missing;
* commit the log to git and, in CI, check that it only ever grows:

```python
# in a CI step (needs the full history: actions/checkout with fetch-depth: 0)
# from honest_backtest import trials
# assert trials.history_problems(".", "trials.jsonl") == []
```

## Limitations

* The trial log can only count the trials you record. Unrecorded trials make every bar
  too low.
* `exit_fn` must reproduce the strategy's exits; if you pass your real results, they are
  compared with `exit_fn` and a difference is a warning.
* The Sidak correction assumes independent trials; with correlated trials it is stricter
  than needed.
* The expected best-of-N Sharpe is computed exactly, by numerical integration, assuming the
  trials' Sharpes are normal. The paper's closed-form approximation (about 8% off at N = 2,
  2% at N = 10) is kept as `overfit.expected_max_sharpe_approx`, only to reproduce the
  paper's example; so deflated Sharpes here differ slightly from the paper's.
* If a process crashes or is killed while writing to the trial log, its lock file stays
  and the log is **blocked**: later writes raise `LogLocked`, with the exact `unlock()` call
  to make once you are sure no other process is writing. Then run `trials.verify`.
* `trials.history_problems` does not follow renames: keep the trial log at one path. It raises
  an error if the path has no committed version (a mistyped or git-ignored path).
* Market-type labels and the concentration check depend on how you choose to slice the
  results. Fix the slices before looking.

## Sources

* Bailey, D. H. and Lopez de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting, and Non-Normality. *The Journal of Portfolio
  Management* 40(5), 94-107. doi:10.3905/jpm.2014.40.5.094
* Bailey, D. H. and Lopez de Prado, M. (2012). The Sharpe Ratio Efficient Frontier. *The
  Journal of Risk* 15(2), 3-44. doi:10.21314/JOR.2012.255
* Ledoit, O. and Wolf, M. (2008). Robust performance hypothesis testing with the Sharpe
  ratio. *Journal of Empirical Finance* 15(5), 850-859. doi:10.1016/j.jempfin.2008.03.002
* Memmel, C. (2003). Performance hypothesis testing with the Sharpe Ratio. *Finance
  Letters* 1, 21-23. (As listed in Ledoit and Wolf (2008); used in the tests.)
* Sidak, Z. (1967). Rectangular Confidence Regions for the Means of Multivariate Normal
  Distributions. *Journal of the American Statistical Association* 62(318), 626-633.
  doi:10.1080/01621459.1967.10482935

## Run the tests

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Licence

MIT. See [LICENSE](LICENSE).
