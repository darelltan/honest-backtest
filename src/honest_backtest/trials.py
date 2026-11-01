"""Trial log: an append-only, tamper-evident record of every backtest you tried.

Why: the evidence bar must rise with the number of things tried (see
:func:`honest_backtest.overfit.deflated_sharpe`). That only works if every trial is
written down, including the ones that failed, and can't be quietly edited later.

The log is one JSON Lines file. Each line is one record, written in canonical JSON
(sorted keys, no spaces, UTF-8), and carries:

* ``prev``: the hash of the line before (64 zeros for the first line), and
* ``hash``: the SHA-256 of its own canonical JSON without the ``hash`` field.

So editing, removing, adding or reordering any line breaks the chain, and
:func:`verify` reports where.

What it does **not** protect against: someone rewriting the *whole* chain from the
point of the edit onwards. Every rewritten line would verify. It is tamper-evident,
not tamper-proof. Two defences:

* keep :meth:`TrialLog.head` (the last hash) somewhere else, e.g. a commit message, and
* commit the log to git and run :func:`history_problems` in CI, which checks that every
  committed version of the log only ever adds lines.

In particular, **cutting lines off the end of the log is not caught by** :func:`verify`
**alone**: the lines that remain still form a valid chain. Only comparing with a stored
:meth:`TrialLog.head` (or record count), or with the git history, shows it.
:func:`honest_backtest.checks.run` takes ``expected_head`` and ``expected_min_records``
for this, and check C fails on a mismatch.

Counting trials
---------------
A *trial* is one (idea, settings) pair. Running the same settings again is the same
trial; changing any setting is a new one. Notes on what counts:

* Idea names are compared exactly: ``"breakout"`` and ``"Breakout"`` are two ideas.
* Settings are compared by their canonical JSON: ``10`` and ``10.0`` are different values,
  so they are different trials. Key order does not matter.
* Both of the above can only *overcount* trials, which raises the bar: the safe direction.
* Settings named in ``exclude`` (for example a random seed, when several seeds are one
  trial) are left out of the comparison. They are still recorded. The trial key is fixed
  when a record is written, so changing ``exclude`` later doesn't change past counts.

IDs are ``T0001``, ``T0002``, ... and grow past four digits (``T10000``). Use
:func:`parse_id` to sort them: plain string sorting puts ``T10000`` before ``T9999``.

Writing
-------
Each append holds a lock file (``<log>.lock``) for a moment. If a process crashes or is
killed while writing, the lock file stays behind and the log is **blocked**: every later
write raises :class:`LogLocked`, whose message says how to call :meth:`TrialLog.unlock`.
Only unlock when you are sure no other process is writing. Then run :func:`verify`: a
crash may also have left a partial last line, which verify reports.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence

__all__ = [
    "GENESIS",
    "LogError",
    "LogLocked",
    "TrialLog",
    "format_id",
    "parse_id",
    "verify",
    "verify_lines",
    "check_appends",
    "history_problems",
    "file_sha256",
    "git_state",
]

GENESIS = "0" * 64
_ID = re.compile(r"T([1-9][0-9]*|0[0-9]{3})\Z")


class LogError(Exception):
    """The log refuses: a record that breaks its rules, or a log that was edited."""


class LogLocked(LogError):
    """Another writer holds the log's lock file."""


# --------------------------------------------------------------------------- ids

def format_id(n: int) -> str:
    """The id of the n-th record: T0001 ... T9999, T10000, ..."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("a record number must be a whole number of at least 1, got %r" % (n,))
    return "T%04d" % n


def parse_id(record_id: str) -> int:
    """The record number in an id; the inverse of :func:`format_id`. Use it to sort ids.
    Refuses anything :func:`format_id` would not have written (e.g. ``T00001``, ``t0001``)."""
    if not isinstance(record_id, str):
        raise ValueError("not a record id: %r" % (record_id,))
    m = _ID.match(record_id)
    if not m or format_id(int(m.group(1))) != record_id:
        raise ValueError("not a record id: %r" % (record_id,))
    return int(m.group(1))


# --------------------------------------------------------------------------- canonical JSON

def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha(obj) -> str:
    return hashlib.sha256(_canon(obj).encode("utf-8")).hexdigest()


def _clean(v):
    """JSON-safe: NaN or infinity becomes a string ("nan", "inf") rather than breaking the
    record; tuples become lists; other objects become their str()."""
    if isinstance(v, float) and not math.isfinite(v):
        return repr(v)
    if isinstance(v, Mapping):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def _seal(rec: dict) -> str:
    """The line for a record: its hash added, in canonical JSON."""
    rec = dict(rec)
    rec.pop("hash", None)
    rec["hash"] = _sha(rec)
    return _canon(rec)


def _split_lines(text: str) -> list[str]:
    """Lines of a log file; tolerant of Windows line endings; blank lines skipped."""
    return [ln.rstrip("\r") for ln in text.split("\n") if ln.strip()]


# --------------------------------------------------------------------------- verification

def verify_lines(lines: Sequence[str]) -> list[str]:
    """Problems with a log's lines (empty list: the chain is intact). Each line must be
    exactly the canonical JSON of its record (so no repeated keys), hash to itself, point at
    the one before, and carry the next id."""
    problems, prev = [], GENESIS
    for i, line in enumerate(lines, 1):
        try:
            rec = json.loads(line)
            canonical = _canon(rec)
            stated = rec.pop("hash")
        except (ValueError, KeyError, AttributeError, TypeError) as e:
            problems.append("line %d is not a record: %s" % (i, e))
            break
        if line != canonical:
            # e.g. a repeated key: JSON readers keep the last one, so the line can say two things
            problems.append("line %d is not in canonical form (a repeated key, extra spaces or reordered "
                            "keys): it was not written by this log" % i)
        if rec.get("prev") != prev:
            problems.append("line %d doesn't follow line %d (a line was removed, added or reordered)" % (i, i - 1))
        if _sha(rec) != stated:
            problems.append("line %d was changed after it was written" % i)
        if rec.get("id") != format_id(i):
            problems.append("line %d has id %r, expected %s" % (i, rec.get("id"), format_id(i)))
        prev = stated
    return problems


def verify(path: str | os.PathLike) -> list[str]:
    """Problems with the log file at ``path`` (empty list: intact)."""
    with open(path, encoding="utf-8", newline="") as fh:
        return verify_lines(_split_lines(fh.read()))


def check_appends(versions: Sequence[str]) -> list[str]:
    """Each version of a log's text, oldest first, must start with the whole version before
    it. Line endings are normalised first, so a CRLF checkout isn't flagged."""
    norm = [v.replace("\r\n", "\n") for v in versions]
    return ["version %d drops or changes lines of version %d" % (i + 1, i)
            for i in range(1, len(norm)) if not norm[i].startswith(norm[i - 1])]


def _git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout


def history_problems(repo: str | os.PathLike, rel_path: str) -> list[str]:
    """For CI: every committed version of the log at ``rel_path`` in the git repo ``repo``,
    then the working copy, only ever appends. Needs the full history (in GitHub Actions,
    ``actions/checkout`` with ``fetch-depth: 0``).

    It does not follow renames: history from before the log was moved to ``rel_path`` is
    not checked. Keep the log at one path. Raises ValueError when ``rel_path`` has no
    committed version at all (a mistyped or git-ignored path would otherwise pass).
    """
    repo = os.fspath(repo)
    versions = []
    commits = _git(repo, "log", "--reverse", "--format=%H", "--", rel_path).split()
    if not commits:
        raise ValueError("%s has no committed version in %s: is the path mistyped, or the file git-ignored "
                         "or not committed yet? There is no history to check" % (rel_path, repo))
    for commit in commits:
        try:
            versions.append(_git(repo, "show", "%s:%s" % (commit, rel_path)))
        except subprocess.CalledProcessError:
            versions.append("")  # deleted in that commit
    path = os.path.join(repo, rel_path)
    if os.path.exists(path):
        with open(path, encoding="utf-8", newline="") as fh:
            versions.append(fh.read())
    elif versions:
        versions.append("")
    return check_appends(versions)


# --------------------------------------------------------------------------- fingerprints

def file_sha256(path: str | os.PathLike) -> str:
    """SHA-256 of a file's bytes: a fingerprint of the data a backtest used."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_state(repo: str | os.PathLike) -> dict:
    """The commit checked out in ``repo`` and whether the working tree has changes, for the
    ``code`` field of a record. On failure: ``{"commit": None, "error": why}``."""
    repo = os.fspath(repo)
    try:
        commit = _git(repo, "rev-parse", "HEAD").strip()
        changed = [ln[3:] for ln in _git(repo, "status", "--porcelain").splitlines() if ln[3:]]
    except (OSError, subprocess.CalledProcessError) as e:
        why = (getattr(e, "stderr", None) or str(e)).strip()
        return {"commit": None, "error": why or type(e).__name__}
    return {"commit": commit, "dirty": bool(changed), "changed": changed}


# --------------------------------------------------------------------------- periods

def _as_datetime(v, name: str) -> tuple[dt.datetime, str]:
    if isinstance(v, str):
        try:
            v = dt.datetime.fromisoformat(v) if "T" in v or " " in v else dt.date.fromisoformat(v)
        except ValueError:
            raise ValueError("%s must be an ISO date or date-time, got %r" % (name, v)) from None
    if isinstance(v, dt.datetime):
        return v, v.isoformat()
    if isinstance(v, dt.date):
        return dt.datetime(v.year, v.month, v.day), v.isoformat()
    raise ValueError("%s must be a date, datetime or ISO string, got %r" % (name, v))


def _period(period) -> dict | None:
    if period is None:
        return None
    try:
        start, end = period
    except (TypeError, ValueError):
        raise ValueError("period must be (start, end), got %r" % (period,)) from None
    s, s_text = _as_datetime(start, "period start")
    e, e_text = _as_datetime(end, "period end")
    if (s.tzinfo is None) != (e.tzinfo is None):
        raise ValueError("period start and end must both have a time zone, or both not")
    if e < s:
        raise ValueError("period ends (%s) before it starts (%s)" % (e_text, s_text))
    return {"start": s_text, "end": e_text}


# --------------------------------------------------------------------------- the log

class TrialLog:
    """An append-only, hash-chained trial log in one JSON Lines file.

    >>> log = TrialLog("trials.jsonl", exclude=("seed",))          # doctest: +SKIP
    >>> log.record("breakout", {"lookback": 20, "seed": 1}, {"sharpe": 0.04})  # doctest: +SKIP
    {'id': 'T0001', 'trial': 1, 'trials_total': 1, 'ideas_total': 1, 'hash': '...'}
    """

    def __init__(self, path: str | os.PathLike, exclude: Iterable[str] = ()):
        self.path = os.fspath(path)
        self.lock_path = self.path + ".lock"
        self.exclude = tuple(exclude)
        if not all(isinstance(k, str) for k in self.exclude):
            raise ValueError("exclude must be setting names (strings)")

    # ---- reading

    def records(self) -> list[dict]:
        """All records, oldest first. Raises LogError if the chain is broken."""
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8", newline="") as fh:
            lines = _split_lines(fh.read())
        problems = verify_lines(lines)
        if problems:
            raise LogError("%s was edited, so it can't be trusted or added to: %s"
                           % (self.path, "; ".join(problems[:5])))
        return [json.loads(x) for x in lines]

    def head(self) -> str:
        """The hash of the last record (GENESIS for an empty log). Keep a copy elsewhere:
        if the whole chain is ever rewritten, this hash will no longer be in it."""
        recs = self.records()
        return recs[-1]["hash"] if recs else GENESIS

    def trial_count(self) -> dict:
        """``{"trials_total": n, "ideas_total": k, "per_idea": {idea: trials}}``."""
        c = _count(self.records())
        return {"trials_total": c["trials_total"], "ideas_total": c["ideas_total"], "per_idea": c["per_idea"]}

    def sharpes(self, key: str = "sharpe") -> list[float]:
        """One Sharpe per trial, for :func:`honest_backtest.overfit.sharpe_variance`: the
        ``results[key]`` of the latest record of each trial that has a finite number there."""
        latest: dict[tuple[str, str], float] = {}
        for r in self.records():
            v = (r.get("results") or {}).get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                latest[(r["idea"], r["trial_key"])] = float(v)
        return list(latest.values())

    # ---- writing

    def record(self, idea: str, settings: Mapping[str, object], results: Mapping[str, object], *,
               data_sha256: str | None = None, period=None, notes: Iterable[str] = (),
               code: Mapping[str, object] | None = None) -> dict:
        """Append one trial's record and return its id and the trial counts.

        ``idea``: what is being tested (compared exactly). ``settings``: the parameters that
        define the trial. ``results``: whatever you measured (e.g. ``{"sharpe": ...}``).
        ``period``: (start, end) of the data, as dates, datetimes or ISO strings.
        ``data_sha256``: a fingerprint of the data (see :func:`file_sha256`).
        ``code``: e.g. :func:`git_state` of your strategy's repo.
        """
        if not isinstance(idea, str) or not idea.strip():
            raise LogError("every record needs an idea (what is being tested)")
        if not isinstance(settings, Mapping) or not isinstance(results, Mapping):
            raise LogError("settings and results must be dicts")
        if data_sha256 is not None and not isinstance(data_sha256, str):
            raise LogError("data_sha256 must be a string or None")
        notes = [notes] if isinstance(notes, str) else [str(n) for n in notes]
        settings = _clean(settings)
        ident = {k: v for k, v in settings.items() if k not in self.exclude}
        rec = {
            "idea": idea,
            "settings": settings,
            "excluded_from_trial": sorted(k for k in self.exclude if k in settings),
            "trial_key": _sha({"settings": ident}),
            "results": _clean(results),
            "data_sha256": data_sha256,
            "period": _period(period),
            "notes": notes,
            "code": _clean(code) if code is not None else None,
        }
        with self._locked():
            old = self.records()
            n = len(old) + 1
            rec["id"] = format_id(n)
            rec["time"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            counts = _count(old + [rec])
            rec["trial"] = counts["trial_of"][(idea, rec["trial_key"])]
            rec["trials_total"] = counts["trials_total"]
            rec["ideas_total"] = counts["ideas_total"]
            rec["prev"] = old[-1]["hash"] if old else GENESIS
            line = _seal(rec)
            with open(self.path, "a", encoding="utf-8", newline="\n") as fh:
                fh.write(line + "\n")
        sealed = json.loads(line)
        return {"id": sealed["id"], "trial": sealed["trial"], "trials_total": sealed["trials_total"],
                "ideas_total": sealed["ideas_total"], "hash": sealed["hash"]}

    # ---- the lock

    def _locked_message(self) -> str:
        try:
            with open(self.lock_path, encoding="utf-8") as fh:
                held = fh.read().strip() or "an empty lock file"
        except OSError:
            held = "a lock file that can't be read"
        return ("the trial log %s is locked (%s). Another process may be writing to it right now. "
                "ONLY if you are sure no other process is writing to this log, clear the lock with:\n"
                "    TrialLog(%r).unlock()" % (self.path, held, self.path))

    class _Lock:
        def __init__(self, log: "TrialLog"):
            self.log = log

        def __enter__(self):
            folder = os.path.dirname(os.path.abspath(self.log.lock_path))
            os.makedirs(folder, exist_ok=True)
            try:
                fd = os.open(self.log.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                raise LogLocked(self.log._locked_message()) from None
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write("held by process %d since %s" % (
                    os.getpid(), dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))
            return self

        def __exit__(self, *exc):
            _remove_with_retry(self.log.lock_path)
            return False

    def _locked(self) -> "TrialLog._Lock":
        return TrialLog._Lock(self)

    def unlock(self) -> str:
        """Remove a lock left behind by a writer that crashed. Call it ONLY when you are sure
        no other process is writing to this log: removing a live writer's lock lets two
        appends clash."""
        if not os.path.exists(self.lock_path):
            return "no lock: nothing to clear"
        _remove_with_retry(self.lock_path)
        return "cleared the lock %s" % self.lock_path


def _remove_with_retry(path: str, tries: int = 50, wait: float = 0.1) -> None:
    """Delete a file; Windows refuses while another process has it open, so retry briefly."""
    for i in range(tries):
        try:
            os.remove(path)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(wait)


def _count(recs: Sequence[Mapping]) -> dict:
    per_idea: dict[str, list[str]] = {}
    trial_of: dict[tuple[str, str], int] = {}
    for r in recs:
        keys = per_idea.setdefault(r["idea"], [])
        if r["trial_key"] not in keys:
            keys.append(r["trial_key"])
            trial_of[(r["idea"], r["trial_key"])] = len(keys)
    return {"trials_total": len(trial_of), "ideas_total": len(per_idea),
            "per_idea": {k: len(v) for k, v in per_idea.items()}, "trial_of": trial_of}
