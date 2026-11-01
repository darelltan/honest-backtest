"""Tests for honest_backtest.trials. No randomness; every test works in a temporary folder."""

import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest

from honest_backtest import trials
from honest_backtest.trials import GENESIS, LogError, LogLocked, TrialLog


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.path = os.path.join(self.tmp, "trials.jsonl")

    def log(self, **kw):
        return TrialLog(self.path, **kw)

    def lines(self):
        with open(self.path, encoding="utf-8", newline="") as fh:
            return fh.read().split("\n")[:-1]

    def write_lines(self, lines, newline="\n"):
        with open(self.path, "w", encoding="utf-8", newline="") as fh:
            fh.write("".join(x + newline for x in lines))


class Ids(unittest.TestCase):
    def test_format_and_parse(self):
        self.assertEqual(trials.format_id(1), "T0001")
        self.assertEqual(trials.format_id(9999), "T9999")
        self.assertEqual(trials.format_id(10000), "T10000")
        for n in (1, 42, 9999, 10000, 123456):
            self.assertEqual(trials.parse_id(trials.format_id(n)), n)

    def test_ids_past_9999_sort_by_number(self):
        ids = ["T10000", "T9999", "T0002", "T10001", "T0100"]
        self.assertEqual(sorted(ids, key=trials.parse_id), ["T0002", "T0100", "T9999", "T10000", "T10001"])
        self.assertNotEqual(sorted(ids), sorted(ids, key=trials.parse_id))  # why parse_id is needed

    def test_ids_format_id_would_not_write_are_refused(self):
        for bad in ("T00001", "T123", "T0000", "t0001", "T-001", "T1e3", "X0001", "", 1, None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                trials.parse_id(bad)
        for bad in (0, -1, 1.0, True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                trials.format_id(bad)


class Format(Base):
    def test_the_line_format_is_pinned(self):
        # The canonical text below is written out by hand, and its hash computed here with
        # hashlib, independently of the module. If this test fails, the format changed and
        # existing logs would no longer verify: don't change it lightly.
        body = ('{"code":null,"data_sha256":null,"excluded_from_trial":[],"id":"T0001",'
                '"idea":"demo","ideas_total":1,"notes":[],"period":null,"prev":"' + GENESIS + '",'
                '"results":{"sharpe":0.05},"settings":{"lookback":20},'
                '"time":"2026-01-01T00:00:00Z","trial":1,"trial_key":"k","trials_total":1}')
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        # keys are sorted, so "hash" goes between "excluded_from_trial" and "id"
        expected = body.replace(',"id":', ',"hash":"%s","id":' % digest)
        self.assertEqual(trials._seal(json.loads(body)), expected)
        self.assertEqual(trials.verify_lines([expected]), [])

    def test_record_fields(self):
        out = self.log().record("breakout", {"lookback": 20}, {"sharpe": 0.05}, data_sha256="ab" * 32,
                                period=("2020-01-01", "2020-12-31"), notes=["first try"],
                                code={"commit": "abc", "dirty": False})
        rec = self.log().records()[0]
        self.assertEqual(out["id"], "T0001")
        self.assertEqual(rec["hash"], out["hash"])
        self.assertEqual(rec["prev"], GENESIS)
        self.assertEqual(rec["period"], {"start": "2020-01-01", "end": "2020-12-31"})
        self.assertEqual(rec["notes"], ["first try"])
        self.assertEqual(rec["code"]["commit"], "abc")
        self.assertTrue(rec["time"].endswith("Z"))

    def test_nan_and_odd_values_are_made_json_safe(self):
        self.log().record("x", {"window": (1, 2)}, {"sharpe": float("nan"), "dd": float("-inf"),
                                                    "when": dt.date(2020, 1, 2)})
        rec = self.log().records()[0]
        self.assertEqual(rec["results"], {"sharpe": "nan", "dd": "-inf", "when": "2020-01-02"})
        self.assertEqual(rec["settings"], {"window": [1, 2]})
        self.assertEqual(trials.verify(self.path), [])

    def test_non_ascii_text_round_trips(self):
        self.log().record("idée", {"name": "Zürich ✓"}, {"note": "日本"})
        self.assertEqual(self.log().records()[0]["settings"]["name"], "Zürich ✓")
        self.assertEqual(trials.verify(self.path), [])

    def test_bad_input_is_refused(self):
        log = self.log()
        for args in (("", {}, {}), ("  ", {}, {}), (None, {}, {}), ("x", [], {}), ("x", {}, 5)):
            with self.assertRaises(LogError, msg=repr(args)):
                log.record(*args)
        for period in (("2020-02-01", "2020-01-01"), ("2020-01-01",), ("soon", "later"),
                       (dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc), dt.datetime(2020, 2, 1))):
            with self.assertRaises(ValueError, msg=repr(period)):
                log.record("x", {}, {}, period=period)
        self.assertFalse(os.path.exists(self.path))


class TrialCounting(Base):
    def test_trials_are_counted_by_the_log(self):
        log = self.log()
        a = log.record("grid", {"levels": 10}, {})
        b = log.record("grid", {"levels": 10}, {})                      # same settings: same trial
        self.assertEqual((a["trial"], b["trial"], b["trials_total"]), (1, 1, 1))
        c = log.record("grid", {"levels": 12}, {})                      # a setting changed: new trial
        self.assertEqual((c["trial"], c["trials_total"]), (2, 2))
        d = log.record("other", {"levels": 10}, {})                     # a new idea doesn't reset the total
        self.assertEqual((d["trial"], d["trials_total"], d["ideas_total"]), (1, 3, 2))
        self.assertEqual(log.trial_count(), {"trials_total": 3, "ideas_total": 2,
                                             "per_idea": {"grid": 2, "other": 1}})

    def test_key_order_does_not_matter(self):
        log = self.log()
        log.record("x", {"a": 1, "b": 2}, {})
        self.assertEqual(log.record("x", {"b": 2, "a": 1}, {})["trial"], 1)

    def test_idea_names_are_compared_exactly(self):
        # Documented: "breakout" and "Breakout" are two ideas (overcounts: the safe side).
        log = self.log()
        log.record("breakout", {"a": 1}, {})
        out = log.record("Breakout", {"a": 1}, {})
        self.assertEqual((out["ideas_total"], out["trials_total"]), (2, 2))

    def test_10_and_10_point_0_are_different_trials(self):
        # Documented: compared as canonical JSON (overcounts: the safe side).
        log = self.log()
        log.record("x", {"a": 10}, {})
        self.assertEqual(log.record("x", {"a": 10.0}, {})["trials_total"], 2)

    def test_excluded_settings_are_kept_but_dont_make_a_new_trial(self):
        log = self.log(exclude=("seed",))
        for seed in (1, 2, 3):
            out = log.record("random-entry", {"hold": 5, "seed": seed}, {})
        self.assertEqual((out["trial"], out["trials_total"]), (1, 1))
        rec = log.records()[-1]
        self.assertEqual(rec["settings"]["seed"], 3)
        self.assertEqual(rec["excluded_from_trial"], ["seed"])

    def test_past_counts_dont_change_with_a_new_exclude(self):
        self.log().record("x", {"seed": 1}, {})
        self.log().record("x", {"seed": 2}, {})
        log = self.log(exclude=("seed",))
        self.assertEqual(log.trial_count()["trials_total"], 2)           # fixed when written
        self.assertEqual(log.record("x", {"seed": 3}, {})["trials_total"], 3)

    def test_sharpes_gives_one_value_per_trial(self):
        log = self.log()
        log.record("x", {"a": 1}, {"sharpe": 0.01})
        log.record("x", {"a": 1}, {"sharpe": 0.02})                      # rerun: latest counts
        log.record("x", {"a": 2}, {"sharpe": 0.05})
        log.record("x", {"a": 3}, {"sharpe": float("nan")})              # not a number: skipped
        log.record("x", {"a": 4}, {})                                    # missing: skipped
        log.record("y", {"a": 1}, {"sr": 0.03})
        self.assertEqual(sorted(log.sharpes()), [0.02, 0.05])
        self.assertEqual(log.sharpes("sr"), [0.03])


class HashChain(Base):
    def three(self):
        log = self.log()
        for i in range(3):
            log.record("x", {"i": i}, {"sharpe": 0.01 * i})
        return self.lines()

    def test_a_clean_log_verifies(self):
        self.three()
        self.assertEqual(trials.verify(self.path), [])

    def test_an_edited_line_is_caught_and_nothing_more_is_added(self):
        lines = self.three()
        lines[1] = lines[1].replace('"sharpe":0.01', '"sharpe":0.5')     # a nicer result
        self.write_lines(lines)
        self.assertTrue(any("changed" in p for p in trials.verify(self.path)))
        with self.assertRaises(LogError):
            self.log().record("x", {"i": 9}, {})
        self.assertEqual(len(self.lines()), 3)

    def test_an_edit_with_its_hash_recomputed_is_caught_at_the_next_line(self):
        lines = self.three()
        rec = json.loads(lines[1])
        rec["results"]["sharpe"] = 0.5
        lines[1] = trials._seal(rec)                                     # its own hash now fits
        self.write_lines(lines)
        problems = trials.verify(self.path)
        self.assertEqual(len(problems), 1)
        self.assertIn("line 3 doesn't follow line 2", problems[0])

    def test_deleted_reordered_and_added_lines_are_caught(self):
        lines = self.three()
        for bad in ([lines[0], lines[2]], [lines[1], lines[0], lines[2]], lines + [lines[2]]):
            self.write_lines(bad)
            self.assertTrue(trials.verify(self.path), msg="should be caught")

    def test_a_garbage_line_is_caught(self):
        lines = self.three()
        self.write_lines(lines[:2] + ["not json"])
        self.assertIn("line 3 is not a record", trials.verify(self.path)[0])
        self.write_lines(lines[:2] + ["[1, 2]"])
        self.assertIn("line 3 is not a record", trials.verify(self.path)[0])

    def test_the_whole_chain_rewritten_still_verifies_but_head_differs(self):
        # Documented limitation: tamper-evident, not tamper-proof. head() is the defence.
        lines = self.three()
        head = self.log().head()
        recs = [json.loads(x) for x in lines]
        recs[1]["results"]["sharpe"] = 0.5
        prev, out = GENESIS, []
        for r in recs:
            r["prev"] = prev
            line = trials._seal(r)
            prev = json.loads(line)["hash"]
            out.append(line)
        self.write_lines(out)
        self.assertEqual(trials.verify(self.path), [])
        self.assertNotEqual(self.log().head(), head)

    def test_windows_line_endings_still_verify(self):
        self.write_lines(self.three(), newline="\r\n")
        self.assertEqual(trials.verify(self.path), [])
        self.assertEqual(len(self.log().records()), 3)

    def test_an_empty_log(self):
        self.assertEqual(self.log().records(), [])
        self.assertEqual(self.log().head(), GENESIS)
        self.assertEqual(self.log().trial_count()["trials_total"], 0)


class PastTenThousand(Base):
    def test_ids_past_9999_append_and_verify(self):
        # 9999 records built directly (record() rereads the log each time, too slow for
        # this many), then two real appends cross into five-digit ids.
        prev, lines = GENESIS, []
        for i in range(1, 10000):
            rec = {"id": trials.format_id(i), "idea": "x", "trial_key": "k", "prev": prev}
            line = trials._seal(rec)
            prev = json.loads(line)["hash"]
            lines.append(line)
        self.write_lines(lines)
        log = self.log()
        self.assertEqual(log.record("x", {}, {})["id"], "T10000")
        self.assertEqual(log.record("x", {}, {})["id"], "T10001")
        self.assertEqual(trials.verify(self.path), [])
        ids = [r["id"] for r in log.records()]
        self.assertEqual(ids[9997:], ["T9998", "T9999", "T10000", "T10001"])
        self.assertEqual(sorted(ids, key=trials.parse_id), ids)


class Appends(unittest.TestCase):
    def test_history_only_ever_appends(self):
        a = '{"id":"T0001"}\n'
        b = a + '{"id":"T0002"}\n'
        self.assertEqual(trials.check_appends([a, b, b]), [])
        self.assertTrue(trials.check_appends([b, a]))                      # last line dropped
        self.assertTrue(trials.check_appends([b, b.replace("2", "3")]))    # a line changed

    def test_crlf_checkouts_are_not_flagged(self):
        a = '{"id":"T0001"}\n'
        b = a + '{"id":"T0002"}\n'
        self.assertEqual(trials.check_appends([a, b.replace("\n", "\r\n")]), [])


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                           "-c", "commit.gpgsign=false", *args], capture_output=True, text=True, check=True).stdout


class GitHistory(Base):
    def test_history_problems_reads_every_committed_version(self):
        git(self.tmp, "init", "-q")
        log = self.log()
        log.record("x", {"i": 1}, {})
        git(self.tmp, "add", "trials.jsonl")
        git(self.tmp, "commit", "-q", "-m", "one")
        log.record("x", {"i": 2}, {})
        git(self.tmp, "commit", "-q", "-am", "two")
        self.assertEqual(trials.history_problems(self.tmp, "trials.jsonl"), [])
        self.write_lines(self.lines()[:1])                                 # drop the newest line
        git(self.tmp, "commit", "-q", "-am", "three")
        self.assertTrue(trials.history_problems(self.tmp, "trials.jsonl"))

    def test_git_state_in_a_repo_and_outside_one(self):
        git(self.tmp, "init", "-q")
        with open(os.path.join(self.tmp, "f.txt"), "w", encoding="utf-8") as fh:
            fh.write("x")
        git(self.tmp, "add", "f.txt")
        git(self.tmp, "commit", "-q", "-m", "one")
        st = trials.git_state(self.tmp)
        self.assertEqual(len(st["commit"]), 40)
        self.assertFalse(st["dirty"])
        with open(os.path.join(self.tmp, "f.txt"), "w", encoding="utf-8") as fh:
            fh.write("y")
        self.assertTrue(trials.git_state(self.tmp)["dirty"])
        outside = trials.git_state(os.path.join(self.tmp, "no-such-folder"))
        self.assertIsNone(outside["commit"])
        self.assertTrue(outside["error"])

    def test_file_sha256(self):
        p = os.path.join(self.tmp, "data.csv")
        with open(p, "wb") as fh:
            fh.write(b"abc")
        # Known value: SHA-256("abc") from FIPS 180-2, Appendix B.1.
        self.assertEqual(trials.file_sha256(p),
                         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


class Lock(Base):
    def hold(self, text="held by process 1 since 2026-01-01T00:00:00Z"):
        with open(self.path + ".lock", "w", encoding="utf-8") as fh:
            fh.write(text)

    def test_a_locked_log_refuses_and_says_how_to_unlock(self):
        self.hold()
        with self.assertRaises(LogLocked) as e:
            self.log().record("x", {}, {})
        msg = str(e.exception)
        self.assertIn("held by process 1", msg)
        self.assertIn("ONLY if you are sure no other process is writing", msg)
        self.assertIn("TrialLog(%r).unlock()" % self.path, msg)
        self.assertFalse(os.path.exists(self.path))                         # nothing written

    def test_the_lock_is_held_only_while_appending(self):
        self.log().record("x", {}, {})
        self.assertFalse(os.path.exists(self.path + ".lock"))

    def test_the_lock_is_let_go_after_an_error(self):
        self.log().record("x", {}, {})
        self.write_lines([self.lines()[0].replace('"x"', '"y"')])          # break the log
        with self.assertRaises(LogError):
            self.log().record("x", {}, {})
        self.assertFalse(os.path.exists(self.path + ".lock"))

    def test_unlock_clears_a_stale_lock(self):
        self.hold("")
        with self.assertRaises(LogLocked) as e:
            self.log().record("x", {}, {})
        self.assertIn("an empty lock file", str(e.exception))
        self.assertIn("cleared", self.log().unlock())
        self.assertEqual(self.log().record("x", {}, {})["id"], "T0001")
        self.assertIn("nothing to clear", self.log().unlock())

    def test_a_lock_can_be_let_go_while_windows_has_it_open(self):
        from unittest import mock
        real, calls = os.remove, []

        def flaky(path):
            calls.append(path)
            if len(calls) < 3:
                raise PermissionError(32, "in use")
            real(path)
        with mock.patch.object(trials.os, "remove", flaky), mock.patch.object(trials.time, "sleep"):
            self.log().record("x", {}, {})
        self.assertEqual(len(calls), 3)
        self.assertFalse(os.path.exists(self.path + ".lock"))


if __name__ == "__main__":
    unittest.main()
