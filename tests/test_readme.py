"""Every Python example in the README runs (synthetic data: shows the code works, not that
any strategy works)."""

import contextlib
import io
import os
import re
import unittest

README = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "README.md")


def blocks():
    with open(README, encoding="utf-8") as fh:
        return re.findall(r"```python\n(.*?)```", fh.read(), re.S)


class ReadmeExamples(unittest.TestCase):
    def test_there_are_examples(self):
        self.assertGreaterEqual(len(blocks()), 5)

    def test_every_python_block_runs(self):
        for i, code in enumerate(blocks(), 1):
            with self.subTest(block=i), contextlib.redirect_stdout(io.StringIO()):
                exec(compile(code, "README block %d" % i, "exec"), {})   # noqa: S102

    def test_the_disclaimer_and_the_trade_floor_note_are_there(self):
        with open(README, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("Passing them is not proof that a strategy works", text)
        self.assertIn("shows the code works, not that any\n  strategy works", text)
        self.assertIn("30 trades is a minimum and not enough on\nits own; 100 or more is much better", text)
        self.assertIn("AN EXAMPLE, NOT A RECOMMENDATION", text)


if __name__ == "__main__":
    unittest.main()
