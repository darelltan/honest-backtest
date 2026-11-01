"""Placeholder test: the package imports and exposes a version string."""

import unittest

import honest_backtest


class TestPackage(unittest.TestCase):
    def test_version_is_a_non_empty_string(self):
        self.assertIsInstance(honest_backtest.__version__, str)
        self.assertTrue(honest_backtest.__version__)


if __name__ == "__main__":
    unittest.main()
