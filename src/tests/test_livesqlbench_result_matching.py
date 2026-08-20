import unittest

from run_livesqlbench_smoke import _results_match


class LiveSQLBenchResultMatchingTest(unittest.TestCase):
    def test_exact_result_still_passes(self):
        self.assertTrue(_results_match([(58.35,)], [(58.35,)], ordered=True))

    def test_missing_constant_text_label_passes(self):
        self.assertTrue(
            _results_match(
                [(58.35,)],
                [("High Risk Routes", 58.35)],
                ordered=True,
            )
        )

    def test_missing_varying_text_column_fails(self):
        self.assertFalse(
            _results_match(
                [(10,), (20,)],
                [("A", 10), ("B", 20)],
                ordered=True,
            )
        )

    def test_missing_numeric_column_fails(self):
        self.assertFalse(
            _results_match(
                [("A",)],
                [("A", 10)],
                ordered=True,
            )
        )

    def test_unordered_projection_preserves_existing_semantics(self):
        self.assertTrue(
            _results_match(
                [(20,), (10,)],
                [("Result", 10), ("Result", 20)],
                ordered=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
