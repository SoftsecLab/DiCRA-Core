import json
import math
import unittest
from pathlib import Path

from src.metrics import (
    aggregate_seeds,
    average_incremental_accuracy,
    backward_transfer,
    final_average,
    summarize_matrix,
    validate_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class MatrixMetricContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = load_json(
            FIXTURES / "metrics" / "three_task_metrics.json"
        )

    def assert_metric_close(self, actual, expected, label):
        self.assertTrue(
            math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
            f"{label}: {actual} != {expected}",
        )

    def test_fixture_and_current_metrics_agree(self):
        num_tasks = self.fixture["num_tasks"]
        for matrix_name, contract in self.fixture["matrices"].items():
            with self.subTest(matrix=matrix_name):
                expected = contract["expected"]
                new = summarize_matrix(contract["values"], num_tasks)
                for metric_name, expected_value in expected.items():
                    self.assert_metric_close(
                        new[metric_name],
                        expected_value,
                        f"new.{matrix_name}.{metric_name}",
                    )

    def test_individual_functions_match_summary(self):
        matrix = self.fixture["matrices"]["global"]["values"]
        num_tasks = self.fixture["num_tasks"]
        summary = summarize_matrix(matrix, num_tasks)
        self.assert_metric_close(
            final_average(matrix, num_tasks), summary["final_avg"], "final_avg"
        )
        self.assert_metric_close(
            average_incremental_accuracy(matrix, num_tasks),
            summary["avg_inc"],
            "avg_inc",
        )
        self.assert_metric_close(
            backward_transfer(matrix, num_tasks), summary["bwt"], "bwt"
        )

    def test_real_result_fixture_and_current_metrics_agree(self):
        result = load_json(
            FIXTURES / "results" / "olora_adapted_clinc150_s0.json"
        )
        matrix = result["matrix"]
        num_tasks = len(matrix)
        new = summarize_matrix(matrix, num_tasks)
        for metric_name in ("final_avg", "avg_inc", "bwt"):
            self.assert_metric_close(
                new[metric_name], result[metric_name], f"new.{metric_name}"
            )

    def test_single_task_bwt_is_zero(self):
        self.assertEqual(backward_transfer([[0.75]], 1), 0.0)

    def test_extra_columns_are_accepted_and_ignored(self):
        matrix = [[0.5, 99.0], [0.4, 0.6, -99.0]]
        self.assertEqual(validate_matrix(matrix, 2), ((0.5, 99.0), (0.4, 0.6)))
        summary = summarize_matrix(matrix, 2)
        for metric_name, expected in {
            "final_avg": 0.5,
            "avg_inc": 0.5,
            "bwt": -0.1,
        }.items():
            self.assert_metric_close(
                summary[metric_name], expected, f"extra_columns.{metric_name}"
            )

    def test_unlearned_upper_triangle_does_not_affect_metrics(self):
        baseline = [[0.5, 0.0, 0.0], [0.4, 0.6, 0.0], [0.3, 0.4, 0.7]]
        changed = [[0.5, 0.9, -0.9], [0.4, 0.6, 0.8], [0.3, 0.4, 0.7]]
        self.assertEqual(summarize_matrix(changed, 3), summarize_matrix(baseline, 3))

    def test_invalid_matrix_shapes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "3 rows"):
            validate_matrix([[0.5]], 3)
        with self.assertRaisesRegex(ValueError, "at least 3 columns"):
            validate_matrix(
                [[0.5, 0.0, 0.0], [0.4, 0.6], [0.3, 0.4, 0.7]],
                3,
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_matrix([], 0)

    def test_non_numeric_and_non_finite_values_are_rejected(self):
        invalid_values = ["0.5", True, None, math.nan, math.inf, -math.inf]
        for value in invalid_values:
            with self.subTest(value=value):
                matrix = [[0.5, 0.0], [0.4, value]]
                with self.assertRaises((TypeError, ValueError)):
                    validate_matrix(matrix, 2)


class SeedAggregationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_json(
            FIXTURES / "metrics" / "three_task_metrics.json"
        )["seed_aggregation"]

    def test_aggregate_uses_sample_standard_deviation(self):
        actual = aggregate_seeds(self.contract["values"])
        self.assertEqual(actual["values"], self.contract["values"])
        self.assertTrue(
            math.isclose(
                actual["mean"],
                self.contract["mean"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )
        self.assertTrue(
            math.isclose(
                actual["std"],
                self.contract["sample_std"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )

    def test_single_seed_uses_zero_standard_deviation(self):
        self.assertEqual(
            aggregate_seeds([0.8]),
            {"mean": 0.8, "std": 0.0, "values": [0.8]},
        )

    def test_aggregate_rejects_empty_or_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            aggregate_seeds([])
        for values in ([0.8, "0.9"], [0.8, True], [0.8, math.nan]):
            with self.subTest(values=values):
                with self.assertRaises((TypeError, ValueError)):
                    aggregate_seeds(values)


if __name__ == "__main__":
    unittest.main()
