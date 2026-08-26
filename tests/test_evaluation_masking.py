import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.report_evaluation_masking import (
    aggregate,
    audit_inputs,
    collect,
    global_rows,
)
from src.evaluation import (
    summarize_evaluation_masking,
    summarize_final_results,
    triangular_avg,
)
from src.metrics import average_incremental_accuracy


class EvaluationMaskingMetricTests(unittest.TestCase):
    def test_main_and_masking_use_the_same_avg_inc_definition(self):
        global_matrix = [[0.5, 0.0], [0.6, 0.7]]
        seen_matrix = [[0.8, 0.0], [0.6, 0.7]]
        ncm_matrix = [[0.7, 0.0], [0.65, 0.75]]

        main = summarize_final_results(
            global_matrix,
            seen_matrix,
            ncm_matrix,
            task_order=[0, 1],
            num_tasks=2,
        )
        masking = summarize_evaluation_masking(
            global_matrix,
            seen_matrix,
            num_tasks=2,
        )

        expected_global = average_incremental_accuracy(global_matrix, 2)
        expected_seen = average_incremental_accuracy(seen_matrix, 2)
        self.assertEqual(triangular_avg(global_matrix, 2), expected_global)
        self.assertEqual(main["avg_inc"], expected_global)
        self.assertEqual(masking["avg_inc_global"], expected_global)
        self.assertEqual(main["avg_inc_seen"], expected_seen)
        self.assertEqual(masking["avg_inc_seen"], expected_seen)

    def test_stage_masking_metrics_and_future_rate_use_distinct_definitions(self):
        report = summarize_evaluation_masking(
            [[0.50, 0.00], [0.60, 0.70]],
            [[0.80, 0.00], [0.60, 0.70]],
            num_tasks=2,
            pred_future_count_matrix=[[3, 0], [0, 0]],
            eval_sample_count_matrix=[[10, 0], [10, 10]],
        )

        self.assertAlmostEqual(report["delta_mask_final"], 0.0)
        self.assertAlmostEqual(report["delta_mask_avg_inc"], 0.15)
        self.assertAlmostEqual(report["bwt_seen"], -0.20)
        self.assertAlmostEqual(report["pred_future_pre_final"], 0.30)
        self.assertAlmostEqual(report["pred_future_final"], 0.0)

    def test_invalid_future_counts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "between zero"):
            summarize_evaluation_masking(
                [[0.5]],
                [[0.5]],
                num_tasks=1,
                pred_future_count_matrix=[[2]],
                eval_sample_count_matrix=[[1]],
            )


class EvaluationMaskingReportTests(unittest.TestCase):
    def _args(self, paths, *, allow_missing=False, strict=False):
        return SimpleNamespace(
            sequential_lora=paths,
            olora=[],
            clora=[],
            recap_wo_refinements=[],
            recap=[],
            allow_missing_pred_future=allow_missing,
            strict_source_consistency=strict,
            audit_only=False,
        )

    def _write_run(self, root, seed, *, include_future=True):
        run = root / f"run_{seed}"
        run.mkdir()
        (run / "config.json").write_text(
            json.dumps(
                {
                    "dataset": "clinc150",
                    "seed": seed,
                    "classifier_protocol": "fixed_global",
                }
            ),
            encoding="utf-8",
        )
        result = {
            "seed": seed,
            "classifier_protocol": "fixed_global",
            "task_order": [0, 1],
            "matrix": [[0.5, 0.0], [0.6, 0.7]],
            "matrix_seen": [[0.8, 0.0], [0.6, 0.7]],
            "final_avg": 0.65,
            "avg_inc": 0.575,
            "bwt": 0.1,
            "bwt_global": 0.1,
            "evaluation_only_seen_masking": {
                "final_avg_global": 0.65,
                "avg_inc_global": 0.575,
                "avg_inc_seen": 0.725,
                "delta_mask_avg_inc": 0.15,
                "bwt_seen": -0.2,
            },
        }
        if include_future:
            result["matrix_pred_future_count"] = [[3, 0], [0, 0]]
            result["matrix_eval_sample_count"] = [[10, 0], [10, 10]]
        (run / "results.json").write_text(
            json.dumps(result),
            encoding="utf-8",
        )
        return run

    def test_three_seed_runs_are_aggregated_with_sample_standard_deviation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [self._write_run(root, seed) for seed in (0, 1, 42)]
            rows = collect(self._args(paths))
            summary = aggregate(rows, [0, 1, 42], allow_incomplete=False)

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["runs"], 3)
        metric = summary[0]["metrics"]["pred_future_pre_final"]
        self.assertAlmostEqual(metric["mean"], 0.3)
        self.assertAlmostEqual(metric["std"], 0.0)
        table2 = global_rows(summary)
        self.assertEqual(
            table2[0]["global_matrix_sha256"],
            summary[0]["global_matrix_sha256"],
        )
        self.assertAlmostEqual(
            table2[0]["metrics"]["avg_inc"]["mean"],
            summary[0]["metrics"]["avg_inc_global"]["mean"],
        )

    def test_strict_source_consistency_accepts_matching_stored_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self._write_run(Path(tmp), 0)
            rows = collect(self._args([run], strict=True))

        self.assertEqual(len(rows), 1)

    def test_strict_source_consistency_rejects_stale_main_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self._write_run(Path(tmp), 0)
            path = run / "results.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            result["avg_inc"] = 0.5
            path.write_text(json.dumps(result), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "results.avg_inc"):
                collect(self._args([run], strict=True))

    def test_legacy_accuracy_matrices_cannot_fabricate_pred_future(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self._write_run(Path(tmp), 0, include_future=False)
            with self.assertRaisesRegex(ValueError, "cannot reconstruct"):
                collect(self._args([run]))

    def test_artifact_audit_identifies_missing_seen_and_future_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self._write_run(Path(tmp), 0, include_future=False)
            result_path = run / "results.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            del result["matrix_seen"]
            result_path.write_text(json.dumps(result), encoding="utf-8")

            failures = audit_inputs(self._args([run]))

        self.assertEqual(failures, 1)


if __name__ == "__main__":
    unittest.main()
