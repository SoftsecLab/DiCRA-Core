import copy
import unittest

import torch
import torch.nn as nn

from src.alignment import (
    align_classifier,
    finalize_alignment_report,
    imprint_classifier_from_prototypes,
    resolve_rem_budget,
)
from src.run_config import AlignmentConfig
from src.sleep import (
    finalize_alignment_report as compatibility_finalize,
    imprint_classifier_from_prototypes as compatibility_imprint,
    resolve_rem_budget as compatibility_budget,
)


class TinyAlignmentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor(
                [
                    [2.0, 0.0],
                    [0.0, 3.0],
                ]
            )
        )
        self.sigma = nn.Parameter(torch.tensor(30.0))

    def forward(self, features):
        return features @ self.weight.t()


class TinyAlignmentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = nn.Linear(2, 2, bias=False)
        self.classifier = TinyAlignmentClassifier()

    @staticmethod
    def global_to_classifier_labels(labels):
        mapping = {10: 0, 20: 1}
        return torch.as_tensor(
            [mapping[int(label)] for label in labels],
            device=labels.device,
            dtype=torch.long,
        )


class RecordingPrototypeMemory:
    def __init__(self):
        self.prototypes = {
            10: {"mean": torch.tensor([1.0, 0.0])},
            20: {"mean": torch.tensor([0.0, 1.0])},
        }
        self.feature_modes = []

    def class_mean(self, label):
        return self.prototypes[label]["mean"]

    def get_prototype_batch(self, batch_size, feature_mode):
        self.feature_modes.append((batch_size, feature_mode))
        return (
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            ),
            torch.tensor([10, 20]),
        )


def make_config(method, **overrides):
    values = {
        "alignment_method": method,
        "rem_schedule": "coverage_clipped",
        "rem_batch_size": 2,
        "rem_cycles_per_class": 1,
        "min_rem_steps": 2,
        "max_rem_steps": 2,
        "rem_classifier_lr": 0.01,
        "rem_noise_std": 0.1,
        "task_id": 0,
    }
    values.update(overrides)
    return AlignmentConfig(**values)


class AlignmentServiceContractTests(unittest.TestCase):
    def test_sleep_keeps_the_existing_alignment_imports(self):
        self.assertIs(compatibility_budget, resolve_rem_budget)
        self.assertIs(
            compatibility_finalize,
            finalize_alignment_report,
        )
        self.assertIs(
            compatibility_imprint,
            imprint_classifier_from_prototypes,
        )

    def test_non_optimizing_methods_report_zero_steps(self):
        for method in (
            "none",
            "direct_ncm",
            "eval_only_imprinting",
        ):
            with self.subTest(method=method):
                reports = []
                model = TinyAlignmentModel()

                returned = align_classifier(
                    model,
                    RecordingPrototypeMemory(),
                    torch.device("cpu"),
                    make_config(method),
                    callback=reports.append,
                )

                self.assertIs(returned, model)
                self.assertEqual(len(reports), 1)
                self.assertEqual(reports[0]["realized_steps"], 0)
                self.assertEqual(
                    reports[0]["realized_feature_draws"],
                    0,
                )
                self.assertEqual(
                    reports[0]["skipped_reason"],
                    method,
                )

    def test_mean_only_uses_exact_budget_and_restores_row_norms(self):
        model = TinyAlignmentModel()
        memory = RecordingPrototypeMemory()
        reports = []
        norms_before = model.classifier.weight.detach().norm(
            dim=1
        )

        align_classifier(
            model,
            memory,
            torch.device("cpu"),
            make_config("mean_only"),
            callback=reports.append,
        )

        self.assertEqual(
            memory.feature_modes,
            [(2, "mean"), (2, "mean")],
        )
        self.assertEqual(reports[0]["realized_steps"], 2)
        self.assertEqual(
            reports[0]["realized_feature_draws"],
            4,
        )
        self.assertEqual(reports[0]["feature_mode"], "mean")
        torch.testing.assert_close(
            model.classifier.weight.detach().norm(dim=1),
            norms_before,
        )
        self.assertTrue(model.classifier.sigma.requires_grad)

    def test_gaussian_alignment_is_reproducible_for_a_fixed_seed(self):
        initial = TinyAlignmentModel()
        first = copy.deepcopy(initial)
        second = copy.deepcopy(initial)
        config = make_config("gaussian")

        torch.manual_seed(123)
        align_classifier(
            first,
            RecordingPrototypeMemory(),
            torch.device("cpu"),
            config,
        )
        torch.manual_seed(123)
        align_classifier(
            second,
            RecordingPrototypeMemory(),
            torch.device("cpu"),
            config,
        )

        torch.testing.assert_close(
            first.classifier.weight,
            second.classifier.weight,
        )
        torch.testing.assert_close(
            first.classifier.sigma,
            second.classifier.sigma,
        )


if __name__ == "__main__":
    unittest.main()
