import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from src.run_config import AlignmentConfig
from src.sleep import stabilization_group as compatibility_group
from src.stabilization import (
    stabilization_group,
    stabilize_model,
)


class TinyStabilizationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = nn.Module()
        self.bert.register_parameter(
            "base_weight",
            nn.Parameter(torch.ones(2, 2)),
        )
        self.bert.register_parameter(
            "lora_weight",
            nn.Parameter(torch.ones(2, 2)),
        )
        self.classifier = nn.Module()
        self.classifier.register_parameter(
            "weight",
            nn.Parameter(torch.ones(2, 2)),
        )
        self.classifier.register_parameter(
            "sigma",
            nn.Parameter(torch.tensor(30.0)),
        )
        self.register_parameter(
            "frozen_other",
            nn.Parameter(torch.ones(2), requires_grad=False),
        )


class StabilizationServiceTests(unittest.TestCase):
    def test_parameter_families_keep_the_existing_decay_rules(self):
        model = TinyStabilizationModel()
        frozen_before = model.frozen_other.detach().clone()
        config = AlignmentConfig(
            alpha=0.25,
            lora_alpha=0.1,
            target_norm=1e9,
        )

        stabilize_model(model, config, num_observed_classes=15)

        torch.testing.assert_close(
            model.bert.base_weight,
            torch.full((2, 2), 0.75),
        )
        torch.testing.assert_close(
            model.bert.lora_weight,
            torch.full((2, 2), 0.9),
        )
        torch.testing.assert_close(
            model.classifier.weight,
            torch.full((2, 2), 0.75),
        )
        self.assertEqual(model.classifier.sigma.item(), 30.0)
        torch.testing.assert_close(model.frozen_other, frozen_before)

    def test_dynamic_target_norm_still_scales_with_observed_classes(self):
        model = TinyStabilizationModel()
        with torch.no_grad():
            model.bert.base_weight.fill_(5.0)
        config = AlignmentConfig(
            alpha=0.0,
            lora_alpha=0.0,
            target_norm=2.0,
        )

        stabilize_model(model, config, num_observed_classes=60)

        self.assertAlmostEqual(
            model.bert.base_weight.norm().item(),
            4.0,
            places=5,
        )

    def test_audit_schema_and_sleep_compatibility_export_are_preserved(self):
        model = TinyStabilizationModel()
        config = AlignmentConfig(
            alpha=0.5,
            lora_alpha=0.1,
            target_norm=1e9,
            audit_stabilization=True,
            task_id=0,
        )

        with tempfile.TemporaryDirectory() as output_dir:
            audit = stabilize_model(
                model,
                config,
                num_observed_classes=2,
                output_dir=output_dir,
            )
            audit_path = (
                Path(output_dir) / "stabilization_audit.jsonl"
            )

            self.assertTrue(audit_path.is_file())
            self.assertEqual(
                audit["groups"]["encoder_lora"]["changed"],
                4,
            )
            self.assertNotIn(
                "pre_norm_sq",
                audit["groups"]["encoder_lora"],
            )
            self.assertIn(
                "relative_l2_delta",
                audit["groups"]["encoder_lora"],
            )

        self.assertIs(compatibility_group, stabilization_group)


if __name__ == "__main__":
    unittest.main()
