import unittest
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

from src.config import (
    RECAP_CANONICAL_CONFIG,
    build_experiment_config,
    build_sleep_config,
    build_wake_config,
)
from src.run_config import AlignmentConfig, ExperimentConfig, WakeConfig
from src.runtime_plugins import OPTIONAL_RUNTIME_PLUGINS, enabled_runtime_plugins


class SleepConfigContractTests(unittest.TestCase):
    def test_minimal_namespace_uses_canonical_sleep_defaults(self):
        args = SimpleNamespace(alpha=0.5, target_norm=11.5)

        config = build_sleep_config(args, task_id=3)

        self.assertIsInstance(config, AlignmentConfig)
        self.assertEqual(config.task_id, 3)
        self.assertEqual(config.lora_alpha, RECAP_CANONICAL_CONFIG["lora_alpha"])
        self.assertEqual(
            config.rem_schedule,
            RECAP_CANONICAL_CONFIG["rem_schedule"],
        )
        self.assertEqual(
            config.rem_cycles_per_class,
            RECAP_CANONICAL_CONFIG["rem_cycles_per_class"],
        )
        self.assertFalse(config.no_rem)
        self.assertFalse(config.exclude_classifier_stabilization)

    def test_explicit_sleep_values_are_preserved(self):
        args = SimpleNamespace(
            alpha=0.25,
            target_norm=9.0,
            no_rem=True,
            alignment_method="mean_only",
            lora_alpha=0.02,
            exclude_classifier_stabilization=True,
            audit_stabilization=True,
            rem_classifier_lr=0.01,
            rem_noise_std=0.2,
            rem_dimp=0.3,
            rem_cycles_per_class=40,
            rem_schedule="coverage_clipped",
            rem_batch_size=16,
            min_rem_steps=20,
            max_rem_steps=80,
        )

        config = build_sleep_config(args, task_id="4")

        self.assertEqual(
            config,
            AlignmentConfig(
                alpha=0.25,
                target_norm=9.0,
                no_rem=True,
                alignment_method="mean_only",
                lora_alpha=0.02,
                exclude_classifier_stabilization=True,
                audit_stabilization=True,
                task_id=4,
                rem_classifier_lr=0.01,
                rem_noise_std=0.2,
                rem_dimp=0.3,
                rem_cycles_per_class=40,
                rem_schedule="coverage_clipped",
                rem_batch_size=16,
                min_rem_steps=20,
                max_rem_steps=80,
            ),
        )

    def test_sleep_config_is_immutable(self):
        config = build_sleep_config(
            SimpleNamespace(alpha=0.5, target_norm=11.5),
            task_id=0,
        )

        with self.assertRaises(FrozenInstanceError):
            config.alpha = 0.1

    def test_invalid_alignment_budget_is_rejected_at_boundary(self):
        with self.assertRaisesRegex(ValueError, "max_rem_steps"):
            AlignmentConfig(min_rem_steps=100, max_rem_steps=99)


class WakeConfigContractTests(unittest.TestCase):
    def test_cli_namespace_is_translated_to_an_immutable_wake_contract(self):
        args = SimpleNamespace(
            epochs=4,
            lr=2e-4,
            weight_decay=0.02,
            precision="BF16",
            grad_accum_steps=3,
            freeze_layers=2,
            llrd_gamma=0.8,
            warmup_ratio=0.1,
            min_lr_ratio=0.05,
            feat_distill_beta=0.2,
            wake_replay_beta=0.3,
            clora_k=64,
            clora_lambda=0.4,
            seed=7,
        )

        config = build_wake_config(args)

        self.assertEqual(
            config,
            WakeConfig(
                epochs=4,
                lr=2e-4,
                weight_decay=0.02,
                precision="bf16",
                grad_accum_steps=3,
                freeze_layers=2,
                llrd_gamma=0.8,
                warmup_ratio=0.1,
                min_lr_ratio=0.05,
                feat_distill_beta=0.2,
                wake_replay_beta=0.3,
                clora_k=64,
                clora_lambda=0.4,
                seed=7,
            ),
        )
        with self.assertRaises(FrozenInstanceError):
            config.lr = 1e-3

    def test_invalid_wake_values_are_rejected_at_the_boundary(self):
        invalid_cases = [
            ({"precision": "fp8"}, "precision"),
            ({"lr": 0.0}, "lr"),
            ({"llrd_gamma": 0.0}, "llrd_gamma"),
            ({"warmup_ratio": 1.1}, "warmup_ratio"),
        ]

        for overrides, message in invalid_cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    WakeConfig(**overrides)


class ExperimentConfigContractTests(unittest.TestCase):
    def test_cli_namespace_is_translated_to_an_immutable_runner_contract(self):
        args = SimpleNamespace(
            seed=42,
            data_root="data/example",
            num_tasks=2,
            num_classes=4,
            max_length=64,
            batch_size=8,
            eval_batch_size=16,
            classifier_protocol="fixed_global",
            allow_missing_tasks=False,
            analyze_stages=False,
            log_rem_diagnostics=False,
            matched_alignment_only=False,
            no_consolidation=False,
            merge_gamma=0.75,
            merge_gamma_min=0.0,
            merge_decay_mode="max_floor",
            use_sleep=True,
            update_prototypes_without_sleep=True,
            save_checkpoints=False,
            audit_imprinting_agreement=False,
            min_imprinting_agreement=0.999,
            evaluate_train_accuracy=False,
            run_reference_diagnostics=False,
            run_prototype_staleness_diagnostics=False,
            num_centroids=1,
            prototype_std_scale=0.5,
            prototype_refresh_split="dev",
            prototype_refresh_protocol="heldout_dev",
            prototype_refresh_data_root=None,
            probe_epochs=30,
            probe_lr=1e-2,
            probe_weight_decay=1e-4,
            probe_batch_size=512,
            probe_eval_batch_size=1024,
            probe_max_train_examples_per_class=0,
            alpha=0.5,
            target_norm=11.5,
        )

        config = build_experiment_config(args)

        self.assertIsInstance(config, ExperimentConfig)
        self.assertEqual(config.data_root, "data/example")
        self.assertEqual(config.alignment_method, "gaussian")
        self.assertEqual(config.alignment.task_id, -1)
        with self.assertRaises(FrozenInstanceError):
            config.num_tasks = 3


class RuntimePluginContractTests(unittest.TestCase):
    def test_plugins_are_opt_in_and_keep_registry_order(self):
        args = SimpleNamespace(
            analyze_stages=True,
            audit_imprinting_agreement=True,
        )

        self.assertEqual(
            enabled_runtime_plugins(args),
            ["stage_analysis", "imprinting_agreement"],
        )
        self.assertEqual(enabled_runtime_plugins(SimpleNamespace()), [])
        self.assertEqual(len(OPTIONAL_RUNTIME_PLUGINS), 8)


if __name__ == "__main__":
    unittest.main()
