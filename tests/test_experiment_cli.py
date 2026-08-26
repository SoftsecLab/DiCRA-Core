import unittest
from pathlib import Path

from src.config import RECAP_CANONICAL_CONFIG
from src.experiment_cli import parse_args, validate_and_normalize_args


ROOT = Path(__file__).resolve().parents[1]


def validated(*arguments):
    return validate_and_normalize_args(
        parse_args(["--exp_name", "cli_contract", *arguments])
    )


class ExperimentCliContractTests(unittest.TestCase):
    def test_minimal_command_uses_canonical_shared_defaults(self):
        args = validated()
        for name, expected in RECAP_CANONICAL_CONFIG.items():
            self.assertEqual(getattr(args, name), expected, name)
        self.assertEqual(args.classifier_protocol, "fixed_global")
        self.assertTrue(args.deterministic)

    def test_boolean_optional_switches_preserve_existing_spellings(self):
        args = validated(
            "--no-deterministic",
            "--no-use_sleep",
            "--pin_memory",
            "--no-update_prototypes_without_sleep",
        )
        self.assertFalse(args.deterministic)
        self.assertFalse(args.use_sleep)
        self.assertTrue(args.pin_memory)
        self.assertFalse(args.update_prototypes_without_sleep)

    def test_no_rem_normalizes_gaussian_alignment_to_none(self):
        args = validated("--no_rem", "--alignment_method", "gaussian")
        self.assertEqual(args.alignment_method, "none")
        with self.assertRaisesRegex(ValueError, "active --alignment_method"):
            validated("--no_rem", "--alignment_method", "mean_only")

    def test_single_prototype_alignment_contract_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "canonical cosine classifier"):
            validated("--alignment_method", "direct_ncm", "--no_cosine")
        with self.assertRaisesRegex(ValueError, "--num_centroids 1"):
            validated(
                "--alignment_method",
                "weight_imprinting",
                "--num_centroids",
                "2",
            )

    def test_clora_cross_option_contract_is_enforced(self):
        args = validated(
            "--clora_lambda",
            "1.0",
            "--no-use_sleep",
            "--alignment_method",
            "none",
        )
        self.assertEqual(args.clora_lambda, 1.0)
        with self.assertRaisesRegex(ValueError, "requires --no-use_sleep"):
            validated("--clora_lambda", "1.0", "--alignment_method", "none")
        with self.assertRaisesRegex(ValueError, "does not permit replay"):
            validated(
                "--clora_lambda",
                "1.0",
                "--no-use_sleep",
                "--alignment_method",
                "none",
                "--wake_replay_beta",
                "0.5",
            )

    def test_slora_pre_cross_option_contract_is_enforced(self):
        args = validated(
            "--slora_mode",
            "pre",
            "--no-use_sleep",
            "--alignment_method",
            "none",
        )
        self.assertEqual(args.slora_mode, "pre")
        self.assertEqual(args.slora_candidate_ratios[-1], 1.0)

        dynamic_args = validated(
            "--classifier_protocol",
            "dynamic_seen",
            "--slora_mode",
            "pre",
            "--no-use_sleep",
            "--alignment_method",
            "none",
        )
        self.assertEqual(dynamic_args.classifier_protocol, "dynamic_seen")
        self.assertEqual(dynamic_args.slora_mode, "pre")

        invalid_commands = (
            ("--use_sleep",),
            ("--alignment_method", "mean_only"),
            ("--no_cosine",),
            ("--wake_replay_beta", "0.1"),
            ("--feat_distill_beta", "0.1"),
            ("--clora_lambda", "1.0"),
            ("--save_checkpoints",),
        )
        for suffix in invalid_commands:
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                validated(
                    "--slora_mode",
                    "pre",
                    "--no-use_sleep",
                    "--alignment_method",
                    "none",
                    *suffix,
                )

        with self.assertRaisesRegex(ValueError, "include 1.0"):
            validated("--slora_candidate_ratios", "0.5")

    def test_eval_only_imprinting_contract_is_enforced(self):
        args = validated(
            "--classifier_protocol",
            "dynamic_seen",
            "--alignment_method",
            "eval_only_imprinting",
        )
        self.assertEqual(args.alignment_method, "eval_only_imprinting")
        with self.assertRaisesRegex(ValueError, "dynamic_seen"):
            validated("--alignment_method", "eval_only_imprinting")
        with self.assertRaisesRegex(ValueError, "wake_replay_beta 0"):
            validated(
                "--classifier_protocol",
                "dynamic_seen",
                "--alignment_method",
                "eval_only_imprinting",
                "--wake_replay_beta",
                "0.1",
            )

    def test_matched_alignment_only_contract_is_enforced(self):
        args = validated(
            "--no-use_sleep",
            "--matched_alignment_only",
            "--no_consolidation",
            "--alignment_method",
            "gaussian",
        )
        self.assertTrue(args.matched_alignment_only)
        self.assertFalse(args.use_sleep)
        self.assertTrue(args.no_consolidation)

        dynamic_args = validated(
            "--classifier_protocol",
            "dynamic_seen",
            "--no-use_sleep",
            "--matched_alignment_only",
            "--no_consolidation",
            "--alignment_method",
            "gaussian",
        )
        self.assertEqual(dynamic_args.classifier_protocol, "dynamic_seen")
        self.assertTrue(dynamic_args.matched_alignment_only)

        invalid_suffixes = (
            ("--use_sleep",),
            ("--alignment_method", "none"),
            ("--no_cosine",),
            ("--num_centroids", "2"),
            ("--wake_replay_beta", "0.1"),
            ("--feat_distill_beta", "0.1"),
            ("--clora_lambda", "1.0"),
            ("--slora_mode", "pre"),
            ("--analyze_stages",),
            ("--log_rem_diagnostics",),
        )
        for suffix in invalid_suffixes:
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                validated(
                    "--no-use_sleep",
                    "--matched_alignment_only",
                    "--no_consolidation",
                    "--alignment_method",
                    "gaussian",
                    *suffix,
                )
        with self.assertRaisesRegex(ValueError, "--no_consolidation"):
            validated(
                "--no-use_sleep",
                "--matched_alignment_only",
                "--alignment_method",
                "gaussian",
            )

    def test_main_keeps_only_the_cli_compatibility_boundary(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertNotIn("argparse.ArgumentParser", source)
        self.assertNotIn("canonical_default(", source)
        self.assertIn("return parse_experiment_args(argv)", source)
        self.assertIn("validate_and_normalize_args(parse_args())", source)


if __name__ == "__main__":
    unittest.main()
