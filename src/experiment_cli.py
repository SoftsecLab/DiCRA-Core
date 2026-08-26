"""Command-line parsing and boundary validation for DiCRA experiments."""

from __future__ import annotations

import argparse

from src.config import canonical_default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DiCRA: representation consolidation and classifier alignment"
    )
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="data/clinc150")
    parser.add_argument("--model_id", type=str, default="bert-base-uncased")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--num_tasks", type=int, default=15)
    parser.add_argument("--num_classes", type=int, default=150)
    parser.add_argument(
        "--classifier_protocol",
        choices=["fixed_global", "dynamic_seen"],
        default="fixed_global",
        help=(
            "fixed_global keeps all dataset classes in the classifier from task 0; "
            "dynamic_seen expands the head at each task and trains/predicts only "
            "over classes observed so far."
        ),
    )
    parser.add_argument(
        "--task_order",
        nargs="+",
        type=int,
        default=None,
        help="Explicit task directory order, e.g. --task_order 3 6 5 2 0 4 1",
    )
    parser.add_argument(
        "--task_order_seed",
        type=int,
        default=None,
        help="Shuffle task order with a fixed seed if --task_order is not provided.",
    )

    parser.add_argument("--lr", type=float, default=canonical_default("lr"))
    parser.add_argument("--epochs", type=int, default=canonical_default("epochs"))
    parser.add_argument(
        "--batch_size",
        type=int,
        default=canonical_default("batch_size"),
    )
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument(
        "--evaluate_train_accuracy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opt-in diagnostic: evaluate historical training splits after every "
            "task. Disabled by default."
        ),
    )
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pad_to_multiple_of", type=int, default=None)
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
    )
    parser.add_argument(
        "--pin_memory",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--allow_missing_tasks",
        action="store_true",
        help=(
            "Skip missing task directories instead of failing. Off by default "
            "because skipped tasks contaminate aggregate metrics."
        ),
    )
    parser.add_argument(
        "--save_checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save per-task model checkpoints. Disabled by default to reduce disk usage.",
    )
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument(
        "--clora_k",
        type=int,
        default=512,
        help="Fixed regularization-subspace width for the Controlled LoRA baseline.",
    )
    parser.add_argument(
        "--clora_lambda",
        type=float,
        default=0.0,
        help="Controlled LoRA regularization weight; zero disables CLoRA.",
    )
    parser.add_argument(
        "--slora_mode",
        choices=["disabled", "pre"],
        default="disabled",
        help=(
            "Enable the online SLoRA-Pre task-boundary denoising baseline. "
            "SLoRA-Post is intentionally outside the online protocol."
        ),
    )
    parser.add_argument(
        "--slora_candidate_ratios",
        nargs="+",
        type=float,
        default=[index / 10 for index in range(1, 11)],
        help="Candidate-rank ratios used by the official SLoRA search.",
    )
    parser.add_argument(
        "--use_sleep",
        action=argparse.BooleanOptionalAction,
        default=canonical_default("use_sleep"),
        help=(
            "Enable DiCRA consolidation, stabilization, and alignment "
            "(canonical default: enabled)."
        ),
    )
    parser.add_argument(
        "--update_prototypes_without_sleep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Update prototype memory after Wake when --use_sleep is disabled, "
            "so NCM/reference diagnostics remain comparable for w/o Sleep runs."
        ),
    )
    parser.add_argument(
        "--matched_alignment_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After each Wake stage, update stored prototypes and run the canonical "
            "classifier-only Gaussian Alignment without consolidation, LoRA reset, "
            "or stabilization. Intended only for the SeqLoRA + Align control."
        ),
    )
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument(
        "--target_norm",
        type=float,
        default=canonical_default("target_norm"),
    )
    parser.add_argument("--alpha", type=float, default=canonical_default("alpha"))
    parser.add_argument(
        "--num_centroids",
        type=int,
        default=canonical_default("num_centroids"),
    )
    parser.add_argument(
        "--rem_classifier_lr",
        type=float,
        default=canonical_default("rem_classifier_lr"),
    )
    parser.add_argument(
        "--rem_cycles_per_class",
        type=int,
        default=canonical_default("rem_cycles_per_class"),
    )
    parser.add_argument(
        "--rem_schedule",
        type=str,
        default=canonical_default("rem_schedule"),
        choices=["fixed_floor", "coverage_clipped"],
        help=(
            "REM replay budget schedule. coverage_clipped is the RECAP canonical "
            "schedule; fixed_floor is retained only for legacy ablations."
        ),
    )
    parser.add_argument(
        "--rem_batch_size",
        type=int,
        default=canonical_default("rem_batch_size"),
    )
    parser.add_argument(
        "--min_rem_steps",
        type=int,
        default=canonical_default("min_rem_steps"),
    )
    parser.add_argument(
        "--max_rem_steps",
        type=int,
        default=canonical_default("max_rem_steps"),
    )
    parser.add_argument("--rem_noise_std", type=float, default=0.05)
    parser.add_argument("--prototype_std_scale", type=float, default=0.5)
    parser.add_argument(
        "--alignment_method",
        choices=[
            "gaussian",
            "mean_only",
            "weight_imprinting",
            "eval_only_imprinting",
            "direct_ncm",
            "none",
        ],
        default=canonical_default("alignment_method"),
        help=(
            "Classifier repair after Stabilization. direct_ncm leaves the learned "
            "classifier unchanged and uses prototype means for primary inference; "
            "eval_only_imprinting imprints a temporary evaluation-only head."
        ),
    )
    parser.add_argument(
        "--audit_imprinting_agreement",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opt-in diagnostic: compare Imprinting and Direct NCM predictions "
            "for every observed task at every stage."
        ),
    )
    parser.add_argument(
        "--min_imprinting_agreement",
        type=float,
        default=0.999,
        help=(
            "Minimum per-stage prediction agreement between single-prototype NCM "
            "and prototype weight imprinting."
        ),
    )

    parser.add_argument(
        "--freeze_layers",
        type=int,
        default=canonical_default("freeze_layers"),
    )
    parser.add_argument("--llrd_gamma", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--min_lr_ratio", type=float, default=0.01)
    parser.add_argument(
        "--rem_dimp",
        type=float,
        default=canonical_default("rem_dimp"),
    )
    parser.add_argument("--feat_distill_beta", type=float, default=0.0)
    parser.add_argument("--no_rem", action="store_true")
    parser.add_argument(
        "--no_consolidation",
        action="store_true",
        help="Skip LoRA-to-backbone consolidation before NREM/REM.",
    )
    parser.add_argument(
        "--merge_gamma",
        type=float,
        default=canonical_default("merge_gamma"),
    )
    parser.add_argument(
        "--merge_gamma_min",
        type=float,
        default=canonical_default("merge_gamma_min"),
        help=(
            "Floor parameter for LoRA merge decay. With --merge_decay_mode max_floor, "
            "decay=max(merge_gamma_min, merge_gamma ** task_id). With affine_floor, "
            "decay=merge_gamma_min + (1 - merge_gamma_min) * merge_gamma ** task_id."
        ),
    )
    parser.add_argument(
        "--merge_decay_mode",
        type=str,
        default=canonical_default("merge_decay_mode"),
        choices=["max_floor", "affine_floor"],
        help=(
            "LoRA merge decay schedule. max_floor preserves previous behavior; "
            "affine_floor uses a smooth asymptotic floor."
        ),
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=canonical_default("lora_alpha"),
    )
    parser.add_argument(
        "--exclude_classifier_stabilization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Exclude classifier parameters from NREM downscaling and norm compression "
            "while retaining LoRA-side stabilization."
        ),
    )
    parser.add_argument(
        "--audit_stabilization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Report and save grouped parameter changes made by NREM stabilization.",
    )
    parser.add_argument("--no_cosine", action="store_true")
    parser.add_argument("--wake_replay_beta", type=float, default=0.0)
    parser.add_argument(
        "--analyze_stages",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--log_rem_diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log before/after REM diagnostics for classifier-feature alignment analysis.",
    )
    parser.add_argument(
        "--print_resource_summary",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print time and parameter profile summary tables at the end of a run.",
    )
    parser.add_argument(
        "--run_reference_diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run post-hoc frozen linear probe diagnostics after each task.",
    )
    parser.add_argument("--probe_epochs", type=int, default=30)
    parser.add_argument("--probe_lr", type=float, default=1e-2)
    parser.add_argument("--probe_weight_decay", type=float, default=1e-4)
    parser.add_argument("--probe_batch_size", type=int, default=512)
    parser.add_argument("--probe_eval_batch_size", type=int, default=1024)
    parser.add_argument(
        "--probe_max_train_examples_per_class",
        type=int,
        default=0,
        help=(
            "Optional cap for diagnostic probe training examples per class. "
            "0 uses all examples."
        ),
    )
    parser.add_argument(
        "--run_oracle_refit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run final classifier-only oracle refit diagnostics.",
    )
    parser.add_argument("--oracle_refit_epochs", type=int, default=30)
    parser.add_argument("--oracle_refit_lr", type=float, default=1e-2)
    parser.add_argument("--oracle_refit_weight_decay", type=float, default=1e-4)
    parser.add_argument("--oracle_refit_batch_size", type=int, default=512)
    parser.add_argument("--oracle_refit_eval_batch_size", type=int, default=1024)
    parser.add_argument(
        "--oracle_refit_max_train_examples_per_class",
        type=int,
        default=0,
        help=(
            "Optional cap for oracle-refit training examples per class. "
            "0 uses all examples."
        ),
    )
    parser.add_argument(
        "--run_prototype_staleness_diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compare stored prototypes with validation-refreshed prototypes at "
            "each post-Alignment checkpoint. This is post-hoc analysis only."
        ),
    )
    parser.add_argument(
        "--prototype_refresh_split",
        choices=["dev", "train"],
        default="dev",
        help="Split used only to rebuild post-hoc diagnostic prototypes.",
    )
    parser.add_argument(
        "--prototype_refresh_protocol",
        choices=["heldout_dev", "historical_train_reencode"],
        default="heldout_dev",
        help=(
            "Declare whether refresh data are held out or are historical train "
            "samples re-encoded strictly for post-hoc diagnosis."
        ),
    )
    parser.add_argument(
        "--prototype_refresh_data_root",
        default=None,
        help=(
            "Optional sidecar data root containing held-out diagnostic splits. "
            "Training and test evaluation continue to use --data_root."
        ),
    )
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def validate_and_normalize_args(args):
    """Validate cross-option contracts and apply intentional CLI normalization."""
    if args.no_rem:
        if args.alignment_method not in {"gaussian", "none"}:
            raise ValueError(
                "--no_rem cannot be combined with an active --alignment_method"
            )
        args.alignment_method = "none"
    if args.run_prototype_staleness_diagnostics:
        if args.prototype_refresh_protocol == "heldout_dev":
            if args.prototype_refresh_split != "dev":
                raise ValueError("heldout_dev refresh requires --prototype_refresh_split dev")
        else:
            if args.prototype_refresh_split != "train":
                raise ValueError(
                    "historical_train_reencode requires --prototype_refresh_split train"
                )
            if args.prototype_refresh_data_root not in {None, args.data_root}:
                raise ValueError(
                    "historical_train_reencode must use the canonical --data_root"
                )
            args.prototype_refresh_data_root = args.data_root
    if not args.use_sleep and args.alignment_method not in {"gaussian", "none"}:
        raise ValueError("Alignment alternatives require --use_sleep")
    if args.matched_alignment_only:
        if args.use_sleep:
            raise ValueError("--matched_alignment_only requires --no-use_sleep")
        if args.alignment_method != "gaussian":
            raise ValueError(
                "--matched_alignment_only requires --alignment_method gaussian"
            )
        if args.no_cosine:
            raise ValueError("--matched_alignment_only requires CosineLinear")
        if not args.no_consolidation:
            raise ValueError(
                "--matched_alignment_only requires --no_consolidation to make the "
                "absence of merge/reset explicit"
            )
        if args.num_centroids != 1:
            raise ValueError("--matched_alignment_only requires --num_centroids 1")
        if args.wake_replay_beta != 0.0 or args.feat_distill_beta != 0.0:
            raise ValueError(
                "--matched_alignment_only does not permit replay or distillation"
            )
        if args.clora_lambda != 0.0 or args.slora_mode != "disabled":
            raise ValueError(
                "--matched_alignment_only cannot be combined with CLoRA or SLoRA"
            )
        if args.analyze_stages or args.log_rem_diagnostics:
            raise ValueError(
                "--matched_alignment_only excludes RECAP stage diagnostics because "
                "they assume consolidation and stabilization"
            )
    if args.alignment_method in {
        "direct_ncm",
        "weight_imprinting",
        "eval_only_imprinting",
    }:
        if args.no_cosine:
            raise ValueError(
                f"{args.alignment_method} requires the canonical cosine classifier"
            )
        if args.num_centroids != 1:
            raise ValueError(
                f"{args.alignment_method} requires --num_centroids 1 for a strict "
                "single-prototype comparison"
            )

    if args.classifier_protocol == "dynamic_seen" and (
        args.run_reference_diagnostics or args.run_oracle_refit
    ):
        raise ValueError(
            "Frozen-probe/oracle-refit diagnostics are not part of the dynamic-head "
            "experiment. Disable --run_reference_diagnostics and --run_oracle_refit."
        )
    if args.clora_k <= 0:
        raise ValueError("--clora_k must be positive")
    if args.clora_lambda < 0:
        raise ValueError("--clora_lambda must be non-negative")
    if args.clora_lambda > 0:
        if args.classifier_protocol != "fixed_global":
            raise ValueError("CLoRA baseline requires --classifier_protocol fixed_global")
        if args.use_sleep:
            raise ValueError("CLoRA baseline requires --no-use_sleep")
        if args.alignment_method != "none":
            raise ValueError("CLoRA baseline requires --alignment_method none")
        if args.no_cosine:
            raise ValueError("CLoRA same-interface baseline requires CosineLinear")
        if args.wake_replay_beta != 0 or args.feat_distill_beta != 0:
            raise ValueError("CLoRA baseline does not permit replay or distillation")
    if any(
        ratio <= 0.0 or ratio > 1.0
        for ratio in args.slora_candidate_ratios
    ):
        raise ValueError("--slora_candidate_ratios values must be in (0, 1]")
    if 1.0 not in args.slora_candidate_ratios:
        raise ValueError("--slora_candidate_ratios must include 1.0")
    if args.slora_mode == "pre":
        if args.use_sleep:
            raise ValueError("SLoRA-Pre baseline requires --no-use_sleep")
        if args.alignment_method != "none":
            raise ValueError("SLoRA-Pre baseline requires --alignment_method none")
        if args.no_cosine:
            raise ValueError("SLoRA-Pre same-interface baseline requires CosineLinear")
        if args.wake_replay_beta != 0 or args.feat_distill_beta != 0:
            raise ValueError("SLoRA-Pre does not permit replay or distillation")
        if args.clora_lambda != 0:
            raise ValueError("SLoRA-Pre cannot be combined with CLoRA regularization")
        if args.save_checkpoints:
            raise ValueError(
                "SLoRA-Pre incremental checkpoints are not yet a supported artifact; "
                "use final results and denoising audits"
            )
    if args.alignment_method == "eval_only_imprinting":
        if args.classifier_protocol != "dynamic_seen":
            raise ValueError(
                "eval_only_imprinting requires --classifier_protocol dynamic_seen"
            )
        if args.wake_replay_beta != 0.0:
            raise ValueError(
                "eval_only_imprinting requires --wake_replay_beta 0 so Acquisition "
                "does not replay historical prototype features"
            )
        if args.run_prototype_staleness_diagnostics:
            raise ValueError(
                "eval_only_imprinting cannot refresh historical prototypes through "
                "--run_prototype_staleness_diagnostics"
            )
    return args
