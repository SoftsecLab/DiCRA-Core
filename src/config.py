"""Single source of truth for DiCRA's canonical training configuration."""

from src.run_config import (
    AlignmentConfig,
    ConsolidationConfig,
    DEFAULT_ALIGNMENT_CONFIG,
    ExperimentConfig,
    WakeConfig,
)


PROJECT_NAME = "DiCRA"
CANONICAL_PRESET_NAME = "recap_coverage_clipped_v1"


# Dataset-specific paths, task counts, class counts, and LLRD factors are added
# by experiment launchers. All shared method defaults must live here.
RECAP_CANONICAL_CONFIG = {
    "lr": 3e-4,
    "epochs": 10,
    "batch_size": 128,
    "use_sleep": True,
    "target_norm": DEFAULT_ALIGNMENT_CONFIG.target_norm,
    "alpha": DEFAULT_ALIGNMENT_CONFIG.alpha,
    "num_centroids": 1,
    "alignment_method": DEFAULT_ALIGNMENT_CONFIG.alignment_method,
    "rem_classifier_lr": DEFAULT_ALIGNMENT_CONFIG.rem_classifier_lr,
    "rem_schedule": DEFAULT_ALIGNMENT_CONFIG.rem_schedule,
    "rem_batch_size": DEFAULT_ALIGNMENT_CONFIG.rem_batch_size,
    "rem_cycles_per_class": DEFAULT_ALIGNMENT_CONFIG.rem_cycles_per_class,
    "min_rem_steps": DEFAULT_ALIGNMENT_CONFIG.min_rem_steps,
    "max_rem_steps": DEFAULT_ALIGNMENT_CONFIG.max_rem_steps,
    "freeze_layers": 0,
    "rem_dimp": DEFAULT_ALIGNMENT_CONFIG.rem_dimp,
    "merge_gamma": 0.75,
    "merge_gamma_min": 0.0,
    "merge_decay_mode": "max_floor",
    "lora_alpha": DEFAULT_ALIGNMENT_CONFIG.lora_alpha,
}


def canonical_default(name):
    """Return a canonical default while failing loudly on misspelled keys."""
    return RECAP_CANONICAL_CONFIG[name]


def canonical_overrides(config):
    """Return shared canonical fields changed by an explicit experiment config."""
    return {
        key: config[key]
        for key, expected in RECAP_CANONICAL_CONFIG.items()
        if key in config and config[key] != expected
    }


def build_sleep_config(args, task_id) -> AlignmentConfig:
    """Build the one canonical Sleep/Alignment runtime configuration."""

    return AlignmentConfig(
        alpha=args.alpha,
        target_norm=args.target_norm,
        no_rem=getattr(args, "no_rem", False),
        alignment_method=getattr(
            args,
            "alignment_method",
            DEFAULT_ALIGNMENT_CONFIG.alignment_method,
        ),
        lora_alpha=getattr(
            args,
            "lora_alpha",
            canonical_default("lora_alpha"),
        ),
        exclude_classifier_stabilization=getattr(
            args,
            "exclude_classifier_stabilization",
            False,
        ),
        audit_stabilization=getattr(args, "audit_stabilization", False),
        task_id=int(task_id),
        rem_classifier_lr=getattr(
            args,
            "rem_classifier_lr",
            canonical_default("rem_classifier_lr"),
        ),
        rem_noise_std=getattr(
            args,
            "rem_noise_std",
            DEFAULT_ALIGNMENT_CONFIG.rem_noise_std,
        ),
        rem_dimp=getattr(
            args,
            "rem_dimp",
            canonical_default("rem_dimp"),
        ),
        rem_cycles_per_class=getattr(
            args,
            "rem_cycles_per_class",
            canonical_default("rem_cycles_per_class"),
        ),
        rem_schedule=getattr(
            args,
            "rem_schedule",
            canonical_default("rem_schedule"),
        ),
        rem_batch_size=getattr(
            args,
            "rem_batch_size",
            canonical_default("rem_batch_size"),
        ),
        min_rem_steps=getattr(
            args,
            "min_rem_steps",
            canonical_default("min_rem_steps"),
        ),
        max_rem_steps=getattr(
            args,
            "max_rem_steps",
            canonical_default("max_rem_steps"),
        ),
    )


def build_wake_config(args) -> WakeConfig:
    """Build the immutable configuration consumed by Wake training."""

    return WakeConfig(
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(getattr(args, "weight_decay", 0.01)),
        precision=str(getattr(args, "precision", "fp32")).lower(),
        grad_accum_steps=int(getattr(args, "grad_accum_steps", 1)),
        freeze_layers=int(getattr(args, "freeze_layers", 0)),
        llrd_gamma=float(getattr(args, "llrd_gamma", 1.0)),
        warmup_ratio=float(getattr(args, "warmup_ratio", 0.0)),
        min_lr_ratio=float(getattr(args, "min_lr_ratio", 0.01)),
        feat_distill_beta=float(
            getattr(args, "feat_distill_beta", 0.0)
        ),
        wake_replay_beta=float(
            getattr(args, "wake_replay_beta", 0.0)
        ),
        clora_k=int(getattr(args, "clora_k", 512)),
        clora_lambda=float(getattr(args, "clora_lambda", 0.0)),
        seed=int(getattr(args, "seed", 0)),
    )


def build_consolidation_config(args) -> ConsolidationConfig:
    """Build the immutable LoRA consolidation configuration."""

    return ConsolidationConfig(
        merge_gamma=float(getattr(args, "merge_gamma", 1.0)),
        merge_gamma_min=float(
            getattr(args, "merge_gamma_min", 0.0)
        ),
        merge_decay_mode=str(
            getattr(args, "merge_decay_mode", "max_floor")
        ),
    )


def build_experiment_config(args) -> ExperimentConfig:
    """Translate the CLI namespace into the task runner's immutable contract."""

    return ExperimentConfig(
        seed=int(args.seed),
        data_root=args.data_root,
        num_tasks=int(args.num_tasks),
        num_classes=int(args.num_classes),
        max_length=int(args.max_length),
        batch_size=int(args.batch_size),
        eval_batch_size=int(args.eval_batch_size),
        classifier_protocol=args.classifier_protocol,
        allow_missing_tasks=bool(args.allow_missing_tasks),
        analyze_stages=bool(args.analyze_stages),
        log_rem_diagnostics=bool(args.log_rem_diagnostics),
        no_consolidation=bool(args.no_consolidation),
        merge_gamma=float(args.merge_gamma),
        merge_gamma_min=float(args.merge_gamma_min),
        merge_decay_mode=args.merge_decay_mode,
        use_sleep=bool(args.use_sleep),
        matched_alignment_only=bool(args.matched_alignment_only),
        update_prototypes_without_sleep=bool(
            args.update_prototypes_without_sleep
        ),
        save_checkpoints=bool(args.save_checkpoints),
        audit_imprinting_agreement=bool(args.audit_imprinting_agreement),
        min_imprinting_agreement=float(args.min_imprinting_agreement),
        evaluate_train_accuracy=bool(args.evaluate_train_accuracy),
        run_reference_diagnostics=bool(args.run_reference_diagnostics),
        run_prototype_staleness_diagnostics=bool(
            args.run_prototype_staleness_diagnostics
        ),
        num_centroids=int(args.num_centroids),
        prototype_std_scale=float(args.prototype_std_scale),
        prototype_refresh_split=args.prototype_refresh_split,
        prototype_refresh_protocol=getattr(
            args, "prototype_refresh_protocol", "heldout_dev"
        ),
        prototype_refresh_data_root=getattr(
            args, "prototype_refresh_data_root", None
        ),
        probe_epochs=int(args.probe_epochs),
        probe_lr=float(args.probe_lr),
        probe_weight_decay=float(args.probe_weight_decay),
        probe_batch_size=int(args.probe_batch_size),
        probe_eval_batch_size=int(args.probe_eval_batch_size),
        probe_max_train_examples_per_class=int(
            args.probe_max_train_examples_per_class
        ),
        alignment=build_sleep_config(args, task_id=-1),
    )
