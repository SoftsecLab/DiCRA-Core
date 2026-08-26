from __future__ import annotations

from dataclasses import dataclass


ALIGNMENT_METHODS = frozenset(
    {
        "gaussian",
        "mean_only",
        "weight_imprinting",
        "eval_only_imprinting",
        "direct_ncm",
        "none",
    }
)
REM_SCHEDULES = frozenset({"fixed_floor", "coverage_clipped"})


@dataclass(frozen=True)
class AlignmentConfig:


    alpha: float = 0.5
    target_norm: float = 11.5
    no_rem: bool = False
    alignment_method: str = "gaussian"
    lora_alpha: float = 0.01
    exclude_classifier_stabilization: bool = False
    audit_stabilization: bool = False
    task_id: int = -1
    rem_classifier_lr: float = 0.005
    rem_noise_std: float = 0.05
    rem_dimp: float = 0.0
    rem_cycles_per_class: int = 80
    rem_schedule: str = "coverage_clipped"
    rem_batch_size: int = 32
    min_rem_steps: int = 100
    max_rem_steps: int = 600

    def __post_init__(self) -> None:
        if self.alignment_method not in ALIGNMENT_METHODS:
            raise ValueError(
                f"Unsupported alignment_method={self.alignment_method!r}. "
                f"Expected one of {sorted(ALIGNMENT_METHODS)}."
            )
        if self.rem_schedule not in REM_SCHEDULES:
            raise ValueError(
                f"Unsupported rem_schedule={self.rem_schedule!r}. "
                f"Expected one of {sorted(REM_SCHEDULES)}."
            )
        if self.rem_batch_size <= 0:
            raise ValueError("rem_batch_size must be positive")
        if self.rem_cycles_per_class < 0:
            raise ValueError("rem_cycles_per_class must be non-negative")
        if self.min_rem_steps < 0:
            raise ValueError("min_rem_steps must be non-negative")
        if self.max_rem_steps < self.min_rem_steps:
            raise ValueError(
                "max_rem_steps must be greater than or equal to min_rem_steps"
            )


DEFAULT_ALIGNMENT_CONFIG = AlignmentConfig()


@dataclass(frozen=True)
class WakeConfig:


    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 0.01
    precision: str = "fp32"
    grad_accum_steps: int = 1
    freeze_layers: int = 0
    llrd_gamma: float = 1.0
    warmup_ratio: float = 0.0
    min_lr_ratio: float = 0.01
    feat_distill_beta: float = 0.0
    wake_replay_beta: float = 0.0
    clora_k: int = 512
    clora_lambda: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")
        if self.grad_accum_steps <= 0:
            raise ValueError("grad_accum_steps must be positive")
        if self.freeze_layers < 0:
            raise ValueError("freeze_layers must be non-negative")
        if self.llrd_gamma <= 0:
            raise ValueError("llrd_gamma must be positive")
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError("warmup_ratio must be between 0 and 1")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be between 0 and 1")
        if self.feat_distill_beta < 0:
            raise ValueError("feat_distill_beta must be non-negative")
        if self.wake_replay_beta < 0:
            raise ValueError("wake_replay_beta must be non-negative")
        if self.clora_k <= 0:
            raise ValueError("clora_k must be positive")
        if self.clora_lambda < 0:
            raise ValueError("clora_lambda must be non-negative")


@dataclass(frozen=True)
class ConsolidationConfig:


    merge_gamma: float = 1.0
    merge_gamma_min: float = 0.0
    merge_decay_mode: str = "max_floor"

    def __post_init__(self) -> None:
        if self.merge_gamma < 0:
            raise ValueError("merge_gamma must be non-negative")
        if self.merge_gamma_min < 0:
            raise ValueError("merge_gamma_min must be non-negative")
        if self.merge_decay_mode not in {
            "max_floor",
            "affine_floor",
        }:
            raise ValueError(
                "merge_decay_mode must be max_floor or affine_floor"
            )


@dataclass(frozen=True)
class ExperimentConfig:


    seed: int
    data_root: str
    num_tasks: int
    num_classes: int
    max_length: int
    batch_size: int
    eval_batch_size: int
    classifier_protocol: str
    allow_missing_tasks: bool
    analyze_stages: bool
    log_rem_diagnostics: bool
    no_consolidation: bool
    merge_gamma: float
    merge_gamma_min: float
    merge_decay_mode: str
    use_sleep: bool
    matched_alignment_only: bool
    update_prototypes_without_sleep: bool
    save_checkpoints: bool
    audit_imprinting_agreement: bool
    min_imprinting_agreement: float
    evaluate_train_accuracy: bool
    run_reference_diagnostics: bool
    run_prototype_staleness_diagnostics: bool
    num_centroids: int
    prototype_std_scale: float
    prototype_refresh_split: str
    prototype_refresh_protocol: str
    prototype_refresh_data_root: str | None
    probe_epochs: int
    probe_lr: float
    probe_weight_decay: float
    probe_batch_size: int
    probe_eval_batch_size: int
    probe_max_train_examples_per_class: int
    alignment: AlignmentConfig

    def __post_init__(self) -> None:
        if self.num_tasks < 0:
            raise ValueError("num_tasks must be non-negative")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.eval_batch_size <= 0:
            raise ValueError("eval_batch_size must be positive")
        if self.classifier_protocol not in {"fixed_global", "dynamic_seen"}:
            raise ValueError(
                "classifier_protocol must be fixed_global or dynamic_seen"
            )
        if self.merge_decay_mode not in {"max_floor", "affine_floor"}:
            raise ValueError(
                "merge_decay_mode must be max_floor or affine_floor"
            )

    @property
    def alignment_method(self) -> str:
        return self.alignment.alignment_method
