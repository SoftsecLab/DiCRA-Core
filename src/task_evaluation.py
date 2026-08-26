from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from src.dataset import JSONLDataset
from src.evaluation import evaluate_learned_tasks, summarize_task_metrics
from src.experiment_state import ExperimentState
from src.reproducibility import preserve_rng_state
from src.run_config import ExperimentConfig
from src.sleep import temporary_imprinted_classifier


@dataclass(frozen=True)
class TaskEvaluationResult:


    test_accs: list[float]
    ncm_accs: list[float]
    train_accs: list[float]


class TaskEvaluationService:


    def __init__(
        self,
        *,
        config: ExperimentConfig,
        task_order: list[int],
        trainer,
        prototype_memory,
        make_loader: Callable[..., Any],
        state: ExperimentState,
        time_profiler,
        get_model: Callable[[], Any],
        get_seen_labels: Callable[[], set[int]],
    ):
        self.config = config
        self.task_order = list(task_order)
        self.trainer = trainer
        self.prototype_memory = prototype_memory
        self.make_loader = make_loader
        self.state = state
        self.time_profiler = time_profiler
        self.get_model = get_model
        self.get_seen_labels = get_seen_labels

    def evaluate_learned_tasks(
        self,
        task_id,
        test_matrix=None,
        seen_matrix=None,
        ncm_matrix=None,
    ):
        return evaluate_learned_tasks(
            task_id=task_id,
            args=self.config,
            task_order=self.task_order,
            model=self.get_model(),
            trainer=self.trainer,
            prototype_memory=self.prototype_memory,
            make_loader=self.make_loader,
            test_matrix=test_matrix,
            seen_matrix=seen_matrix,
            ncm_matrix=ncm_matrix,
            seen_labels=sorted(self.get_seen_labels()),
            pred_future_count_matrix=(
                self.state.pred_future_count_matrix
                if test_matrix is self.state.test_matrix
                else None
            ),
            eval_sample_count_matrix=(
                self.state.eval_sample_count_matrix
                if test_matrix is self.state.test_matrix
                else None
            ),
        )

    def summarize_metrics(
        self,
        task_id,
        test_accs,
        ncm_accs,
        train_accs,
        test_matrix,
        seen_matrix,
        ncm_matrix,
    ):
        return summarize_task_metrics(
            task_id=task_id,
            task_order=self.task_order,
            test_accs=test_accs,
            ncm_accs=ncm_accs,
            train_accs=train_accs,
            test_matrix=test_matrix,
            seen_matrix=seen_matrix,
            ncm_matrix=ncm_matrix,
        )

    def _audit_classifier_ncm_agreement(
        self,
        task_id: int,
    ) -> dict[str, Any]:
        task_agreements = []
        agreed_samples = 0.0
        total_samples = 0
        for eval_step in range(task_id + 1):
            eval_dir = self.task_order[eval_step]
            test_path = os.path.join(
                self.config.data_root,
                f"task_{eval_dir}",
                "test.json",
            )
            if not os.path.exists(test_path):
                continue
            agreement_loader = self.make_loader(
                JSONLDataset(
                    test_path,
                    max_len=self.config.max_length,
                    encode_on_getitem=False,
                ),
                batch_size=self.config.eval_batch_size,
                shuffle=False,
            )
            audit = self.trainer.evaluate_classifier_ncm_agreement(
                agreement_loader,
                self.prototype_memory,
                seen_labels=self.get_seen_labels(),
            )
            samples = int(audit["samples"])
            agreement = float(audit["agreement"])
            task_agreements.append(
                {
                    "eval_step": eval_step,
                    "task_dir": eval_dir,
                    "agreement": agreement,
                    "samples": samples,
                }
            )
            agreed_samples += agreement * samples
            total_samples += samples
        return {
            "task_id": task_id,
            "task_dir": self.task_order[task_id],
            "agreement": (
                agreed_samples / total_samples if total_samples else 0.0
            ),
            "samples": total_samples,
            "task_agreements": task_agreements,
        }

    def evaluate_stage(self, task_id: int) -> TaskEvaluationResult:
        evaluation_head_context = (
            temporary_imprinted_classifier(
                self.get_model(),
                self.prototype_memory,
            )
            if self.config.alignment_method == "eval_only_imprinting"
            else nullcontext(None)
        )
        with evaluation_head_context as temporary_head_report:
            with self.time_profiler.track("eval_sec"):
                test_accs, ncm_accs, train_accs = (
                    self.evaluate_learned_tasks(
                        task_id,
                        test_matrix=self.state.test_matrix,
                        seen_matrix=self.state.seen_matrix,
                        ncm_matrix=self.state.ncm_matrix,
                    )
                )
                self.state.train_matrix[
                    task_id,
                    : len(train_accs),
                ] = train_accs
                if (
                    len(self.get_seen_labels()) == self.config.num_classes
                    and not np.allclose(
                        self.state.test_matrix[task_id, : task_id + 1],
                        self.state.seen_matrix[task_id, : task_id + 1],
                    )
                ):
                    raise RuntimeError(
                        "Global and seen-space accuracies must match once all "
                        "classes are observed"
                    )
                stage_future = int(
                    np.sum(
                        self.state.pred_future_count_matrix[
                            task_id, : task_id + 1
                        ]
                    )
                )
                stage_samples = int(
                    np.sum(
                        self.state.eval_sample_count_matrix[
                            task_id, : task_id + 1
                        ]
                    )
                )
                stage_global = float(
                    np.mean(self.state.test_matrix[task_id, : task_id + 1])
                )
                stage_seen = float(
                    np.mean(self.state.seen_matrix[task_id, : task_id + 1])
                )
                pred_future = (
                    stage_future / stage_samples if stage_samples else 0.0
                )
                print(
                    "[Seen-mask Audit] "
                    f"global={stage_global * 100:.2f}%, "
                    f"seen={stage_seen * 100:.2f}%, "
                    f"delta={(stage_seen - stage_global) * 100:+.2f}%, "
                    f"PredFuture={pred_future * 100:.2f}%"
                )
                if (
                    self.config.audit_imprinting_agreement
                    and self.config.alignment_method
                    in {"weight_imprinting", "eval_only_imprinting"}
                ):
                    with preserve_rng_state():
                        agreement_report = (
                            self._audit_classifier_ncm_agreement(task_id)
                        )
                    self.state.imprinting_agreement_by_stage.append(
                        agreement_report
                    )
                    stage_agreement = agreement_report["agreement"]
                    total_samples = agreement_report["samples"]
                    print(
                        "[Imprinting Audit] classifier/NCM prediction "
                        f"agreement={stage_agreement * 100:.4f}% "
                        f"({total_samples} samples)"
                    )
                    if (
                        stage_agreement
                        < self.config.min_imprinting_agreement
                    ):
                        raise RuntimeError(
                            f"{self.config.alignment_method} does not reproduce "
                            "single-prototype NCM: "
                            f"agreement={stage_agreement:.6f}, required>="
                            f"{self.config.min_imprinting_agreement:.6f}"
                        )

        if temporary_head_report is not None:
            if not temporary_head_report["persistent_classifier_restored"]:
                raise RuntimeError(
                    "Eval-only Imprinting failed to restore the persistent "
                    "classifier"
                )
            self.state.eval_only_imprinting_by_stage.append(
                {
                    "task_id": task_id,
                    "task_dir": self.task_order[task_id],
                    "imprinted_labels": temporary_head_report[
                        "imprinted_labels"
                    ],
                    "build_sec": temporary_head_report["build_sec"],
                    "temporary_parameters": temporary_head_report[
                        "temporary_parameters"
                    ],
                    "persistent_classifier_restored": temporary_head_report[
                        "persistent_classifier_restored"
                    ],
                    "persistent_state_unchanged": temporary_head_report[
                        "persistent_state_unchanged"
                    ],
                }
            )
        return TaskEvaluationResult(
            test_accs=test_accs,
            ncm_accs=ncm_accs,
            train_accs=train_accs,
        )
