"""Diagnostic service coordination for continual experiments."""

from __future__ import annotations

from typing import Any, Callable

from src.diagnostics import StageDiagnostics
from src.experiment_state import ExperimentState
from src.prototype_staleness import PrototypeStalenessDiagnostics
from src.reference_diagnostics import (
    run_frozen_linear_probe,
    summarize_probe_diagnostics,
    write_reference_diagnostics,
)
from src.reproducibility import preserve_rng_state
from src.run_config import ExperimentConfig


class ExperimentDiagnosticsCoordinator:
    """Own optional diagnostics while preserving their existing artifacts."""

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        output_dir: str,
        task_order: list[int],
        get_model: Callable[[], Any],
        evaluator,
        prototype_memory,
        make_loader: Callable[..., Any],
        device,
        state: ExperimentState,
        time_profiler,
        evaluate_learned_tasks: Callable[..., Any],
        summarize_metrics: Callable[..., Any],
    ):
        self.config = config
        self.output_dir = output_dir
        self.task_order = list(task_order)
        self.get_model = get_model
        self.prototype_memory = prototype_memory
        self.make_loader = make_loader
        self.device = device
        self.state = state
        self.time_profiler = time_profiler

        self.stage_diagnostics = StageDiagnostics(
            args=config,
            output_dir=output_dir,
            task_order=self.task_order,
            get_model=get_model,
            prototype_memory=prototype_memory,
            evaluate_learned_tasks=evaluate_learned_tasks,
            summarize_metrics=summarize_metrics,
            make_loader=make_loader,
        )
        self.prototype_staleness_diagnostics = None
        if config.run_prototype_staleness_diagnostics:
            self.prototype_staleness_diagnostics = (
                PrototypeStalenessDiagnostics(
                    args=config,
                    output_dir=output_dir,
                    task_order=self.task_order,
                    model=get_model(),
                    evaluator=evaluator,
                    stored_memory=prototype_memory,
                    make_loader=make_loader,
                    device=device,
                )
            )

    @preserve_rng_state()
    def record_staleness(
        self,
        task_id: int,
        seen_labels: set[int],
        current_labels: set[int],
    ):
        if self.prototype_staleness_diagnostics is None:
            return None
        with self.time_profiler.track("prototype_staleness_sec"):
            return self.prototype_staleness_diagnostics.record_stage(
                task_id=task_id,
                seen_labels=seen_labels,
                current_labels=current_labels,
                expected_stored_row=self.state.ncm_matrix[
                    task_id,
                    : task_id + 1,
                ],
            )

    @preserve_rng_state()
    def run_reference_probe(self, task_id: int):
        if not self.config.run_reference_diagnostics:
            return None
        print("[Reference Diagnostics] Frozen linear probe...")
        probe_accs = run_frozen_linear_probe(
            task_id=task_id,
            args=self.config,
            task_order=self.task_order,
            model=self.get_model(),
            make_loader=self.make_loader,
            device=self.device,
        )
        self.state.probe_matrix[task_id, : len(probe_accs)] = probe_accs
        probe_payload = summarize_probe_diagnostics(
            probe_matrix=self.state.probe_matrix,
            ncm_matrix=self.state.ncm_matrix,
            num_tasks=task_id + 1,
        )
        probe_payload["task_order"] = self.task_order
        probe_payload["completed_tasks"] = task_id + 1
        write_reference_diagnostics(self.output_dir, probe_payload)
        print(
            "[Reference Diagnostics] "
            f"probe_avg={probe_payload['final_probe_avg'] * 100:.2f}%, "
            f"F_probe={probe_payload['f_probe'] * 100:.2f}%, "
            f"F_feat={probe_payload['f_feat'] * 100:.2f}%"
        )
        return probe_payload
