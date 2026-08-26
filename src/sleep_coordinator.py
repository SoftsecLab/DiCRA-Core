"""Task-level Sleep orchestration around the core Sleep algorithm."""

from __future__ import annotations

from dataclasses import replace

from src.alignment import align_classifier
from src.run_config import ExperimentConfig
from src.sleep import merge_and_reinit_lora, sleep_phase


class SleepCoordinator:
    """Run the configured Sleep path and return the active model."""

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        tokenizer,
        trainer,
        prototype_memory,
        device,
        output_dir: str,
        diagnostics,
        time_profiler,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.trainer = trainer
        self.prototype_memory = prototype_memory
        self.device = device
        self.output_dir = output_dir
        self.diagnostics = diagnostics
        self.time_profiler = time_profiler

    def run(self, model, task_id: int, prototype_loader):
        if self.config.matched_alignment_only:
            return self._run_matched_alignment_only(
                model,
                task_id,
                prototype_loader,
            )

        if not self.config.use_sleep:
            if self.config.update_prototypes_without_sleep:
                print("[Memory] update prototypes without Sleep for diagnostics")
                self.prototype_memory.update_prototypes(
                    model,
                    prototype_loader,
                    self.device,
                )
            return model

        with self.time_profiler.track("sleep_sec"):
            if self.config.analyze_stages or self.config.log_rem_diagnostics:
                return self._run_diagnostic_sleep(
                    model,
                    task_id,
                    prototype_loader,
                )
            return self.trainer.sleep(
                self.tokenizer,
                self.prototype_memory,
                prototype_loader=prototype_loader,
                task_id=task_id,
                output_dir=self.output_dir,
                alignment_callback=self.time_profiler.record_alignment,
            )

    def _run_diagnostic_sleep(
        self,
        model,
        task_id: int,
        prototype_loader,
    ):
        print("[Sleep] Diagnostic mode: explicit merge -> NREM -> REM")
        if self.config.no_consolidation:
            print("[Consolidation] skipped by --no_consolidation")
        else:
            merge_and_reinit_lora(
                model,
                self.config,
                task_id=task_id,
            )
        if self.diagnostics.has_stage("post_merge"):
            self.prototype_memory.update_prototypes(
                model,
                prototype_loader,
                self.device,
            )
            self.diagnostics.record_stage("post_merge", task_id)

        alignment_config = replace(
            self.config.alignment,
            task_id=task_id,
        )
        updated_model = sleep_phase(
            model,
            self.tokenizer,
            self.device,
            alignment_config,
            self.prototype_memory,
            prototype_loader=prototype_loader,
            before_rem_callback=(
                (
                    lambda: self.diagnostics.record_stage(
                        "before_rem",
                        task_id,
                    )
                )
                if self.diagnostics.has_stage("before_rem")
                else None
            ),
            alignment_callback=self.time_profiler.record_alignment,
            output_dir=self.output_dir,
        )
        self.trainer.bind_model(updated_model)
        if self.diagnostics.has_stage("post_rem"):
            self.diagnostics.record_stage("post_rem", task_id)
        return updated_model

    def _run_matched_alignment_only(
        self,
        model,
        task_id: int,
        prototype_loader,
    ):
        print(
            "[Matched Alignment] update stored prototypes -> "
            "classifier-only Gaussian repair"
        )
        with self.time_profiler.track("sleep_sec"):
            self.prototype_memory.update_prototypes(
                model,
                prototype_loader,
                self.device,
            )
            alignment_config = replace(
                self.config.alignment,
                task_id=task_id,
            )
            updated_model = align_classifier(
                model,
                self.prototype_memory,
                self.device,
                alignment_config,
                output_dir=self.output_dir,
                callback=self.time_profiler.record_alignment,
            )
            self.trainer.bind_model(updated_model)
        return updated_model
