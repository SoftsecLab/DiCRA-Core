"""Task-stream orchestration for continual RECAP experiments."""

from __future__ import annotations

import json
import os
from typing import Any, Callable

import numpy as np
import torch

from src.classifier_protocol import ClassifierProtocolCoordinator
from src.evaluation import bwt_from_matrix
from src.experiment_diagnostics import ExperimentDiagnosticsCoordinator
from src.experiment_state import ExperimentState
from src.profiling import TimeProfiler
from src.run_config import ExperimentConfig
from src.sleep_coordinator import SleepCoordinator
from src.task_evaluation import TaskEvaluationService
from src.task_stream import TaskStream


def save_incremental_checkpoint(model, task_ckpt_dir: str) -> None:
    """Save only the incremental encoder state, classifier, and class mapping."""

    os.makedirs(task_ckpt_dir, exist_ok=True)

    if hasattr(model.bert, "peft_config"):
        model.bert.save_pretrained(task_ckpt_dir, safe_serialization=False)
    else:
        trainable_state = {
            name: param.detach().cpu()
            for name, param in model.bert.named_parameters()
            if param.requires_grad
        }
        torch.save(
            trainable_state,
            os.path.join(task_ckpt_dir, "bert_trainable.pth"),
        )

    torch.save(
        model.classifier.state_dict(),
        os.path.join(task_ckpt_dir, "head.pth"),
    )
    with open(
        os.path.join(task_ckpt_dir, "class_ids.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(list(model.class_ids), handle, indent=2)

    for filename in ("model.safetensors", "pytorch_model.bin"):
        file_path = os.path.join(task_ckpt_dir, filename)
        if os.path.exists(file_path):
            os.remove(file_path)


class ContinualExperimentRunner:
    """Coordinate task loading, Wake/Sleep stages, evaluation, and diagnostics."""

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        task_order: list[int],
        model,
        tokenizer,
        trainer,
        prototype_memory,
        make_loader: Callable[..., Any],
        output_dir: str,
        device,
        task_boundary_transform: Callable[..., Any] | None = None,
    ):
        if len(task_order) != config.num_tasks:
            raise ValueError(
                "task_order length must match ExperimentConfig.num_tasks"
            )

        self.config = config
        self.task_order = list(task_order)
        self.model = model
        self.tokenizer = tokenizer
        self.trainer = trainer
        self.prototype_memory = prototype_memory
        self.make_loader = make_loader
        self.task_stream = TaskStream(config, make_loader)
        self.output_dir = output_dir
        self.device = device
        self.task_boundary_transform = task_boundary_transform

        self.state = ExperimentState.create(config.num_tasks)
        self.classifier_protocol = ClassifierProtocolCoordinator(
            config=config,
            state=self.state,
        )
        self.seen_labels = self.classifier_protocol.seen_labels
        self.time_profiler = TimeProfiler()
        self.task_evaluation = TaskEvaluationService(
            config=config,
            task_order=self.task_order,
            trainer=trainer,
            prototype_memory=prototype_memory,
            make_loader=make_loader,
            state=self.state,
            time_profiler=self.time_profiler,
            get_model=lambda: self.model,
            get_seen_labels=lambda: self.seen_labels,
        )
        self.experiment_diagnostics = ExperimentDiagnosticsCoordinator(
            config=config,
            output_dir=output_dir,
            task_order=self.task_order,
            get_model=lambda: self.model,
            evaluator=(
                trainer.evaluator
                if config.run_prototype_staleness_diagnostics
                else None
            ),
            prototype_memory=prototype_memory,
            make_loader=make_loader,
            device=device,
            state=self.state,
            time_profiler=self.time_profiler,
            evaluate_learned_tasks=(
                self.task_evaluation.evaluate_learned_tasks
            ),
            summarize_metrics=self.task_evaluation.summarize_metrics,
        )
        self.diagnostics = self.experiment_diagnostics.stage_diagnostics
        self.sleep_coordinator = SleepCoordinator(
            config=config,
            tokenizer=tokenizer,
            trainer=trainer,
            prototype_memory=prototype_memory,
            device=device,
            output_dir=output_dir,
            diagnostics=self.diagnostics,
            time_profiler=self.time_profiler,
        )

    def _print_task_summary(
        self,
        task_id: int,
        test_accs,
        ncm_accs,
        train_accs,
    ) -> None:
        avg_test = np.mean(test_accs) * 100
        avg_ncm = np.mean(ncm_accs) * 100
        avg_train = np.mean(train_accs) * 100
        train_display = (
            f"{avg_train:.2f}%"
            if self.config.evaluate_train_accuracy
            else "N/A"
        )
        current_bwt_seen = (
            bwt_from_matrix(self.state.seen_matrix, task_id) * 100
        )
        if self.config.alignment_method == "direct_ncm":
            current_bwt_ncm = (
                bwt_from_matrix(self.state.ncm_matrix, task_id) * 100
            )
            print(
                f"[Task {task_id}] primary_ncm={avg_ncm:.2f}%, "
                f"classifier={avg_test:.2f}%, train={train_display}, "
                f"BWT_ncm={current_bwt_ncm:.2f}%"
            )
        else:
            print(
                f"[Task {task_id}] test={avg_test:.2f}%, "
                f"ncm={avg_ncm:.2f}%, train={train_display}, "
                f"BWT_seen={current_bwt_seen:.2f}%"
            )

    def run_task(self, task_id: int, task_dir: int) -> None:
        print(
            "\n"
            + "=" * 20
            + f" Task {task_id} (task_dir={task_dir}) "
            + "=" * 20
        )
        self.time_profiler.start_task(task_id, task_dir)

        prepared_task = self.task_stream.prepare_task(task_id, task_dir)
        if prepared_task is None:
            self.time_profiler.finish_task()
            return

        current_labels = set(prepared_task.current_labels)
        self.classifier_protocol.activate_task(
            self.model,
            current_labels,
        )
        task_loaders = self.task_stream.build_loaders(prepared_task)
        train_loader = task_loaders.train_loader
        prototype_loader = task_loaders.prototype_loader

        print("[Wake] Training current task...")
        with self.time_profiler.track("wake_sec"):
            self.trainer.train_task(
                train_loader,
                task_id=task_id,
                prototype_memory=self.prototype_memory,
            )

        if self.task_boundary_transform is not None:
            with self.time_profiler.track("consolidation_sec"):
                self.task_boundary_transform(self.model, task_id)
            self.trainer.bind_model(self.model)

        if self.config.analyze_stages:
            self.prototype_memory.update_prototypes(
                self.model,
                prototype_loader,
                self.device,
            )
            self.diagnostics.record_stage("post_wake", task_id)

        self.model = self.sleep_coordinator.run(
            self.model,
            task_id,
            prototype_loader,
        )

        if self.config.save_checkpoints:
            task_ckpt_dir = os.path.join(
                self.output_dir,
                f"task_{task_id}",
            )
            save_incremental_checkpoint(self.model, task_ckpt_dir)

        evaluation = self.task_evaluation.evaluate_stage(task_id)
        self.experiment_diagnostics.record_staleness(
            task_id,
            self.seen_labels,
            current_labels,
        )
        self.time_profiler.finish_task()
        self._print_task_summary(
            task_id,
            evaluation.test_accs,
            evaluation.ncm_accs,
            evaluation.train_accs,
        )
        self.experiment_diagnostics.run_reference_probe(task_id)

    def run(self) -> ExperimentState:
        for task_id, task_dir in enumerate(self.task_order):
            self.run_task(task_id, task_dir)
        return self.state
