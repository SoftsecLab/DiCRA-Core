from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from src.dataset import JSONLDataset
from src.run_config import ExperimentConfig


@dataclass(frozen=True)
class PreparedTask:


    task_id: int
    task_dir: int
    train_path: str
    train_dataset: Any
    current_labels: frozenset[int]


@dataclass(frozen=True)
class TaskLoaders:


    train_loader: Any
    prototype_loader: Any


class TaskStream:


    def __init__(
        self,
        config: ExperimentConfig,
        make_loader: Callable[..., Any],
        *,
        dataset_factory: Callable[..., Any] = JSONLDataset,
    ):
        self.config = config
        self.make_loader = make_loader
        self.dataset_factory = dataset_factory

    def prepare_task(
        self,
        task_id: int,
        task_dir: int,
    ) -> PreparedTask | None:
        train_path = os.path.join(
            self.config.data_root,
            f"task_{task_dir}",
            "train.json",
        )
        if not os.path.exists(train_path):
            message = f"Task data not found: {train_path}"
            if not self.config.allow_missing_tasks:
                raise FileNotFoundError(
                    message
                    + ". Use --allow_missing_tasks only for intentional "
                    "partial/debug runs."
                )
            print(
                f"{message}, skip because --allow_missing_tasks is set."
            )
            return None

        train_dataset = self.dataset_factory(
            train_path,
            max_len=self.config.max_length,
            encode_on_getitem=False,
        )
        current_labels = frozenset(
            int(item["label"]) for item in train_dataset.data
        )
        return PreparedTask(
            task_id=int(task_id),
            task_dir=int(task_dir),
            train_path=train_path,
            train_dataset=train_dataset,
            current_labels=current_labels,
        )

    def build_loaders(self, task: PreparedTask) -> TaskLoaders:
        return TaskLoaders(
            train_loader=self.make_loader(
                task.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
            ),
            prototype_loader=self.make_loader(
                task.train_dataset,
                batch_size=min(32, self.config.eval_batch_size),
                shuffle=False,
            ),
        )
