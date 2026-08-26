from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ExperimentState:


    test_matrix: np.ndarray
    seen_matrix: np.ndarray
    ncm_matrix: np.ndarray
    pred_future_count_matrix: np.ndarray
    eval_sample_count_matrix: np.ndarray
    train_matrix: np.ndarray
    probe_matrix: np.ndarray
    seen_labels_by_stage: list[list[int]]
    classifier_class_ids_by_stage: list[list[int]]
    classifier_output_dims_by_stage: list[int]
    imprinting_agreement_by_stage: list[dict[str, Any]]
    eval_only_imprinting_by_stage: list[dict[str, Any]]

    @classmethod
    def create(cls, num_tasks: int) -> "ExperimentState":
        shape = (num_tasks, num_tasks)
        return cls(
            test_matrix=np.zeros(shape),
            seen_matrix=np.zeros(shape),
            ncm_matrix=np.zeros(shape),
            pred_future_count_matrix=np.zeros(shape, dtype=np.int64),
            eval_sample_count_matrix=np.zeros(shape, dtype=np.int64),
            train_matrix=np.zeros(shape),
            probe_matrix=np.zeros(shape),
            seen_labels_by_stage=[],
            classifier_class_ids_by_stage=[],
            classifier_output_dims_by_stage=[],
            imprinting_agreement_by_stage=[],
            eval_only_imprinting_by_stage=[],
        )
