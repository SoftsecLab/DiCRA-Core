"""Dependency-free metric primitives for continual-learning result matrices.

All metric values are represented as fractions in the inclusive range used by
the caller (normally ``[0, 1]``), never as percentages.  This module is not
wired into the training or reporting paths yet; it provides a single,
behavior-locked implementation for a later migration.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence


MetricSummary = dict[str, float]


def _validate_num_tasks(num_tasks: int) -> None:
    if isinstance(num_tasks, bool) or not isinstance(num_tasks, int):
        raise TypeError("num_tasks must be an integer")
    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive")


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _finite_float(value: object, *, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{location} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{location} must be finite")
    return normalized


def validate_matrix(
    matrix: Sequence[Sequence[float]],
    num_tasks: int,
) -> tuple[tuple[float, ...], ...]:
    """Validate and normalize the leading ``num_tasks`` square matrix.

    Exactly ``num_tasks`` rows are required.  Rows may contain trailing
    columns for compatibility with existing result files, but only their first
    ``num_tasks`` values are validated and returned.
    """

    _validate_num_tasks(num_tasks)
    if not _is_sequence(matrix) or len(matrix) != num_tasks:
        raise ValueError(f"matrix must have {num_tasks} rows")

    normalized_rows = []
    for row_index, row in enumerate(matrix):
        if not _is_sequence(row) or len(row) < num_tasks:
            raise ValueError(
                f"every matrix row must have at least {num_tasks} columns"
            )
        normalized_rows.append(
            tuple(
                _finite_float(value, location=f"matrix[{row_index}][{column_index}]")
                for column_index, value in enumerate(row[:num_tasks])
            )
        )
    return tuple(normalized_rows)


def _final_average(matrix: tuple[tuple[float, ...], ...], num_tasks: int) -> float:
    return float(statistics.mean(matrix[num_tasks - 1][:num_tasks]))


def _average_incremental_accuracy(
    matrix: tuple[tuple[float, ...], ...],
    num_tasks: int,
) -> float:
    return float(
        statistics.mean(
            statistics.mean(matrix[stage][: stage + 1])
            for stage in range(num_tasks)
        )
    )


def _backward_transfer(
    matrix: tuple[tuple[float, ...], ...],
    num_tasks: int,
) -> float:
    if num_tasks == 1:
        return 0.0
    final_stage = num_tasks - 1
    return float(
        statistics.mean(
            matrix[final_stage][task] - matrix[task][task]
            for task in range(final_stage)
        )
    )


def final_average(matrix: Sequence[Sequence[float]], num_tasks: int) -> float:
    """Return the equally weighted task accuracy at the final stage."""

    validated = validate_matrix(matrix, num_tasks)
    return _final_average(validated, num_tasks)


def average_incremental_accuracy(
    matrix: Sequence[Sequence[float]],
    num_tasks: int,
) -> float:
    """Average each stage over seen tasks, then average over all stages."""

    validated = validate_matrix(matrix, num_tasks)
    return _average_incremental_accuracy(validated, num_tasks)


def backward_transfer(matrix: Sequence[Sequence[float]], num_tasks: int) -> float:
    """Return final-stage forgetting relative to each old task's diagonal."""

    validated = validate_matrix(matrix, num_tasks)
    return _backward_transfer(validated, num_tasks)


def summarize_matrix(
    matrix: Sequence[Sequence[float]],
    num_tasks: int,
) -> MetricSummary:
    """Compute the canonical Final Avg, Avg Inc, and BWT metrics once."""

    validated = validate_matrix(matrix, num_tasks)
    return {
        "final_avg": _final_average(validated, num_tasks),
        "avg_inc": _average_incremental_accuracy(validated, num_tasks),
        "bwt": _backward_transfer(validated, num_tasks),
    }


def aggregate_seeds(values: Sequence[float]) -> dict[str, object]:
    """Return seed values with their mean and sample standard deviation."""

    if not _is_sequence(values):
        raise TypeError("values must be a sequence")
    if not values:
        raise ValueError("at least one seed value is required")
    normalized = [
        _finite_float(value, location=f"values[{index}]")
        for index, value in enumerate(values)
    ]
    return {
        "mean": float(statistics.mean(normalized)),
        "std": float(statistics.stdev(normalized)) if len(normalized) > 1 else 0.0,
        "values": normalized,
    }
