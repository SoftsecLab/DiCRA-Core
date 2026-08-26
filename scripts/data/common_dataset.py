#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def task_sort_key(path: Path) -> int:

    return int(path.name.rsplit("_", 1)[-1])


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: Iterable[Mapping[str, Any]], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def load_json(path: Path | str, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(payload: Mapping[str, Any], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def group_rows_by_label(rows: Iterable[Mapping[str, Any]], label_key: str = "label"):
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        normalized = dict(row)
        label = int(normalized[label_key])
        normalized[label_key] = label
        grouped[label].append(normalized)
    return grouped


def split_labels(class_order: Sequence[int], num_tasks: int) -> list[list[int]]:
    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive")
    if num_tasks > len(class_order):
        raise ValueError(f"num_tasks={num_tasks} exceeds num_classes={len(class_order)}")

    base = len(class_order) // num_tasks
    remainder = len(class_order) % num_tasks
    groups = []
    cursor = 0
    for task_id in range(num_tasks):
        size = base + (1 if task_id < remainder else 0)
        groups.append(list(class_order[cursor: cursor + size]))
        cursor += size
    return groups


def shuffled_labels(labels: Iterable[int], seed: int) -> list[int]:
    ordered = sorted(int(label) for label in labels)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return ordered


def write_task_splits(
    output_root: Path | str,
    task_groups: Sequence[Sequence[int]],
    rows_by_split_label: Mapping[str, Mapping[int, Sequence[Mapping[str, Any]]]],
    *,
    seed: int,
    include_empty_dev: bool = False,
) -> dict[str, int]:
    output_root = Path(output_root)
    rng = random.Random(seed)
    totals = {split: 0 for split in rows_by_split_label}

    for task_id, labels in enumerate(task_groups):
        for split, rows_by_label in rows_by_split_label.items():
            rows = []
            for label in labels:
                rows.extend(dict(row) for row in rows_by_label.get(int(label), []))
            rng.shuffle(rows)
            if rows or split != "dev" or include_empty_dev:
                save_jsonl(rows, output_root / f"task_{task_id}" / f"{split}.json")
            totals[split] += len(rows)

    return totals


def build_meta(
    *,
    dataset: str,
    num_tasks: int,
    num_classes: int,
    task_groups: Sequence[Sequence[int]],
    id2label: Mapping[int | str, str] | None = None,
    seed: int | None = None,
    source: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_id2label = {}
    if id2label:
        normalized_id2label = {str(int(k)): str(v) for k, v in id2label.items()}

    meta: dict[str, Any] = {
        "dataset": dataset,
        "num_tasks": int(num_tasks),
        "num_classes": int(num_classes),
        "classes_per_task": [len(group) for group in task_groups],
        "task_class_order": [list(map(int, group)) for group in task_groups],
        "format": "jsonl:text,label",
    }
    if normalized_id2label:
        meta["id2label"] = normalized_id2label
        meta["label2id"] = {name: int(label) for label, name in normalized_id2label.items()}
    if seed is not None:
        meta["seed"] = int(seed)
    if source is not None:
        meta["source"] = source
    if extra:
        meta.update(dict(extra))
    return meta
