#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.common_dataset import save_json, save_jsonl, split_labels


def group_rows(rows):
    grouped = defaultdict(list)
    for item in rows:
        label = int(item["label"])
        grouped[label].append({"text": item["text"], "label": label})
    return grouped


def prepare_banking77(output_root, num_tasks, seed, overwrite=False):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Preparing Banking77 requires the `datasets` package. "
            "Install project requirements first."
        ) from exc

    output_root = Path(output_root)
    dataset = load_dataset("mteb/banking77", trust_remote_code=True)
    train_data = dataset["train"]
    test_data = dataset["test"]

    labels = sorted({int(label) for label in train_data["label"]})
    if num_tasks > len(labels):
        raise ValueError(f"--num_tasks={num_tasks} exceeds num_classes={len(labels)}")

    rng = random.Random(seed)
    class_order = list(labels)
    rng.shuffle(class_order)
    task_groups = split_labels(class_order, num_tasks)

    rows_by_split_label = {
        "train": group_rows(train_data),
        "test": group_rows(test_data),
    }

    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    totals = {split: 0 for split in rows_by_split_label}
    for task_id, task_labels in enumerate(task_groups):
        task_dir = output_root / f"task_{task_id}"
        split_counts = {}
        for split, grouped_rows in rows_by_split_label.items():
            rows = []
            for label_id in task_labels:
                rows.extend(dict(row) for row in grouped_rows.get(int(label_id), []))
            rng.shuffle(rows)
            save_jsonl(rows, task_dir / f"{split}.json")
            split_counts[split] = len(rows)
            totals[split] += len(rows)
        print(
            f"Task {task_id}: classes={len(task_labels)} | "
            f"train={split_counts.get('train', 0)} | test={split_counts.get('test', 0)}"
        )

    id2label = {}
    try:
        names = train_data.features["label"].names
        id2label = {idx: name for idx, name in enumerate(names)}
    except Exception:
        id2label = {idx: str(idx) for idx in labels}

    meta = {
        "dataset": "banking77",
        "source": "huggingface:mteb/banking77",
        "seed": int(seed),
        "num_tasks": int(num_tasks),
        "num_classes": len(labels),
        "classes_per_task": [len(group) for group in task_groups],
        "task_class_order": task_groups,
        "id2label": {str(k): v for k, v in id2label.items()},
        "label2id": {v: int(k) for k, v in id2label.items()},
        "totals": totals,
        "format": "jsonl:text,label",
    }
    save_json(meta, output_root / "meta.json")

    print(f"Saved Banking77 to {output_root}")
    print(f"Classes={len(labels)}, Tasks={num_tasks}, seed={seed}")
    print(f"Examples={totals}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Banking77 CIL task stream.")
    parser.add_argument("--output_root", type=Path, default=Path("data/banking77"))
    parser.add_argument("--num_tasks", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    prepare_banking77(
        output_root=args.output_root,
        num_tasks=args.num_tasks,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
