#!/usr/bin/env python3
"""Prepare CLINC150 as a class-incremental JSONL dataset."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.common_dataset import save_json, save_jsonl, split_labels


def load_clinc_raw(raw_path: Path):
    if not raw_path.exists():
        raise FileNotFoundError(
            f"CLINC150 raw file not found: {raw_path}. "
            "Download data_full.json first and pass it with --raw_path."
        )
    with raw_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def group_split(rows, label2id):
    grouped = defaultdict(list)
    for text, label_name in rows:
        if label_name not in label2id:
            continue
        label_id = label2id[label_name]
        grouped[label_id].append({"text": text, "label": label_id})
    return grouped


def prepare_clinc150(raw_path, output_root, num_tasks, seed, overwrite=False):
    raw_path = Path(raw_path)
    output_root = Path(output_root)
    raw_data = load_clinc_raw(raw_path)

    for split in ("train", "test"):
        if split not in raw_data:
            raise ValueError(f"CLINC150 raw file missing required split: {split}")

    labels = sorted({label for _text, label in raw_data["train"] if label != "oos"})
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    if num_tasks > len(labels):
        raise ValueError(f"--num_tasks={num_tasks} exceeds num_classes={len(labels)}")

    rng = random.Random(seed)
    class_order = list(range(len(labels)))
    rng.shuffle(class_order)
    task_groups = split_labels(class_order, num_tasks)

    rows_by_split_label = {
        "train": group_split(raw_data["train"], label2id),
        "test": group_split(raw_data["test"], label2id),
    }
    if "val" in raw_data:
        rows_by_split_label["dev"] = group_split(raw_data["val"], label2id)

    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    totals = {split: 0 for split in rows_by_split_label}
    for task_id, task_labels in enumerate(task_groups):
        task_dir = output_root / f"task_{task_id}"
        for split, grouped_rows in rows_by_split_label.items():
            rows = []
            for label_id in task_labels:
                rows.extend(dict(row) for row in grouped_rows.get(label_id, []))
            rng.shuffle(rows)
            save_jsonl(rows, task_dir / f"{split}.json")
            totals[split] += len(rows)

        print(
            f"Task {task_id}: classes={len(task_labels)} | "
            f"train={len((task_dir / 'train.json').read_text(encoding='utf-8').splitlines())} | "
            f"test={len((task_dir / 'test.json').read_text(encoding='utf-8').splitlines())}"
        )

    meta = {
        "dataset": "clinc150",
        "source": str(raw_path),
        "seed": int(seed),
        "num_tasks": int(num_tasks),
        "num_classes": len(labels),
        "classes_per_task": [len(group) for group in task_groups],
        "task_class_order": task_groups,
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "totals": totals,
        "format": "jsonl:text,label",
    }
    save_json(meta, output_root / "meta.json")

    print(f"Saved CLINC150 to {output_root}")
    print(f"Classes={len(labels)}, Tasks={num_tasks}, seed={seed}")
    print(f"Examples={totals}")


def main():
    parser = argparse.ArgumentParser(description="Prepare CLINC150 CIL task stream.")
    parser.add_argument(
        "--raw_path",
        type=Path,
        required=True,
        help="Path to CLINC150 data_full.json.",
    )
    parser.add_argument("--output_root", type=Path, default=Path("data/clinc150"))
    parser.add_argument("--num_tasks", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    prepare_clinc150(
        raw_path=args.raw_path,
        output_root=args.output_root,
        num_tasks=args.num_tasks,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
