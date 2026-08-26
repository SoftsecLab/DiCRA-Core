#!/usr/bin/env python3
import argparse
import pickle
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.common_dataset import save_json, save_jsonl


NUM_CLASSES = 80
NUM_TASKS = 8
CLASSES_PER_TASK = 10


def candidate_pkl_paths(repo_root):
    return [
        repo_root / "FewRel-2021.pkl",
        repo_root / "dataset" / "fewrel" / "FewRel-2021.pkl",
        repo_root / "codebase" / "dataset" / "fewrel" / "FewRel-2021.pkl",
        repo_root / "tmp_vag_repo" / "VAG-main" / "data" / "FewRel-2021.pkl",
    ]


def resolve_source_pkl(repo_root, source_pkl):
    if source_pkl:
        path = Path(source_pkl)
        path = path if path.is_absolute() else repo_root / path
        if not path.exists():
            raise FileNotFoundError(f"FewRel source pkl not found: {path}")
        return path

    for path in candidate_pkl_paths(repo_root):
        if path.exists():
            return path
    return None


def validate_official_pkl(all_data):
    if not isinstance(all_data, tuple) or len(all_data) < 3:
        raise ValueError("FewRel-2021.pkl should be a tuple/list with train/dev/test splits.")
    for split_name, split, expected_per_class in [
        ("train", all_data[0], 420),
        ("dev", all_data[1], 140),
        ("test", all_data[2], 140),
    ]:
        if len(split) != NUM_CLASSES:
            raise ValueError(f"{split_name} split should contain {NUM_CLASSES} classes, got {len(split)}.")
        bad = [idx for idx, rows in enumerate(split) if len(rows) != expected_per_class]
        if bad:
            raise ValueError(
                f"{split_name} split expected {expected_per_class} samples per class; "
                f"bad class ids={bad[:10]}"
            )


def row_from_instance(instance, label_id):
    return {
        "text": instance["text"],
        "label": label_id,
    }


def format_relation_text(tokens, head_indices, tail_indices):
    h_positions = head_indices[0]
    t_positions = tail_indices[0]
    h_start, h_end = min(h_positions), max(h_positions) + 1
    t_start, t_end = min(t_positions), max(t_positions) + 1

    markers = sorted(
        [
            (h_start, h_end, "[E1]", "[/E1]"),
            (t_start, t_end, "[E2]", "[/E2]"),
        ],
        key=lambda item: item[0],
        reverse=True,
    )
    marked_tokens = list(tokens)
    for start, end, open_tag, close_tag in markers:
        marked_tokens.insert(end, close_tag)
        marked_tokens.insert(start, open_tag)
    return " ".join(marked_tokens)


def load_hf_class_data():
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "FewRel-2021.pkl was not found and HuggingFace datasets is unavailable. "
            "Install it with `pip install datasets`, or pass --source_pkl /path/to/FewRel-2021.pkl."
        ) from exc

    print("[FewRel] FewRel-2021.pkl not found; falling back to HuggingFace thunlp/few_rel.")
    print("[FewRel] This keeps the ACL2024 counts/protocol, but the official pkl is the most faithful source.")

    ds_train = load_dataset("thunlp/few_rel", split="train_wiki", trust_remote_code=True)
    ds_val = load_dataset("thunlp/few_rel", split="val_wiki", trust_remote_code=True)

    class_data = {}
    malformed = 0
    for item in list(ds_train) + list(ds_val):
        relation = item["relation"]
        try:
            text = format_relation_text(item["tokens"], item["head"]["indices"], item["tail"]["indices"])
        except (IndexError, ValueError, TypeError):
            malformed += 1
            text = " ".join(item["tokens"])
        class_data.setdefault(relation, []).append(text)

    if malformed:
        print(f"[FewRel] Warning: {malformed} samples had malformed entity indices; used plain text fallback.")
    if len(class_data) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} FewRel classes from HF, got {len(class_data)}.")
    bad = [label for label, rows in class_data.items() if len(rows) < 700]
    if bad:
        raise ValueError(f"Expected at least 700 samples per FewRel class; bad labels={bad[:10]}")
    return class_data


def prepare_from_huggingface(output_root, include_dev=True, shuffle_within_task=True, seed=42):
    class_data = load_hf_class_data()
    output_root = Path(output_root)
    rng = random.Random(seed)

    class_order = sorted(class_data.keys())
    label2id = {label: idx for idx, label in enumerate(class_order)}
    id2label = {idx: label for label, idx in label2id.items()}

    total_train = 0
    total_dev = 0
    total_test = 0
    for task_id in range(NUM_TASKS):
        task_labels = class_order[task_id * CLASSES_PER_TASK : (task_id + 1) * CLASSES_PER_TASK]
        train_rows = []
        dev_rows = []
        test_rows = []

        for label in task_labels:
            texts = list(class_data[label])
            rng.shuffle(texts)
            train_texts = texts[:420]
            dev_texts = texts[420:560]
            test_texts = texts[560:700]
            label_id = label2id[label]
            train_rows.extend({"text": text, "label": label_id} for text in train_texts)
            dev_rows.extend({"text": text, "label": label_id} for text in dev_texts)
            test_rows.extend({"text": text, "label": label_id} for text in test_texts)

        if shuffle_within_task:
            rng.shuffle(train_rows)
            rng.shuffle(dev_rows)
            rng.shuffle(test_rows)

        task_dir = output_root / f"task_{task_id}"
        save_jsonl(train_rows, task_dir / "train.json")
        save_jsonl(test_rows, task_dir / "test.json")
        if include_dev:
            save_jsonl(dev_rows, task_dir / "dev.json")

        total_train += len(train_rows)
        total_dev += len(dev_rows)
        total_test += len(test_rows)
        print(
            f"Task {task_id}: classes={len(task_labels)} | "
            f"train={len(train_rows)} | dev={len(dev_rows)} | test={len(test_rows)}"
        )

    meta = {
        "dataset": "fewrel",
        "protocol": "acl2024_learn_or_recall_task8",
        "source": "huggingface:thunlp/few_rel",
        "source_note": "Fallback split with ACL2024 counts. Use FewRel-2021.pkl for exact official split/order.",
        "num_classes": NUM_CLASSES,
        "num_tasks": NUM_TASKS,
        "classes_per_task": CLASSES_PER_TASK,
        "train_per_class": 420,
        "dev_per_class": 140,
        "test_per_class": 140,
        "total_train": total_train,
        "total_dev": total_dev,
        "total_test": total_test,
        "class_order": [label2id[label] for label in class_order],
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
    }
    save_json(meta, output_root / "meta.json")

    print("\nFewRel ACL2024 task8 split prepared from HuggingFace fallback.")
    print(f"Output: {output_root}")
    print(f"Train/Test: {total_train}/{total_test}")
    if include_dev:
        print(f"Dev: {total_dev}")


def prepare_from_official_pkl(source_pkl, output_root, include_dev=True, shuffle_within_task=True, seed=42):
    with open(source_pkl, "rb") as f:
        all_data = pickle.load(f)
    validate_official_pkl(all_data)

    train_split, dev_split, test_split = all_data[0], all_data[1], all_data[2]
    output_root = Path(output_root)
    rng = random.Random(seed)

    label2id = {}
    id2label = {}
    label_explanations = {}
    for class_id in range(NUM_CLASSES):
        semantic_label = train_split[class_id][0]["semantic_label"]
        label2id[semantic_label] = class_id
        id2label[class_id] = semantic_label
        label_explanations[semantic_label] = train_split[class_id][0].get("label_explanation", "")

    total_train = 0
    total_dev = 0
    total_test = 0

    for task_id in range(NUM_TASKS):
        class_ids = list(range(task_id * CLASSES_PER_TASK, (task_id + 1) * CLASSES_PER_TASK))

        train_rows = []
        dev_rows = []
        test_rows = []
        for class_id in class_ids:
            train_rows.extend(row_from_instance(instance, class_id) for instance in train_split[class_id])
            dev_rows.extend(row_from_instance(instance, class_id) for instance in dev_split[class_id])
            test_rows.extend(row_from_instance(instance, class_id) for instance in test_split[class_id])

        if shuffle_within_task:
            rng.shuffle(train_rows)
            rng.shuffle(dev_rows)
            rng.shuffle(test_rows)

        task_dir = output_root / f"task_{task_id}"
        save_jsonl(train_rows, task_dir / "train.json")
        save_jsonl(test_rows, task_dir / "test.json")
        if include_dev:
            save_jsonl(dev_rows, task_dir / "dev.json")

        total_train += len(train_rows)
        total_dev += len(dev_rows)
        total_test += len(test_rows)
        print(
            f"Task {task_id}: classes={len(class_ids)} | "
            f"train={len(train_rows)} | dev={len(dev_rows)} | test={len(test_rows)}"
        )

    meta = {
        "dataset": "fewrel",
        "protocol": "acl2024_learn_or_recall_task8",
        "source": str(source_pkl),
        "num_classes": NUM_CLASSES,
        "num_tasks": NUM_TASKS,
        "classes_per_task": CLASSES_PER_TASK,
        "train_per_class": 420,
        "dev_per_class": 140,
        "test_per_class": 140,
        "total_train": total_train,
        "total_dev": total_dev,
        "total_test": total_test,
        "class_order": list(range(NUM_CLASSES)),
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "label_explanations": label_explanations,
    }
    save_json(meta, output_root / "meta.json")

    print("\nFewRel ACL2024 task8 split prepared.")
    print(f"Output: {output_root}")
    print(f"Train/Test: {total_train}/{total_test}")
    if include_dev:
        print(f"Dev: {total_dev}")


def main():
    parser = argparse.ArgumentParser(description="Prepare FewRel with ACL 2024 task8 protocol.")
    parser.add_argument("--output_root", type=str, default="data/fewrel_acl2024")
    parser.add_argument("--source_pkl", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_dev", action="store_true")
    parser.add_argument("--no_shuffle_within_task", action="store_true")
    args = parser.parse_args()

    repo_root = Path.cwd()
    source_pkl = resolve_source_pkl(repo_root, args.source_pkl)
    if source_pkl is not None:
        prepare_from_official_pkl(
            source_pkl=source_pkl,
            output_root=args.output_root,
            include_dev=not args.no_dev,
            shuffle_within_task=not args.no_shuffle_within_task,
            seed=args.seed,
        )
    else:
        prepare_from_huggingface(
            output_root=args.output_root,
            include_dev=not args.no_dev,
            shuffle_within_task=not args.no_shuffle_within_task,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
