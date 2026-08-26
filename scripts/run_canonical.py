#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments" / "documented_dev_selection_v1.json"
OUTPUT_ROOT = ROOT / "outputs" / "canonical"
SEEDS = (0, 1, 42)
DATASETS = {
    "clinc150": (15, 150),
    "banking77": (7, 77),
    "fewrel_acl2024": (8, 80),
}
METHODS = (
    "sequential_lora",
    "seq_lora_matched_alignment",
    "olora",
    "clora",
    "slora_pre",
    "recap_wo_refinements",
    "recap",
)
LABELS = {
    "sequential_lora": "Sequential LoRA",
    "seq_lora_matched_alignment": "SeqLoRA + Align",
    "olora": "O-LoRA",
    "clora": "CLoRA",
    "slora_pre": "SLoRA-Pre",
    "recap_wo_refinements": "DiCRA w/o Refinements",
    "recap": "DiCRA",
}


def canonical_sha256(value) -> str:
    import hashlib

    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_freeze(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    stored = value.pop("freeze_content_sha256", None)
    actual = canonical_sha256(value)
    value["freeze_content_sha256"] = stored
    if stored != actual:
        raise ValueError(f"configuration freeze hash mismatch: {path}")
    return value


def common_args(exp_name, data_root, dataset, seed, lr, llrd):
    tasks, classes = DATASETS[dataset]
    return [
        "--exp_name", exp_name,
        "--data_root", str(data_root),
        "--model_id", "bert-base-uncased",
        "--seed", str(seed),
        "--num_tasks", str(tasks),
        "--num_classes", str(classes),
        "--classifier_protocol", "fixed_global",
        "--lr", str(lr),
        "--epochs", "10",
        "--batch_size", "128",
        "--eval_batch_size", "256",
        "--max_length", "128",
        "--num_workers", "0",
        "--lora_rank", "16",
        "--precision", "fp32",
        "--weight_decay", "0.01",
        "--llrd_gamma", str(llrd),
    ]


def build_command(freeze, method, dataset, seed, python):
    params = freeze["selected_configurations"][method][dataset]["parameters"]
    exp_name = f"canonical/{method}_{dataset}_s{seed}"
    data_root = ROOT / "data" / dataset
    common = common_args(
        exp_name,
        data_root,
        dataset,
        seed,
        params["lr"],
        params.get("llrd_gamma", 1.0),
    )
    if method == "sequential_lora":
        return [python, "baselines/run_sequential_lora.py", *common, "--classifier", "cosine", "--deterministic", "--no-pin_memory"]
    if method == "seq_lora_matched_alignment":
        return [python, "baselines/run_sequential_lora_matched_alignment.py", *common, "--deterministic", "--no-pin_memory"]
    if method == "olora":
        return [
            python,
            "baselines/run_olora.py",
            *common,
            "--target_modules", "query", "value",
            "--lora_dropout", "0.1",
            "--olora_lambda1", str(params["olora_lambda1"]),
            "--global_interface",
        ]
    if method == "clora":
        return [python, "baselines/run_clora.py", *common, "--clora_k", "512", "--clora_lambda", "1.0", "--deterministic", "--no-pin_memory"]
    if method == "slora_pre":
        return [
            python,
            "baselines/run_slora.py",
            *common,
            "--candidate_ratios",
            *[str(index / 10) for index in range(1, 11)],
            "--deterministic",
            "--no-pin_memory",
        ]

    rem_lr = "0.01" if method == "recap_wo_refinements" else "0.005"
    rem_cycles = "30" if method == "recap_wo_refinements" else "80"
    merge_gamma = "1.0" if method == "recap_wo_refinements" else "0.75"
    return [
        python,
        "main.py",
        *common,
        "--use_sleep",
        "--alpha", "0.5",
        "--target_norm", "11.5",
        "--num_centroids", "1",
        "--prototype_std_scale", "0.5",
        "--rem_noise_std", "0.05",
        "--alignment_method", "gaussian",
        "--rem_classifier_lr", rem_lr,
        "--rem_schedule", "coverage_clipped",
        "--rem_batch_size", "32",
        "--rem_cycles_per_class", rem_cycles,
        "--min_rem_steps", "100",
        "--max_rem_steps", "600",
        "--rem_dimp", "0.0",
        "--merge_gamma", merge_gamma,
        "--merge_gamma_min", "0.0",
        "--merge_decay_mode", "max_floor",
        "--lora_alpha", "0.01",
        "--no-analyze_stages",
        "--no-log_rem_diagnostics",
        "--no-evaluate_train_accuracy",
        "--no-save_checkpoints",
        "--deterministic",
        "--no-pin_memory",
    ]


def result_path(method, dataset, seed):
    return OUTPUT_ROOT / f"{method}_{dataset}_s{seed}" / "results.json"


def matrix_metrics(matrix):
    tasks = len(matrix)
    final = statistics.mean(matrix[-1][:tasks])
    avg_inc = statistics.mean(
        statistics.mean(matrix[stage][: stage + 1]) for stage in range(tasks)
    )
    bwt = (
        statistics.mean(matrix[-1][task] - matrix[task][task] for task in range(tasks - 1))
        if tasks > 1
        else 0.0
    )
    return final, avg_inc, bwt


def complete(path, tasks):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        matrix = value["matrix"]
        return len(matrix) == tasks and all(len(row) >= tasks for row in matrix)
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False


def report(methods, datasets, seeds):
    rows = []
    for dataset in datasets:
        for method in methods:
            values = []
            for seed in seeds:
                path = result_path(method, dataset, seed)
                if not path.is_file():
                    raise FileNotFoundError(path)
                result = json.loads(path.read_text(encoding="utf-8"))
                values.append(matrix_metrics(result["matrix"]))
            row = {
                "dataset": dataset,
                "method": method,
                "label": LABELS[method],
                "seeds": list(seeds),
            }
            for index, key in enumerate(("final_avg", "avg_inc", "bwt")):
                samples = [item[index] for item in values]
                row[key] = {
                    "mean": statistics.mean(samples),
                    "std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
                }
            rows.append(row)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (OUTPUT_ROOT / "summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("Dataset", "Method", "Final Avg", "Avg Inc", "BWT"))
        for row in rows:
            writer.writerow(
                [
                    row["dataset"],
                    row["label"],
                    *[
                        f"{row[key]['mean'] * 100:.2f} +/- {row[key]['std'] * 100:.2f}"
                        for key in ("final_avg", "avg_inc", "bwt")
                    ],
                ]
            )
    for row in rows:
        print(
            f"{row['dataset']} | {row['label']} | "
            f"Final={row['final_avg']['mean'] * 100:.2f} +/- {row['final_avg']['std'] * 100:.2f} | "
            f"AvgInc={row['avg_inc']['mean'] * 100:.2f} +/- {row['avg_inc']['std'] * 100:.2f} | "
            f"BWT={row['bwt']['mean'] * 100:.2f} +/- {row['bwt']['std'] * 100:.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Run the canonical DiCRA comparison.")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()
    if any(seed not in SEEDS for seed in args.seeds):
        raise ValueError(f"supported seeds: {SEEDS}")
    freeze = load_freeze(FREEZE)

    failures = []
    if not args.report_only:
        planned = [
            (method, dataset, seed, build_command(freeze, method, dataset, seed, args.python))
            for method in args.methods
            for dataset in args.datasets
            for seed in args.seeds
        ]
        for index, (method, dataset, seed, command) in enumerate(planned, 1):
            path = result_path(method, dataset, seed)
            tasks = DATASETS[dataset][0]
            if complete(path, tasks):
                print(f"[{index}/{len(planned)}] skip complete: {method}/{dataset}/s{seed}")
                continue
            print(f"[{index}/{len(planned)}] $ " + subprocess.list2cmdline(command))
            if args.dry_run:
                continue
            outcome = subprocess.run(command, cwd=ROOT)
            if outcome.returncode:
                failures.append((method, dataset, seed, outcome.returncode))
                if not args.keep_going:
                    break
        if args.dry_run:
            print(f"planned_runs={len(planned)}")
            return
        if failures:
            raise SystemExit(f"failed runs: {failures}")
    report(args.methods, args.datasets, args.seeds)


if __name__ == "__main__":
    main()
