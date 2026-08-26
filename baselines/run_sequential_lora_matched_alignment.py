"""Sequential LoRA with RECAP's matched classifier Alignment only."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "main.py",
    "src/alignment.py",
    "src/config.py",
    "src/evaluation.py",
    "src/experiment_cli.py",
    "src/experiment_runner.py",
    "src/memory.py",
    "src/result_reporting.py",
    "src/run_config.py",
    "src/sleep_coordinator.py",
    "src/trainer.py",
    "baselines/run_sequential_lora_matched_alignment.py",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sequential LoRA + canonical matched Gaussian Alignment"
    )
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--data_root", default="data/clinc150")
    parser.add_argument("--model_id", default="bert-base-uncased")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_tasks", type=int, default=15)
    parser.add_argument("--num_classes", type=int, default=150)
    parser.add_argument(
        "--classifier_protocol",
        choices=["fixed_global", "dynamic_seen"],
        default="fixed_global",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--llrd_gamma", type=float, default=1.0)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--pin_memory",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def source_file_hashes():
    return {
        relative: hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
        for relative in SOURCE_FILES
    }


def git_provenance():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"source_commit": commit, "source_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"source_commit": None, "source_dirty": None}


def build_command(args):
    command = [
        sys.executable,
        "main.py",
        "--exp_name",
        args.exp_name,
        "--data_root",
        args.data_root,
        "--model_id",
        args.model_id,
        "--seed",
        str(args.seed),
        "--num_tasks",
        str(args.num_tasks),
        "--num_classes",
        str(args.num_classes),
        "--classifier_protocol",
        args.classifier_protocol,
        "--lr",
        str(args.lr),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--max_length",
        str(args.max_length),
        "--num_workers",
        str(args.num_workers),
        "--precision",
        args.precision,
        "--lora_rank",
        str(args.lora_rank),
        "--weight_decay",
        str(args.weight_decay),
        "--freeze_layers",
        "0",
        "--llrd_gamma",
        str(args.llrd_gamma),
        "--feat_distill_beta",
        "0.0",
        "--wake_replay_beta",
        "0.0",
        "--clora_lambda",
        "0.0",
        "--slora_mode",
        "disabled",
        "--no-use_sleep",
        "--matched_alignment_only",
        "--no_consolidation",
        "--alignment_method",
        "gaussian",
        "--num_centroids",
        "1",
        "--prototype_std_scale",
        "0.5",
        "--rem_noise_std",
        "0.05",
        "--rem_classifier_lr",
        "0.005",
        "--rem_schedule",
        "coverage_clipped",
        "--rem_batch_size",
        "32",
        "--rem_cycles_per_class",
        "80",
        "--min_rem_steps",
        "100",
        "--max_rem_steps",
        "600",
        "--rem_dimp",
        "0.0",
        "--no-update_prototypes_without_sleep",
        "--no-analyze_stages",
        "--no-log_rem_diagnostics",
        "--no-evaluate_train_accuracy",
        "--no-save_checkpoints",
    ]
    command.append("--deterministic" if args.deterministic else "--no-deterministic")
    command.append("--pin_memory" if args.pin_memory else "--no-pin_memory")
    return command


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def annotate_artifacts(args, provenance):
    output_dir = REPO_ROOT / "outputs" / args.exp_name
    budget_path = output_dir / "alignment_budget.jsonl"
    profile_path = output_dir / "time_profile.json"
    if not budget_path.is_file() or not profile_path.is_file():
        raise FileNotFoundError("matched Alignment budget or time profile is missing")
    budgets = _read_jsonl(budget_path)
    profile = _read_json(profile_path)
    hidden_size = 768
    stored_floats_per_class = 3 * hidden_size + 1
    prototype_storage_bytes = (
        args.num_classes * stored_floats_per_class * 4
    )
    metadata = {
        "method": "Sequential LoRA + Matched Alignment",
        "method_id": "seq_lora_matched_alignment",
        "variant": "class-incremental matched classifier-repair control",
        "classifier_protocol": args.classifier_protocol,
        "classifier": "CosineLinear",
        "primary_metric_space": (
            "dynamic_seen"
            if args.classifier_protocol == "dynamic_seen"
            else "global_unmasked"
        ),
        "training_protocol": args.classifier_protocol,
        "evaluation_protocol": args.classifier_protocol,
        "future_classifier_rows_present_during_training": (
            args.classifier_protocol == "fixed_global"
        ),
        "encoder_update": "persistent_sequential_lora",
        "consolidation_enabled": False,
        "lora_merge_reset_enabled": False,
        "stabilization_enabled": False,
        "matched_alignment_only": True,
        "alignment_method": "gaussian",
        "alignment_updates": "classifier_only",
        "alignment_label_smoothing": 0.1,
        "prototype_memory_role": "training_only_classifier_alignment",
        "prototype_memory_required_at_inference": False,
        "prototype_storage_budget": {
            "num_classes": args.num_classes,
            "num_centroids": 1,
            "hidden_size": hidden_size,
            "dtype": "float32",
            "stored_unique_tensors": [
                "centroid_mean",
                "centroid_std",
                "centroid_weight",
                "class_mean",
            ],
            "bytes": prototype_storage_bytes,
            "mib": prototype_storage_bytes / (1024 ** 2),
            "accounting": "tensor payload only; excludes Python container overhead and views",
        },
        "historical_text_replay": False,
        "sample_exemplars": False,
        "sample_level_task_id_at_inference": False,
        "peft_target_modules": ["query", "value"],
        "peft_lora_rank": args.lora_rank,
        "llrd_gamma": args.llrd_gamma,
        "additional_inference_parameters": 0,
        "source_files_sha256": source_file_hashes(),
        **provenance,
    }
    resource = {
        "alignment_reports": budgets,
        "time_profile_summary": profile.get("summary", {}),
    }
    for filename in ("config.json", "results.json"):
        path = output_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"matched Alignment output is missing: {path}")
        payload = _read_json(path)
        payload.update(metadata)
        if filename == "results.json":
            payload["matched_alignment"] = resource
        path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if args.lora_rank != 16:
        raise ValueError("the matched control locks --lora_rank 16")
    command = build_command(args)
    provenance = git_provenance()
    print("[SeqLoRA + Align] matched classifier-repair control")
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    annotate_artifacts(args, provenance)


if __name__ == "__main__":
    main()
