from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


PAPER_URL = "https://aclanthology.org/2025.acl-long.940/"
OFFICIAL_REPOSITORY = "https://github.com/sutakori/CLoRA"
OFFICIAL_SOURCE_COMMIT = "802cda88cd21e839326701ba5c2ba48cbd317be0"
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "main.py",
    "src/experiment_cli.py",
    "src/result_reporting.py",
    "src/evaluation.py",
    "src/evaluator.py",
    "src/experiment_state.py",
    "src/task_evaluation.py",
    "src/trainer.py",
    "src/wake_training.py",
    "src/clora_regularizer.py",
    "baselines/run_clora.py",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Controlled LoRA class-incremental adapted reimplementation"
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
    parser.add_argument("--clora_k", type=int, default=512)
    parser.add_argument("--clora_lambda", type=float, default=1.0)
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


def git_provenance():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"source_commit": commit, "source_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"source_commit": None, "source_dirty": None}


def source_file_hashes():
    return {
        relative_path: hashlib.sha256(
            (REPO_ROOT / relative_path).read_bytes()
        ).hexdigest()
        for relative_path in SOURCE_FILES
    }


def annotate_artifacts(args, provenance):
    dynamic_seen = args.classifier_protocol == "dynamic_seen"
    metadata = {
        "method": "Controlled LoRA",
        "method_id": "clora_dynamic_seen" if dynamic_seen else "clora_global",
        "variant": "class-incremental adapted reimplementation",
        "paper_url": PAPER_URL,
        "official_repository": OFFICIAL_REPOSITORY,
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "classifier_protocol": args.classifier_protocol,
        "classifier": "CosineLinear",
        "primary_metric_space": (
            "dynamic_seen" if dynamic_seen else "global_unmasked"
        ),
        "future_classifier_rows_present_during_training": not dynamic_seen,
        "future_class_masking": False if not dynamic_seen else None,
        "classifier_head_expands_with_seen_classes": dynamic_seen,
        "training_protocol": args.classifier_protocol,
        "evaluation_protocol": args.classifier_protocol,
        "historical_text_replay": False,
        "sample_exemplars": False,
        "sample_level_task_id_at_inference": False,
        "prototype_memory_role": "diagnostics_only",
        "peft_target_modules": ["query", "value"],
        "peft_lora_rank": args.lora_rank,
        "peft_lora_alpha": args.lora_rank * 2,
        "peft_lora_dropout": 0.1,
        "clora_k": args.clora_k,
        "clora_lambda": args.clora_lambda,
        **provenance,
    }
    output_dir = os.path.join("outputs", args.exp_name)
    for filename in ("config.json", "results.json"):
        path = os.path.join(output_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"CLoRA expected output artifact: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload.update(metadata)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=4)
            handle.write("\n")


def build_command(args):
    return [
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
        "--clora_k",
        str(args.clora_k),
        "--clora_lambda",
        str(args.clora_lambda),
        "--weight_decay",
        str(args.weight_decay),
        "--freeze_layers",
        "0",
        "--llrd_gamma",
        str(getattr(args, "llrd_gamma", 1.0)),
        "--feat_distill_beta",
        "0.0",
        "--wake_replay_beta",
        "0.0",
        "--alignment_method",
        "none",
        "--no-use_sleep",
        "--update_prototypes_without_sleep",
        "--no-analyze_stages",
        "--no-log_rem_diagnostics",
        "--no-evaluate_train_accuracy",
    ]


def main():
    args = parse_args()
    if args.clora_k <= 0:
        raise ValueError("--clora_k must be positive")
    if args.clora_lambda <= 0:
        raise ValueError("--clora_lambda must be positive for the CLoRA baseline")

    provenance = {
        **git_provenance(),
        "source_files_sha256": source_file_hashes(),
    }
    command = build_command(args)
    command.append("--deterministic" if args.deterministic else "--no-deterministic")
    command.append("--pin_memory" if args.pin_memory else "--no-pin_memory")

    print("[CLoRA] Controlled LoRA class-incremental adapted reimplementation")
    print("$ " + " ".join(command))
    subprocess.run(command, check=True)
    annotate_artifacts(args, provenance)


if __name__ == "__main__":
    main()
