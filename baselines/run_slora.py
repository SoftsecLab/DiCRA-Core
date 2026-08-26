from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path


PAPER_TITLE = (
    "SLoRA: Balancing Plasticity and Forgetting in Large Language Models "
    "for Continual Learning"
)
PAPER_VENUE = "ACL 2026 Main Conference, Long Paper"
PAPER_PDF_SHA256 = (
    "0e0d689a90669fb0750ac459eebec88366d5029d64192eb6527348a0abb06763"
)
OFFICIAL_SOURCE_ARCHIVE = "SLoRA-main.zip"
OFFICIAL_SOURCE_ARCHIVE_SHA256 = (
    "91140958447f931ac941be96b616044f9e860bfc105b6d3f44457e2c35529005"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "main.py",
    "src/experiment_cli.py",
    "src/experiment_runner.py",
    "src/profiling.py",
    "src/slora.py",
    "src/result_reporting.py",
    "src/evaluation.py",
    "src/evaluator.py",
    "src/task_evaluation.py",
    "src/wake_training.py",
    "baselines/run_slora.py",
)
DEFAULT_CANDIDATE_RATIOS = tuple(index / 10 for index in range(1, 11))


def parse_args():
    parser = argparse.ArgumentParser(
        description="SLoRA-Pre class-incremental adapted reimplementation"
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
        "--candidate_ratios",
        nargs="+",
        type=float,
        default=list(DEFAULT_CANDIDATE_RATIOS),
    )
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
        "--merge_gamma",
        "1.0",
        "--feat_distill_beta",
        "0.0",
        "--wake_replay_beta",
        "0.0",
        "--clora_lambda",
        "0.0",
        "--alignment_method",
        "none",
        "--slora_mode",
        "pre",
        "--slora_candidate_ratios",
        *[str(value) for value in args.candidate_ratios],
        "--no-use_sleep",
        "--update_prototypes_without_sleep",
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
    reference_path = output_dir / "slora_reference.json"
    denoising_path = output_dir / "slora_denoising.jsonl"
    if not reference_path.exists() or not denoising_path.exists():
        raise FileNotFoundError("SLoRA reference or denoising audit is missing")
    reference = _read_json(reference_path)
    task_reports = _read_jsonl(denoising_path)
    chosen_ranks = Counter(
        int(module["chosen_rank"])
        for report in task_reports
        for module in report["modules"]
    )
    slora_summary = {
        "variant": "SLoRA-Pre",
        "reference": reference,
        "task_reports": task_reports,
        "total_denoising_sec": sum(
            float(report["denoising_sec"]) for report in task_reports
        ),
        "chosen_rank_histogram": {
            str(rank): count for rank, count in sorted(chosen_ranks.items())
        },
    }
    metadata = {
        "method": "SLoRA-Pre",
        "method_id": (
            "slora_pre_dynamic_seen"
            if args.classifier_protocol == "dynamic_seen"
            else "slora_pre_global"
        ),
        "variant": "class-incremental adapted reimplementation",
        "paper_title": PAPER_TITLE,
        "paper_venue": PAPER_VENUE,
        "paper_pdf_sha256": PAPER_PDF_SHA256,
        "official_source_archive": OFFICIAL_SOURCE_ARCHIVE,
        "official_source_archive_sha256": OFFICIAL_SOURCE_ARCHIVE_SHA256,
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
        "future_class_masking": False,
        "historical_text_replay": False,
        "historical_gradient_storage": False,
        "sample_exemplars": False,
        "sample_level_task_id_at_inference": False,
        "prototype_memory_role": "diagnostics_only",
        "peft_target_modules": ["query", "value"],
        "peft_lora_rank": args.lora_rank,
        "peft_lora_alpha": args.lora_rank * 2,
        "peft_lora_dropout": 0.1,
        "llrd_gamma": args.llrd_gamma,
        "slora_mode": "pre",
        "slora_candidate_ratios": list(args.candidate_ratios),
        "slora_update_definition": "effective_peft_scaling_times_BA",
        "slora_reference_scope": "frozen_pretrained_base_weights",
        "persistent_encoder_updates": "denoised_task_updates",
        "inference_path": "consolidated_encoder_plus_global_classifier",
        "training_only_reference_memory_bytes": reference[
            "reference_memory_bytes"
        ],
        "additional_inference_parameters": 0,
        **provenance,
    }
    for filename in ("config.json", "results.json"):
        path = output_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"SLoRA expected output artifact: {path}")
        payload = _read_json(path)
        payload.update(metadata)
        if filename == "results.json":
            payload["slora"] = slora_summary
        else:
            payload["slora_reference_sha256"] = reference[
                "reference_sha256"
            ]
        path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if args.lora_rank <= 0:
        raise ValueError("--lora_rank must be positive")
    if any(value <= 0 or value > 1 for value in args.candidate_ratios):
        raise ValueError("--candidate_ratios values must be in (0, 1]")
    if 1.0 not in args.candidate_ratios:
        raise ValueError("--candidate_ratios must include 1.0")

    provenance = {
        **git_provenance(),
        "source_files_sha256": source_file_hashes(),
    }
    command = build_command(args)
    print("[SLoRA-Pre] class-incremental adapted reimplementation")
    print("$ " + " ".join(command))
    subprocess.run(command, check=True, cwd=REPO_ROOT)
    annotate_artifacts(args, provenance)


if __name__ == "__main__":
    main()
