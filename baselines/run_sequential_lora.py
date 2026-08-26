import argparse
import json
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Sequential LoRA baseline")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="data/clinc150")
    parser.add_argument("--model_id", type=str, default="bert-base-uncased")
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
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pad_to_multiple_of", type=int, default=None)
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--llrd_gamma", type=float, default=1.0)
    parser.add_argument(
        "--classifier",
        type=str,
        default="linear",
        choices=["linear", "cosine"],
        help="Plain Sequential LoRA uses linear by default. Use cosine only for an extra diagnostic.",
    )
    return parser.parse_args()


def build_command(args):
    cmd = [
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
        "--grad_accum_steps",
        str(args.grad_accum_steps),
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
        "--rem_dimp",
        "0.0",
        "--feat_distill_beta",
        "0.0",
        "--wake_replay_beta",
        "0.0",
        "--no-use_sleep",
    ]

    if args.pad_to_multiple_of is not None:
        cmd.extend(["--pad_to_multiple_of", str(args.pad_to_multiple_of)])
    if args.pin_memory:
        cmd.append("--pin_memory")
    else:
        cmd.append("--no-pin_memory")
    if args.deterministic:
        cmd.append("--deterministic")
    else:
        cmd.append("--no-deterministic")
    if args.gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    else:
        cmd.append("--no-gradient_checkpointing")
    if args.classifier == "linear":
        cmd.append("--no_cosine")
    return cmd


def main():
    args = parse_args()
    cmd = build_command(args)

    print("[Sequential LoRA] Delegating to main.py with no Sleep/replay/consolidation.")
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    output_dir = os.path.join("outputs", args.exp_name)
    for filename in ("config.json", "results.json"):
        path = os.path.join(output_dir, filename)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["method"] = "Sequential LoRA"
        payload["variant"] = f"LoRA+{'Linear' if args.classifier == 'linear' else 'CosineLinear'}+NoSleep"
        payload["rehearsal"] = "none"
        payload["uses_sleep"] = False
        payload["classifier_protocol"] = args.classifier_protocol
        payload["llrd_gamma"] = args.llrd_gamma
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)
            f.write("\n")


if __name__ == "__main__":
    main()
