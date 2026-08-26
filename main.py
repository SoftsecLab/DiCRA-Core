import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.config import (
    CANONICAL_PRESET_NAME,
    PROJECT_NAME,
    RECAP_CANONICAL_CONFIG,
    build_experiment_config,
    canonical_overrides,
)
from src.dataset import JSONLDataset, JSONLBatchCollator
from src.experiment_cli import (
    parse_args as parse_experiment_args,
    validate_and_normalize_args,
)
from src.experiment_runner import ContinualExperimentRunner
from src.memory import PrototypeMemory
from src.model import RECAPBertClassifier, get_bert_path
from src.profiling import (
    make_param_profile,
    print_param_profile,
    write_param_profile,
)
from src.result_reporting import (
    ResultOptions,
    finalize_experiment,
    write_initial_clora_audit,
)
from src.runtime_plugins import enabled_runtime_plugins
from src.slora import SLoRAPreConsolidator
from src.source_provenance import source_provenance_snapshot
from src.trainer import RECAPTrainer


def setup_seed(seed, deterministic=True):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    print(f"Seed fixed: {seed}")


def parse_args(argv=None):
    """Compatibility entrypoint for callers that import ``main.parse_args``."""
    return parse_experiment_args(argv)


def resolve_task_order(args):
    if args.task_order is not None:
        task_order = list(args.task_order)
    else:
        task_order = list(range(args.num_tasks))
        if args.task_order_seed is not None:
            rng = random.Random(args.task_order_seed)
            rng.shuffle(task_order)

    expected = list(range(args.num_tasks))
    if sorted(task_order) != expected:
        raise ValueError(
            f"Invalid task_order {task_order}. Expected a permutation of {expected}."
        )
    return task_order


def initial_classifier_labels(args, task_order):
    if args.classifier_protocol == "fixed_global":
        return list(range(args.num_classes))

    first_task = task_order[0]
    train_path = os.path.join(args.data_root, f"task_{first_task}", "train.json")
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            "The dynamic-seen protocol requires the first task before model "
            f"construction, but it is missing: {train_path}"
        )
    dataset = JSONLDataset(train_path, max_len=args.max_length, encode_on_getitem=False)
    labels = sorted({int(item["label"]) for item in dataset.data})
    if not labels:
        raise ValueError(f"No class labels found in first task: {train_path}")
    return labels


def main():
    args = validate_and_normalize_args(parse_args())
    setup_seed(args.seed, deterministic=args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_order = resolve_task_order(args)
    classifier_initial_labels = initial_classifier_labels(args, task_order)

    experiment_config = build_experiment_config(args)
    output_dir = os.path.join("outputs", args.exp_name)
    os.makedirs(output_dir, exist_ok=True)
    saved_config = vars(args).copy()
    saved_config.update(
        {
            **source_provenance_snapshot(os.path.dirname(__file__)),
            "project_name": PROJECT_NAME,
            "canonical_preset": CANONICAL_PRESET_NAME,
            "canonical_shared_config": RECAP_CANONICAL_CONFIG,
            "canonical_overrides": canonical_overrides(saved_config),
            "classifier_initial_class_ids": classifier_initial_labels,
            "classifier_output_policy": (
                "fixed_full_dataset"
                if args.classifier_protocol == "fixed_global"
                else "expand_on_class_arrival"
            ),
            "training_candidate_space": (
                "fixed_global_unmasked"
                if args.classifier_protocol == "fixed_global"
                else "dynamic_seen"
            ),
            "prediction_candidate_space": (
                "fixed_global_unmasked"
                if args.classifier_protocol == "fixed_global"
                else "dynamic_seen"
            ),
            "future_class_rows_present": args.classifier_protocol == "fixed_global",
            "enabled_runtime_plugins": enabled_runtime_plugins(args),
        }
    )
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(saved_config, f, indent=4)

    print(f"{PROJECT_NAME} experiment start: {args.exp_name}")
    print(f"Task order: {task_order}")

    model_path = get_bert_path(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = RECAPBertClassifier(
        model_path,
        num_classes=len(classifier_initial_labels),
        use_lora=True,
        use_cosine=not args.no_cosine,
        lora_rank=args.lora_rank,
        gradient_checkpointing=args.gradient_checkpointing,
        class_ids=classifier_initial_labels,
    ).to(device)
    slora_consolidator = (
        SLoRAPreConsolidator(
            model,
            candidate_ratios=args.slora_candidate_ratios,
            seed=args.seed,
            output_dir=output_dir,
        )
        if args.slora_mode == "pre"
        else None
    )
    wake_profile = make_param_profile(model, stage="wake_train")
    print_param_profile(wake_profile)
    write_param_profile(output_dir, wake_profile)

    inference_profile = make_param_profile(
        model,
        stage="inference",
        trainable_param_names=set(),
    )
    inference_profile.update(
        {
            "inference_modules": (
                "shared backbone + Prototype Memory"
                if args.alignment_method == "direct_ncm"
                else (
                    "shared backbone + temporary prototype-imprinted classifier"
                    if args.alignment_method == "eval_only_imprinting"
                    else "shared backbone + unified classifier"
                )
            ),
            "needs_task_id": False,
            "uses_prototype_memory_at_inference": (
                args.alignment_method == "direct_ncm"
            ),
            "uses_prototype_memory_for_eval_head_construction": (
                args.alignment_method == "eval_only_imprinting"
            ),
            "growing_task_modules": args.classifier_protocol == "dynamic_seen",
            "classifier_protocol": args.classifier_protocol,
        }
    )
    write_param_profile(output_dir, inference_profile)

    prototype_memory = PrototypeMemory(
        args.num_classes,
        768,
        device,
        num_centroids=args.num_centroids,
        prototype_std_scale=args.prototype_std_scale,
    )
    trainer = RECAPTrainer(model, device, args)
    write_initial_clora_audit(output_dir, trainer)
    collator = JSONLBatchCollator(
        tokenizer,
        max_len=args.max_length,
        pad_to_multiple_of=args.pad_to_multiple_of,
    )

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory and device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    def make_loader(dataset, batch_size, shuffle):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collator,
            **loader_kwargs,
        )

    runner = ContinualExperimentRunner(
        config=experiment_config,
        task_order=task_order,
        model=model,
        tokenizer=tokenizer,
        trainer=trainer,
        prototype_memory=prototype_memory,
        make_loader=make_loader,
        output_dir=output_dir,
        device=device,
        task_boundary_transform=(
            slora_consolidator.consolidate
            if slora_consolidator is not None
            else None
        ),
    )
    run_state = runner.run()
    model = runner.model
    finalize_experiment(
        run_state=run_state,
        task_order=task_order,
        options=ResultOptions.from_args(
            args,
            enabled_runtime_plugins(args),
        ),
        output_dir=output_dir,
        trainer=trainer,
        time_profiler=runner.time_profiler,
        diagnostics=runner.diagnostics,
        oracle_args=args,
        model=model,
        make_loader=make_loader,
        device=device,
    )


if __name__ == "__main__":
    main()
