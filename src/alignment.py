"""Prototype-based classifier alignment services."""

from __future__ import annotations

import copy
import json
import math
import os
import time
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.profiling import (
    make_param_profile,
    print_param_profile,
    write_param_profile,
)
from src.run_config import AlignmentConfig


SUPPORTED_ALIGNMENT_METHODS = frozenset(
    {
        "gaussian",
        "mean_only",
        "weight_imprinting",
        "eval_only_imprinting",
        "direct_ncm",
        "none",
    }
)


def effective_alignment_method(config: AlignmentConfig) -> str:
    method = "none" if config.no_rem else config.alignment_method
    if method not in SUPPORTED_ALIGNMENT_METHODS:
        raise ValueError(f"Unsupported alignment_method={method!r}")
    return method


def resolve_rem_budget(num_classes, config: AlignmentConfig):
    """Resolve how many REM prototype batches to replay."""

    num_classes = max(0, int(num_classes))
    rem_cycles_per_class = int(config.rem_cycles_per_class)
    rem_batch_size = int(config.rem_batch_size)
    rem_schedule = config.rem_schedule

    if rem_schedule == "fixed_floor":
        raw_steps = num_classes * rem_cycles_per_class
        dream_steps = max(800, raw_steps)
        min_rem_steps = None
        max_rem_steps = None
        clip_reason = "fixed_floor" if raw_steps < 800 else "none"
    elif rem_schedule == "coverage_clipped":
        min_rem_steps = int(config.min_rem_steps)
        max_rem_steps = int(config.max_rem_steps)
        raw_steps = math.ceil(
            num_classes
            * rem_cycles_per_class
            / rem_batch_size
        )
        dream_steps = min(
            max(raw_steps, min_rem_steps),
            max_rem_steps,
        )
        if raw_steps < min_rem_steps:
            clip_reason = "min_steps"
        elif raw_steps > max_rem_steps:
            clip_reason = "max_steps"
        else:
            clip_reason = "none"
    else:
        raise ValueError(
            f"Unsupported rem_schedule={rem_schedule!r}. "
            "Expected 'fixed_floor' or 'coverage_clipped'."
        )

    planned_feature_draws = int(dream_steps) * rem_batch_size
    return {
        "schedule": rem_schedule,
        "num_observed_classes": num_classes,
        "steps": int(dream_steps),
        "raw_steps": int(raw_steps),
        "batch_size": rem_batch_size,
        "cycles_per_class": rem_cycles_per_class,
        "min_steps": min_rem_steps,
        "max_steps": max_rem_steps,
        "clip_reason": clip_reason,
        "planned_feature_draws": planned_feature_draws,
        "planned_draws_per_observed_class": (
            planned_feature_draws / num_classes
            if num_classes > 0
            else 0.0
        ),
    }


def _synchronize_device(device):
    if (
        getattr(device, "type", None) == "cuda"
        and torch.cuda.is_available()
    ):
        torch.cuda.synchronize(device)


def write_alignment_budget_log(report, output_dir):
    if output_dir is None:
        return None
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "alignment_budget.jsonl")
    mode = "w" if report["task_id"] == 0 else "a"
    with open(path, mode, encoding="utf-8") as handle:
        handle.write(json.dumps(report) + "\n")
    return path


def finalize_alignment_report(
    budget,
    task_id,
    realized_steps,
    realized_feature_draws,
    alignment_sec,
    output_dir=None,
    callback=None,
    skipped_reason=None,
):
    num_classes = budget["num_observed_classes"]
    report = {
        "task_id": int(task_id),
        **budget,
        "realized_steps": int(realized_steps),
        "realized_feature_draws": int(
            realized_feature_draws
        ),
        "realized_draws_per_observed_class": (
            realized_feature_draws / num_classes
            if num_classes > 0
            else 0.0
        ),
        "alignment_sec": float(alignment_sec),
        "skipped_reason": skipped_reason,
    }
    path = write_alignment_budget_log(report, output_dir)
    method = report.get("alignment_method", "gaussian")
    print(
        "   [Alignment Profile] "
        f"method={method}, "
        f"steps={report['realized_steps']}/{report['steps']}, "
        f"draws={report['realized_feature_draws']}, "
        f"coverage="
        f"{report['realized_draws_per_observed_class']:.2f}/class, "
        f"clip={report['clip_reason']}, "
        f"time={report['alignment_sec']:.3f}s"
    )
    if path is not None and task_id == 0:
        print(f"   [Alignment Profile] budget log: {path}")
    if callback is not None:
        callback(report)
    return report


def imprint_classifier_from_prototypes(
    model,
    prototype_memory,
    *,
    normalize_weights=False,
):
    """Overwrite every stored class row with its class prototype mean."""

    classifier = model.classifier
    if not hasattr(classifier, "weight"):
        raise TypeError(
            "Prototype weight imprinting requires a weighted classifier"
        )

    labels = sorted(prototype_memory.prototypes)
    rows = model.classifier_rows_for_global_labels(
        labels,
        device=classifier.weight.device,
    )
    means = torch.stack(
        [
            prototype_memory.class_mean(label)
            for label in labels
        ],
        dim=0,
    ).to(
        device=classifier.weight.device,
        dtype=classifier.weight.dtype,
    )
    if normalize_weights:
        means = F.normalize(means, p=2, dim=1, eps=1e-8)
    if (
        means.shape
        != classifier.weight.index_select(0, rows).shape
    ):
        raise ValueError(
            "Prototype means do not match classifier weight "
            "dimensionality"
        )
    with torch.no_grad():
        classifier.weight.index_copy_(0, rows, means)
    return labels


@contextmanager
def temporary_imprinted_classifier(model, prototype_memory):
    """Temporarily evaluate with an imprinted copy of the persistent head."""

    persistent_classifier = model.classifier
    persistent_state = {
        name: tensor.detach().clone()
        for name, tensor in persistent_classifier.state_dict().items()
    }
    device = persistent_classifier.weight.device
    _synchronize_device(device)
    start = time.perf_counter()
    temporary_classifier = copy.deepcopy(persistent_classifier)
    model.classifier = temporary_classifier
    report = {
        "imprinted_labels": [],
        "build_sec": 0.0,
        "temporary_parameters": int(
            sum(
                param.numel()
                for param in temporary_classifier.parameters()
            )
        ),
        "persistent_state_unchanged": False,
        "persistent_classifier_restored": False,
    }
    try:
        imprinted_labels = imprint_classifier_from_prototypes(
            model,
            prototype_memory,
            normalize_weights=True,
        )
        if set(imprinted_labels) != set(model.class_ids):
            raise RuntimeError(
                "Eval-only Imprinting requires one stored prototype "
                "for every dynamic classifier row"
            )
        _synchronize_device(device)
        build_sec = time.perf_counter() - start
        report["imprinted_labels"] = imprinted_labels
        report["build_sec"] = float(build_sec)
        yield report
    finally:
        model.classifier = persistent_classifier
        report["persistent_classifier_restored"] = (
            model.classifier is persistent_classifier
        )
        report["persistent_state_unchanged"] = all(
            torch.equal(
                persistent_classifier.state_dict()[name],
                expected,
            )
            for name, expected in persistent_state.items()
        )
        if not report["persistent_state_unchanged"]:
            raise RuntimeError(
                "Eval-only Imprinting modified the persistent "
                "training classifier"
            )


def _zero_step_budget(
    num_classes,
    config,
    *,
    alignment_method,
    feature_mode,
):
    budget = resolve_rem_budget(num_classes, config)
    budget["steps"] = 0
    budget["raw_steps"] = 0
    budget["planned_feature_draws"] = 0
    budget["planned_draws_per_observed_class"] = 0.0
    budget["alignment_method"] = alignment_method
    budget["feature_mode"] = feature_mode
    budget["clip_reason"] = "not_applicable"
    return budget


def align_classifier(
    model,
    prototype_memory,
    device,
    config: AlignmentConfig,
    *,
    output_dir=None,
    callback=None,
):
    """Run the selected post-stabilization classifier alignment."""

    alignment_method = effective_alignment_method(config)
    num_classes = (
        len(prototype_memory.prototypes)
        if prototype_memory
        else 0
    )
    task_id = int(config.task_id)

    if num_classes == 0 or alignment_method in {
        "none",
        "direct_ncm",
        "eval_only_imprinting",
    }:
        empty_budget = _zero_step_budget(
            num_classes,
            config,
            alignment_method=alignment_method,
            feature_mode=None,
        )
        finalize_alignment_report(
            empty_budget,
            task_id=task_id,
            realized_steps=0,
            realized_feature_draws=0,
            alignment_sec=0.0,
            output_dir=output_dir,
            callback=callback,
            skipped_reason=(
                "no_prototypes"
                if num_classes == 0
                else alignment_method
            ),
        )
        return model

    if alignment_method == "weight_imprinting":
        imprint_budget = _zero_step_budget(
            num_classes,
            config,
            alignment_method=alignment_method,
            feature_mode="class_mean",
        )
        _synchronize_device(device)
        imprint_start = time.perf_counter()
        imprinted_labels = imprint_classifier_from_prototypes(
            model,
            prototype_memory,
        )
        _synchronize_device(device)
        alignment_sec = time.perf_counter() - imprint_start
        finalize_alignment_report(
            imprint_budget,
            task_id=task_id,
            realized_steps=0,
            realized_feature_draws=0,
            alignment_sec=alignment_sec,
            output_dir=output_dir,
            callback=callback,
            skipped_reason=None,
        )
        print(
            "   [Alignment] imprinted prototype means: "
            f"classes={len(imprinted_labels)}"
        )
        return model

    feature_mode = (
        "mean"
        if alignment_method == "mean_only"
        else "gaussian"
    )
    print(
        "   [Alignment] classifier repair: "
        f"method={alignment_method}, classes={num_classes}"
    )
    rem_budget = resolve_rem_budget(num_classes, config)
    rem_budget["alignment_method"] = alignment_method
    rem_budget["feature_mode"] = feature_mode
    dream_steps = rem_budget["steps"]
    rem_batch_size = rem_budget["batch_size"]
    print(
        "   [REM Budget] "
        f"schedule={rem_budget['schedule']}, "
        f"steps={dream_steps}, "
        f"raw_steps={rem_budget['raw_steps']}, "
        f"classes={num_classes}, "
        f"batch_size={rem_batch_size}, "
        f"cycles_per_class={rem_budget['cycles_per_class']}, "
        f"clip={rem_budget['clip_reason']}"
    )

    realized_steps = 0
    realized_feature_draws = 0
    alignment_sec = 0.0

    classifier_params = []
    rem_trainable_names = set()
    for name, param in model.classifier.named_parameters():
        if "sigma" in name:
            param.requires_grad = False
        else:
            classifier_params.append(param)
            rem_trainable_names.add(f"classifier.{name}")

    if classifier_params:
        rem_profile = make_param_profile(
            model,
            stage="rem_classifier_only",
            trainable_param_names=rem_trainable_names,
        )
        print_param_profile(rem_profile)
        if output_dir is not None:
            write_param_profile(output_dir, rem_profile)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": classifier_params,
                    "lr": config.rem_classifier_lr,
                }
            ],
            weight_decay=0.0,
        )
        _synchronize_device(device)
        alignment_start = time.perf_counter()

        saved_norms = {}
        with torch.no_grad():
            for name, param in model.classifier.named_parameters():
                if "weight" in name and param.dim() > 1:
                    norm = param.norm(dim=1, keepdim=True)
                    saved_norms[name] = norm.clone()
                    param.data /= norm + 1e-8

        rem_noise_std = (
            0.0
            if alignment_method == "mean_only"
            else config.rem_noise_std
        )
        rem_dimp = config.rem_dimp

        for _ in range(dream_steps):
            proto_feats, proto_labels = (
                prototype_memory.get_prototype_batch(
                    batch_size=rem_batch_size,
                    feature_mode=feature_mode,
                )
            )
            if proto_feats is None:
                break

            proto_feats = proto_feats.to(device)
            proto_labels = proto_labels.to(device)
            classifier_labels = (
                model.global_to_classifier_labels(proto_labels)
            )
            if rem_noise_std > 0:
                proto_feats = (
                    proto_feats
                    + torch.randn_like(proto_feats)
                    * rem_noise_std
                )

            optimizer.zero_grad()
            logits = model.classifier(proto_feats)

            if rem_dimp > 0:
                with torch.no_grad():
                    probs = F.softmax(logits.detach(), dim=1)
                    correct_probs = probs.gather(
                        1,
                        classifier_labels.unsqueeze(1),
                    ).squeeze(1)
                    raw_weights = (
                        1.0 / (correct_probs + 1e-6)
                    ).clamp(min=0.1, max=10.0)
                    weights = raw_weights / raw_weights.mean()
                    weights = (
                        rem_dimp * weights
                        + (1.0 - rem_dimp)
                    )

                per_sample_loss = F.cross_entropy(
                    logits,
                    classifier_labels,
                    reduction="none",
                    label_smoothing=0.1,
                )
                loss = (weights * per_sample_loss).mean()
            else:
                loss = F.cross_entropy(
                    logits,
                    classifier_labels,
                    label_smoothing=0.1,
                )

            loss.backward()
            nn.utils.clip_grad_norm_(
                classifier_params,
                max_norm=1.0,
            )
            optimizer.step()
            realized_steps += 1
            realized_feature_draws += int(
                proto_labels.numel()
            )

        with torch.no_grad():
            for name, param in model.classifier.named_parameters():
                if "weight" in name and param.dim() > 1:
                    current_norm = param.norm(
                        dim=1,
                        keepdim=True,
                    )
                    param.data /= current_norm + 1e-8
                    param.data *= saved_norms[name]

        for name, param in model.classifier.named_parameters():
            if "sigma" in name:
                param.requires_grad = True

        _synchronize_device(device)
        alignment_sec = (
            time.perf_counter() - alignment_start
        )

    finalize_alignment_report(
        rem_budget,
        task_id=task_id,
        realized_steps=realized_steps,
        realized_feature_draws=realized_feature_draws,
        alignment_sec=alignment_sec,
        output_dir=output_dir,
        callback=callback,
        skipped_reason=(
            None
            if classifier_params
            else "no_classifier_parameters"
        ),
    )

    print(
        f"   [Alignment] {alignment_method} repair complete"
    )
    return model
