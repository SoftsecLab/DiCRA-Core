"""NREM-style parameter stabilization and its audit trail."""

from __future__ import annotations

import json
import math
import os

import torch

from src.run_config import AlignmentConfig


STABILIZATION_GROUPS = (
    "encoder_base",
    "encoder_lora",
    "classifier_weight",
    "classifier_sigma",
    "other",
)


def stabilization_group(name):
    if name.startswith("bert."):
        return "encoder_lora" if "lora" in name else "encoder_base"
    if name.startswith("classifier."):
        return (
            "classifier_sigma"
            if "sigma" in name
            else "classifier_weight"
        )
    return "other"


def init_stabilization_audit(model, task_id, exclude_classifier):
    groups = {
        group: {
            "tensors": 0,
            "numel": 0,
            "changed": 0,
            "pre_norm_sq": 0.0,
            "post_norm_sq": 0.0,
            "delta_norm_sq": 0.0,
        }
        for group in STABILIZATION_GROUPS
    }
    for name, param in model.named_parameters():
        group = groups[stabilization_group(name)]
        norm = float(
            torch.linalg.vector_norm(
                param.detach().float()
            ).item()
        )
        group["tensors"] += 1
        group["numel"] += int(param.numel())
        group["pre_norm_sq"] += norm * norm
        group["post_norm_sq"] += norm * norm
    return {
        "task_id": int(task_id),
        "exclude_classifier_stabilization": bool(
            exclude_classifier
        ),
        "groups": groups,
    }


def record_stabilization_change(audit, name, before, after):
    if audit is None:
        return
    group = audit["groups"][stabilization_group(name)]
    before_norm = float(
        torch.linalg.vector_norm(before.float()).item()
    )
    after_norm = float(
        torch.linalg.vector_norm(after.detach().float()).item()
    )
    difference = after.detach().float() - before.float()
    delta_norm = float(
        torch.linalg.vector_norm(difference).item()
    )
    group["post_norm_sq"] += (
        after_norm * after_norm - before_norm * before_norm
    )
    group["delta_norm_sq"] += delta_norm * delta_norm
    group["changed"] += int(
        torch.count_nonzero(difference).item()
    )


def finalize_stabilization_audit(audit, output_dir=None):
    for group in audit["groups"].values():
        pre_norm = math.sqrt(
            max(group.pop("pre_norm_sq"), 0.0)
        )
        post_norm = math.sqrt(
            max(group.pop("post_norm_sq"), 0.0)
        )
        delta_norm = math.sqrt(
            max(group.pop("delta_norm_sq"), 0.0)
        )
        group["relative_l2_delta"] = (
            delta_norm / (pre_norm + 1e-12)
        )
        group["norm_pre"] = pre_norm
        group["norm_post"] = post_norm

    print("\n[Stabilization Audit]")
    print(
        f"{'Group':<24} {'Tensors':>8} {'Numel':>12} "
        f"{'Changed':>12} {'Relative L2 Delta':>18} "
        f"{'Norm Pre -> Post':>28}"
    )
    print("-" * 108)
    for name in STABILIZATION_GROUPS:
        group = audit["groups"][name]
        print(
            f"{name:<24} {group['tensors']:>8} "
            f"{group['numel']:>12,} "
            f"{group['changed']:>12,} "
            f"{group['relative_l2_delta']:>18.6e} "
            f"{group['norm_pre']:>12.6f} -> "
            f"{group['norm_post']:<12.6f}"
        )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(
            output_dir,
            "stabilization_audit.jsonl",
        )
        mode = "w" if audit["task_id"] == 0 else "a"
        with open(path, mode, encoding="utf-8") as handle:
            handle.write(json.dumps(audit) + "\n")
        print(f"[Stabilization Audit] Saved to: {path}")


def stabilize_model(
    model,
    config: AlignmentConfig,
    num_observed_classes,
    *,
    output_dir=None,
    audit=None,
):
    """Apply the existing NREM compression rule to trainable parameters."""

    if audit is None and config.audit_stabilization:
        audit = init_stabilization_audit(
            model,
            task_id=config.task_id,
            exclude_classifier=(
                config.exclude_classifier_stabilization
            ),
        )

    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "sigma" in name:
                continue
            if (
                config.exclude_classifier_stabilization
                and name.startswith("classifier.")
            ):
                continue

            before = (
                param.detach().clone()
                if audit is not None
                else None
            )
            if "lora" in name:
                param.data *= 1.0 - config.lora_alpha
                record_stabilization_change(
                    audit,
                    name,
                    before,
                    param,
                )
                continue

            param.data *= 1.0 - config.alpha
            if param.dim() > 1:
                curr_norm = param.norm()
                dynamic_target_norm = (
                    config.target_norm
                    * math.sqrt(
                        max(1, int(num_observed_classes)) / 15.0
                    )
                )
                if curr_norm > dynamic_target_norm:
                    param.data *= (
                        dynamic_target_norm / (curr_norm + 1e-8)
                    )
            record_stabilization_change(
                audit,
                name,
                before,
                param,
            )

    print("   [NREM] compression complete")
    if audit is not None:
        finalize_stabilization_audit(
            audit,
            output_dir=output_dir,
        )
    return audit
