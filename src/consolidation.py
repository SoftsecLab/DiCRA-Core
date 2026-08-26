from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.run_config import ConsolidationConfig


def resolve_merge_decay(
    config: ConsolidationConfig,
    task_id: int,
):


    raw_decay = config.merge_gamma ** int(task_id)
    if config.merge_decay_mode == "affine_floor":
        decay = config.merge_gamma_min + (
            (1.0 - config.merge_gamma_min) * raw_decay
        )
    else:
        decay = max(config.merge_gamma_min, raw_decay)
    return raw_decay, decay


def consolidate_lora(
    model,
    config: ConsolidationConfig,
    task_id=0,
):


    if not isinstance(config, ConsolidationConfig):
        raise TypeError("config must be a ConsolidationConfig")

    merged_count = 0
    raw_decay, decay = resolve_merge_decay(config, task_id)

    if (
        config.merge_decay_mode == "max_floor"
        and config.merge_gamma_min > 0.0
        and decay != raw_decay
    ):
        print(
            "   [Consolidation] merge decay clipped: "
            f"raw={raw_decay:.6g}, "
            f"min={config.merge_gamma_min:.6g}"
        )

    if config.merge_decay_mode == "affine_floor":
        print(
            "   [Consolidation] affine merge decay: "
            f"floor={config.merge_gamma_min:.6g}, "
            f"raw={raw_decay:.6g}, decay={decay:.6g}"
        )

    with torch.no_grad():
        for module in model.bert.modules():
            if not (
                hasattr(module, "lora_A")
                and hasattr(module, "lora_B")
            ):
                continue
            if not hasattr(module, "scaling"):
                continue

            for adapter_name in list(module.lora_A.keys()):
                lora_a_weight = (
                    module.lora_A[adapter_name].weight
                )
                lora_b_weight = (
                    module.lora_B[adapter_name].weight
                )
                scaling = module.scaling[adapter_name]

                if hasattr(module, "get_base_layer"):
                    base_weight = (
                        module.get_base_layer().weight
                    )
                elif hasattr(module, "base_layer"):
                    base_weight = module.base_layer.weight
                else:
                    continue

                base_weight.data += (
                    lora_b_weight @ lora_a_weight
                ) * scaling * decay
                nn.init.kaiming_uniform_(
                    lora_a_weight,
                    a=math.sqrt(5),
                )
                lora_b_weight.zero_()
                merged_count += 1

    print(
        "   [Consolidation] "
        f"merged_lora_layers={merged_count}, "
        f"task={task_id}, decay={decay:.3f}"
    )
    return merged_count, decay
