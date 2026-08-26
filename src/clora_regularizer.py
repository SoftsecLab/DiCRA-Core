from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class _LoRARecord:
    module_name: str
    lora_a: nn.Parameter
    lora_b: nn.Parameter
    p_input_name: str
    p_output_name: str


def _lora_parameter_groups(model: nn.Module):
    groups = {}
    for name, parameter in model.named_parameters():
        if ".lora_A." in name:
            module_name = name.split(".lora_A.", 1)[0]
            group = groups.setdefault(module_name, {})
            if "a" in group:
                raise ValueError(
                    f"{module_name}: CLoRA supports exactly one active LoRA adapter"
                )
            group["a"] = (name, parameter)
        elif ".lora_B." in name:
            module_name = name.split(".lora_B.", 1)[0]
            group = groups.setdefault(module_name, {})
            if "b" in group:
                raise ValueError(
                    f"{module_name}: CLoRA supports exactly one active LoRA adapter"
                )
            group["b"] = (name, parameter)

    incomplete = sorted(
        module_name
        for module_name, group in groups.items()
        if set(group) != {"a", "b"}
    )
    if incomplete:
        raise ValueError(
            "CLoRA found incomplete PEFT LoRA parameter pairs: "
            + ", ".join(incomplete)
        )
    if not groups:
        raise ValueError("CLoRA requires PEFT parameters named lora_A/lora_B")
    return groups


class CLoRARegularizer(nn.Module):


    OFFICIAL_REPOSITORY = "https://github.com/sutakori/CLoRA"
    OFFICIAL_SOURCE_COMMIT = "802cda88cd21e839326701ba5c2ba48cbd317be0"
    PAPER_URL = "https://aclanthology.org/2025.acl-long.940/"

    def __init__(
        self,
        model: nn.Module,
        *,
        k: int = 512,
        seed: int = 0,
        expected_target_modules: tuple[str, ...] | None = ("query", "value"),
    ):
        super().__init__()
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("CLoRA k must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("CLoRA seed must be a non-negative integer")

        self.k = k
        self.seed = seed
        self._records: list[_LoRARecord] = []
        groups = _lora_parameter_groups(model)
        target_modules = sorted(
            {module_name.rsplit(".", 1)[-1] for module_name in groups}
        )
        if expected_target_modules is not None and set(target_modules) != set(
            expected_target_modules
        ):
            raise ValueError(
                "CLoRA target-module mismatch: "
                f"found={target_modules}, expected={sorted(expected_target_modules)}"
            )
        self.target_modules = tuple(target_modules)


        parameter_devices = {
            group["a"][1].device for group in groups.values()
        } | {
            group["b"][1].device for group in groups.values()
        }
        if len(parameter_devices) != 1:
            raise ValueError("CLoRA requires all active LoRA parameters on one device")
        initialization_device = next(iter(parameter_devices))
        fork_devices = (
            [initialization_device.index or torch.cuda.current_device()]
            if initialization_device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(seed)
            if initialization_device.type == "cuda":
                torch.cuda.manual_seed(seed)
            for index, module_name in enumerate(sorted(groups)):
                lora_a = groups[module_name]["a"][1]
                lora_b = groups[module_name]["b"][1]
                if lora_a.ndim != 2 or lora_b.ndim != 2:
                    raise ValueError(
                        f"{module_name}: CLoRA expects two-dimensional LoRA weights"
                    )
                rank, input_dim = lora_a.shape
                output_dim, b_rank = lora_b.shape
                if rank != b_rank:
                    raise ValueError(
                        f"{module_name}: inconsistent LoRA ranks {rank} and {b_rank}"
                    )

                p_input = torch.empty(
                    input_dim,
                    k,
                    dtype=torch.float32,
                    device=initialization_device,
                )
                p_output = torch.empty(
                    output_dim,
                    k,
                    dtype=torch.float32,
                    device=initialization_device,
                )
                nn.init.orthogonal_(p_input)
                nn.init.orthogonal_(p_output)
                p_input = p_input.to(device=lora_a.device, dtype=lora_a.dtype)
                p_output = p_output.to(device=lora_b.device, dtype=lora_b.dtype)

                p_input_name = f"p_input_{index}"
                p_output_name = f"p_output_{index}"
                self.register_buffer(p_input_name, p_input, persistent=True)
                self.register_buffer(p_output_name, p_output, persistent=True)
                self._records.append(
                    _LoRARecord(
                        module_name=module_name,
                        lora_a=lora_a,
                        lora_b=lora_b,
                        p_input_name=p_input_name,
                        p_output_name=p_output_name,
                    )
                )

        self.initial_subspace_sha256 = self.subspace_sha256()

    @property
    def num_lora_modules(self) -> int:
        return len(self._records)

    def forward(self) -> torch.Tensor:
        loss = None
        for record in self._records:
            p_input = getattr(self, record.p_input_name)
            p_output = getattr(self, record.p_output_name)

            a_projection = record.lora_a @ p_input
            b_projection = record.lora_b.transpose(0, 1) @ p_output
            layer_loss = 0.5 * (
                a_projection.square().sum() + b_projection.square().sum()
            )
            loss = layer_loss if loss is None else loss + layer_loss
        if loss is None:
            raise RuntimeError("CLoRA regularizer has no LoRA parameter records")
        return loss

    def subspace_sha256(self) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(self.named_buffers()):
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def audit_report(self) -> dict:
        final_hash = self.subspace_sha256()
        return {
            "paper_url": self.PAPER_URL,
            "official_repository": self.OFFICIAL_REPOSITORY,
            "official_source_commit": self.OFFICIAL_SOURCE_COMMIT,
            "k": self.k,
            "seed": self.seed,
            "num_lora_modules": self.num_lora_modules,
            "target_modules": list(self.target_modules),
            "module_names": [record.module_name for record in self._records],
            "regularization_buffers": sum(
                tensor.numel() for tensor in self.buffers()
            ),
            "regularization_buffers_trainable": any(
                tensor.requires_grad for tensor in self.buffers()
            ),
            "initial_subspace_sha256": self.initial_subspace_sha256,
            "final_subspace_sha256": final_hash,
            "subspaces_unchanged": final_hash == self.initial_subspace_sha256,
        }
