from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


DEFAULT_CANDIDATE_RATIOS = tuple(index / 10 for index in range(1, 11))


def resolve_candidate_ranks(rank: int, ratios=DEFAULT_CANDIDATE_RATIOS):


    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("rank must be a positive integer")
    values = []
    for ratio in ratios:
        ratio = float(ratio)
        if not 0.0 < ratio <= 1.0:
            raise ValueError("SLoRA candidate ratios must be in (0, 1]")
        values.append(max(1, min(rank, int(rank * ratio))))
    ranks = tuple(sorted(set(values)))
    if rank not in ranks:
        raise ValueError("SLoRA candidate ratios must include the full rank")
    return ranks


def _stable_seed(seed: int, task_id: int, module_name: str, rank: int) -> int:
    payload = f"{seed}:{task_id}:{module_name}:{rank}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _local_generator(device, seed):
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def randomized_rank_approximation(update, candidate_rank, *, seed):


    if update.ndim != 2:
        raise ValueError("LoRA update must be a matrix")
    out_features, in_features = update.shape
    if not 0 < candidate_rank <= min(out_features, in_features):
        raise ValueError("candidate rank is incompatible with the update shape")

    generator = _local_generator(update.device, seed)
    projection = torch.randn(
        in_features,
        candidate_rank,
        device=update.device,
        dtype=update.dtype,
        generator=generator,
    )
    sketch = update @ projection
    basis, _ = torch.linalg.qr(sketch, mode="reduced")
    projected = basis.transpose(0, 1) @ update
    projected_u, singular_values, right_vectors = torch.linalg.svd(
        projected,
        full_matrices=False,
    )
    left_vectors = basis @ projected_u
    approximation = (
        left_vectors[:, :candidate_rank]
        @ torch.diag(singular_values[:candidate_rank])
        @ right_vectors[:candidate_rank, :]
    )
    return approximation


def fixed_rank_subspace_similarity(update, reference_basis, rank):


    update_u, _, _ = torch.linalg.svd(update, full_matrices=False)
    update_basis = update_u[:, :rank]
    reference = reference_basis[:, :rank].to(
        device=update.device,
        dtype=update.dtype,
    )
    overlap = update_basis.transpose(0, 1) @ reference
    return float(torch.linalg.matrix_norm(overlap, ord="fro").item() ** 2)


def denoise_lora_update(
    update,
    reference_basis,
    *,
    rank,
    candidate_ranks,
    seed,
    task_id,
    module_name,
):


    scores = []
    best_score = -math.inf
    best_rank = None
    best_update = None
    for candidate_rank in candidate_ranks:
        approximation = randomized_rank_approximation(
            update,
            candidate_rank,
            seed=_stable_seed(seed, task_id, module_name, candidate_rank),
        )
        score = fixed_rank_subspace_similarity(
            approximation,
            reference_basis,
            rank,
        )
        scores.append({"rank": int(candidate_rank), "similarity": score})
        if score > best_score:
            best_score = score
            best_rank = int(candidate_rank)
            best_update = approximation

    if best_update is None:
        raise RuntimeError("SLoRA candidate search produced no update")
    return best_update, best_rank, best_score, scores


def _base_weight(module):
    if hasattr(module, "get_base_layer"):
        return module.get_base_layer().weight
    if hasattr(module, "base_layer"):
        return module.base_layer.weight
    raise TypeError("PEFT LoRA module does not expose its base-layer weight")


def iter_peft_lora_modules(model):
    for module_name, module in model.bert.named_modules():
        if not all(hasattr(module, field) for field in ("lora_A", "lora_B", "scaling")):
            continue
        for adapter_name in module.lora_A.keys():
            if adapter_name not in module.lora_B:
                raise RuntimeError(
                    f"{module_name}: missing lora_B adapter {adapter_name!r}"
                )
            yield module_name, adapter_name, module


@dataclass(frozen=True)
class ReferenceSubspace:
    module_name: str
    adapter_name: str
    rank: int
    basis: torch.Tensor


class SLoRAPreConsolidator:


    def __init__(
        self,
        model,
        *,
        candidate_ratios=DEFAULT_CANDIDATE_RATIOS,
        seed=0,
        output_dir=None,
    ):
        self.seed = int(seed)
        self.candidate_ratios = tuple(float(value) for value in candidate_ratios)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.references = {}
        self._capture_reference_subspaces(model)
        self._write_reference_audit()

    @property
    def num_modules(self):
        return len(self.references)

    @property
    def reference_memory_bytes(self):
        return sum(
            reference.basis.numel() * reference.basis.element_size()
            for reference in self.references.values()
        )

    @property
    def reference_sha256(self):
        digest = hashlib.sha256()
        for key in sorted(self.references):
            reference = self.references[key]
            digest.update(f"{key[0]}:{key[1]}:{reference.rank}".encode("utf-8"))
            digest.update(reference.basis.contiguous().numpy().tobytes())
        return digest.hexdigest()

    def _capture_reference_subspaces(self, model):
        with torch.no_grad():
            for module_name, adapter_name, module in iter_peft_lora_modules(model):
                lora_a = module.lora_A[adapter_name].weight
                rank = int(lora_a.shape[0])
                resolve_candidate_ranks(rank, self.candidate_ratios)
                weight = _base_weight(module).detach().float()
                left_vectors, _, _ = torch.linalg.svd(weight, full_matrices=False)
                basis = left_vectors[:, :rank].cpu().contiguous()
                key = (module_name, adapter_name)
                self.references[key] = ReferenceSubspace(
                    module_name=module_name,
                    adapter_name=adapter_name,
                    rank=rank,
                    basis=basis,
                )
        if not self.references:
            raise RuntimeError("SLoRA found no PEFT LoRA modules")

    def _write_reference_audit(self):
        if self.output_dir is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "method": "SLoRA-Pre",
            "reference": "frozen_pretrained_base_weights",
            "reference_dtype": "float32",
            "num_modules": self.num_modules,
            "candidate_ratios": list(self.candidate_ratios),
            "candidate_search": "official_ratio_grid_max_similarity",
            "svd_backend": "randomized_projection_qr_projected_svd",
            "similarity": "squared_frobenius_overlap_of_fixed_rank_left_bases",
            "update_definition": "effective_peft_scaling_times_BA",
            "reference_memory_bytes": self.reference_memory_bytes,
            "reference_memory_mib": self.reference_memory_bytes / (1024**2),
            "reference_sha256": self.reference_sha256,
            "modules": [
                {
                    "module_name": item.module_name,
                    "adapter_name": item.adapter_name,
                    "rank": item.rank,
                    "shape": list(item.basis.shape),
                    "dtype": str(item.basis.dtype).replace("torch.", ""),
                }
                for item in self.references.values()
            ],
        }
        (self.output_dir / "slora_reference.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    def consolidate(self, model, task_id):
        start = time.perf_counter()
        module_reports = []
        seen_keys = set()
        with torch.no_grad():
            for module_name, adapter_name, module in iter_peft_lora_modules(model):
                key = (module_name, adapter_name)
                if key not in self.references:
                    raise RuntimeError(f"SLoRA reference missing for {key}")
                seen_keys.add(key)
                reference = self.references[key]
                lora_a = module.lora_A[adapter_name].weight
                lora_b = module.lora_B[adapter_name].weight
                scaling = float(module.scaling[adapter_name])
                raw_update = (lora_b.float() @ lora_a.float()) * scaling
                candidate_ranks = resolve_candidate_ranks(
                    reference.rank,
                    self.candidate_ratios,
                )
                denoised, chosen_rank, chosen_score, scores = denoise_lora_update(
                    raw_update,
                    reference.basis,
                    rank=reference.rank,
                    candidate_ranks=candidate_ranks,
                    seed=self.seed,
                    task_id=int(task_id),
                    module_name=f"{module_name}:{adapter_name}",
                )
                raw_norm = float(torch.linalg.vector_norm(raw_update).item())
                denoised_norm = float(torch.linalg.vector_norm(denoised).item())
                _base_weight(module).add_(denoised.to(_base_weight(module).dtype))
                nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5))
                lora_b.zero_()
                module_reports.append(
                    {
                        "module_name": module_name,
                        "adapter_name": adapter_name,
                        "lora_rank": reference.rank,
                        "candidate_ranks": list(candidate_ranks),
                        "chosen_rank": chosen_rank,
                        "chosen_similarity": chosen_score,
                        "candidate_scores": scores,
                        "raw_update_norm": raw_norm,
                        "denoised_update_norm": denoised_norm,
                        "retained_norm_ratio": (
                            denoised_norm / raw_norm if raw_norm > 0 else 0.0
                        ),
                        "peft_scaling": scaling,
                    }
                )

        if seen_keys != set(self.references):
            missing = sorted(set(self.references) - seen_keys)
            raise RuntimeError(f"SLoRA modules disappeared before task {task_id}: {missing}")

        report = {
            "schema_version": 1,
            "task_id": int(task_id),
            "method": "SLoRA-Pre",
            "reference_sha256": self.reference_sha256,
            "num_modules": len(module_reports),
            "denoising_sec": time.perf_counter() - start,
            "chosen_rank_mean": sum(row["chosen_rank"] for row in module_reports)
            / len(module_reports),
            "modules": module_reports,
        }
        if self.output_dir is not None:
            path = self.output_dir / "slora_denoising.jsonl"
            mode = "w" if int(task_id) == 0 else "a"
            with path.open(mode, encoding="utf-8") as handle:
                handle.write(json.dumps(report) + "\n")
        print(
            "[SLoRA-Pre] denoised and merged "
            f"{len(module_reports)} modules, task={task_id}, "
            f"mean_rank={report['chosen_rank_mean']:.2f}, "
            f"time={report['denoising_sec']:.3f}s"
        )
        return report
