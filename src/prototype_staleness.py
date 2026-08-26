from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset

from src.dataset import JSONLDataset
from src.evaluation import bwt_from_matrix
from src.memory import PrototypeMemory


def _row_keys(dataset):
    return {(str(row["text"]), int(row["label"])) for row in dataset.data}


def validate_diagnostic_splits(
    data_root,
    task_order,
    split_name="dev",
    refresh_data_root=None,
    refresh_protocol="heldout_dev",
):

    root = Path(data_root)
    refresh_root = Path(refresh_data_root) if refresh_data_root else root
    validated = []
    for task_dir in task_order:
        task_root = root / f"task_{task_dir}"
        train_path = task_root / "train.json"
        test_path = task_root / "test.json"
        refresh_path = refresh_root / f"task_{task_dir}" / f"{split_name}.json"
        paths = {"train": train_path, "refresh": refresh_path, "test": test_path}
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Prototype staleness diagnostics require train/dev/test files. "
                f"Missing: {', '.join(missing)}"
            )

        datasets = {name: JSONLDataset(path, encode_on_getitem=False) for name, path in paths.items()}
        if not datasets["refresh"].data:
            raise ValueError(f"Diagnostic split is empty: {refresh_path}")

        split_keys = {name: _row_keys(dataset) for name, dataset in datasets.items()}
        if refresh_protocol == "historical_train_reencode":
            if refresh_path.resolve() != train_path.resolve():
                raise ValueError(
                    "historical_train_reencode must read canonical train.json"
                )
        elif refresh_protocol == "heldout_dev":
            for other in ("train", "test"):
                overlap = split_keys["refresh"].intersection(split_keys[other])
                if overlap:
                    raise ValueError(
                        f"{refresh_path} overlaps {paths[other]} by "
                        f"{len(overlap)} text/label pairs"
                    )
        else:
            raise ValueError(f"Unknown prototype refresh protocol: {refresh_protocol}")

        train_labels = {int(row["label"]) for row in datasets["train"].data}
        refresh_labels = {int(row["label"]) for row in datasets["refresh"].data}
        if refresh_labels != train_labels:
            raise ValueError(
                f"Label mismatch in {task_root}: train={sorted(train_labels)}, "
                f"refresh={sorted(refresh_labels)}"
            )
        validated.append(str(refresh_path))
    return validated


def _memory_centers(memory, labels, device):
    centers = []
    for label in labels:
        if label not in memory.prototypes:
            raise KeyError(f"Prototype memory is missing observed label {label}")
        means = memory.prototypes[label].get("means")
        if means is None:
            means = memory.prototypes[label]["mean"].unsqueeze(0)
        if means.size(0) != 1:
            raise ValueError("Prototype staleness diagnostics currently require K=1")
        centers.append(means[0].to(device))
    return F.normalize(torch.stack(centers), dim=1, eps=1e-8)


def _ncm_accuracy(features, targets, centers, labels):
    if targets.numel() == 0:
        return 0.0
    normalized = F.normalize(features, dim=1, eps=1e-8)
    label_tensor = torch.tensor(labels, device=features.device, dtype=torch.long)
    predictions = label_tensor[(normalized @ centers.to(normalized.dtype).t()).argmax(dim=1)]
    return float((predictions == targets).float().mean().item())


def prototype_cosine_drift(stored_memory, refreshed_memory, labels, device):

    if not labels:
        return None
    stored = _memory_centers(stored_memory, labels, device)
    refreshed = _memory_centers(refreshed_memory, labels, device)
    return float((1.0 - (stored * refreshed).sum(dim=1)).mean().item())


def validate_stored_ncm_reproduction(diagnostic, main, sample_counts):

    diagnostic = np.asarray(diagnostic, dtype=float)
    main = np.asarray(main, dtype=float)
    sample_counts = np.asarray(sample_counts, dtype=int)
    if diagnostic.shape != main.shape or diagnostic.shape != sample_counts.shape:
        raise ValueError("Stored-NCM reproduction inputs must have matching shapes")
    if np.any(sample_counts <= 0):
        raise ValueError("Stored-NCM reproduction requires non-empty test sets")

    deltas = np.abs(diagnostic - main)
    tolerances = 1.0 / sample_counts.astype(float) + 1e-7
    failures = np.flatnonzero(deltas > tolerances)
    if failures.size:
        details = [
            {
                "task": int(index),
                "diagnostic": float(diagnostic[index]),
                "main": float(main[index]),
                "delta": float(deltas[index]),
                "allowed": float(tolerances[index]),
            }
            for index in failures
        ]
        raise RuntimeError(
            "Stored-NCM diagnostic differs from the main NCM row by more than "
            f"one prediction: {details}"
        )
    return {
        "absolute_deltas": deltas.tolist(),
        "per_task_tolerances": tolerances.tolist(),
        "max_absolute_delta": float(deltas.max()),
        "within_one_prediction_per_task": True,
    }


class PrototypeStalenessDiagnostics:


    def __init__(
        self,
        *,
        args,
        output_dir,
        task_order,
        model,
        evaluator,
        stored_memory,
        make_loader,
        device,
    ):
        if int(args.num_centroids) != 1:
            raise ValueError(
                "--run_prototype_staleness_diagnostics requires --num_centroids 1"
            )
        self.args = args
        self.output_path = Path(output_dir) / "prototype_staleness.json"
        self.task_order = list(task_order)
        self.model = model
        self.evaluator = evaluator
        self.stored_memory = stored_memory
        self.make_loader = make_loader
        self.device = device
        self.split_name = args.prototype_refresh_split
        self.refresh_protocol = args.prototype_refresh_protocol
        self.refresh_data_root = (
            args.prototype_refresh_data_root or args.data_root
        )
        self.validated_splits = validate_diagnostic_splits(
            args.data_root,
            self.task_order,
            self.split_name,
            self.refresh_data_root,
            self.refresh_protocol,
        )
        shape = (args.num_tasks, args.num_tasks)
        self.matrix_stored = np.zeros(shape, dtype=float)
        self.matrix_refreshed = np.zeros(shape, dtype=float)
        self.stages = []

    def _dataset(self, task_dir, split_name, *, data_root=None):
        path = os.path.join(
            data_root or self.args.data_root,
            f"task_{task_dir}",
            f"{split_name}.json",
        )
        return JSONLDataset(
            path,
            max_len=self.args.max_length,
            encode_on_getitem=False,
        )

    def _build_refreshed_memory(self, task_id):
        dev_datasets = [
            self._dataset(
                self.task_order[index],
                self.split_name,
                data_root=self.refresh_data_root,
            )
            for index in range(task_id + 1)
        ]
        loader = self.make_loader(
            ConcatDataset(dev_datasets),
            batch_size=self.args.eval_batch_size,
            shuffle=False,
        )
        refreshed = PrototypeMemory(
            self.args.num_classes,
            self.model.config.hidden_size,
            self.device,
            num_centroids=1,
            prototype_std_scale=self.args.prototype_std_scale,
        )
        refreshed.update_prototypes(self.model, loader, self.device)
        return refreshed

    def _extract_test_features(self, task_dir):
        loader = self.make_loader(
            self._dataset(task_dir, "test"),
            batch_size=self.args.eval_batch_size,
            shuffle=False,
        )
        feature_chunks = []
        target_chunks = []
        self.model.eval()
        self.evaluator.bind_model(self.model)
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                with self.evaluator.inference_context():
                    features = self.model.get_features(input_ids, attention_mask)
                feature_chunks.append(features.detach())
                target_chunks.append(batch["labels"].to(self.device))
        return torch.cat(feature_chunks), torch.cat(target_chunks)

    def record_stage(self, task_id, seen_labels, current_labels, expected_stored_row):
        print("[Prototype Staleness] stored vs refreshed NCM...")
        labels = sorted(int(label) for label in seen_labels)
        current_labels = {int(label) for label in current_labels}
        old_labels = [label for label in labels if label not in current_labels]
        refreshed_memory = self._build_refreshed_memory(task_id)
        if sorted(refreshed_memory.prototypes) != labels:
            raise RuntimeError(
                "Refreshed prototypes do not exactly cover observed labels: "
                f"expected={labels}, actual={sorted(refreshed_memory.prototypes)}"
            )

        stored_centers = _memory_centers(self.stored_memory, labels, self.device)
        refreshed_centers = _memory_centers(refreshed_memory, labels, self.device)
        stored_accs = []
        refreshed_accs = []
        test_sample_counts = []
        for eval_step in range(task_id + 1):
            features, targets = self._extract_test_features(self.task_order[eval_step])
            test_sample_counts.append(int(targets.numel()))
            stored_accs.append(
                _ncm_accuracy(features, targets, stored_centers, labels)
            )
            refreshed_accs.append(
                _ncm_accuracy(features, targets, refreshed_centers, labels)
            )

        self.matrix_stored[task_id, : task_id + 1] = stored_accs
        self.matrix_refreshed[task_id, : task_id + 1] = refreshed_accs
        expected = np.asarray(expected_stored_row, dtype=float)
        reproduction = validate_stored_ncm_reproduction(
            stored_accs,
            expected,
            test_sample_counts,
        )
        if reproduction["max_absolute_delta"] > 1e-7:
            print(
                "[Prototype Staleness] stored/main NCM differs by at most one "
                "boundary prediction; paired stored/refreshed features are shared."
            )

        old_stored = float(np.mean(stored_accs[:-1])) if task_id > 0 else None
        old_refreshed = float(np.mean(refreshed_accs[:-1])) if task_id > 0 else None
        old_gain = (
            float(np.mean(np.asarray(refreshed_accs[:-1]) - np.asarray(stored_accs[:-1])))
            if task_id > 0
            else None
        )
        stage = {
            "task_id": int(task_id),
            "task_dir": int(self.task_order[task_id]),
            "seen_labels": labels,
            "current_labels": sorted(current_labels),
            "stored_task_accuracies": stored_accs,
            "refreshed_task_accuracies": refreshed_accs,
            "test_sample_counts": test_sample_counts,
            "stored_main_reproduction": reproduction,
            "old_stored_ncm": old_stored,
            "old_refreshed_ncm": old_refreshed,
            "old_refresh_gain": old_gain,
            "bwt_feat_stored": bwt_from_matrix(self.matrix_stored, task_id),
            "bwt_feat_refreshed": bwt_from_matrix(self.matrix_refreshed, task_id),
            "old_prototype_cosine_drift": prototype_cosine_drift(
                self.stored_memory,
                refreshed_memory,
                old_labels,
                self.device,
            ),
        }
        self.stages.append(stage)
        self.write()
        if task_id > 0:
            print(
                f"[Prototype Staleness] old_ncm={old_stored*100:.2f}% -> "
                f"{old_refreshed*100:.2f}% (gain={old_gain*100:+.2f}%), "
                f"drift={stage['old_prototype_cosine_drift']:.4f}"
            )
        return stage

    def payload(self):
        final = self.stages[-1] if self.stages else None
        stage_gains = [
            stage["old_refresh_gain"]
            for stage in self.stages
            if stage["old_refresh_gain"] is not None
        ]
        summary = None
        if final is not None:
            summary = {
                "final_old_stored_ncm": final["old_stored_ncm"],
                "final_old_refreshed_ncm": final["old_refreshed_ncm"],
                "final_old_refresh_gain": final["old_refresh_gain"],
                "stage_equal_old_refresh_gain": (
                    float(np.mean(stage_gains)) if stage_gains else None
                ),
                "bwt_feat_stored": final["bwt_feat_stored"],
                "bwt_feat_refreshed": final["bwt_feat_refreshed"],
                "bwt_feat_refresh_delta": (
                    final["bwt_feat_refreshed"] - final["bwt_feat_stored"]
                ),
                "final_old_prototype_cosine_drift": final[
                    "old_prototype_cosine_drift"
                ],
            }
        return {
            "protocol": {
                "name": "prototype_staleness_stored_vs_refreshed",
                "checkpoint": "post_alignment",
                "encoder_scope": (
                    "stage_specific_post_alignment; fixed only within each "
                    "stored-vs-refreshed pair"
                ),
                "matrix_row_definition": (
                    "row t uses the post-alignment encoder and prototype "
                    "memories available at stage t"
                ),
                "bwt_definition": (
                    "computed across the stage-specific stored or refreshed "
                    "NCM matrix, not with one fixed final encoder"
                ),
                "prototype_count_per_class": 1,
                "refresh_split": self.split_name,
                "refresh_protocol": self.refresh_protocol,
                "refresh_data_role": (
                    "post_hoc_historical_train_reencoding"
                    if self.refresh_protocol == "historical_train_reencode"
                    else "post_hoc_heldout_diagnostic_only"
                ),
                "historical_text_access": (
                    "diagnostic_only_not_available_to_method"
                    if self.refresh_protocol == "historical_train_reencode"
                    else "none"
                ),
                "test_features_shared_within_pair": True,
                "main_ncm_reproduction_tolerance": "at most one prediction per task",
                "candidate_space": "observed_classes",
                "distance": "cosine",
                "diagnostic_data_root": str(self.args.data_root),
                "refresh_data_root": str(self.refresh_data_root),
                "training_and_test_data_root": str(self.args.data_root),
                "validated_split_files": self.validated_splits,
            },
            "task_order": self.task_order,
            "matrix_stored": self.matrix_stored.tolist(),
            "matrix_refreshed": self.matrix_refreshed.tolist(),
            "stages": self.stages,
            "summary": summary,
        }

    def write(self):
        with self.output_path.open("w", encoding="utf-8") as handle:
            json.dump(self.payload(), handle, indent=2)
