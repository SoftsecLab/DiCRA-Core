import json
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from src.dataset import JSONLDataset
from src.evaluation import bwt_from_matrix, triangular_avg


def extract_features(model, loader, device):

    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            feats = model.get_features(input_ids, attention_mask)
            features.append(feats.detach().cpu())
            labels.append(batch["labels"].detach().cpu())

    if not features:
        return None, None
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def _cap_examples_per_class(features, labels, max_examples_per_class):
    if max_examples_per_class <= 0:
        return features, labels

    keep_indices = []
    for label in torch.unique(labels).tolist():
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        keep_indices.append(indices[:max_examples_per_class])

    if not keep_indices:
        return features, labels
    keep_indices = torch.cat(keep_indices, dim=0)
    return features[keep_indices], labels[keep_indices]


def fit_linear_probe(
    features,
    labels,
    num_classes,
    device,
    epochs=30,
    lr=1e-2,
    weight_decay=1e-4,
    batch_size=512,
):

    probe = nn.Linear(features.size(1), num_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    num_examples = features.size(0)

    for _ in range(epochs):
        permutation = torch.randperm(num_examples)
        for start in range(0, num_examples, batch_size):
            indices = permutation[start : start + batch_size]
            batch_features = features[indices].to(device)
            batch_labels = labels[indices].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(probe(batch_features), batch_labels)
            loss.backward()
            optimizer.step()

    return probe


def _make_fresh_classifier_like(model, in_features, num_classes):
    current = model.classifier
    if isinstance(current, nn.Linear):
        return nn.Linear(in_features, num_classes, bias=current.bias is not None)

    if hasattr(current, "in_features") and hasattr(current, "out_features"):
        try:
            return current.__class__(in_features, num_classes)
        except TypeError:
            pass

    return nn.Linear(in_features, num_classes)


def fit_classifier_refit(
    model,
    features,
    labels,
    num_classes,
    device,
    epochs=30,
    lr=1e-2,
    weight_decay=1e-4,
    batch_size=512,
):

    classifier = _make_fresh_classifier_like(model, features.size(1), num_classes).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    num_examples = features.size(0)

    for _ in range(epochs):
        permutation = torch.randperm(num_examples)
        for start in range(0, num_examples, batch_size):
            indices = permutation[start : start + batch_size]
            batch_features = features[indices].to(device)
            batch_labels = labels[indices].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(classifier(batch_features), batch_labels)
            loss.backward()
            optimizer.step()

    return classifier


def evaluate_feature_classifier(probe, features, labels, device, batch_size=1024):
    if features is None or labels is None or features.size(0) == 0:
        return 0.0

    probe.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, features.size(0), batch_size):
            batch_features = features[start : start + batch_size].to(device)
            batch_labels = labels[start : start + batch_size].to(device)
            preds = probe(batch_features).argmax(dim=1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.numel()
    return correct / total if total else 0.0


def evaluate_current_classifier(model, features, labels, device, batch_size=1024):
    if features is None or labels is None or features.size(0) == 0:
        return 0.0

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, features.size(0), batch_size):
            batch_features = features[start : start + batch_size].to(device)
            batch_labels = labels[start : start + batch_size].to(device)
            logits = model.classifier(batch_features)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.numel()
    return correct / total if total else 0.0


def run_frozen_linear_probe(
    task_id,
    args,
    task_order,
    model,
    make_loader,
    device,
):






    train_features, train_labels, test_features_by_task, test_labels_by_task = (
        _collect_diagnostic_features(task_id, args, task_order, model, make_loader, device)
    )
    if train_features is None:
        return [0.0 for _ in range(task_id + 1)]

    train_features, train_labels = _cap_examples_per_class(
        train_features,
        train_labels,
        getattr(args, "probe_max_train_examples_per_class", 0),
    )

    probe = fit_linear_probe(
        train_features,
        train_labels,
        num_classes=args.num_classes,
        device=device,
        epochs=args.probe_epochs,
        lr=args.probe_lr,
        weight_decay=args.probe_weight_decay,
        batch_size=args.probe_batch_size,
    )

    row = []
    for eval_step in range(task_id + 1):
        acc = evaluate_feature_classifier(
            probe,
            test_features_by_task.get(eval_step),
            test_labels_by_task.get(eval_step),
            device=device,
            batch_size=args.probe_eval_batch_size,
        )
        row.append(float(acc))

    del probe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def _collect_diagnostic_features(task_id, args, task_order, model, make_loader, device):
    train_features: List[torch.Tensor] = []
    train_labels: List[torch.Tensor] = []
    test_features_by_task: Dict[int, torch.Tensor] = {}
    test_labels_by_task: Dict[int, torch.Tensor] = {}

    for eval_step in range(task_id + 1):
        task_dir = task_order[eval_step]
        train_path = os.path.join(args.data_root, f"task_{task_dir}", "train.json")
        test_path = os.path.join(args.data_root, f"task_{task_dir}", "test.json")

        if os.path.exists(train_path):
            train_loader = make_loader(
                JSONLDataset(train_path, max_len=args.max_length, encode_on_getitem=False),
                batch_size=args.eval_batch_size,
                shuffle=False,
            )
            features, labels = extract_features(model, train_loader, device)
            if features is not None:
                train_features.append(features)
                train_labels.append(labels)

        if os.path.exists(test_path):
            test_loader = make_loader(
                JSONLDataset(test_path, max_len=args.max_length, encode_on_getitem=False),
                batch_size=args.eval_batch_size,
                shuffle=False,
            )
            features, labels = extract_features(model, test_loader, device)
            if features is not None:
                test_features_by_task[eval_step] = features
                test_labels_by_task[eval_step] = labels

    if not train_features:
        return None, None, test_features_by_task, test_labels_by_task

    return (
        torch.cat(train_features, dim=0),
        torch.cat(train_labels, dim=0),
        test_features_by_task,
        test_labels_by_task,
    )


def run_classifier_oracle_refit(
    task_id,
    args,
    task_order,
    model,
    make_loader,
    device,
):






    train_features, train_labels, test_features_by_task, test_labels_by_task = (
        _collect_diagnostic_features(task_id, args, task_order, model, make_loader, device)
    )
    if train_features is None:
        return {
            "oracle_refit_final_avg": 0.0,
            "current_classifier_final_avg": 0.0,
            "oracle_repair_gain": 0.0,
            "oracle_task_accs": [0.0 for _ in range(task_id + 1)],
            "current_task_accs": [0.0 for _ in range(task_id + 1)],
        }

    train_features, train_labels = _cap_examples_per_class(
        train_features,
        train_labels,
        getattr(args, "oracle_refit_max_train_examples_per_class", 0),
    )

    oracle = fit_classifier_refit(
        model=model,
        features=train_features,
        labels=train_labels,
        num_classes=args.num_classes,
        device=device,
        epochs=args.oracle_refit_epochs,
        lr=args.oracle_refit_lr,
        weight_decay=args.oracle_refit_weight_decay,
        batch_size=args.oracle_refit_batch_size,
    )

    oracle_accs = []
    current_accs = []
    for eval_step in range(task_id + 1):
        features = test_features_by_task.get(eval_step)
        labels = test_labels_by_task.get(eval_step)
        oracle_accs.append(
            float(
                evaluate_feature_classifier(
                    oracle,
                    features,
                    labels,
                    device=device,
                    batch_size=args.oracle_refit_eval_batch_size,
                )
            )
        )
        current_accs.append(
            float(
                evaluate_current_classifier(
                    model,
                    features,
                    labels,
                    device=device,
                    batch_size=args.oracle_refit_eval_batch_size,
                )
            )
        )

    oracle_avg = float(np.mean(oracle_accs)) if oracle_accs else 0.0
    current_avg = float(np.mean(current_accs)) if current_accs else 0.0

    del oracle
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "oracle_refit_final_avg": oracle_avg,
        "current_classifier_final_avg": current_avg,
        "oracle_repair_gain": float(oracle_avg - current_avg),
        "oracle_task_accs": oracle_accs,
        "current_task_accs": current_accs,
        "note": (
            "Classifier-only oracle refit is diagnostic-only. It freezes the "
            "current encoder and retrains a fresh classifier from dataset files "
            "for post-hoc validation only."
        ),
    }


def summarize_probe_diagnostics(probe_matrix, ncm_matrix, num_tasks):
    probe_arr = np.asarray(probe_matrix, dtype=float)
    ncm_arr = np.asarray(ncm_matrix, dtype=float)

    bwt_probe = bwt_from_matrix(probe_arr, num_tasks - 1) if num_tasks > 1 else 0.0
    bwt_feat = bwt_from_matrix(ncm_arr, num_tasks - 1) if num_tasks > 1 else 0.0

    return {
        "final_probe_avg": float(np.mean(probe_arr[num_tasks - 1, :num_tasks]))
        if num_tasks > 0
        else 0.0,
        "avg_inc_probe": triangular_avg(probe_arr, num_tasks) or 0.0,
        "bwt_probe": float(bwt_probe),
        "bwt_feat": float(bwt_feat),
        "f_probe": float(-bwt_probe),
        "f_feat": float(-bwt_feat),
        "matrix_probe": probe_arr.tolist(),
        "note": (
            "Frozen linear probe is diagnostic-only. It uses dataset files for "
            "post-hoc validation and is never used by RECAP training or memory."
        ),
    }


def write_reference_diagnostics(output_dir, payload):
    path = os.path.join(output_dir, "reference_diagnostics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    return path
