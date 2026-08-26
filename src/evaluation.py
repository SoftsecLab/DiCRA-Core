import os

import numpy as np
import torch

from src.dataset import JSONLDataset


def evaluate_learned_tasks(
    task_id,
    args,
    task_order,
    model,
    trainer,
    prototype_memory,
    make_loader,
    test_matrix=None,
    seen_matrix=None,
    ncm_matrix=None,
    seen_labels=None,
    pred_future_count_matrix=None,
    eval_sample_count_matrix=None,
):

    seen_count = len(seen_labels) if seen_labels is not None else 0
    print(
        f"Evaluating learned tasks (0 -> {task_id}); "
        f"candidate_space=global:{args.num_classes}, seen:{seen_count}..."
    )
    if seen_matrix is not None and not seen_labels:
        raise ValueError("seen_labels are required when seen_matrix is provided")
    test_accs = []
    ncm_accs = []
    train_accs = []
    model.eval()

    with torch.no_grad():
        for eval_step in range(task_id + 1):
            eval_dir = task_order[eval_step]
            test_path = os.path.join(args.data_root, f"task_{eval_dir}", "test.json")
            eval_train_path = os.path.join(args.data_root, f"task_{eval_dir}", "train.json")

            if not os.path.exists(test_path):
                test_accs.append(0.0)
                ncm_accs.append(0.0)
                train_accs.append(0.0)
                continue

            test_loader = make_loader(
                JSONLDataset(test_path, max_len=args.max_length, encode_on_getitem=False),
                batch_size=args.eval_batch_size,
                shuffle=False,
            )
            if seen_matrix is not None:
                if (
                    pred_future_count_matrix is not None
                    or eval_sample_count_matrix is not None
                ):
                    audit = trainer.evaluate_global_seen_and_future(
                        test_loader,
                        seen_labels,
                    )
                    acc_test = audit["global_accuracy"]
                    acc_seen = audit["seen_accuracy"]
                    if pred_future_count_matrix is not None:
                        pred_future_count_matrix[task_id, eval_step] = int(
                            audit["pred_future_count"]
                        )
                    if eval_sample_count_matrix is not None:
                        eval_sample_count_matrix[task_id, eval_step] = int(
                            audit["sample_count"]
                        )
                else:
                    acc_test, acc_seen = trainer.evaluate_global_and_seen(
                        test_loader,
                        seen_labels,
                    )
                seen_matrix[task_id, eval_step] = acc_seen
            else:
                acc_test = trainer.evaluate(test_loader)
            test_accs.append(acc_test)
            if test_matrix is not None:
                test_matrix[task_id, eval_step] = acc_test

            acc_ncm = trainer.evaluate_ncm(test_loader, prototype_memory)
            ncm_accs.append(acc_ncm)
            if ncm_matrix is not None:
                ncm_matrix[task_id, eval_step] = acc_ncm

            if getattr(args, "evaluate_train_accuracy", True) and os.path.exists(
                eval_train_path
            ):
                train_loader_eval = make_loader(
                    JSONLDataset(
                        eval_train_path,
                        max_len=args.max_length,
                        encode_on_getitem=False,
                    ),
                    batch_size=args.eval_batch_size,
                    shuffle=False,
                )
                acc_train = trainer.evaluate(train_loader_eval)
                train_accs.append(acc_train)
            else:
                train_accs.append(0.0)

    return test_accs, ncm_accs, train_accs


def mean_or_zero(values):
    return float(np.mean(values)) if values else 0.0


def bwt_from_matrix(matrix, task_id):
    if task_id <= 0:
        return 0.0
    return float(np.mean([matrix[task_id, i] - matrix[i, i] for i in range(task_id)]))


def triangular_avg(matrix, num_tasks=None):
    if matrix is None:
        return None
    arr = np.asarray(matrix, dtype=float)
    if arr.size == 0:
        return None
    if num_tasks is None:
        num_tasks = min(arr.shape[0], arr.shape[1])
    if num_tasks <= 0:
        return None
    return float(np.mean([np.mean(arr[t, : t + 1]) for t in range(num_tasks)]))


def summarize_evaluation_masking(
    global_matrix,
    seen_matrix,
    *,
    num_tasks=None,
    pred_future_count_matrix=None,
    eval_sample_count_matrix=None,
):


    global_arr = np.asarray(global_matrix, dtype=float)
    seen_arr = np.asarray(seen_matrix, dtype=float)
    if global_arr.shape != seen_arr.shape or global_arr.ndim != 2:
        raise ValueError("global and seen matrices must be equally shaped 2D arrays")
    if num_tasks is None:
        num_tasks = min(global_arr.shape)
    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive")
    if global_arr.shape[0] < num_tasks or global_arr.shape[1] < num_tasks:
        raise ValueError("metric matrices are smaller than num_tasks")
    for stage in range(num_tasks):
        if np.any(
            seen_arr[stage, : stage + 1]
            < global_arr[stage, : stage + 1] - 1e-12
        ):
            raise ValueError("seen masking cannot reduce accuracy on seen-class data")

    final_idx = num_tasks - 1
    final_global = float(np.mean(global_arr[final_idx, :num_tasks]))
    final_seen = float(np.mean(seen_arr[final_idx, :num_tasks]))
    avg_inc_global = triangular_avg(global_arr, num_tasks) or 0.0
    avg_inc_seen = triangular_avg(seen_arr, num_tasks) or 0.0
    summary = {
        "final_avg_global": final_global,
        "final_avg_seen": final_seen,
        "delta_mask_final": final_seen - final_global,
        "avg_inc_global": avg_inc_global,
        "avg_inc_seen": avg_inc_seen,
        "delta_mask_avg_inc": avg_inc_seen - avg_inc_global,
        "bwt_global": bwt_from_matrix(global_arr, final_idx),
        "bwt_seen": bwt_from_matrix(seen_arr, final_idx),
        "pred_future_pre_final": None,
        "pred_future_final": None,
        "pred_future_by_stage": None,
        "pred_future_scope": (
            "sample-pooled over all seen-class test evaluations before the "
            "final all-class stage"
        ),
        "final_delta_is_structurally_zero_under_full_coverage": True,
    }

    if pred_future_count_matrix is None and eval_sample_count_matrix is None:
        return summary
    if pred_future_count_matrix is None or eval_sample_count_matrix is None:
        raise ValueError(
            "future prediction counts and evaluation sample counts must be provided together"
        )

    future = np.asarray(pred_future_count_matrix, dtype=float)
    samples = np.asarray(eval_sample_count_matrix, dtype=float)
    if future.shape != global_arr.shape or samples.shape != global_arr.shape:
        raise ValueError("future-count matrices must match the accuracy matrices")
    if not np.all(np.isfinite(future)) or not np.all(np.isfinite(samples)):
        raise ValueError("future prediction and sample counts must be finite")
    if np.any(future != np.floor(future)) or np.any(samples != np.floor(samples)):
        raise ValueError("future prediction and sample counts must be integers")
    if np.any(future < 0) or np.any(samples < 0) or np.any(future > samples):
        raise ValueError("future prediction counts must lie between zero and sample counts")
    future = future.astype(np.int64)
    samples = samples.astype(np.int64)

    rate_matrix = np.zeros(global_arr.shape, dtype=float)
    np.divide(future, samples, out=rate_matrix, where=samples > 0)
    stages = []
    for stage in range(num_tasks):
        stage_future = int(np.sum(future[stage, : stage + 1]))
        stage_samples = int(np.sum(samples[stage, : stage + 1]))
        stage_global = float(np.mean(global_arr[stage, : stage + 1]))
        stage_seen = float(np.mean(seen_arr[stage, : stage + 1]))
        stages.append(
            {
                "stage": stage,
                "global_avg": stage_global,
                "seen_avg": stage_seen,
                "delta_mask": stage_seen - stage_global,
                "pred_future_count": stage_future,
                "sample_count": stage_samples,
                "pred_future_rate": (
                    stage_future / stage_samples if stage_samples else 0.0
                ),
            }
        )

    pre_final_future = sum(row["pred_future_count"] for row in stages[:-1])
    pre_final_samples = sum(row["sample_count"] for row in stages[:-1])
    summary.update(
        {
            "pred_future_pre_final": (
                pre_final_future / pre_final_samples
                if pre_final_samples
                else 0.0
            ),
            "pred_future_final": stages[-1]["pred_future_rate"],
            "pred_future_by_stage": stages,
            "matrix_pred_future_rate": rate_matrix.tolist(),
        }
    )
    return summary


def summarize_task_metrics(
    task_id,
    task_order,
    test_accs,
    ncm_accs,
    train_accs,
    test_matrix,
    seen_matrix,
    ncm_matrix,
):
    bwt_global = bwt_from_matrix(test_matrix, task_id)
    bwt_seen = bwt_from_matrix(seen_matrix, task_id)
    bwt_features = bwt_from_matrix(ncm_matrix, task_id)
    bwt_classifier = bwt_seen - bwt_features
    seen_accs = np.asarray(seen_matrix[task_id, : task_id + 1], dtype=float)
    return {
        "task_id": task_id,
        "task_dir": task_order[task_id],
        "avg_test": mean_or_zero(test_accs),
        "avg_seen": float(np.mean(seen_accs)) if seen_accs.size else 0.0,
        "avg_ncm": mean_or_zero(ncm_accs),
        "avg_train": mean_or_zero(train_accs),
        "current_task_test": float(test_accs[-1]) if test_accs else 0.0,
        "current_task_seen": float(seen_accs[-1]) if seen_accs.size else 0.0,
        "current_task_ncm": float(ncm_accs[-1]) if ncm_accs else 0.0,
        "current_task_train": float(train_accs[-1]) if train_accs else 0.0,
        "old_task_test_avg": float(np.mean(test_accs[:-1])) if task_id > 0 else None,
        "old_task_seen_avg": float(np.mean(seen_accs[:-1])) if task_id > 0 else None,
        "old_task_ncm_avg": float(np.mean(ncm_accs[:-1])) if task_id > 0 else None,
        "old_task_train_avg": float(np.mean(train_accs[:-1])) if task_id > 0 else None,
        "bwt": float(bwt_global),
        "bwt_global": float(bwt_global),
        "bwt_seen": float(bwt_seen),
        "bwt_features": float(bwt_features),
        "bwt_classifier": float(bwt_classifier),
        "test_accs": [float(x) for x in test_accs],
        "ncm_accs": [float(x) for x in ncm_accs],
        "train_accs": [float(x) for x in train_accs],
    }


def summarize_final_results(
    test_matrix,
    seen_matrix,
    ncm_matrix,
    task_order,
    num_tasks=None,
):
    test_arr = np.asarray(test_matrix, dtype=float)
    seen_arr = np.asarray(seen_matrix, dtype=float)
    ncm_arr = np.asarray(ncm_matrix, dtype=float)
    if num_tasks is None:
        num_tasks = min(test_arr.shape[0], test_arr.shape[1]) if test_arr.size else 0

    if num_tasks <= 0:
        return {
            "final_avg": 0.0,
            "final_avg_ncm": 0.0,
            "avg_inc": 0.0,
            "final_avg_seen": 0.0,
            "avg_inc_seen": 0.0,
            "avg_inc_ncm": 0.0,
            "bwt": 0.0,
            "bwt_global": 0.0,
            "bwt_seen": 0.0,
            "bwt_features": 0.0,
            "bwt_classifier": 0.0,
            "matrix": test_arr.tolist(),
            "matrix_seen": seen_arr.tolist(),
            "matrix_ncm": ncm_arr.tolist(),
            "task_order": task_order,
        }

    final_idx = num_tasks - 1
    final_avg = float(np.mean(test_arr[final_idx, :num_tasks]))
    final_avg_seen = float(np.mean(seen_arr[final_idx, :num_tasks]))
    final_avg_ncm = float(np.mean(ncm_arr[final_idx, :num_tasks]))
    avg_inc = triangular_avg(test_arr, num_tasks) or 0.0
    avg_inc_seen = triangular_avg(seen_arr, num_tasks) or 0.0
    avg_inc_ncm = triangular_avg(ncm_arr, num_tasks) or 0.0
    bwt_global = (
        float(np.mean([test_arr[final_idx, i] - test_arr[i, i] for i in range(num_tasks - 1)]))
        if num_tasks > 1
        else 0.0
    )
    bwt_seen = (
        float(np.mean([seen_arr[final_idx, i] - seen_arr[i, i] for i in range(num_tasks - 1)]))
        if num_tasks > 1
        else 0.0
    )
    bwt_features = (
        float(np.mean([ncm_arr[final_idx, i] - ncm_arr[i, i] for i in range(num_tasks - 1)]))
        if num_tasks > 1
        else 0.0
    )
    bwt_classifier = bwt_seen - bwt_features

    return {
        "final_avg": final_avg,
        "final_avg_seen": final_avg_seen,
        "final_avg_ncm": final_avg_ncm,
        "avg_inc": avg_inc,
        "avg_inc_seen": avg_inc_seen,
        "avg_inc_ncm": avg_inc_ncm,

        "bwt": bwt_global,
        "bwt_global": bwt_global,
        "bwt_seen": bwt_seen,
        "bwt_features": bwt_features,
        "bwt_classifier": bwt_classifier,
        "matrix": test_arr.tolist(),
        "matrix_seen": seen_arr.tolist(),
        "matrix_ncm": ncm_arr.tolist(),
        "task_order": task_order,
    }
