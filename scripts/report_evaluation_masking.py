#!/usr/bin/env python3
"""Aggregate evaluation-only seen-class masking controls across seeds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import summarize_evaluation_masking
from src.metrics import aggregate_seeds, summarize_matrix


METHOD_GROUPS = (
    ("sequential_lora", "Sequential LoRA"),
    ("seq_lora_matched_alignment", "SeqLoRA + Align"),
    ("olora", "O-LoRA"),
    ("clora", "CLoRA"),
    ("slora_pre", "SLoRA-Pre"),
    ("recap_wo_refinements", "RECAP w/o Refinements"),
    ("recap", "RECAP"),
)
METRICS = (
    "final_avg_global",
    "final_avg_seen",
    "delta_mask_final",
    "avg_inc_global",
    "avg_inc_seen",
    "delta_mask_avg_inc",
    "bwt_global",
    "bwt_seen",
    "pred_future_pre_final",
    "pred_future_final",
)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path):
    return _sha256_bytes(path.read_bytes())


def _matrix_sha256(matrix):
    canonical = json.dumps(
        matrix,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return _sha256_bytes(canonical)


def _require_close(actual, expected, location):
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        raise ValueError(f"{location} must be a real number")
    if not math.isclose(
        float(actual),
        float(expected),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"{location}={actual!r} disagrees with matrix-derived "
            f"value {expected!r}"
        )


def _resolve_run(path: Path):
    result_path = path if path.name == "results.json" else path / "results.json"
    if not result_path.exists():
        raise FileNotFoundError(f"missing results.json: {result_path}")
    run_dir = result_path.parent
    config_path = run_dir / "config.json"
    return (
        run_dir,
        _load_json(result_path),
        _load_json(config_path) if config_path.exists() else {},
    )


def _dataset_id(result, config):
    value = result.get("dataset") or config.get("dataset")
    if value is None and config.get("data_root"):
        value = Path(str(config["data_root"])).name
    if not value:
        raise ValueError("dataset must be explicit in results/config data")
    value = str(value)
    aliases = {
        "fewrel_acl2024": "fewrel",
        "CLINC150": "clinc150",
        "Banking77": "banking77",
        "FewRel": "fewrel",
    }
    return aliases.get(value, value)


def _seed(result, config):
    values = [source["seed"] for source in (result, config) if "seed" in source]
    if not values:
        raise ValueError("seed must be explicit in results or config")
    normalized = [int(value) for value in values]
    if len(set(normalized)) != 1:
        raise ValueError(f"results/config seed mismatch: {normalized}")
    return normalized[0]


def _require_fixed_global(result, config):
    protocol = result.get("classifier_protocol", config.get("classifier_protocol"))
    if protocol is not None and protocol != "fixed_global":
        raise ValueError(f"expected fixed_global classifier protocol, got {protocol!r}")
    if result.get("legacy_global_fields_are_dynamic_aliases"):
        raise ValueError("dynamic-seen result cannot be used for this control")


def _masking_summary(result, *, allow_missing_pred_future):
    global_matrix = result.get("matrix")
    seen_matrix = result.get("matrix_seen")
    if global_matrix is None or seen_matrix is None:
        raise ValueError("results must contain matrix and matrix_seen")
    num_tasks = len(result.get("task_order", [])) or len(global_matrix)
    future = result.get("matrix_pred_future_count")
    samples = result.get("matrix_eval_sample_count")
    if (future is None or samples is None) and not allow_missing_pred_future:
        raise ValueError(
            "PredFuture is unavailable: this run predates future-prediction "
            "instrumentation. Accuracy differences cannot reconstruct it."
        )
    if future is None or samples is None:
        future = samples = None
    summary = summarize_evaluation_masking(
        global_matrix,
        seen_matrix,
        num_tasks=num_tasks,
        pred_future_count_matrix=future,
        eval_sample_count_matrix=samples,
    )
    if abs(summary["delta_mask_final"]) > 1e-10:
        raise ValueError(
            "final global and seen averages differ; the final stage does not "
            "appear to cover the complete fixed-global output space"
        )
    return summary


def _audit_global_source(result, masking, *, strict):
    matrix = result["matrix"]
    num_tasks = len(result.get("task_order", [])) or len(matrix)
    global_metrics = summarize_matrix(matrix, num_tasks)
    _require_close(
        masking["final_avg_global"],
        global_metrics["final_avg"],
        "masking.final_avg_global",
    )
    _require_close(
        masking["avg_inc_global"],
        global_metrics["avg_inc"],
        "masking.avg_inc_global",
    )
    _require_close(
        masking["bwt_global"],
        global_metrics["bwt"],
        "masking.bwt_global",
    )
    if strict:
        stored = {
            "results.final_avg": result.get("final_avg"),
            "results.avg_inc": result.get("avg_inc"),
            "results.bwt_global": result.get(
                "bwt_global",
                result.get("bwt"),
            ),
        }
        expected = {
            "results.final_avg": global_metrics["final_avg"],
            "results.avg_inc": global_metrics["avg_inc"],
            "results.bwt_global": global_metrics["bwt"],
        }
        for location, value in stored.items():
            if value is None:
                raise ValueError(f"{location} is required in strict mode")
            _require_close(value, expected[location], location)

        stored_masking = result.get("evaluation_only_seen_masking")
        if not isinstance(stored_masking, dict):
            raise ValueError(
                "evaluation_only_seen_masking is required in strict mode"
            )
        for key in (
            "final_avg_global",
            "avg_inc_global",
            "avg_inc_seen",
            "delta_mask_avg_inc",
            "bwt_seen",
        ):
            if key not in stored_masking:
                raise ValueError(
                    f"evaluation_only_seen_masking.{key} is required "
                    "in strict mode"
                )
            _require_close(
                stored_masking[key],
                masking[key],
                f"evaluation_only_seen_masking.{key}",
            )
        if "bwt_global" in stored_masking:
            _require_close(
                stored_masking["bwt_global"],
                masking["bwt_global"],
                "evaluation_only_seen_masking.bwt_global",
            )
    return global_metrics


def collect(args):
    rows = []
    seen_keys = set()
    for argument, method in METHOD_GROUPS:
        for path in getattr(args, argument, ()):
            run_dir, result, config = _resolve_run(path)
            _require_fixed_global(result, config)
            dataset = _dataset_id(result, config)
            seed = _seed(result, config)
            key = (dataset, method, seed)
            if key in seen_keys:
                raise ValueError(f"duplicate run for {dataset}/{method}/seed={seed}")
            seen_keys.add(key)
            try:
                summary = _masking_summary(
                    result,
                    allow_missing_pred_future=args.allow_missing_pred_future,
                )
                global_metrics = _audit_global_source(
                    result,
                    summary,
                    strict=getattr(args, "strict_source_consistency", False),
                )
            except Exception as exc:
                raise ValueError(
                    f"{run_dir} ({method}, dataset={dataset}, seed={seed}): {exc}"
                ) from exc
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "metrics": summary,
                    "global_metrics": global_metrics,
                    "results_sha256": _sha256_file(run_dir / "results.json"),
                    "global_matrix_sha256": _matrix_sha256(result["matrix"]),
                }
            )
    return rows


def audit_inputs(args):
    """Print artifact coverage without attempting metric aggregation."""

    failures = 0
    print("\nEvaluation Masking Artifact Audit")
    print(
        f"{'Method':<24} {'Run directory':<62} "
        f"{'matrix':>7} {'seen':>7} {'future':>7} {'samples':>8}"
    )
    print("-" * 122)
    for argument, method in METHOD_GROUPS:
        for path in getattr(args, argument, ()):
            try:
                run_dir, result, _config = _resolve_run(path)
            except Exception as exc:
                failures += 1
                print(f"{method:<24} {str(path):<62} ERROR: {exc}")
                continue
            availability = {
                "matrix": result.get("matrix") is not None,
                "seen": result.get("matrix_seen") is not None,
                "future": result.get("matrix_pred_future_count") is not None,
                "samples": result.get("matrix_eval_sample_count") is not None,
            }
            if not all(availability.values()):
                failures += 1
            marks = {key: ("yes" if value else "NO") for key, value in availability.items()}
            print(
                f"{method:<24} {str(run_dir):<62} "
                f"{marks['matrix']:>7} {marks['seen']:>7} "
                f"{marks['future']:>7} {marks['samples']:>8}"
            )
    print(f"\nArtifact sets with missing fields: {failures}")
    return failures


def aggregate(rows, expected_seeds, allow_incomplete):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["method"]), []).append(row)
    output = []
    expected = set(expected_seeds)
    for (dataset, method), group in sorted(grouped.items()):
        seeds = {row["seed"] for row in group}
        if seeds != expected and not allow_incomplete:
            raise ValueError(
                f"{dataset}/{method}: seeds={sorted(seeds)}, "
                f"expected={sorted(expected)}"
            )
        metrics = {}
        for metric in METRICS:
            values = [row["metrics"][metric] for row in group]
            if any(value is None for value in values):
                metrics[metric] = None
            else:
                metrics[metric] = aggregate_seeds(values)
        output.append(
            {
                "dataset": dataset,
                "method": method,
                "seeds": sorted(seeds),
                "runs": len(group),
                "metrics": metrics,
                "run_dirs": [row["run_dir"] for row in group],
                "results_sha256": [row["results_sha256"] for row in group],
                "global_matrix_sha256": [
                    row["global_matrix_sha256"] for row in group
                ],
            }
        )
    return output


def _formatted(metric):
    if metric is None:
        return "N/A"
    return f"{metric['mean'] * 100:.2f} +/- {metric['std'] * 100:.2f}"


def print_report(rows):
    print("\nEvaluation-only Seen-class Masking Control")
    header = (
        f"{'Dataset':<12} {'Method':<24} {'Runs':>4} "
        f"{'Final global':<18} {'Final seen':<18} {'dFinal':<18} "
        f"{'AvgInc global':<18} {'AvgInc seen':<18} {'dAvgInc':<18} "
        f"{'BWT_seen':<18} {'PredFuture*':<18}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        metric = row["metrics"]
        print(
            f"{row['dataset']:<12} {row['method']:<24} {row['runs']:>4} "
            f"{_formatted(metric['final_avg_global']):<18} "
            f"{_formatted(metric['final_avg_seen']):<18} "
            f"{_formatted(metric['delta_mask_final']):<18} "
            f"{_formatted(metric['avg_inc_global']):<18} "
            f"{_formatted(metric['avg_inc_seen']):<18} "
            f"{_formatted(metric['delta_mask_avg_inc']):<18} "
            f"{_formatted(metric['bwt_seen']):<18} "
            f"{_formatted(metric['pred_future_pre_final']):<18}"
        )
    print("\n* PredFuture pools samples over stages before the final all-class stage.")
    print("  Final seen/global values are equal by construction when all classes are observed.")


def write_outputs(rows, json_path: Path, tsv_path: Path):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        columns = ["dataset", "method", "runs", "seeds"]
        for metric in METRICS:
            columns.extend([f"{metric}_mean", f"{metric}_std"])
        writer.writerow(columns)
        for row in rows:
            values = [
                row["dataset"],
                row["method"],
                row["runs"],
                ",".join(str(seed) for seed in row["seeds"]),
            ]
            for metric in METRICS:
                aggregate = row["metrics"][metric]
                values.extend(
                    [None, None]
                    if aggregate is None
                    else [aggregate["mean"], aggregate["std"]]
                )
            writer.writerow(values)


def global_rows(masking_rows):
    rows = []
    for row in masking_rows:
        metrics = row["metrics"]
        rows.append(
            {
                "dataset": row["dataset"],
                "method": row["method"],
                "runs": row["runs"],
                "seeds": row["seeds"],
                "metrics": {
                    "final_avg": metrics["final_avg_global"],
                    "avg_inc": metrics["avg_inc_global"],
                    "bwt_global": metrics["bwt_global"],
                },
                "run_dirs": row["run_dirs"],
                "results_sha256": row["results_sha256"],
                "global_matrix_sha256": row["global_matrix_sha256"],
            }
        )
    return rows


def write_global_outputs(rows, json_path: Path, tsv_path: Path):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "dataset",
                "method",
                "runs",
                "seeds",
                "final_avg_mean",
                "final_avg_std",
                "avg_inc_mean",
                "avg_inc_std",
                "bwt_global_mean",
                "bwt_global_std",
            ]
        )
        for row in rows:
            metrics = row["metrics"]
            writer.writerow(
                [
                    row["dataset"],
                    row["method"],
                    row["runs"],
                    ",".join(str(seed) for seed in row["seeds"]),
                    metrics["final_avg"]["mean"],
                    metrics["final_avg"]["std"],
                    metrics["avg_inc"]["mean"],
                    metrics["avg_inc"]["std"],
                    metrics["bwt_global"]["mean"],
                    metrics["bwt_global"]["std"],
                ]
            )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument, method in METHOD_GROUPS:
        parser.add_argument(
            f"--{argument.replace('_', '-')}",
            nargs="+",
            type=Path,
            default=[],
            help=f"Run directories for {method}.",
        )
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[0, 1, 42])
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--allow-missing-pred-future", action="store_true")
    parser.add_argument(
        "--strict-source-consistency",
        action="store_true",
        help=(
            "Require stored main and masking metrics to exactly match the "
            "single global matrix used for both exports."
        ),
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Only report artifact-field coverage; do not aggregate metrics.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/evaluation_masking_summary.json"),
    )
    parser.add_argument(
        "--tsv",
        type=Path,
        default=Path("outputs/evaluation_masking_summary.tsv"),
    )
    parser.add_argument(
        "--global-out",
        type=Path,
        default=Path("outputs/table2_global_from_masking_runs.json"),
    )
    parser.add_argument(
        "--global-tsv",
        type=Path,
        default=Path("outputs/table2_global_from_masking_runs.tsv"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.audit_only:
        raise SystemExit(1 if audit_inputs(args) else 0)
    rows = collect(args)
    if not rows:
        raise SystemExit("No run directories were provided")
    summary = aggregate(rows, args.expected_seeds, args.allow_incomplete)
    print_report(summary)
    write_outputs(summary, args.out, args.tsv)
    table2_rows = global_rows(summary)
    write_global_outputs(
        table2_rows,
        args.global_out,
        args.global_tsv,
    )
    print(f"\nSaved JSON: {args.out}")
    print(f"Saved TSV:  {args.tsv}")
    print(f"Saved global JSON: {args.global_out}")
    print(f"Saved global TSV:  {args.global_tsv}")


if __name__ == "__main__":
    main()
