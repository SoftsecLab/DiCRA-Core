import argparse
import json
import os
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional


class TimeProfiler:
    def __init__(self):
        self.rows = []
        self._active = None

    def start_task(self, task_id, task_dir):
        self._active = {
            "task_id": int(task_id),
            "task_dir": int(task_dir),
            "wake_sec": 0.0,
            "consolidation_sec": 0.0,
            "sleep_sec": 0.0,
            "alignment_sec": 0.0,
            "alignment_steps": 0,
            "alignment_feature_draws": 0,
            "eval_sec": 0.0,
            "prototype_staleness_sec": 0.0,
            "total_sec": 0.0,
        }

    @contextmanager
    def track(self, key):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            if self._active is not None:
                self._active[key] = self._active.get(key, 0.0) + elapsed

    def record_alignment(self, report):
        if self._active is None:
            return
        self._active["alignment_sec"] += float(report.get("alignment_sec", 0.0))
        self._active["alignment_steps"] += int(report.get("realized_steps", 0))
        self._active["alignment_feature_draws"] += int(
            report.get("realized_feature_draws", 0)
        )
        self._active["alignment_budget"] = dict(report)

    def finish_task(self):
        if self._active is None:
            return
        self._active["total_sec"] = (
            self._active.get("wake_sec", 0.0)
            + self._active.get("consolidation_sec", 0.0)
            + self._active.get("sleep_sec", 0.0)
            + self._active.get("eval_sec", 0.0)
        )
        self.rows.append(self._active)
        self._active = None

    @staticmethod
    def _mean(rows, key):
        vals = [row.get(key, 0.0) for row in rows]
        return statistics.mean(vals) if vals else 0.0

    @staticmethod
    def _std(rows, key):
        vals = [row.get(key, 0.0) for row in rows]
        return statistics.stdev(vals) if len(vals) > 1 else 0.0

    def summary(self):
        return {
            "schema_version": 2,
            "num_tasks": len(self.rows),
            "avg_wake_sec": self._mean(self.rows, "wake_sec"),
            "std_wake_sec": self._std(self.rows, "wake_sec"),
            "avg_consolidation_sec": self._mean(
                self.rows,
                "consolidation_sec",
            ),
            "std_consolidation_sec": self._std(
                self.rows,
                "consolidation_sec",
            ),
            "total_consolidation_sec": sum(
                row.get("consolidation_sec", 0.0) for row in self.rows
            ),
            "avg_sleep_sec": self._mean(self.rows, "sleep_sec"),
            "std_sleep_sec": self._std(self.rows, "sleep_sec"),
            "avg_alignment_sec": self._mean(self.rows, "alignment_sec"),
            "std_alignment_sec": self._std(self.rows, "alignment_sec"),
            "total_alignment_sec": sum(
                row.get("alignment_sec", 0.0) for row in self.rows
            ),
            "total_alignment_steps": sum(
                row.get("alignment_steps", 0) for row in self.rows
            ),
            "total_alignment_feature_draws": sum(
                row.get("alignment_feature_draws", 0) for row in self.rows
            ),
            "total_profiled_sec": sum(
                row.get("total_sec", 0.0) for row in self.rows
            ),
            "avg_eval_sec": self._mean(self.rows, "eval_sec"),
            "std_eval_sec": self._std(self.rows, "eval_sec"),
            "total_prototype_staleness_sec": sum(
                row.get("prototype_staleness_sec", 0.0) for row in self.rows
            ),
            "avg_total_sec": self._mean(self.rows, "total_sec"),
            "std_total_sec": self._std(self.rows, "total_sec"),
        }

    def payload(self):
        return {
            "schema_version": 2,
            "tasks": self.rows,
            "summary": self.summary(),
        }

    def write(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "time_profile.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.payload(), f, indent=4)
        return path

    def print_summary(self, output_dir=None):
        if not self.rows:
            return
        summary = self.summary()
        print("\n" + "=" * 124)
        print("Time Profile Summary")
        print("=" * 124)
        print(
            f"{'Tasks':<8} {'Wake/task':>14} {'Consol./task':>16} "
            f"{'Sleep/task':>14} "
            f"{'Align/task':>14} {'Eval/task':>14} {'Total/task':>14}"
        )
        print("-" * 124)
        print(
            f"{summary['num_tasks']:<8} "
            f"{summary['avg_wake_sec']:>11.2f}s "
            f"{summary['avg_consolidation_sec']:>13.2f}s "
            f"{summary['avg_sleep_sec']:>11.2f}s "
            f"{summary['avg_alignment_sec']:>11.2f}s "
            f"{summary['avg_eval_sec']:>11.2f}s "
            f"{summary['avg_total_sec']:>11.2f}s"
        )
        print("-" * 124)
        if output_dir is not None:
            print(f"[TimeProfile] Saved to: {os.path.join(output_dir, 'time_profile.json')}")


def count_parameters(model, trainable_only=False):
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def count_named_parameters(named_parameters: Iterable, trainable_only=False):
    total = 0
    for _, param in named_parameters:
        if trainable_only and not param.requires_grad:
            continue
        total += param.numel()
    return total


def _is_profile_trainable(name, param, trainable_param_names):
    if trainable_param_names is None:
        return param.requires_grad
    return name in trainable_param_names


def make_param_profile(model, stage, trainable_param_names: Optional[set] = None):
    total_params = count_parameters(model, trainable_only=False)
    profile_trainable_params = sum(
        param.numel()
        for name, param in model.named_parameters()
        if _is_profile_trainable(name, param, trainable_param_names)
    )
    classifier_total_params = count_named_parameters(model.classifier.named_parameters())
    classifier_trainable_params = sum(
        param.numel()
        for name, param in model.classifier.named_parameters()
        if _is_profile_trainable(f"classifier.{name}", param, trainable_param_names)
    )
    backbone_total_params = count_named_parameters(model.bert.named_parameters())
    backbone_trainable_params = sum(
        param.numel()
        for name, param in model.bert.named_parameters()
        if _is_profile_trainable(f"bert.{name}", param, trainable_param_names)
    )

    lora_trainable_params = 0
    non_lora_trainable_params = 0
    for name, param in model.named_parameters():
        if not _is_profile_trainable(name, param, trainable_param_names):
            continue
        if "lora" in name.lower():
            lora_trainable_params += param.numel()
        else:
            non_lora_trainable_params += param.numel()

    trainable_ratio = (
        profile_trainable_params / total_params * 100.0 if total_params else 0.0
    )
    backbone_trainable_ratio = (
        backbone_trainable_params / backbone_total_params * 100.0
        if backbone_total_params
        else 0.0
    )

    return {
        "stage": stage,
        "total_params": int(total_params),
        "backbone_total_params": int(backbone_total_params),
        "classifier_total_params": int(classifier_total_params),
        "trainable_params": int(profile_trainable_params),
        "trainable_ratio_percent": trainable_ratio,
        "backbone_trainable_params": int(backbone_trainable_params),
        "backbone_trainable_ratio_percent": backbone_trainable_ratio,
        "classifier_trainable_params": int(classifier_trainable_params),
        "lora_trainable_params": int(lora_trainable_params),
        "non_lora_trainable_params": int(non_lora_trainable_params),
    }


def print_param_profile(profile):
    print(
        "[ParamProfile] "
        f"{profile['stage']}: "
        f"trainable={profile['trainable_params']:,}/"
        f"{profile['total_params']:,} "
        f"({profile['trainable_ratio_percent']:.4f}%), "
        f"backbone_total={profile['backbone_total_params']:,}, "
        f"backbone_trainable={profile['backbone_trainable_params']:,}, "
        f"classifier_trainable={profile['classifier_trainable_params']:,}, "
        f"lora_trainable={profile['lora_trainable_params']:,}"
    )


def write_param_profile(output_dir, profile):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "param_profile.json")
    profiles = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
        profiles = existing if isinstance(existing, list) else [existing]

    profiles = [p for p in profiles if p.get("stage") != profile.get("stage")]
    profiles.append(profile)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=4)


def load_param_profiles(output_dir):
    path = os.path.join(output_dir, "param_profile.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        profiles = json.load(f)
    return profiles if isinstance(profiles, list) else [profiles]


def print_param_profile_summary(output_dir):
    profiles = load_param_profiles(output_dir)
    if not profiles:
        return

    print("\n" + "=" * 90)
    print("Parameter Profile Summary")
    print("=" * 90)
    print(
        f"{'Stage':<22} {'Trainable':>14} {'Ratio':>10} "
        f"{'Backbone':>14} {'LoRA':>14} {'Classifier':>14}"
    )
    print("-" * 90)
    for profile in profiles:
        print(
            f"{profile['stage']:<22} "
            f"{profile['trainable_params']:>14,} "
            f"{profile['trainable_ratio_percent']:>9.4f}% "
            f"{profile['backbone_trainable_params']:>14,} "
            f"{profile['lora_trainable_params']:>14,} "
            f"{profile['classifier_trainable_params']:>14,}"
        )
    print("-" * 90)
    print(f"[ParamProfile] Saved to: {os.path.join(output_dir, 'param_profile.json')}")


def print_resource_summary(output_dir, time_profiler=None):

    if time_profiler is not None:
        time_profiler.print_summary(output_dir)
    print_param_profile_summary(output_dir)


def _std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _mean(values):
    return statistics.mean(values) if values else 0.0


def exp_name_for_seed(template, seed):
    if "{seed}" in template:
        return template.format(seed=seed)
    return f"{template}_s{seed}"


def exp_name_for_run(template, seed, dataset=None, multi_dataset=False):
    name = template
    if dataset is not None and "{dataset}" in name:
        name = name.format(dataset=dataset, seed="{seed}")
    elif dataset is not None and multi_dataset:
        name = f"{dataset}_{name}"
    return exp_name_for_seed(name, seed)


def load_time_profile(run_dir):
    path = Path(run_dir) / "time_profile.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_runs(run_items):

    metrics = [
        "avg_wake_sec",
        "avg_consolidation_sec",
        "avg_sleep_sec",
        "avg_alignment_sec",
        "avg_eval_sec",
        "avg_total_sec",
        "total_alignment_sec",
        "total_alignment_steps",
        "total_alignment_feature_draws",
        "total_consolidation_sec",
        "total_profiled_sec",
    ]
    available = [item for item in run_items if item.get("summary")]
    summary = {
        "num_runs": len(available),
        "seeds": [item["seed"] for item in available],
        "missing": [item for item in run_items if not item.get("summary")],
    }
    for metric in metrics:
        values = [
            item["summary"][metric]
            for item in available
            if metric in item["summary"]
        ]
        summary[metric] = {
            "mean": _mean(values) if values else None,
            "std": _std(values) if values else None,
            "values": values,
        }
    return summary


def collect_time_profiles(exp_name, datasets, seeds, outputs_dir="outputs"):
    multi_dataset = len(datasets) > 1
    payload = []

    for dataset in datasets:
        run_items = []
        for seed in seeds:
            name = exp_name_for_run(
                exp_name,
                seed,
                dataset=dataset,
                multi_dataset=multi_dataset,
            )
            run_dir = Path(outputs_dir) / name
            profile = load_time_profile(run_dir)
            if profile is None:
                run_items.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "exp_name": name,
                        "run_dir": str(run_dir),
                        "summary": None,
                    }
                )
                continue
            run_items.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "exp_name": name,
                    "run_dir": str(run_dir),
                    "summary": profile.get("summary", {}),
                }
            )

        dataset_summary = summarize_runs(run_items)
        dataset_summary.update(
            {
                "dataset": dataset,
                "exp_name_template": exp_name,
                "runs": run_items,
            }
        )
        payload.append(dataset_summary)

    return payload


def _fmt_mean_std(metric):
    if metric["mean"] is None:
        return "N/A"
    return f"{metric['mean']:.2f} +/- {metric['std']:.2f}"


def print_time_profile_report(summaries):
    print("\n" + "=" * 148)
    print("Time Profile Report")
    print("=" * 148)
    print(
        f"{'Dataset':<18} {'Seeds':<12} {'Wake/task':>18} "
        f"{'Consol./task':>18} "
        f"{'Sleep/task':>18} {'Align/task':>18} {'Eval/task':>18} "
        f"{'Total/task':>18} {'Missing':>8}"
    )
    print("-" * 148)
    for item in summaries:
        missing = len(item.get("missing", []))
        print(
            f"{item['dataset']:<18} "
            f"{','.join(map(str, item['seeds'])):<12} "
            f"{_fmt_mean_std(item['avg_wake_sec']):>18} "
            f"{_fmt_mean_std(item['avg_consolidation_sec']):>18} "
            f"{_fmt_mean_std(item['avg_sleep_sec']):>18} "
            f"{_fmt_mean_std(item['avg_alignment_sec']):>18} "
            f"{_fmt_mean_std(item['avg_eval_sec']):>18} "
            f"{_fmt_mean_std(item['avg_total_sec']):>18} "
            f"{missing:>8}"
        )
    print("-" * 148)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate RECAP time_profile.json files across datasets/seeds."
    )
    parser.add_argument(
        "--exp_name",
        required=True,
        help=(
            "Experiment name template. Matches scripts/run_multiseed.py naming: "
            "with multiple datasets, outputs are {dataset}_{exp_name}_s{seed} "
            "unless {dataset}/{seed} placeholders are used."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["clinc150", "banking77", "fewrel_acl2024"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 42])
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--out", default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    summaries = collect_time_profiles(
        exp_name=args.exp_name,
        datasets=args.datasets,
        seeds=args.seeds,
        outputs_dir=args.outputs_dir,
    )
    print_time_profile_report(summaries)

    out_path = args.out
    if out_path is None:
        safe_name = args.exp_name.replace("{", "").replace("}", "").replace("/", "_")
        out_path = str(Path(args.outputs_dir) / f"time_profile_{safe_name}_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=4)
    print(f"[TimeProfile] Summary saved to: {out_path}")


if __name__ == "__main__":
    main()
