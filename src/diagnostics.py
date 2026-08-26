import csv
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.dataset import JSONLDataset
from src.reproducibility import preserve_rng_state


class StageDiagnostics:


    stage_display = {
        "post_wake": "post-wake / pre-merge",
        "post_merge": "post-merge / pre-NREM",
        "before_rem": "post-NREM / before-REM",
        "post_rem": "after-REM",
    }

    def __init__(
        self,
        args,
        output_dir,
        task_order,
        get_model,
        prototype_memory,
        evaluate_learned_tasks,
        summarize_metrics,
        make_loader=None,
    ):
        self.args = args
        self.output_dir = output_dir
        self.task_order = task_order
        self.get_model = get_model
        self.prototype_memory = prototype_memory
        self.evaluate_learned_tasks = evaluate_learned_tasks
        self.summarize_metrics = summarize_metrics
        self.make_loader = make_loader
        self.task_label_cache = {}
        self.rem_diagnostics_rows = []
        self.rem_diag_path = os.path.join(output_dir, "rem_diagnostics.csv")

        self.tracked_stages = self._resolve_tracked_stages()
        self.process_analysis = {
            stage: {
                "matrix": np.zeros((args.num_tasks, args.num_tasks)),
                "matrix_seen": np.zeros((args.num_tasks, args.num_tasks)),
                "matrix_ncm": np.zeros((args.num_tasks, args.num_tasks)),
                "steps": [],
            }
            for stage in self.tracked_stages
        }

        if args.log_rem_diagnostics and os.path.exists(self.rem_diag_path):
            os.remove(self.rem_diag_path)

    def _resolve_tracked_stages(self):
        if self.args.analyze_stages:
            return (
                ["post_wake", "post_merge", "before_rem", "post_rem"]
                if self.args.use_sleep
                else ["post_wake"]
            )
        if self.args.log_rem_diagnostics and self.args.use_sleep:
            return ["before_rem", "post_rem"]
        return []

    def has_stage(self, stage_name):
        return stage_name in self.process_analysis

    def get_task_labels_by_step(self, step):
        if step in self.task_label_cache:
            return self.task_label_cache[step]

        task_dir = self.task_order[step]
        train_path = os.path.join(
            self.args.data_root,
            f"task_{task_dir}",
            "train.json",
        )
        labels = set()
        if os.path.exists(train_path):
            with open(train_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        labels.add(int(json.loads(line)["label"]))

        self.task_label_cache[step] = labels
        return labels

    def labels_until_step(self, step):
        labels = set()
        for idx in range(step + 1):
            labels.update(self.get_task_labels_by_step(idx))
        return labels

    def old_sample_classifier_diagnostics(self, task_id):

        empty = {
            "old_sample_seen_acc": None,
            "align_pos_old": None,
            "align_margin_old": None,
            "classifier_weight_norm_old": None,
            "old_pred_old_rate": None,
            "old_pred_current_task_rate": None,
            "old_pred_seen_rate": None,
            "old_pred_future_unseen_rate": None,
            "old_routing_samples": 0,
            "eval_candidate_mode": "seen_classes",
            "alignment_scope": "old_samples",
        }
        if task_id <= 0 or self.make_loader is None:
            return empty

        model = self.get_model()
        classifier = model.classifier
        if isinstance(classifier, torch.nn.Sequential):
            classifier = classifier[0]
        if not hasattr(classifier, "weight"):
            return empty

        old_labels = set(self.labels_until_step(task_id - 1))
        current_labels = set(self.get_task_labels_by_step(task_id))
        seen_labels = old_labels | current_labels
        if not old_labels:
            return empty

        weights = classifier.weight.detach()
        classifier_class_ids = list(
            getattr(model, "class_ids", range(weights.shape[0]))
        )
        class_to_row = {
            int(label): row for row, label in enumerate(classifier_class_ids)
        }
        seen_labels = sorted(label for label in seen_labels if label in class_to_row)
        if len(seen_labels) < 2:
            return empty

        total = 0
        correct = 0
        pred_old = 0
        pred_current = 0
        pred_seen = 0
        pred_future = 0
        pos_sum = 0.0
        margin_sum = 0.0
        device = next(model.parameters()).device
        seen_row_tensor = torch.tensor(
            [class_to_row[label] for label in seen_labels],
            device=device,
            dtype=torch.long,
        )
        seen_label_tensor = torch.tensor(seen_labels, device=device, dtype=torch.long)
        seen_label_to_col = {label: idx for idx, label in enumerate(seen_labels)}
        weights_norm = F.normalize(weights[seen_row_tensor], dim=1, eps=1e-8)
        old_weight_norm = weights[
            torch.tensor(
                [class_to_row[label] for label in sorted(old_labels)],
                device=device,
                dtype=torch.long,
            )
        ].norm(dim=1).mean().item()
        model.eval()

        with torch.no_grad():
            for eval_step in range(task_id):
                eval_dir = self.task_order[eval_step]
                test_path = os.path.join(
                    self.args.data_root,
                    f"task_{eval_dir}",
                    "test.json",
                )
                if not os.path.exists(test_path):
                    continue

                loader = self.make_loader(
                    JSONLDataset(
                        test_path,
                        max_len=self.args.max_length,
                        encode_on_getitem=False,
                    ),
                    batch_size=self.args.eval_batch_size,
                    shuffle=False,
                )
                for batch in loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    features = model.get_features(input_ids, attention_mask)
                    logits = model.classifier(features)
                    seen_logits = logits.index_select(1, seen_row_tensor)
                    pred_cols = seen_logits.argmax(dim=1)
                    preds = seen_label_tensor[pred_cols]

                    correct += int((preds == labels).sum().item())
                    pred_old += sum(int(pred) in old_labels for pred in preds.tolist())
                    pred_current += sum(
                        int(pred) in current_labels for pred in preds.tolist()
                    )
                    pred_seen += int(preds.numel())

                    label_cols = torch.tensor(
                        [seen_label_to_col[int(label)] for label in labels.tolist()],
                        device=device,
                        dtype=torch.long,
                    )
                    feature_norm = F.normalize(features, dim=1, eps=1e-8)
                    similarities = feature_norm @ weights_norm.t()
                    positive = similarities.gather(1, label_cols.unsqueeze(1)).squeeze(1)
                    competitors = similarities.clone()
                    competitors.scatter_(1, label_cols.unsqueeze(1), float("-inf"))
                    margins = positive - competitors.max(dim=1).values
                    pos_sum += float(positive.sum().item())
                    margin_sum += float(margins.sum().item())
                    total += int(labels.numel())

        if total == 0:
            return empty

        return {
            "old_sample_seen_acc": correct / total,
            "align_pos_old": pos_sum / total,
            "align_margin_old": margin_sum / total,
            "classifier_weight_norm_old": float(old_weight_norm),
            "old_pred_old_rate": pred_old / total,
            "old_pred_current_task_rate": pred_current / total,
            "old_pred_seen_rate": pred_seen / total,
            "old_pred_future_unseen_rate": pred_future / total,
            "old_routing_samples": total,
            "eval_candidate_mode": "seen_classes",
            "alignment_scope": "old_samples",
        }

    def write_rem_diagnostic_row(self, summary):
        row = {
            "dataset": os.path.basename(os.path.normpath(self.args.data_root)),
            "seed": self.args.seed,
            "task_id": summary["task_id"],
            "task_dir": summary["task_dir"],
            "stage": summary["stage"],
            "old_acc": summary.get("old_sample_seen_acc"),
            "new_acc": summary["current_task_seen"],
            "seen_avg": summary["avg_seen"],
            "ncm": summary["avg_ncm"],
            "old_ncm": summary["old_task_ncm_avg"],

            "bwt": summary["bwt_seen"],
            "bwt_global": summary["bwt_global"],
            "bwt_seen": summary["bwt_seen"],
            "bwt_feat": summary["bwt_features"],
            "bwt_cls": summary["bwt_classifier"],
            "bwt_reference": summary.get("bwt_reference"),
            "align_pos_old": summary.get("align_pos_old"),
            "align_margin_old": summary.get("align_margin_old"),
            "classifier_weight_norm_old": summary.get("classifier_weight_norm_old"),
            "old_pred_old_rate": summary.get("old_pred_old_rate"),
            "old_pred_current_task_rate": summary.get("old_pred_current_task_rate"),
            "old_pred_seen_rate": summary.get("old_pred_seen_rate"),
            "old_pred_future_unseen_rate": summary.get("old_pred_future_unseen_rate"),
            "old_routing_samples": summary.get("old_routing_samples"),
            "eval_candidate_mode": summary.get("eval_candidate_mode"),
            "alignment_scope": summary.get("alignment_scope"),
        }
        self.rem_diagnostics_rows.append(row)
        fieldnames = list(row.keys())
        file_exists = os.path.exists(self.rem_diag_path)
        with open(self.rem_diag_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def mean_optional(rows, key):
        values = [row[key] for row in rows if row.get(key) is not None]
        if not values:
            return None
        return float(np.mean(values))

    def summarize_rem_diagnostics(self):
        if not self.rem_diagnostics_rows:
            return []

        summaries = []
        for stage in ("before_rem", "after_rem"):
            rows = [
                row
                for row in self.rem_diagnostics_rows
                if row["stage"] == stage and row["task_id"] > 0
            ]
            if not rows:
                continue
            summaries.append(
                {
                    "stage": stage,
                    "num_tasks": len(rows),
                    "old_acc": self.mean_optional(rows, "old_acc"),
                    "new_acc": self.mean_optional(rows, "new_acc"),
                    "seen_avg": self.mean_optional(rows, "seen_avg"),
                    "ncm": self.mean_optional(rows, "ncm"),
                    "old_ncm": self.mean_optional(rows, "old_ncm"),
                    "bwt": self.mean_optional(rows, "bwt"),
                    "bwt_global": self.mean_optional(rows, "bwt_global"),
                    "bwt_seen": self.mean_optional(rows, "bwt_seen"),
                    "bwt_feat": self.mean_optional(rows, "bwt_feat"),
                    "bwt_cls": self.mean_optional(rows, "bwt_cls"),
                    "align_pos_old": self.mean_optional(rows, "align_pos_old"),
                    "align_margin_old": self.mean_optional(rows, "align_margin_old"),
                    "classifier_weight_norm_old": self.mean_optional(
                        rows,
                        "classifier_weight_norm_old",
                    ),
                }
            )

        by_stage = {row["stage"]: row for row in summaries}
        if "before_rem" in by_stage and "after_rem" in by_stage:
            before = by_stage["before_rem"]
            after = by_stage["after_rem"]
            delta = {"stage": "delta_after_minus_before", "num_tasks": after["num_tasks"]}
            for key in before:
                if key in {"stage", "num_tasks"}:
                    continue
                if before[key] is None or after.get(key) is None:
                    delta[key] = None
                else:
                    delta[key] = float(after[key] - before[key])
            summaries.append(delta)

        return summaries

    @staticmethod
    def format_pct(value):
        return "N/A" if value is None else f"{value * 100:.2f}"

    @staticmethod
    def format_float(value):
        return "N/A" if value is None else f"{value:.4f}"

    def print_rem_diagnostics_summary(self):
        summaries = self.summarize_rem_diagnostics()
        if not summaries:
            return

        print("\n" + "=" * 120)
        print("REM Diagnostics Summary (averaged over incremental tasks, task_id > 0)")
        print("=" * 120)
        header = (
            f"{'Stage':<26} {'Old Acc':>8} {'New Acc':>8} {'Seen Avg':>9} "
            f"{'NCM':>8} {'Old NCM':>8} {'BWT_s':>8} {'BWT_f':>8} {'BWT_c':>8} "
            f"{'AlignPos':>9} {'AlignMargin':>12}"
        )
        print(header)
        print("-" * len(header))
        for row in summaries:
            print(
                f"{row['stage']:<26} "
                f"{self.format_pct(row.get('old_acc')):>8} "
                f"{self.format_pct(row.get('new_acc')):>8} "
                f"{self.format_pct(row.get('seen_avg')):>9} "
                f"{self.format_pct(row.get('ncm')):>8} "
                f"{self.format_pct(row.get('old_ncm')):>8} "
                f"{self.format_pct(row.get('bwt')):>8} "
                f"{self.format_pct(row.get('bwt_feat')):>8} "
                f"{self.format_pct(row.get('bwt_cls')):>8} "
                f"{self.format_float(row.get('align_pos_old')):>9} "
                f"{self.format_float(row.get('align_margin_old')):>12}"
            )

        out_path = os.path.join(self.output_dir, "rem_diagnostics_summary.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=4)
        print(f"\n[REM Diagnostics] Summary saved to: {out_path}")

    @preserve_rng_state()
    def record_stage(self, stage_name, task_id):


        test_accs, ncm_accs, train_accs = self.evaluate_learned_tasks(
            task_id,
            test_matrix=self.process_analysis[stage_name]["matrix"],
            seen_matrix=self.process_analysis[stage_name]["matrix_seen"],
            ncm_matrix=self.process_analysis[stage_name]["matrix_ncm"],
        )
        summary = self.summarize_metrics(
            task_id,
            test_accs,
            ncm_accs,
            train_accs,
            self.process_analysis[stage_name]["matrix"],
            self.process_analysis[stage_name]["matrix_seen"],
            self.process_analysis[stage_name]["matrix_ncm"],
        )
        summary["stage"] = stage_name
        if stage_name in {"before_rem", "post_rem"}:
            summary["bwt_reference"] = "checkpoint_specific_diagonal"
            summary.update(self.old_sample_classifier_diagnostics(task_id))
        self.process_analysis[stage_name]["steps"].append(summary)

        if self.args.log_rem_diagnostics and stage_name in {"before_rem", "post_rem"}:
            csv_summary = dict(summary)
            csv_summary["stage"] = "after_rem" if stage_name == "post_rem" else stage_name
            self.write_rem_diagnostic_row(csv_summary)

        print(f"[Stage Analysis] {self.stage_display[stage_name]}")
        print(
            f"  Current Task Test/NCM: {summary['current_task_test']*100:.2f}% / "
            f"{summary['current_task_ncm']*100:.2f}%"
        )
        if summary.get("old_sample_seen_acc") is not None:
            print(
                f"  Old Samples Seen-Acc/NCM: {summary['old_sample_seen_acc']*100:.2f}% / "
                f"{summary['old_task_ncm_avg']*100:.2f}%"
            )
        print(
            f"  BWT_seen / BWT_feat / BWT_cls: {summary['bwt_seen']*100:.2f}% / "
            f"{summary['bwt_features']*100:.2f}% / {summary['bwt_classifier']*100:.2f}%"
        )
        if summary.get("align_pos_old") is not None:
            print(
                f"  Alignment old pos/margin/norm: "
                f"{summary['align_pos_old']:.4f} / "
                f"{summary['align_margin_old']:.4f} / "
                f"{summary['classifier_weight_norm_old']:.4f}"
            )
        if summary.get("old_pred_current_task_rate") is not None:
            print(
                f"  Old-sample routing ({summary.get('eval_candidate_mode', 'seen_classes')}): "
                f"old={summary['old_pred_old_rate']*100:.2f}%, "
                f"current={summary['old_pred_current_task_rate']*100:.2f}%, "
                f"future={summary['old_pred_future_unseen_rate']*100:.2f}%"
            )
        return test_accs, ncm_accs, train_accs

    def process_payload(self):
        payload = {}
        for stage_name, stage_data in self.process_analysis.items():
            payload[stage_name] = {
                "display_name": self.stage_display[stage_name],
                "matrix": stage_data["matrix"].tolist(),
                "matrix_seen": stage_data["matrix_seen"].tolist(),
                "matrix_ncm": stage_data["matrix_ncm"].tolist(),
                "steps": stage_data["steps"],
                "task_order": self.task_order,
            }
        return payload

    def write_rem_json(self):
        with open(
            os.path.join(self.output_dir, "rem_diagnostics.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(self.rem_diagnostics_rows, f, indent=4)
