#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from figure_evidence import (
    configure_matplotlib,
    ensure_keys,
    find_file,
    materialize_evidence,
    resolve_evidence,
    save_figure,
)


DATASETS = ["clinc150", "banking77", "fewrel_acl2024"]
DATASET_LABELS = {
    "clinc150": "CLINC150",
    "banking77": "Banking77",
    "fewrel_acl2024": "FewRel",
}
VARIANTS = ["no_merge_decay", "no_stabilization", "no_alignment"]
VARIANT_LABELS = {
    "no_merge_decay": "No merge decay",
    "no_stabilization": "No Stabilization",
    "no_alignment": "No Alignment",
}
METRICS = ["final_avg", "bwt_features", "bwt_classifier"]
TITLES = [r"(a) $\Delta$ Final Avg", r"(b) $\Delta\,\mathrm{BWT}_{feat}$", r"(c) $\Delta\,\mathrm{BWT}_{cls}$"]
XLIMS = [(-85, 4), (-23, 3), (-95, 5)]
XTICKS = [[-80, -60, -40, -20, 0], [-20, -15, -10, -5, 0], [-90, -60, -30, 0]]
STYLES = {
    "clinc150": ("#0072B2", "o"),
    "banking77": ("#E69F00", "s"),
    "fewrel_acl2024": ("#009E73", "D"),
}


def load_data(evidence: Path):
    with materialize_evidence(evidence) as root:
        summary_path = find_file(root, "single_variable_ablation_summary.json")
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    records = {
        (row["dataset"], row["variant"]): row
        for row in payload["paired_deltas"]
        if row["variant"] in VARIANTS
    }
    ensure_keys(records, {(d, v) for d in DATASETS for v in VARIANTS}, "ablation grid")
    return records


def plot(records, output: str, png: str | None) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.25), sharey=True)
    y = np.arange(len(VARIANTS))
    offsets = {"clinc150": -0.15, "banking77": 0.0, "fewrel_acl2024": 0.15}

    for metric, title, xlim, xticks, ax in zip(METRICS, TITLES, XLIMS, XTICKS, axes):
        for dataset in DATASETS:
            means = [records[(dataset, variant)]["metrics"][metric]["mean"] * 100 for variant in VARIANTS]
            stds = [records[(dataset, variant)]["metrics"][metric]["std"] * 100 for variant in VARIANTS]
            color, marker = STYLES[dataset]
            ax.errorbar(
                means,
                y + offsets[dataset],
                xerr=stds,
                fmt=marker,
                markersize=5.3,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.1,
                ecolor=color,
                elinewidth=0.9,
                capsize=2.0,
                linestyle="none",
                zorder=3,
            )
        ax.axvline(0, color="#888888", linestyle="--", linewidth=0.7)
        ax.set_xlim(*xlim)
        ax.set_xticks(xticks)
        ax.set_title(title, fontweight="bold", pad=5)
        ax.grid(axis="x", color="#D7D7D7", linewidth=0.55, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=2.5, width=0.6)

    axes[0].set_yticks(y, [VARIANT_LABELS[item] for item in VARIANTS])
    axes[0].invert_yaxis()
    plt.setp(axes[1].get_yticklabels(), visible=False)
    plt.setp(axes[2].get_yticklabels(), visible=False)

    legend = [
        Line2D([0], [0], marker=STYLES[d][1], color="none", markerfacecolor="white", markeredgecolor=STYLES[d][0], markeredgewidth=1.1, label=DATASET_LABELS[d])
        for d in DATASETS
    ]
    fig.legend(handles=legend, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.text(0.52, 0.02, "Change relative to full DiCRA (percentage points; higher is better)", ha="center", fontsize=7)
    fig.subplots_adjust(left=0.16, right=0.995, top=0.78, bottom=0.22, wspace=0.22)
    save_figure(fig, output, png)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the DiCRA component-ablation figure from verified evidence.")
    parser.add_argument("--evidence", default=None)
    parser.add_argument("--output", default="figures/dicra-ablation-effects.pdf")
    parser.add_argument("--png", default=None)
    args = parser.parse_args()
    evidence = resolve_evidence(
        args.evidence,
        [
            "evidence_documented_dev_single_variable_ablation_45_anonymous.tar.gz",
            "evidence_documented_dev_single_variable_ablation_45.tar.gz",
        ],
    )
    plot(load_data(evidence), args.output, args.png)


if __name__ == "__main__":
    main()
