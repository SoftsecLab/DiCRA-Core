#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figure_evidence import (
    configure_matplotlib,
    ensure_keys,
    find_file,
    keyed_rows,
    materialize_evidence,
    read_tsv,
    resolve_evidence,
    save_figure,
)


DATASETS = ["CLINC150", "Banking77", "FewRel"]
METHODS = [
    "Sequential LoRA",
    "SeqLoRA + Align",
    "O-LoRA",
    "CLoRA",
    "SLoRA-Pre",
    "RECAP w/o Refinements",
    "RECAP",
]
LABELS = {
    "RECAP w/o Refinements": "DiCRA w/o Refinements",
    "RECAP": "DiCRA",
}
STYLES = {
    "Sequential LoRA": ("#0072B2", "o", 5.2),
    "SeqLoRA + Align": ("#E69F00", "s", 5.2),
    "O-LoRA": ("#009E73", "D", 5.2),
    "CLoRA": ("#56B4E9", "^", 5.4),
    "SLoRA-Pre": ("#CC79A7", "v", 5.4),
    "RECAP w/o Refinements": ("#888888", "P", 5.4),
    "RECAP": ("#D55E00", "*", 8.5),
}


def load_data(evidence: Path):
    with materialize_evidence(evidence) as root:
        table = find_file(root, "all_diagnostics.tsv")
        rows = keyed_rows(read_tsv(table), ("Dataset", "Method"))
    expected = {(dataset, method) for dataset in DATASETS for method in METHODS}
    ensure_keys({key: rows[key] for key in expected}, expected, "diagnostic grid")
    return rows


def plot(rows, output: str, png: str | None) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.45), sharex=True, sharey=True)

    for index, (dataset, ax) in enumerate(zip(DATASETS, axes)):
        for method in METHODS:
            row = rows[(dataset, method)]
            color, marker, size = STYLES[method]
            ax.errorbar(
                float(row["bwt_feat Mean"]) * 100,
                float(row["bwt_cls Mean"]) * 100,
                xerr=float(row["bwt_feat Std"]) * 100,
                yerr=float(row["bwt_cls Std"]) * 100,
                fmt=marker,
                markersize=size,
                markerfacecolor=color if method == "RECAP" else "white",
                markeredgecolor="#202020" if method == "RECAP" else color,
                markeredgewidth=0.9,
                ecolor=color,
                elinewidth=0.75,
                capsize=1.8,
                linestyle="none",
                zorder=4 if method == "RECAP" else 3,
            )
        ax.axvline(0, color="#888888", linestyle="--", linewidth=0.65)
        ax.axhline(0, color="#888888", linestyle="--", linewidth=0.65)
        ax.set_title(f"({chr(97 + index)}) {dataset}", fontweight="bold", pad=5)
        ax.set_xlim(-100, 3)
        ax.set_ylim(-75, 25)
        ax.set_xticks([-100, -75, -50, -25, 0])
        ax.set_yticks([-75, -50, -25, 0, 20])
        ax.grid(color="#D7D7D7", linewidth=0.5, alpha=0.75)
        ax.set_axisbelow(True)
        ax.tick_params(length=2.5, width=0.6)

    axes[0].set_ylabel(r"Readout residual: $\mathrm{BWT}_{seen}-\mathrm{BWT}_{feat}$")
    fig.supxlabel(r"$\mathrm{BWT}_{feat}$ (%)   Higher feature retention $\longrightarrow$", y=0.05, fontsize=7.2)

    handles = []
    for method in METHODS:
        color, marker, size = STYLES[method]
        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="none",
                markerfacecolor=color if method == "RECAP" else "white",
                markeredgecolor="#202020" if method == "RECAP" else color,
                markeredgewidth=0.9,
                markersize=size,
                label=LABELS.get(method, method),
            )
        )
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01), columnspacing=1.0, handletextpad=0.4)
    fig.subplots_adjust(left=0.12, right=0.995, top=0.77, bottom=0.23, wspace=0.15)
    save_figure(fig, output, png)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the DiCRA matched diagnostic plane from verified evidence.")
    parser.add_argument("--evidence", default=None)
    parser.add_argument("--output", default="figures/dicra-diagnostic-plane.pdf")
    parser.add_argument("--png", default=None)
    args = parser.parse_args()
    evidence = resolve_evidence(
        args.evidence,
        [
            "evidence_63_documented_dev_v1_anonymous.tar.gz",
            "evidence_63_documented_dev_v1.tar.gz",
        ],
    )
    plot(load_data(evidence), args.output, args.png)


if __name__ == "__main__":
    main()
