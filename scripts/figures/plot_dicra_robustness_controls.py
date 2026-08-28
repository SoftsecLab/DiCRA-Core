#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from figure_evidence import (
    configure_matplotlib,
    find_file,
    keyed_rows,
    materialize_evidence,
    parse_interval,
    parse_mean_std,
    read_tsv,
    resolve_evidence,
    save_figure,
)


DATASETS = ["CLINC150", "Banking77", "FewRel"]
COLORS = {
    "dicra": "#0072B2",
    "seq_align": "#E69F00",
    "no_stab": "#009E73",
    "prototype": "#777777",
}


def ci_error(values, lower, upper):
    values = np.asarray(values, dtype=float)
    return np.vstack((values - np.asarray(lower), np.asarray(upper) - values))


def load_cross_interface(evidence: Path):
    with materialize_evidence(evidence) as root:
        expanding = keyed_rows(
            read_tsv(find_file(root, "expanding_head_matched.tsv")),
            ("Dataset", "Method"),
        )
        masking = read_tsv(find_file(root, "masking_control.tsv", "fixed_global"))

    control_final = []
    control_final_sd = []
    dicra_final = []
    dicra_final_sd = []
    control_bwt = []
    control_bwt_sd = []
    dicra_bwt = []
    dicra_bwt_sd = []
    for dataset in DATASETS:
        control = expanding[(dataset, "SeqLoRA + Align")]
        dicra = expanding[(dataset, "RECAP")]
        control_final.append(float(control["Final Avg Mean"]) * 100)
        control_final_sd.append(float(control["Final Avg Std"]) * 100)
        dicra_final.append(float(dicra["Final Avg Mean"]) * 100)
        dicra_final_sd.append(float(dicra["Final Avg Std"]) * 100)
        control_bwt.append(float(control["BWT feat Mean"]) * 100)
        control_bwt_sd.append(float(control["BWT feat Std"]) * 100)
        dicra_bwt.append(float(dicra["BWT feat Mean"]) * 100)
        dicra_bwt_sd.append(float(dicra["BWT feat Std"]) * 100)

    max_delta_avg_inc = max(abs(float(row["Delta AvgInc Mean"])) for row in masking) * 100
    max_pred_future = max(float(row["PredFuture Mean"]) for row in masking) * 100
    return {
        "control_final": control_final,
        "control_final_error": np.asarray(control_final_sd),
        "dicra_final": dicra_final,
        "dicra_final_error": np.asarray(dicra_final_sd),
        "final_delta": np.asarray(dicra_final) - np.asarray(control_final),
        "control_bwt": control_bwt,
        "control_bwt_error": np.asarray(control_bwt_sd),
        "dicra_bwt": dicra_bwt,
        "dicra_bwt_error": np.asarray(dicra_bwt_sd),
        "bwt_delta": np.asarray(dicra_bwt) - np.asarray(control_bwt),
        "max_delta_avg_inc": max_delta_avg_inc,
        "max_pred_future": max_pred_future,
    }


def load_multi_order(evidence: Path):
    with materialize_evidence(evidence) as root:
        summary = keyed_rows(
            read_tsv(find_file(root, "multi_order_summary.tsv")),
            ("Dataset", "Method"),
        )
        paired = keyed_rows(
            read_tsv(find_file(root, "multi_order_paired_deltas.tsv")),
            ("Dataset", "Difference"),
        )

    result = {name: [] for name in [
        "control_final", "control_final_low", "control_final_high",
        "dicra_final", "dicra_final_low", "dicra_final_high",
        "control_bwt", "control_bwt_low", "control_bwt_high",
        "dicra_bwt", "dicra_bwt_low", "dicra_bwt_high",
        "final_delta", "bwt_delta",
    ]}
    for dataset in DATASETS:
        control = summary[(dataset, "RECAP w/o Stabilization")]
        dicra = summary[(dataset, "RECAP")]
        c_final = parse_interval(control["final_avg"])
        d_final = parse_interval(dicra["final_avg"])
        c_bwt = parse_interval(control["bwt_features"])
        d_bwt = parse_interval(dicra["bwt_features"])
        delta = paired[(dataset, "RECAP - RECAP w/o Stabilization")]
        for prefix, values in [("control_final", c_final), ("dicra_final", d_final), ("control_bwt", c_bwt), ("dicra_bwt", d_bwt)]:
            result[prefix].append(values[0])
            result[f"{prefix}_low"].append(values[1])
            result[f"{prefix}_high"].append(values[2])
        result["final_delta"].append(parse_interval(delta["final_avg"])[0])
        result["bwt_delta"].append(parse_interval(delta["bwt_features"])[0])

    result["control_final_error"] = ci_error(result["control_final"], result["control_final_low"], result["control_final_high"])
    result["dicra_final_error"] = ci_error(result["dicra_final"], result["dicra_final_low"], result["dicra_final_high"])
    result["control_bwt_error"] = ci_error(result["control_bwt"], result["control_bwt_low"], result["control_bwt_high"])
    result["dicra_bwt_error"] = ci_error(result["dicra_bwt"], result["dicra_bwt_low"], result["dicra_bwt_high"])
    return result


def load_prototype_control(evidence: Path):
    with materialize_evidence(evidence) as root:
        rows = keyed_rows(
            read_tsv(find_file(root, "single_variable_ablation_summary.tsv")),
            ("Dataset", "Variant"),
        )
    control = []
    control_sd = []
    dicra = []
    dicra_sd = []
    for dataset in DATASETS:
        c_mean, c_std = parse_mean_std(rows[(dataset, "Decayed LoRA Merge + NCM")]["Final Avg"])
        d_mean, d_std = parse_mean_std(rows[(dataset, "Full RECAP")]["Final Avg"])
        control.append(c_mean)
        control_sd.append(c_std)
        dicra.append(d_mean)
        dicra_sd.append(d_std)
    return {
        "control_final": control,
        "control_final_error": np.asarray(control_sd),
        "dicra_final": dicra,
        "dicra_final_error": np.asarray(dicra_sd),
        "final_delta": np.asarray(dicra) - np.asarray(control),
    }


def draw_comparison(
    ax,
    control,
    dicra,
    control_error,
    dicra_error,
    deltas,
    control_color,
    control_hatch,
    title,
    ylabel,
    ylim,
    yticks,
):
    x = np.arange(len(DATASETS))
    width = 0.34
    error_style = {"elinewidth": 0.8, "capthick": 0.8, "capsize": 2.2, "ecolor": "#303030"}
    control_bars = ax.bar(
        x - width / 2,
        control,
        width,
        yerr=control_error,
        color=control_color,
        edgecolor="#202020",
        linewidth=0.65,
        hatch=control_hatch,
        error_kw=error_style,
        zorder=3,
    )
    dicra_bars = ax.bar(
        x + width / 2,
        dicra,
        width,
        yerr=dicra_error,
        color=COLORS["dicra"],
        edgecolor="#202020",
        linewidth=0.65,
        error_kw=error_style,
        zorder=3,
    )
    ax.bar_label(control_bars, fmt="%.1f", padding=2, fontsize=5.7)
    ax.bar_label(dicra_bars, fmt="%.1f", padding=2, fontsize=5.7)
    span = ylim[1] - ylim[0]
    delta_y = ylim[1] - 0.065 * span
    for index, delta in enumerate(deltas):
        ax.text(
            x[index],
            delta_y,
            rf"$\Delta$ +{delta:.2f}",
            ha="center",
            va="center",
            fontsize=5.8,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.85},
            zorder=5,
        )
    ax.set_title(title, fontweight="bold", pad=5)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, DATASETS)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.axhline(0, color="#303030", linewidth=0.75, zorder=2)
    ax.grid(axis="y", color="#D7D7D7", linewidth=0.55, linestyle="--", zorder=0)
    ax.tick_params(axis="both", length=2.5, width=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot(cross, multi, prototype, output: str, png: str | None) -> None:
    configure_matplotlib()
    fig = plt.figure(figsize=(7.05, 4.45))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.08], hspace=0.52, wspace=0.34)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[0, 1], sharey=ax_a)
    ax_d = fig.add_subplot(grid[1, 1], sharey=ax_b)
    ax_e = fig.add_subplot(grid[:, 2])

    draw_comparison(ax_a, cross["control_final"], cross["dicra_final"], cross["control_final_error"], cross["dicra_final_error"], cross["final_delta"], COLORS["seq_align"], "///", "(a) Expanding head: Final Avg", "Final Avg (%)", (0, 110), [0, 25, 50, 75, 100])
    draw_comparison(ax_b, cross["control_bwt"], cross["dicra_bwt"], cross["control_bwt_error"], cross["dicra_bwt_error"], cross["bwt_delta"], COLORS["seq_align"], "///", r"(b) Expanding head: $\mathrm{BWT}_{feat}$", r"$\mathrm{BWT}_{feat}$ (%)", (-80, 12), [-80, -60, -40, -20, 0])
    draw_comparison(ax_c, multi["control_final"], multi["dicra_final"], multi["control_final_error"], multi["dicra_final_error"], multi["final_delta"], COLORS["no_stab"], "\\\\", "(c) New class orders: Final Avg", "", (0, 110), [0, 25, 50, 75, 100])
    draw_comparison(ax_d, multi["control_bwt"], multi["dicra_bwt"], multi["control_bwt_error"], multi["dicra_bwt_error"], multi["bwt_delta"], COLORS["no_stab"], "\\\\", r"(d) New class orders: $\mathrm{BWT}_{feat}$", "", (-80, 12), [-80, -60, -40, -20, 0])
    plt.setp(ax_c.get_yticklabels(), visible=False)
    plt.setp(ax_d.get_yticklabels(), visible=False)
    draw_comparison(ax_e, prototype["control_final"], prototype["dicra_final"], prototype["control_final_error"], prototype["dicra_final_error"], prototype["final_delta"], COLORS["prototype"], "xx", "(e) Prototype-only readout\nFinal Avg", "Final Avg (%)", (0, 100), [0, 25, 50, 75, 100])

    handles = [
        Patch(facecolor=COLORS["dicra"], edgecolor="#202020", label="DiCRA"),
        Patch(facecolor=COLORS["seq_align"], edgecolor="#202020", hatch="///", label="SeqLoRA + Align"),
        Patch(facecolor=COLORS["no_stab"], edgecolor="#202020", hatch="\\\\", label="DiCRA w/o Stabilization"),
        Patch(facecolor=COLORS["prototype"], edgecolor="#202020", hatch="xx", label="Prototype control"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=False, handlelength=1.4, columnspacing=1.2)
    fig.text(
        0.5,
        0.025,
        rf"Masking sanity check: max $|\Delta \mathrm{{AvgInc}}|={cross['max_delta_avg_inc']:.2f}$ pp; future-class predictions $\leq {cross['max_pred_future']:.2f}\%$.",
        ha="center",
        va="bottom",
        fontsize=6.4,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.89, bottom=0.13)
    save_figure(fig, output, png)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the DiCRA robustness-control figure from verified evidence.")
    parser.add_argument("--cross-interface-evidence", default=None)
    parser.add_argument("--multi-order-evidence", default=None)
    parser.add_argument("--ablation-evidence", default=None)
    parser.add_argument("--output", default="figures/dicra-robustness-controls.pdf")
    parser.add_argument("--png", default=None)
    args = parser.parse_args()
    cross_path = resolve_evidence(args.cross_interface_evidence, ["evidence_99_documented_dev_v1_anonymous.tar.gz", "evidence_99_documented_dev_v1.tar.gz"])
    multi_path = resolve_evidence(args.multi_order_evidence, ["evidence_documented_dev_multi_order_81_anonymous.tar.gz", "evidence_documented_dev_multi_order_81.tar.gz"])
    ablation_path = resolve_evidence(args.ablation_evidence, ["evidence_documented_dev_single_variable_ablation_45_anonymous.tar.gz", "evidence_documented_dev_single_variable_ablation_45.tar.gz"])
    plot(load_cross_interface(cross_path), load_multi_order(multi_path), load_prototype_control(ablation_path), args.output, args.png)


if __name__ == "__main__":
    main()
