"""
analysis/graphs.py — generate publication-ready figures from aggregated results.

Reads results/aggregated/<run-id>_summary.json and produces PDF figures under:
  results/aggregated/figures/<run-id>/

Figures:
  fig1_t1_performance.pdf   — T1 throughput and fields-present / latency
  fig1b_t1_boxplot.pdf      — T1 generation-time distribution (boxplot)
  fig2_t1_scatter.pdf       — T1/T4: speed vs schema breadth scatter
  fig3_t2_reliability.pdf   — T2 reliability stacked bar
  fig4_t3_discovery.pdf     — T3 F1 / precision / recall (full + py scope)
  fig5_t3_counts.pdf        — T3 TP / FP / FN absolute counts
  fig6_t4_schema.pdf        — T4 schema breadth, completeness & conformance

Usage:
    python -m analysis.graphs --run-id <run-id>
    uv run eval graphs --run-id <run-id>
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path
from typing import Any

import click
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from constants import (
    AGGREGATED_DIR,
    EC_CRASH,
    EC_UNAUTHORIZED_DOWNLOAD,
    FIGURES_DIR,
    RESULTS_DIR,
    SUBTASK_T1,
    SUBTASK_T2,
    SUBTASK_T3,
    SUBTASK_T4G1,
    SUBTASK_T4G2,
)

# ── Style ──────────────────────────────────────────────────────────────────

_RC = {
    "figure.dpi":        150,
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
}

# ── Palette & labels ────────────────────────────────────────────────────────

PALETTE: dict[str, str] = {
    "aibomgen-cli":      "#1f77b4",
    "aibom-generator":   "#ff7f0e",
    "ALOHA":             "#2ca02c",
    "lab700xorg-aisbom": "#d62728",
    "akumar-aibom":      "#9467bd",
    "jorisbot-aibom":    "#8c564b",
    "trusera-aibom":     "#e377c2",
}

TOOL_LABELS: dict[str, str] = {
    "aibomgen-cli":      "aibomgen-cli",
    "aibom-generator":   "aibom-generator",
    "ALOHA":             "ALOHA",
    "lab700xorg-aisbom": "lab700xorg",
    "akumar-aibom":      "akumar-aibom",
    "jorisbot-aibom":    "jorisbot-aibom",
    "trusera-aibom":     "trusera-aibom",
}

# ── Tool groups (mirrors the notebook) ─────────────────────────────────────

T1_TOOLS  = ["aibomgen-cli", "aibom-generator", "ALOHA", "lab700xorg-aisbom"]
T2_TOOLS  = ["aibomgen-cli", "aibom-generator", "ALOHA", "lab700xorg-aisbom"]
T3_TOOLS  = ["aibomgen-cli", "trusera-aibom", "jorisbot-aibom", "akumar-aibom"]
T4G1_TOOLS = ["aibomgen-cli", "aibom-generator", "ALOHA", "lab700xorg-aisbom"]
T4G2_TOOLS = ["aibomgen-cli", "akumar-aibom", "trusera-aibom"]


# ================================================================== #
# CLI entry point                                                       #
# ================================================================== #

@click.command()
@click.option("--run-id", required=True, help="Run ID to generate figures for.")
def main(run_id: str) -> None:
    """Generate publication-ready figures for a completed aggregated run."""
    summary_path = AGGREGATED_DIR / f"{run_id}_summary.json"
    if not summary_path.exists():
        click.echo(f"Summary not found: {summary_path}", err=True)
        sys.exit(1)

    with open(summary_path) as f:
        S: dict[str, Any] = json.load(f)

    out_dir = FIGURES_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(_RC)

    _fig1_t1_performance(S, out_dir)
    _fig1b_t1_boxplot(S, run_id, out_dir)
    _fig2_speed_breadth(S, out_dir)
    _fig3_t2_reliability(S, out_dir)
    _fig4_t3_discovery(S, out_dir)
    _fig5_t3_counts(S, out_dir)
    _fig6_t4_schema(S, out_dir)

    click.echo(f"Figures written to: {out_dir}")


# ================================================================== #
# Figure helpers                                                        #
# ================================================================== #

def _fig1_t1_performance(S: dict, out: Path) -> None:
    """Figure 1 — T1: throughput and fields-present / latency."""
    tools  = [t for t in T1_TOOLS if f"{t}::{SUBTASK_T1}" in S]
    labels = [TOOL_LABELS[t] for t in tools]
    colors = [PALETTE[t] for t in tools]

    throughput = [S[f"{t}::{SUBTASK_T1}"]["metrics"]["normalized_throughput"]["mean"] for t in tools]
    fields     = [S[f"{t}::{SUBTASK_T1}"]["metrics"]["fields_present"]["mean"]        for t in tools]
    duration   = [S[f"{t}::{SUBTASK_T1}"]["metrics"]["duration_seconds"]["mean"]      for t in tools]

    x = np.arange(len(tools))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))

    # left: throughput
    ax = axes[0]
    bars = ax.bar(x, throughput, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Throughput (fields / s)")
    ax.set_title("(a) Processing Throughput")
    for bar, v in zip(bars, throughput):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.1f}",
                ha="center", va="bottom", fontsize=8)

    # right: fields present vs duration
    ax2 = axes[1]
    ax2.bar(x - w / 2, fields,   width=w, label="Fields present (mean)",
            color=colors, edgecolor="white", linewidth=0.6)
    ax3 = ax2.twinx()
    ax3.spines["top"].set_visible(False)
    ax3.bar(x + w / 2, duration, width=w, label="Duration (s, mean)",
            color=colors, edgecolor="white", linewidth=0.6, alpha=0.55, hatch="//")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylabel("Fields present (mean)")
    ax3.set_ylabel("Duration (s, mean)")
    ax2.set_title("(b) Fields Present vs. Latency")

    patch1 = mpatches.Patch(facecolor="grey",                       label="Fields present")
    patch2 = mpatches.Patch(facecolor="grey", alpha=0.55, hatch="//", label="Duration (s)")
    ax2.legend(handles=[patch1, patch2], loc="best", frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out / "fig1_t1_performance.pdf", bbox_inches="tight")
    plt.close(fig)
    click.echo("  Saved fig1_t1_performance.pdf")


def _fig1b_t1_boxplot(S: dict, run_id: str, out: Path) -> None:
    """Figure 1b — T1: generation-time distribution (boxplot)."""
    run_base = RESULTS_DIR / run_id
    tools = [t for t in T1_TOOLS if f"{t}::{SUBTASK_T1}" in S]

    def _load_durations(tool: str) -> list[float]:
        pattern = str(run_base / tool / "T1_T2" / "popular" / "*" / "metrics.json")
        durations: list[float] = []
        for path in glob.glob(pattern):
            with open(path) as fh:
                m = json.load(fh)
            if m.get("success"):
                durations.append(m["duration_seconds"])
        return sorted(durations)

    dur_data   = {t: _load_durations(t) for t in tools}
    box_labels = [f"{TOOL_LABELS[t]}\n(n={len(dur_data[t])})" for t in tools]
    box_colors = [PALETTE[t] for t in tools]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    bp = ax.boxplot(
        [dur_data[t] for t in tools],
        patch_artist=True,
        notch=False,
        showmeans=True,
        meanprops=dict(marker="x", markerfacecolor="black", markeredgecolor="black",
                       markersize=8, markeredgewidth=2, zorder=5),
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=3.5, linestyle="none", alpha=0.5),
        widths=0.5,
    )

    for patch, fc, flier in zip(bp["boxes"], box_colors, bp["fliers"]):
        patch.set_facecolor(fc)
        patch.set_alpha(0.55)
        flier.set_markerfacecolor(fc)
        flier.set_markeredgecolor(fc)

    for i, (t, med_line, mean_marker) in enumerate(
            zip(tools, bp["medians"], bp["means"]), start=1):
        med_val  = med_line.get_ydata()[0]
        mean_val = mean_marker.get_ydata()[0]
        ax.text(i + 0.28, med_val,  f"med {med_val:.2f}s",
                ha="left", va="center", fontsize=7.5, color="dimgrey")
        ax.text(i + 0.28, mean_val, f"μ {mean_val:.2f}s",
                ha="left", va="center", fontsize=7.5, color="black")

    ax.set_yscale("log")
    ax.set_ylabel("Generation time (s, log scale)")
    ax.set_xticklabels(box_labels)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:g}s"))

    med_patch  = plt.Line2D([0], [0], color="black", linewidth=1.2, label="Median")
    mean_patch = plt.Line2D([0], [0], marker="x", color="w", markerfacecolor="black",
                            markeredgecolor="black", markersize=8, markeredgewidth=2,
                            label="Mean (μ)")
    ax.legend(handles=[med_patch, mean_patch], frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(out / "fig1b_t1_boxplot.pdf", bbox_inches="tight")
    plt.close(fig)
    click.echo("  Saved fig1b_t1_boxplot.pdf")


def _fig2_speed_breadth(S: dict, out: Path) -> None:
    """Figure 2 — T1/T4: speed vs schema breadth scatter."""
    tools = [t for t in T1_TOOLS if f"{t}::{SUBTASK_T1}" in S and f"{t}::{SUBTASK_T4G1}" in S]

    breadth_vals  = {t: S[f"{t}::{SUBTASK_T4G1}"]["metrics"]["cdx_breadth_pct_v16"]["distinct_paths_filled"] for t in tools}
    duration_vals = {t: S[f"{t}::{SUBTASK_T1}"]["metrics"]["duration_seconds"]["mean"]  for t in tools}

    fig, ax = plt.subplots(figsize=(4.8, 3.8))

    for t in tools:
        dur = duration_vals[t]
        brd = breadth_vals[t]
        ax.scatter(dur, brd, color=PALETTE[t], s=100, zorder=3)
        ax.annotate(TOOL_LABELS[t], (dur, brd),
                    textcoords="offset points", xytext=(8, 0),
                    color=PALETTE[t], fontsize=8, va="center", ha="left")

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Mean generation time (s)")
    ax.set_ylabel("Distinct CycloneDX 1.6 paths filled (breadth)")

    fig.tight_layout()
    fig.savefig(out / "fig2_t1_scatter.pdf", bbox_inches="tight")
    plt.close(fig)
    click.echo("  Saved fig2_t1_scatter.pdf")


def _fig3_t2_reliability(S: dict, out: Path) -> None:
    """Figure 3 — T2: reliability stacked bar (success / crash / auth error)."""
    tools = [t for t in T2_TOOLS if f"{t}::{SUBTASK_T2}" in S]
    labels = [TOOL_LABELS[t] for t in tools]
    colors = [PALETTE[t] for t in tools]

    def _frac(tool: str, ec: str) -> float:
        entry = S[f"{tool}::{SUBTASK_T2}"]
        total = max(entry.get("total_runs", 1), 1)
        return entry.get("error_breakdown", {}).get(ec, 0) / total

    success_frac = [S[f"{t}::{SUBTASK_T2}"]["success_rate"] for t in tools]
    crash_frac   = [_frac(t, EC_CRASH) for t in tools]
    unauth_frac  = [_frac(t, EC_UNAUTHORIZED_DOWNLOAD) for t in tools]

    x2 = np.arange(len(tools))

    fig, ax = plt.subplots(figsize=(5, 3.5))

    ax.bar(x2, success_frac, color=colors, edgecolor="white", label="Success")
    ax.bar(x2, crash_frac,   bottom=success_frac,
           color="black", alpha=0.65, edgecolor="white", label="Crash")
    ax.bar(x2, unauth_frac,
           bottom=[s + c for s, c in zip(success_frac, crash_frac)],
           color="grey",  alpha=0.65, edgecolor="white", hatch="xx", label="Auth error")

    ax.set_xticks(x2)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Fraction of runs")
    ax.legend(frameon=False)

    for i, v in enumerate(success_frac):
        ax.text(i, v / 2, f"{v * 100:.0f}%", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out / "fig3_t2_reliability.pdf", bbox_inches="tight")
    plt.close(fig)
    click.echo("  Saved fig3_t2_reliability.pdf")


def _fig4_t3_discovery(S: dict, out: Path) -> None:
    """Figure 4 — T3: F1 / precision / recall (full scope + python-literal scope)."""
    tools  = [t for t in T3_TOOLS if f"{t}::{SUBTASK_T3}" in S]
    labels = [TOOL_LABELS[t] for t in tools]
    colors = [PALETTE[t] for t in tools]

    f1_full   = [S[f"{t}::{SUBTASK_T3}"]["metrics"]["f1"]["mean"]        for t in tools]
    prec_full = [S[f"{t}::{SUBTASK_T3}"]["metrics"]["precision"]["mean"]  for t in tools]
    rec_full  = [S[f"{t}::{SUBTASK_T3}"]["metrics"]["recall"]["mean"]     for t in tools]
    f1_py     = [S[f"{t}::{SUBTASK_T3}"]["metrics"]["f1_py"]["mean"]        for t in tools]
    prec_py   = [S[f"{t}::{SUBTASK_T3}"]["metrics"]["precision_py"]["mean"]  for t in tools]
    rec_py    = [S[f"{t}::{SUBTASK_T3}"]["metrics"]["recall_py"]["mean"]     for t in tools]

    x3 = np.arange(len(tools))
    w3 = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), sharey=False)

    def _grouped_bar(ax: plt.Axes, data_sets: list, title: str) -> None:
        offsets = np.linspace(-(len(data_sets) - 1) * w3 / 2,
                              (len(data_sets) - 1) * w3 / 2, len(data_sets))
        for offset, (vals, hatch) in zip(offsets, data_sets):
            ax.bar(x3 + offset, vals, width=w3,
                   color=colors, edgecolor="white",
                   hatch=hatch, alpha=0.9 if hatch else 1.0)
        ax.set_xticks(x3)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title(title)
        patches = [mpatches.Patch(facecolor="grey",           label="F1"),
                   mpatches.Patch(facecolor="grey", hatch="//", label="Precision"),
                   mpatches.Patch(facecolor="grey", hatch="xx", label="Recall")]
        ax.legend(handles=patches, frameon=False, fontsize=8)

    _grouped_bar(axes[0],
                 [(f1_full, None), (prec_full, "//"), (rec_full, "xx")],
                 "(a) Full scope (all file types)")
    _grouped_bar(axes[1],
                 [(f1_py, None), (prec_py, "//"), (rec_py, "xx")],
                 "(b) Python-literal scope (fair subset)")

    tool_patches = [mpatches.Patch(color=PALETTE[t], label=TOOL_LABELS[t]) for t in tools]
    axes[0].legend(handles=tool_patches, frameon=False, fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(out / "fig4_t3_discovery.pdf", bbox_inches="tight")
    plt.close(fig)
    click.echo("  Saved fig4_t3_discovery.pdf")


def _fig5_t3_counts(S: dict, out: Path) -> None:
    """Figure 5 — T3: TP / FP / FN absolute counts (stacked bar)."""
    tools  = [t for t in T3_TOOLS if f"{t}::{SUBTASK_T3}" in S]
    labels = [TOOL_LABELS[t] for t in tools]
    colors = [PALETTE[t] for t in tools]

    tp = [S[f"{t}::{SUBTASK_T3}"]["metrics"]["tp_total"] for t in tools]
    fp = [S[f"{t}::{SUBTASK_T3}"]["metrics"]["fp_total"] for t in tools]
    fn = [S[f"{t}::{SUBTASK_T3}"]["metrics"]["fn_total"] for t in tools]

    x5 = np.arange(len(tools))
    w5 = 0.55

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.bar(x5, tp, width=w5, label="TP (correct detections)", color=colors)
    ax.bar(x5, fp, width=w5, bottom=tp,
           color=colors, alpha=0.45, hatch="//", label="FP (false alarms)")
    ax.bar(x5, fn, width=w5, bottom=[t_ + f_ for t_, f_ in zip(tp, fp)],
           color=colors, alpha=0.25, hatch="xx", label="FN (missed models)")

    ax.set_xticks(x5)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Count (across all targets)")

    tp_patch = mpatches.Patch(facecolor="grey",                        label="TP")
    fp_patch = mpatches.Patch(facecolor="grey", alpha=0.45, hatch="//", label="FP")
    fn_patch = mpatches.Patch(facecolor="grey", alpha=0.25, hatch="xx", label="FN")
    ax.legend(handles=[tp_patch, fp_patch, fn_patch], frameon=False, fontsize=9)

    for i, v in enumerate(tp):
        ax.text(i, v / 2, str(v), ha="center", va="center", fontsize=9,
                color="white", fontweight="bold")

    fig.tight_layout()
    fig.savefig(out / "fig5_t3_counts.pdf", bbox_inches="tight")
    plt.close(fig)
    click.echo("  Saved fig5_t3_counts.pdf")


def _fig6_t4_schema(S: dict, out: Path) -> None:
    """Figure 6 — T4: CycloneDX schema breadth, completeness & conformance."""

    def _t4_data(tools: list[str], subtask: str) -> tuple:
        breadth      = [S[f"{t}::{subtask}"]["metrics"]["cdx_breadth_pct_v16"]["mean"]           for t in tools]
        completeness = [S[f"{t}::{subtask}"]["metrics"]["cdx_completeness_pct_v16"]["mean"]       for t in tools]
        distinct     = [S[f"{t}::{subtask}"]["metrics"]["cdx_breadth_pct_v16"]["distinct_paths_filled"] for t in tools]
        conform      = [S[f"{t}::{subtask}"]["metrics"]["cdx_conformance_pass_rate_v16"]          for t in tools]
        return breadth, completeness, distinct, conform

    g1_tools = [t for t in T4G1_TOOLS if f"{t}::{SUBTASK_T4G1}" in S]
    g2_tools = [t for t in T4G2_TOOLS if f"{t}::{SUBTASK_T4G2}" in S]

    b1, comp1, d1, c1 = _t4_data(g1_tools, SUBTASK_T4G1)
    b2, comp2, d2, c2 = _t4_data(g2_tools, SUBTASK_T4G2)

    def _plot_t4(ax: plt.Axes, tools: list[str], breadth: list, completeness: list,
                 distinct: list, conform: list, title: str) -> None:
        x   = np.arange(len(tools))
        w   = 0.28
        col = [PALETTE[t] for t in tools]
        lbl = [TOOL_LABELS[t] for t in tools]

        bars_b = ax.bar(x - w / 2, breadth,      width=w, color=col, edgecolor="white")
        ax.bar(       x + w / 2, completeness, width=w, color=col,
                      edgecolor="white", hatch="//", alpha=0.6)

        for bar, d in zip(bars_b, distinct):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.04,
                    f"{d}p", ha="center", va="bottom", fontsize=7)

        ax2 = ax.twinx()
        ax2.spines["top"].set_visible(False)
        ax2.plot(x, conform, "D-", color="black", markersize=7,
                 linewidth=1.4, zorder=5)
        for xi, cv in zip(x, conform):
            ax2.text(xi, cv + 0.04, f"{cv:.0%}", ha="center", va="bottom",
                     fontsize=7.5, color="black")
        ax2.set_ylim(0, 1.35)
        ax2.set_ylabel("Conformance pass rate", fontsize=9)
        ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax2.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])

        ax.set_xticks(x)
        ax.set_xticklabels(lbl, rotation=18, ha="right")
        ax.set_ylabel("CycloneDX 1.6 coverage score (×100)", fontsize=9)
        ax.set_title(title)

        b_patch = mpatches.Patch(facecolor="grey", label="Breadth (distinct paths filled)")
        c_patch = mpatches.Patch(facecolor="grey", hatch="//", alpha=0.6, label="Completeness (per-BOM mean)")
        d_patch = mpatches.Patch(facecolor="black", label="Conformance rate (right axis)")
        ax.legend(handles=[b_patch, c_patch, d_patch], frameon=False, fontsize=7.5, loc="upper right")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    _plot_t4(axes[0], g1_tools, b1, comp1, d1, c1, "(a) G1: HF Model ID → AIBOM")
    _plot_t4(axes[1], g2_tools, b2, comp2, d2, c2, "(b) G2: Source-code scan → AIBOM")

    fig.tight_layout()
    fig.savefig(out / "fig6_t4_schema.pdf", bbox_inches="tight")
    plt.close(fig)
    click.echo("  Saved fig6_t4_schema.pdf")
