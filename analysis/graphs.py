"""
analysis/graphs.py — generate publication-ready figures from aggregated results.

Reads results/aggregated/data/<run-id>/<run-id>_summary.json and produces PDF figures under:
  results/aggregated/figures/<run-id>/

Figures:
  fig1a_t1_latency.pdf        — T1 latency (mean duration)
  fig1b_t1_throughput.pdf     — T1 throughput
  fig2_t1_boxplot.pdf         — T1 generation-time distribution (boxplot)
  fig3_t2_reliability.pdf     — T2 success rate bar
  fig4a_t3_discovery_full.pdf — T3 F1 / precision / recall (full scope)
    fig4b_t3_discovery_py.pdf   — T3 F1 / precision / recall (python scope)
    fig4c_t3_discovery_mixed.pdf — T3 F1 / precision / recall (mixed scope)
    fig4d_t3_discovery_config.pdf — T3 F1 / precision / recall (config scope)
    fig4e_t3_discovery_cloud_api.pdf — T3 F1 / precision / recall (cloud_api scope)
    fig5a_t4_conformance.pdf — T4 CycloneDX conformance (G1 + G2)
    fig5b_t4_completeness.pdf — T4 CycloneDX completeness & breadth (G1 + G2)

Usage:
    python -m analysis.graphs --run-id <run-id>
    uv run eval aggregate --run-id <run-id>
"""

from __future__ import annotations

import glob
import json
import re
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
T3_TOOLS  = ["aibomgen-cli", "trusera-aibom", "akumar-aibom"]
T3_PY_TOOLS = ["aibomgen-cli", "trusera-aibom", "akumar-aibom", "jorisbot-aibom"]
T4G1_TOOLS = ["aibomgen-cli", "aibom-generator", "ALOHA", "lab700xorg-aisbom"]
T4G2_TOOLS = ["aibomgen-cli", "akumar-aibom", "trusera-aibom"]


# ================================================================== #
# CLI entry point                                                       #
# ================================================================== #

@click.command()
@click.option("--run-id", required=True, help="Run ID to generate figures for.")
def main(run_id: str) -> None:
    """Generate publication-ready figures for a completed aggregated run."""
    summary_path = AGGREGATED_DIR / "data" / run_id / f"{run_id}_summary.json"
    if not summary_path.exists():
        click.echo(f"Summary not found: {summary_path}", err=True)
        sys.exit(1)

    with open(summary_path) as f:
        S: dict[str, Any] = json.load(f)

    meta_path = AGGREGATED_DIR / f"{run_id}_meta.json"
    group_meta: dict[str, Any] | None = None
    if meta_path.exists():
        with open(meta_path) as f:
            group_meta = json.load(f)

    out_dir = FIGURES_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(_RC)

    _fig1a_t1_latency(S, out_dir)
    _fig1b_t1_throughput(S, out_dir)
    _fig2_t1_boxplot(S, run_id, out_dir, group_meta)
    _fig3_t2_reliability(S, out_dir)
    _fig4a_t3_discovery_full(S, out_dir)
    _fig4b_t3_discovery_py(S, out_dir)
    _fig4c_t3_discovery_mixed(S, out_dir)
    _fig4d_t3_discovery_config(S, out_dir)
    _fig4e_t3_discovery_cloud_api(S, out_dir)
    _fig5a_t4_conformance(S, out_dir)
    _fig5b_t4_completeness(S, out_dir)

    click.echo(f"Figures written to: {out_dir}")


# ================================================================== #
# Figure helpers                                                        #
# ================================================================== #

def _fig1a_t1_latency(S: dict, out: Path) -> None:
    """Figure 1a — T1: latency (mean duration)."""
    tools  = [t for t in T1_TOOLS if f"{t}::{SUBTASK_T1}" in S]
    labels = [TOOL_LABELS[t] for t in tools]
    colors = [PALETTE[t] for t in tools]
    duration = [S[f"{t}::{SUBTASK_T1}"]["metrics"]["duration_seconds"]["mean"] for t in tools]
    x = np.arange(len(tools))

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    bars = ax.bar(x, duration, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean duration (s)")
    ax.set_title("Latency")
    for bar, v in zip(bars, duration):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}s",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig1a_t1_latency.pdf", bbox_inches="tight")
    plt.close(fig)
    click.echo("  Saved fig1a_t1_latency.pdf")


def _fig1b_t1_throughput(S: dict, out: Path) -> None:
    """Figure 1b — T1: throughput."""
    tools  = [t for t in T1_TOOLS if f"{t}::{SUBTASK_T1}" in S]
    labels = [TOOL_LABELS[t] for t in tools]
    colors = [PALETTE[t] for t in tools]
    throughput = [S[f"{t}::{SUBTASK_T1}"]["metrics"]["normalized_throughput"]["mean"] for t in tools]
    x = np.arange(len(tools))

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    bars = ax.bar(x, throughput, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Throughput (fields / s)")
    ax.set_title("Throughput")
    for bar, v in zip(bars, throughput):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.1f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig1b_t1_throughput.pdf", bbox_inches="tight")
    plt.close(fig)
    click.echo("  Saved fig1b_t1_throughput.pdf")


def _fig2_t1_boxplot(
    S: dict,
    run_id: str,
    out: Path,
    group_meta: dict[str, Any] | None = None,
) -> None:
    """Figure 2 — T1: generation-time distribution (boxplot)."""
    run_ids = [run_id]
    if isinstance(group_meta, dict):
        maybe_ids = group_meta.get("run_ids")
        if isinstance(maybe_ids, list) and maybe_ids:
            run_ids = [rid for rid in maybe_ids if isinstance(rid, str)]

    tools = [t for t in T1_TOOLS if f"{t}::{SUBTASK_T1}" in S]

    def _load_durations_and_repeats(tool: str) -> tuple[list[float], set[int]]:
        durations: list[float] = []
        repeat_ids: set[int] = set()
        for rid in run_ids:
            run_base = RESULTS_DIR / rid
            pattern = str(run_base / tool / "T1_T2" / "**" / "metrics.json")
            for path in glob.glob(pattern, recursive=True):
                repeat_dir = Path(path).parent.name
                m_repeat = re.fullmatch(r"r(\d+)", repeat_dir)
                if m_repeat:
                    repeat_ids.add(int(m_repeat.group(1)))
                with open(path) as fh:
                    m = json.load(fh)
                if m.get("task") != "T1_T2":
                    continue
                if not m.get("success"):
                    continue
                dur = m.get("duration_seconds")
                if isinstance(dur, (int, float)):
                    durations.append(float(dur))
        return sorted(durations), repeat_ids

    tool_stats = {t: _load_durations_and_repeats(t) for t in tools}
    dur_data = {t: stats[0] for t, stats in tool_stats.items()}
    tools = [t for t in tools if dur_data.get(t)]
    if not tools:
        click.echo("  Skipped fig2_t1_boxplot.pdf (no successful T1 samples found)")
        return

    def _repeat_n(tool: str) -> int:
        repeats = tool_stats.get(tool, ([], set()))[1]
        return len(repeats) if repeats else 1

    repeat_n = max((_repeat_n(t) for t in tools), default=1)
    box_labels = [
        f"{TOOL_LABELS[t]}\n(repeats={repeat_n}, N={len(dur_data[t])})"
        for t in tools
    ]
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

        # Stagger labels if mean/median are visually too close on a log scale.
        close_on_log_scale = abs(np.log10(mean_val) - np.log10(med_val)) < 0.08
        if close_on_log_scale:
            med_x, mean_x = i + 0.26, i + 0.33
            med_y, mean_y = med_val * 0.92, mean_val * 1.08
        else:
            med_x = mean_x = i + 0.28
            med_y, mean_y = med_val, mean_val

        ax.text(med_x, med_y, f"med {med_val:.2f}s",
            ha="left", va="center", fontsize=7.5, color="dimgrey")
        ax.text(mean_x, mean_y, f"μ {mean_val:.2f}s",
            ha="left", va="center", fontsize=7.5, color="black")

    ax.set_yscale("log")
    ax.set_ylabel("Generation time (s, log scale)")
    ax.set_xticklabels(box_labels, fontsize=7.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:g}s"))

    med_patch  = plt.Line2D([0], [0], color="black", linewidth=1.2, label="Median")
    mean_patch = plt.Line2D([0], [0], marker="x", color="w", markerfacecolor="black",
                            markeredgecolor="black", markersize=8, markeredgewidth=2,
                            label="Mean (μ)")
    ax.legend(handles=[med_patch, mean_patch], frameon=False, fontsize=7.5, loc="upper left")
    fig.text(0.99, 0.01, "N = models x repeats", ha="right", va="bottom", fontsize=7, color="dimgrey")

    fig.tight_layout()
    fig.savefig(out / "fig2_t1_boxplot.pdf", bbox_inches="tight")
    plt.close(fig)
    click.echo("  Saved fig2_t1_boxplot.pdf")


def _fig3_t2_reliability(S: dict, out: Path) -> None:
    """Figure 3 — T2: success rate bar (with crash / auth-error overlay)."""
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
    ax.bar(x2, crash_frac, bottom=success_frac,
           color="black", alpha=0.65, edgecolor="white", label="Crash")
    ax.bar(x2, unauth_frac,
           bottom=[s + c for s, c in zip(success_frac, crash_frac)],
           color="grey", alpha=0.65, edgecolor="white", hatch="xx", label="Auth error")

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


def _fig4a_t3_discovery_full(S: dict, out: Path) -> None:
    """Figure 4a — T3: F1 / precision / recall (full scope)."""
    _fig4_t3_panel(
        S,
        out,
        filename="fig4a_t3_discovery_full.pdf",
        title="Full scope (all file types)",
        tools=T3_TOOLS,
        f1_key="f1",
        precision_key="precision",
        recall_key="recall",
    )


def _fig4b_t3_discovery_py(S: dict, out: Path) -> None:
    """Figure 4b — T3: python-scope metrics."""
    _fig4_t3_panel(
        S,
        out,
        filename="fig4b_t3_discovery_py.pdf",
        title="Python scope",
        tools=T3_PY_TOOLS,
        f1_key="f1_scope_python",
        precision_key="precision_scope_python",
        recall_key="recall_scope_python",
    )


def _fig4c_t3_discovery_mixed(S: dict, out: Path) -> None:
    """Figure 4c — T3: mixed-scope metrics."""
    _fig4_t3_panel(
        S,
        out,
        filename="fig4c_t3_discovery_mixed.pdf",
        title="Mixed scope",
        tools=T3_TOOLS,
        f1_key="f1_scope_mixed",
        precision_key="precision_scope_mixed",
        recall_key="recall_scope_mixed",
    )


def _fig4d_t3_discovery_config(S: dict, out: Path) -> None:
    """Figure 4d — T3: config-scope metrics."""
    _fig4_t3_panel(
        S,
        out,
        filename="fig4d_t3_discovery_config.pdf",
        title="Config scope",
        tools=T3_TOOLS,
        f1_key="f1_scope_config",
        precision_key="precision_scope_config",
        recall_key="recall_scope_config",
    )


def _fig4e_t3_discovery_cloud_api(S: dict, out: Path) -> None:
    """Figure 4e — T3: cloud_api-scope metrics."""
    _fig4_t3_panel(
        S,
        out,
        filename="fig4e_t3_discovery_cloud_api.pdf",
        title="Cloud API scope",
        tools=T3_TOOLS,
        f1_key="f1_scope_cloud_api",
        precision_key="precision_scope_cloud_api",
        recall_key="recall_scope_cloud_api",
    )


def _fig4_t3_panel(
    S: dict,
    out: Path,
    filename: str,
    title: str,
    tools: list[str],
    f1_key: str,
    precision_key: str,
    recall_key: str,
) -> None:
    """Render a T3 panel with F1/precision/recall for one metric key set."""

    def _metric_mean(tool: str, key: str) -> float | None:
        entry = S.get(f"{tool}::{SUBTASK_T3}", {})
        metric = entry.get("metrics", {}).get(key)
        if isinstance(metric, dict):
            value = metric.get("mean")
            return float(value) if isinstance(value, (int, float)) else None
        if isinstance(metric, (int, float)):
            return float(metric)
        return None

    panel_tools = [t for t in tools if _metric_mean(t, f1_key) is not None]
    if not panel_tools:
        click.echo(f"  Skipped {filename} (no data)")
        return

    labels = [TOOL_LABELS[t] for t in panel_tools]
    colors = [PALETTE[t] for t in panel_tools]

    f1_vals = [_metric_mean(t, f1_key) or 0.0 for t in panel_tools]
    prec_vals = [_metric_mean(t, precision_key) or 0.0 for t in panel_tools]
    rec_vals = [_metric_mean(t, recall_key) or 0.0 for t in panel_tools]

    x3 = np.arange(len(panel_tools))
    w3 = 0.25

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    offsets = np.linspace(-w3, w3, 3)
    for offset, (vals, hatch) in zip(offsets, [(f1_vals, None), (prec_vals, "//"), (rec_vals, "xx")]):
        ax.bar(
            x3 + offset,
            vals,
            width=w3,
            color=colors,
            edgecolor="white",
            hatch=hatch,
            alpha=0.9 if hatch else 1.0,
        )

    ax.set_xticks(x3)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)

    tool_patches = [mpatches.Patch(color=PALETTE[t], label=TOOL_LABELS[t]) for t in panel_tools]
    metric_patches = [
        mpatches.Patch(facecolor="grey", label="F1"),
        mpatches.Patch(facecolor="grey", hatch="//", label="Precision"),
        mpatches.Patch(facecolor="grey", hatch="xx", label="Recall"),
    ]
    ax.add_artist(ax.legend(handles=metric_patches, frameon=False, fontsize=8, loc="upper left"))
    ax.legend(handles=tool_patches, frameon=False, fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(out / filename, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"  Saved {filename}")


def _fig5a_t4_conformance(S: dict, out: Path) -> None:
    """Figure 5a — T4 conformance side-by-side panels for G1 and G2."""

    def _panel(ax: Any, subtask: str, tool_group: list[str], title: str, show_ylabel: bool) -> bool:
        tools = [t for t in tool_group if f"{t}::{subtask}" in S]
        if not tools:
            return False

        labels = [TOOL_LABELS[t] for t in tools]
        colors = [PALETTE[t] for t in tools]
        x = np.arange(len(tools))
        w2 = 0.35

        def _metric(t: str, key: str) -> float:
            val = S[f"{t}::{subtask}"]["metrics"][key]
            return val["mean"] if isinstance(val, dict) else float(val)

        conform_v16 = [_metric(t, "cdx_conformance_pass_rate_v16") for t in tools]
        conform_v17 = [_metric(t, "cdx_conformance_pass_rate_v17") for t in tools]

        ax.bar(x - w2 / 2, conform_v16, width=w2, color=colors, edgecolor="white")
        ax.bar(x + w2 / 2, conform_v17, width=w2, color=colors, edgecolor="white", hatch="//", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylim(0, 1.15)
        if show_ylabel:
            ax.set_ylabel("Conformance pass rate")
        ax.set_title(title)
        return True

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), sharey=True)
    has_g1 = _panel(axes[0], SUBTASK_T4G1, T4G1_TOOLS, "G1", True)
    has_g2 = _panel(axes[1], SUBTASK_T4G2, T4G2_TOOLS, "G2", False)

    if not has_g1 and not has_g2:
        plt.close(fig)
        click.echo("  Skipped fig5a_t4_conformance.pdf (no data)")
        return

    v16_patch = mpatches.Patch(facecolor="grey", label="v1.6")
    v17_patch = mpatches.Patch(facecolor="grey", hatch="//", alpha=0.7, label="v1.7")
    fig.legend(handles=[v16_patch, v17_patch], frameon=False, fontsize=8, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_name = "fig5a_t4_conformance.pdf"
    fig.savefig(out / out_name, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"  Saved {out_name}")


def _fig5b_t4_completeness(S: dict, out: Path) -> None:
    """Figure 5b — T4 completeness/breadth side-by-side panels for G1 and G2."""

    def _panel(ax: Any, subtask: str, tool_group: list[str], title: str, show_ylabel: bool) -> bool:
        tools = [t for t in tool_group if f"{t}::{subtask}" in S]
        if not tools:
            return False

        labels = [TOOL_LABELS[t] for t in tools]
        colors = [PALETTE[t] for t in tools]
        x = np.arange(len(tools))
        w4 = 0.18

        def _metric(t: str, key: str) -> float:
            val = S[f"{t}::{subtask}"]["metrics"][key]
            return val["mean"] if isinstance(val, dict) else float(val)

        complt_v16 = [_metric(t, "cdx_completeness_pct_v16") for t in tools]
        complt_v17 = [_metric(t, "cdx_completeness_pct_v17") for t in tools]
        breadth_v16 = [_metric(t, "cdx_breadth_pct_v16") for t in tools]
        breadth_v17 = [_metric(t, "cdx_breadth_pct_v17") for t in tools]

        bar_groups = [
            (complt_v16, {}),
            (complt_v17, {"hatch": "//", "alpha": 0.7}),
            (breadth_v16, {"hatch": "xx"}),
            (breadth_v17, {"hatch": "xx", "alpha": 0.7}),
        ]
        offsets = [-1.5 * w4, -0.5 * w4, 0.5 * w4, 1.5 * w4]

        for offset, (vals, kwargs) in zip(offsets, bar_groups):
            ax.bar(x + offset, vals, width=w4, color=colors, edgecolor="white", **kwargs)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        if show_ylabel:
            ax.set_ylabel("Coverage score (\u00d7100 = %)")
        ax.set_title(title)
        return True

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.8), sharey=True)
    has_g1 = _panel(axes[0], SUBTASK_T4G1, T4G1_TOOLS, "G1", True)
    has_g2 = _panel(axes[1], SUBTASK_T4G2, T4G2_TOOLS, "G2", False)

    if not has_g1 and not has_g2:
        plt.close(fig)
        click.echo("  Skipped fig5b_t4_completeness.pdf (no data)")
        return

    patches = [
        mpatches.Patch(facecolor="grey", label="Completeness v1.6"),
        mpatches.Patch(facecolor="grey", hatch="//", alpha=0.7, label="Completeness v1.7"),
        mpatches.Patch(facecolor="grey", hatch="xx", label="Breadth v1.6"),
        mpatches.Patch(facecolor="grey", hatch="xx", alpha=0.7, label="Breadth v1.7"),
    ]
    fig.legend(handles=patches, frameon=False, fontsize=7.5, loc="upper center", ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out_name = "fig5b_t4_completeness.pdf"
    fig.savefig(out / out_name, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"  Saved {out_name}")
