"""
analysis/graphs.py — generate publication-ready figures from aggregated results.

Reads <root>/aggregated/data/<run-id>/<run-id>_summary.json (and, for the T1
boxplots, the run's metrics.json files under <root>/runs/<run-id>/) and writes PDF +
PNG figures to <root>/aggregated/figures/<run-id>/ (root = --results-dir).

Figures (run type = what the tool was given):
  fig1_t1_generation_time            model ID      T1 generation-time distribution
                                                   (+ median Hub requests per run)
  fig2_t1_throughput                 model ID      T1 throughput distribution
  fig3_t2_reliability                model ID      T2 share of models that succeed / fail
  fig4a_t3_discovery                 source scan   T3 precision / recall / F1, all targets
  fig4b_t3_recall_by_scope           source scan   T3 recall per target scope
  fig5a_t4_conformance_model_id / _source_scan      CycloneDX conformance pass rate
  fig5b_t4_completeness_model_id / _source_scan     full-BOM completeness and breadth
  fig5c_t4_ai_completeness_model_id / _source_scan      AI-relevant completeness and breadth

Style rules (all figures)
  * No plot titles — the LaTeX caption provides them.  The only title is a short
    subtitle naming the run type (and, for T3, the scope).
  * One visual encoding everywhere: colour = tool, shade (dark → light) = metric,
    diagonal hatch = CycloneDX v1.7 (solid = v1.6).  Hatching means nothing else.
  * One shared legend row below the plot in black/white/grey explains the shades and
    hatching for every tool (each tool keeps its own colour).  Legends never widen
    a figure.

Usage:
    uv run eval aggregate --run-id <run-id>
"""

from __future__ import annotations

import glob
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import click
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from constants import (
    EVAL_ROOT,
    EC_CRASH,
    EC_GATED_OR_UNAUTHORIZED,
    K_NORMALIZED_THROUGHPUT,
    TASK_T1,
    TASK_T2,
    TASK_T3,
    RUN_MODEL_ID,
    RUN_SOURCE_SCAN,
    RUN_TYPE_LABELS,
    T4_RUN_TYPE,
    TOOL_LABELS,
    TOOL_ORDER,
)
from scoring.aggregate import hub_request_medians

# ── Style ──────────────────────────────────────────────────────────────────

_RC = {
    "figure.dpi":        150,
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    9,
    "axes.labelsize":    10,
    "legend.fontsize":   7,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
}

# Colour = tool.
PALETTE: dict[str, str] = {
    "aibomgen-cli":      "#1f77b4",
    "aibom-generator":   "#ff7f0e",
    "ALOHA":             "#2ca02c",
    "lab700xorg-aisbom": "#d62728",
    "akumar-aibom":      "#9467bd",
    "jorisbot-aibom":    "#8c564b",
    "trusera-aibom":     "#e377c2",
    "cdxgen-aibom":      "#17becf",
    "hts-aibom":         "#bcbd22",
}

# Shade = metric: share of white blended into the tool colour (0 = the tool colour itself).
_SHADE_WHITE = {"dark": 0.0, "mid": 0.40, "light": 0.68}

# Hatch = CycloneDX v1.7 (the only use of hatching in the whole figure set).
HATCH_V17 = "///"

plt.rcParams["hatch.linewidth"] = 1.1

_INK = "#333333"
_SUBTITLE_INK = "#555555"

# Every figure shows the tools that have data for its task in the summary,
# in TOOL_ORDER.



def shade(tool: str, level: str) -> tuple[float, float, float]:
    """Tool colour blended toward white: level in {dark, mid, light}."""
    base = np.array(mcolors.to_rgb(PALETTE[tool]))
    t = _SHADE_WHITE[level]
    r, g, b = base * (1 - t) + t
    return float(r), float(g), float(b)


# ================================================================== #
# Entry point                                                           #
# ================================================================== #

def make_figures(run_id: str, results_root: Path) -> None:
    """Generate the figures of an aggregated run (see the module docstring)."""
    summary_path = results_root / "aggregated" / "data" / run_id / f"{run_id}_summary.json"
    S: dict[str, Any] = json.loads(summary_path.read_text())

    out_dir = results_root / "aggregated" / "figures" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(_RC)

    run_dir = results_root / "runs" / run_id
    _fig1_t1_generation_time(S, run_dir, out_dir)
    _fig2_t1_throughput(S, run_dir, out_dir)
    _fig3_t2_reliability(S, out_dir)
    _fig4_t3_discovery(S, out_dir)
    _fig5a_t4_conformance(S, out_dir)
    _fig5b_t4_completeness(S, out_dir)
    _fig5c_t4_ai_completeness(S, out_dir)

    click.echo(f"Figures written to: {out_dir}")


# ================================================================== #
# Shared helpers                                                        #
# ================================================================== #

def _save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    """Write a figure to both PDF and PNG outputs."""
    for suffix in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{suffix}", bbox_inches="tight")


def _tick_label(tool: str) -> str:
    """Horizontal tick label: a tool name wrapped at its first space, and every line
    longer than 10 characters at its last hyphen, so neighbouring labels never overlap."""
    lines = []
    for line in TOOL_LABELS[tool].split(" ", 1):
        if len(line) > 10 and "-" in line:
            head, tail = line.rsplit("-", 1)
            lines += [f"{head}-", tail]
        else:
            lines.append(line)
    return "\n".join(lines)


def _subtitle(ax: Any, text: str) -> None:
    ax.set_title(text, fontsize=9, color=_SUBTITLE_INK, pad=6)


def _label_bars(
    ax: Any,
    xs: Any,
    values: list[float],
    fmt: Callable[[float], str],
    pad: float,
    fontsize: float = 7,
    rotation: float = 0,
    yerr: list[float] | None = None,
) -> None:
    """Write each bar's value just above it (above the error cap if present)."""
    for i, (x, v) in enumerate(zip(xs, values)):
        err = yerr[i] if yerr is not None and not np.isnan(yerr[i]) else 0.0
        ax.text(x, v + err + pad, fmt(v), ha="center", va="bottom",
                fontsize=fontsize, rotation=rotation, color=_INK)


@dataclass(frozen=True)
class Series:
    """One bar series of a grouped-bar figure."""
    key: str                 # metric key in the summary
    label: str               # legend text
    level: str               # shade level: dark | mid | light
    v17: bool = False        # hatched (CycloneDX v1.7)?
    err: str | None = None   # None | "std" | "ci"


def _bar_kwargs(tool: str, s: Series) -> dict[str, Any]:
    """Bar style.  Hatch lines are white on dark/mid shades and the tool's own
    colour on the light shade, so v1.7 reads as "same shade, with diagonals"."""
    if not s.v17:
        return {"facecolor": shade(tool, s.level), "edgecolor": "white",
                "linewidth": 0.6, "hatch": None}
    line = shade(tool, "dark") if s.level == "light" else "white"
    return {"facecolor": shade(tool, s.level), "edgecolor": line,
            "linewidth": 0.0, "hatch": HATCH_V17}


def _below_axes_anchor(fig: plt.Figure, ax: Any, gap: float = 0.04) -> float:
    """y (axes coordinates) just below the lowest tick label / axis text."""
    fig.canvas.draw()
    box = ax.get_tightbbox(fig.canvas.get_renderer())
    y_below = ax.transAxes.inverted().transform((box.x0, box.y0))[1]
    return y_below - gap


# Neutral swatches for the shared legend: the shade of a bar is explained once, in
# black/white/grey, and applies to every tool (each tool keeps its own colour).
_LEGEND_GREY = {"dark": "#4d4d4d", "mid": "#9a9a9a", "light": "#d6d6d6"}


def _legend_patch(s: Series) -> mpatches.Patch:
    """Grey swatch for one series: shade = metric, diagonals = CycloneDX v1.7."""
    if not s.v17:
        return mpatches.Patch(facecolor=_LEGEND_GREY[s.level], edgecolor="white",
                              linewidth=0.6, label=s.label)
    line = "#4d4d4d" if s.level == "light" else "white"
    return mpatches.Patch(facecolor=_LEGEND_GREY[s.level], edgecolor=line,
                          linewidth=0.0, hatch=HATCH_V17, label=s.label)


def _shared_legend(fig: plt.Figure, ax: Any, series: list[Series]) -> None:
    """One compact black/white/grey legend row below the plot, valid for all tools."""
    legend = ax.legend(
        handles=[_legend_patch(s) for s in series],
        ncol=len(series), loc="upper center",
        bbox_to_anchor=(0.5, _below_axes_anchor(fig, ax)), frameon=False,
        columnspacing=1.2, handlelength=1.4, handleheight=0.9,
        handletextpad=0.5, fontsize=7,
    )
    for text in legend.get_texts():
        text.set_color(_INK)


def _metric_entry(S: dict, tool: str, task: str, key: str) -> dict | float | None:
    return (S.get(f"{tool}::{task}") or {}).get("metrics", {}).get(key)


def _entry_mean(entry: Any) -> float | None:
    if isinstance(entry, dict):
        entry = entry.get("mean")
    return float(entry) if isinstance(entry, (int, float)) else None


def _entry_err(entry: Any, kind: str | None) -> tuple[float, float]:
    """(lower, upper) error-bar lengths for a summary metric entry."""
    nan = float("nan")
    if kind is None or not isinstance(entry, dict):
        return nan, nan
    mean = entry.get("mean")
    if kind == "std":
        std = entry.get("std")
        return (float(std), float(std)) if isinstance(std, (int, float)) else (nan, nan)
    lo, hi = entry.get("ci_low"), entry.get("ci_high")
    if isinstance(mean, (int, float)) and isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
        return max(mean - lo, 0.0), max(hi - mean, 0.0)
    return nan, nan


def _grouped_bar_figure(
    S: dict,
    out: Path,
    stem: str,
    task: str,
    tools: list[str],
    series: list[Series],
    ylabel: str,
    subtitle: str,
    figsize: tuple[float, float],
    fmt: Callable[[float], str] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Grouped bars: one group per tool, one bar per series.

    Encoding: colour = tool, shade = metric, hatch = CycloneDX v1.7 (explained by the
    shared grey legend).  ``fmt`` (when
    given) writes each bar's value on top of it.  ``err`` on a series draws
    ±SD ("std") or 95 % CI ("ci") error bars.
    """
    tools = [t for t in tools if _metric_entry(S, t, task, series[0].key) is not None]
    if not tools:
        click.echo(f"  Skipped {stem} (no data)")
        return

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(tools))
    width = 0.78 / len(series)
    offsets = [(i - (len(series) - 1) / 2) * width for i in range(len(series))]

    means: dict[str, list[float]] = {}
    tops: list[float] = []
    for s, off in zip(series, offsets):
        entries = [_metric_entry(S, t, task, s.key) for t in tools]
        vals = [_entry_mean(e) or 0.0 for e in entries]
        errs = [_entry_err(e, s.err) for e in entries]
        lo = np.array([e[0] for e in errs])
        hi = np.array([e[1] for e in errs])
        means[s.key] = vals
        for i, tool in enumerate(tools):
            kw = _bar_kwargs(tool, s)
            ax.bar(x[i] + off, vals[i], width=width, **kw)
            if s.err and not np.isnan(lo[i]):
                ax.errorbar(x[i] + off, vals[i], yerr=[[lo[i]], [hi[i]]], fmt="none",
                            ecolor=_INK, elinewidth=0.8, capsize=2, capthick=0.8)
        top = [v + (0.0 if np.isnan(h) else h) for v, h in zip(vals, hi)]
        tops.extend(top)
        if fmt:
            _label_bars(ax, x + off, vals, fmt, pad=(ylim[1] if ylim else max(tops)) * 0.012,
                        fontsize=6.5, rotation=90 if len(series) > 1 else 0,
                        yerr=[0.0 if np.isnan(h) else float(h) for h in hi])

    ax.set_xticks(x)
    ax.set_xticklabels([_tick_label(t) for t in tools], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*(ylim if ylim else (0, (max(tops, default=0) or 1.0) * 1.25)))
    _subtitle(ax, subtitle)

    fig.tight_layout()
    _shared_legend(fig, ax, series)
    _save_figure(fig, out, stem)
    plt.close(fig)
    click.echo(f"  Saved {stem}.pdf/.png")


# ================================================================== #
# T1 — boxplots                                                         #
# ================================================================== #

def _load_t1_samples(
    tool: str,
    run_dir: Path,
    extract: Callable[[dict], Any],
) -> tuple[list[float], set[int]]:
    """Collect one value per successful model-ID run, plus the repeat indices seen."""
    values: list[float] = []
    repeat_ids: set[int] = set()
    pattern = str(run_dir / tool / RUN_MODEL_ID / "**" / "metrics.json")
    for path in glob.glob(pattern, recursive=True):
        m_repeat = re.fullmatch(r"r(\d+)", Path(path).parent.name)
        if m_repeat:
            repeat_ids.add(int(m_repeat.group(1)))
        with open(path) as fh:
            m = json.load(fh)
        if m.get("run_type") != RUN_MODEL_ID or not m.get("success"):
            continue
        val = extract(m)
        if isinstance(val, (int, float)):
            values.append(float(val))
    return sorted(values), repeat_ids


def _t1_boxplot(
    S: dict,
    run_dir: Path,
    out: Path,
    extract: Callable[[dict], Any],
    ylabel: str,
    unit: str,
    value_fmt: str,
    stem: str,
    hub_requests: dict | None = None,
) -> None:
    """Render a log-scale per-tool boxplot of one per-run T1 quantity.  With
    ``hub_requests`` ({tool: median}), each tool's median number of Hub
    requests per run is written small under its box and explained in the legend."""
    tools = [t for t in TOOL_ORDER if f"{t}::{TASK_T1}" in S]

    tool_stats = {t: _load_t1_samples(t, run_dir, extract) for t in tools}
    data = {t: stats[0] for t, stats in tool_stats.items()}
    tools = [t for t in tools if data.get(t)]
    if not tools:
        click.echo(f"  Skipped {stem}.pdf (no successful T1 samples found)")
        return

    def _repeat_n(tool: str) -> int:
        repeats = tool_stats.get(tool, ([], set()))[1]
        return len(repeats) if repeats else 1

    repeat_n = max((_repeat_n(t) for t in tools), default=1)
    box_labels = [f"{_tick_label(t)}\nN={len(data[t])}" for t in tools]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    bp = ax.boxplot(
        [data[t] for t in tools],
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

    for patch, tool, flier in zip(bp["boxes"], tools, bp["fliers"]):
        patch.set_facecolor(PALETTE[tool])
        patch.set_alpha(0.55)
        flier.set_markerfacecolor(PALETTE[tool])
        flier.set_markeredgecolor(PALETTE[tool])

    for i, (med_line, mean_marker) in enumerate(zip(bp["medians"], bp["means"]), start=1):
        med_val = med_line.get_ydata()[0]
        mean_val = mean_marker.get_ydata()[0]

        # Stagger labels if mean/median are visually too close on a log scale:
        # centre them on their log-midpoint, at least _MIN_SEP decades apart,
        # keeping the larger value's label on top.
        _MIN_SEP = 0.12
        log_med, log_mean = np.log10(med_val), np.log10(mean_val)
        if abs(log_mean - log_med) < _MIN_SEP:
            mid = (log_med + log_mean) / 2
            sign = 1 if mean_val >= med_val else -1
            med_x = mean_x = i + 0.28
            med_y = 10 ** (mid - sign * _MIN_SEP / 2)
            mean_y = 10 ** (mid + sign * _MIN_SEP / 2)
        else:
            med_x = mean_x = i + 0.28
            med_y, mean_y = med_val, mean_val

        ax.text(med_x, med_y, f"med {med_val:{value_fmt}}{unit}",
                ha="left", va="center", fontsize=7.5, color="dimgrey")
        ax.text(mean_x, mean_y, f"μ {mean_val:{value_fmt}}{unit}",
                ha="left", va="center", fontsize=7.5, color="black")

    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(box_labels, fontsize=7.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:g}{unit}"))
    _subtitle(ax, RUN_TYPE_LABELS[RUN_MODEL_ID])

    med_patch = plt.Line2D([0], [0], color="black", linewidth=1.2, label="Median")
    mean_patch = plt.Line2D([0], [0], marker="x", color="w", markerfacecolor="black",
                            markeredgecolor="black", markersize=8, markeredgewidth=2,
                            label="Mean (μ)")
    handles = [med_patch, mean_patch]
    hub = {t: hub_requests[t] for t in tools if t in (hub_requests or {})}
    if hub:
        # Free a strip at the bottom (8 % of the log axis) for the numbers.
        lo, hi = (np.log10(v) for v in ax.get_ylim())
        ax.set_ylim(bottom=10 ** ((lo - 0.08 * hi) / 0.92))
        for i, t in enumerate(tools, start=1):
            if t in hub:
                ax.text(i, 0.025, f"{hub[t]:g}", transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=7.5, color="dimgrey")
        handles.append(plt.Line2D([0], [0], marker="$n$", color="w", markerfacecolor="dimgrey",
                                  markeredgecolor="dimgrey", markersize=7,
                                  label="Hub requests per run (median)"))
    ax.set_xlabel(f"N = models x repeats ({repeat_n} repeats)", fontsize=7, color="dimgrey")
    fig.tight_layout()
    # One legend row below the plot (as in the other figures): it never covers a box.
    ax.legend(handles=handles, frameon=False, fontsize=7.5, ncol=len(handles),
              loc="upper center", bbox_to_anchor=(0.5, _below_axes_anchor(fig, ax)))
    _save_figure(fig, out, stem)
    plt.close(fig)
    click.echo(f"  Saved {stem}.pdf/.png")


def _fig1_t1_generation_time(S: dict, run_dir: Path, out: Path) -> None:
    """Figure 1 — T1: generation-time distribution (boxplot) with the median number
    of Hub requests per run (measured around each job), when available."""
    _t1_boxplot(
        S, run_dir, out,
        extract=lambda m: m.get("duration_seconds"),
        ylabel="Generation time (s, log scale)",
        unit="s",
        value_fmt=".2f",
        stem="fig1_t1_generation_time",
        hub_requests=hub_request_medians(S),
    )


def _fig2_t1_throughput(S: dict, run_dir: Path, out: Path) -> None:
    """Figure 2 — T1: normalised-throughput distribution (boxplot) with the median
    number of Hub requests per run, when available."""
    _t1_boxplot(
        S, run_dir, out,
        extract=lambda m: (m.get("scores") or {}).get(K_NORMALIZED_THROUGHPUT),
        ylabel="Throughput (fields / s, log scale)",
        unit=" f/s",
        value_fmt=".1f",
        stem="fig2_t1_throughput",
        hub_requests=hub_request_medians(S),
    )


# ================================================================== #
# T2 — reliability                                                      #
# ================================================================== #

def _fig3_t2_reliability(S: dict, out: Path) -> None:
    """Figure 3 — T2: share of unique models that succeed, by tool.

    A model counts as failed only if ALL its repeats failed.  Failure classes are
    stacked above the success bar in neutral greys (solid, no hatch).
    """
    tools = [t for t in TOOL_ORDER if f"{t}::{TASK_T2}" in S]
    if not tools:
        click.echo("  Skipped fig3_t2_reliability (no data)")
        return

    grey = {"gated": "#9a9a9a", "crash": "#4d4d4d", "other": "#d2d2d2"}

    def _frac(tool: str, ec: str) -> float:
        e = S[f"{tool}::{TASK_T2}"]
        return e.get("error_breakdown_models", {}).get(ec, 0) / max(e.get("n_models", 1), 1)

    success = [S[f"{t}::{TASK_T2}"]["model_success_rate"] for t in tools]
    gated = [_frac(t, EC_GATED_OR_UNAUTHORIZED) for t in tools]
    crash = [_frac(t, EC_CRASH) for t in tools]
    other = [max(1 - s - g - c, 0.0) for s, g, c in zip(success, gated, crash)]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(tools))
    ax.bar(x, success, color=[PALETTE[t] for t in tools], edgecolor="white")
    bottom = np.array(success)
    for vals, key in ((gated, "gated"), (crash, "crash"), (other, "other")):
        ax.bar(x, vals, bottom=bottom, color=grey[key], edgecolor="white")
        bottom = bottom + np.array(vals)

    for i, v in enumerate(success):
        ax.text(i, v / 2, f"{v * 100:.0f}%", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold")

    n_models = [S[f"{t}::{TASK_T2}"].get("n_models", 0) for t in tools]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{_tick_label(t)}\nN={n}" for t, n in zip(tools, n_models)], fontsize=7.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share of models")
    _subtitle(ax, RUN_TYPE_LABELS[RUN_MODEL_ID])
    fig.tight_layout()
    ax.legend(
        handles=[
            mpatches.Patch(facecolor=grey["gated"], label="gated / unauthorized"),
            mpatches.Patch(facecolor=grey["crash"], label="crash"),
            mpatches.Patch(facecolor=grey["other"], label="other error"),
        ],
        ncol=3, loc="upper center", bbox_to_anchor=(0.5, _below_axes_anchor(fig, ax)),
        frameon=False,
    )
    _save_figure(fig, out, "fig3_t2_reliability")
    plt.close(fig)
    click.echo("  Saved fig3_t2_reliability.pdf/.png")


# ================================================================== #
# T3 — discovery                                                        #
# ================================================================== #

_T3_SCOPES = [
    # (scope bucket, x-axis label)
    ("python",    "Python"),
    ("mixed",     "Mixed"),
    ("config",    "Config files"),
    ("cloud_api", "Cloud API"),
]


def _fig4_t3_discovery(S: dict, out: Path) -> None:
    """Figure 4a — T3: pooled precision / recall / F1 over all targets.

    Bars are micro (pooled over targets) scores; whiskers are the 95 % bootstrap
    CI over targets.  Shade = metric (dark F1, mid precision, light recall).
    """
    tools = [t for t in TOOL_ORDER if f"{t}::{TASK_T3}" in S]
    series = [
        Series("f1",        "F1",        "dark",  err="ci"),
        Series("precision", "Precision", "mid",   err="ci"),
        Series("recall",    "Recall",    "light", err="ci"),
    ]
    _grouped_bar_figure(
        S, out, "fig4a_t3_discovery", TASK_T3, tools, series,
        ylabel="Score", subtitle=f"{RUN_TYPE_LABELS[RUN_SOURCE_SCAN]} — all targets",
        figsize=(5.0, 3.6), ylim=(0, 1.05),
    )
    _fig4b_t3_recall_by_scope(S, out, tools)


_CLAIMS_PATH = EVAL_ROOT / "benchmarks" / "tool-claims.yaml"
# Claim marker under each bar: what the tool's own documentation says it supports.
_CLAIM_STYLE = {
    "yes":     dict(marker="o", markersize=5.5, markerfacecolor="black", markeredgecolor="black"),
    "partial": dict(marker="o", markersize=5.5, fillstyle="left", markerfacecolor="black",
                    markerfacecoloralt="white", markeredgecolor="black"),
    "no":      dict(marker="o", markersize=5.5, markerfacecolor="white", markeredgecolor="black"),
}
_CLAIM_LABEL = {"yes": "claims support", "partial": "claims partial support", "no": "no claim"}


def _load_claims() -> dict[str, dict[str, str]]:
    """tool -> scope -> yes|partial|no, from benchmarks/tool-claims.yaml."""
    try:
        import yaml

        raw = yaml.safe_load(_CLAIMS_PATH.read_text()) or {}
    except Exception:
        return {}
    def norm(value: Any) -> str:
        # YAML 1.1 reads bare yes/no as booleans.
        if isinstance(value, bool):
            return "yes" if value else "no"
        return str(value or "").strip().lower()

    return {
        tool: {scope: norm((v or {}).get("claim")) for scope, v in scopes.items()}
        for tool, scopes in raw.items() if isinstance(scopes, dict)
    }


def _fig4b_t3_recall_by_scope(S: dict, out: Path, tools: list[str]) -> None:
    """Figure 4b — T3: recall per target scope, with each tool's documented claim.

    x = scope, one bar per tool in its own colour, value on each bar.  Under each
    bar a marker shows what the tool's documentation claims for that scope
    (filled = claims support, half = partial, open = no claim; source:
    benchmarks/tool-claims.yaml), so a low bar outside a tool's stated scope is
    distinguishable from a miss inside it.  Precision is not split by scope:
    per scope it rests on a handful of detections.
    """
    stem = "fig4b_t3_recall_by_scope"
    if not tools:
        click.echo(f"  Skipped {stem} (no data)")
        return
    claims = _load_claims()
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    x = np.arange(len(_T3_SCOPES))
    width = 0.8 / len(tools)
    marker_y = -0.07
    for i, tool in enumerate(tools):
        by_scope = _metric_entry(S, tool, TASK_T3, "by_scope") or {}
        vals = [_entry_mean((by_scope.get(k) or {}).get("recall")) or 0.0 for k, _ in _T3_SCOPES]
        xs = x + (i - (len(tools) - 1) / 2) * width
        ax.bar(xs, vals, width=width, color=PALETTE[tool], edgecolor="white", linewidth=0.6,
               label=TOOL_LABELS[tool])
        _label_bars(ax, xs, vals, lambda v: f"{v:.2f}", pad=0.012, fontsize=6.5, rotation=90)
        for xi, (key, _) in zip(xs, _T3_SCOPES):
            claim = claims.get(tool, {}).get(key)
            if claim in _CLAIM_STYLE:
                ax.plot([xi], [marker_y], linestyle="none", clip_on=False, **_CLAIM_STYLE[claim])
    ax.axhline(0, color=_INK, linewidth=0.8)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in _T3_SCOPES])
    ax.tick_params(axis="x", length=0, pad=14)
    ax.set_ylabel("Recall")
    ax.set_ylim(-0.12, 1.12)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.spines["left"].set_bounds(0, 1.0)
    _subtitle(ax, f"{RUN_TYPE_LABELS[RUN_SOURCE_SCAN]} — recall by target scope")
    fig.tight_layout()

    tool_handles = [mpatches.Patch(facecolor=PALETTE[t], edgecolor="white", label=TOOL_LABELS[t]) for t in tools]
    claim_handles = [plt.Line2D([0], [0], linestyle="none", label=_CLAIM_LABEL[c], **_CLAIM_STYLE[c])
                     for c in ("yes", "partial", "no")]
    y0 = _below_axes_anchor(fig, ax)
    ncol = len(tools) if len(tools) <= 4 else 3        # wrap so the legend never widens the figure
    rows = -(-len(tools) // ncol)
    leg1 = ax.legend(handles=tool_handles, ncol=ncol, loc="upper center", frameon=False,
                     fontsize=7, bbox_to_anchor=(0.5, y0), handlelength=1.4, columnspacing=1.2)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=claim_handles, ncol=3, loc="upper center", frameon=False, fontsize=7,
                     bbox_to_anchor=(0.5, y0 - 0.045 - 0.04 * rows), handlelength=1.0, columnspacing=1.4,
                     title="Documented support (marker under each bar)", title_fontsize=7)
    for leg in (leg1, leg2):
        for text in leg.get_texts():
            text.set_color(_INK)
    _save_figure(fig, out, stem)
    plt.close(fig)
    click.echo(f"  Saved {stem}.pdf/.png")


# ================================================================== #
# T4 — CycloneDX quality                                                #
# ================================================================== #

def _fig5a_t4_conformance(S: dict, out: Path) -> None:
    """Figure 5a — T4: CycloneDX conformance pass rate (v1.6 solid, v1.7 hatched)."""
    for task, run_type in T4_RUN_TYPE.items():
        suffix = run_type.replace("-", "_")          # model_id / source_scan
        _grouped_bar_figure(
            S, out, f"fig5a_t4_conformance_{suffix}", task, TOOL_ORDER,
            series=[
                Series("cdx_conformance_pass_rate_v16", "v1.6", "dark"),
                Series("cdx_conformance_pass_rate_v17", "v1.7", "dark", v17=True),
            ],
            ylabel="Conformance pass rate", subtitle=RUN_TYPE_LABELS[run_type],
            figsize=(5.4, 3.8), fmt=lambda v: f"{v * 100:.1f}%".replace(".0%", "%"),
            ylim=(0, 1.12),
        )


def _fig5b_t4_completeness(S: dict, out: Path) -> None:
    """Figure 5b — T4: full-BOM completeness (dark, ±1 SD across inputs) and
    breadth (light).  Hatched = CycloneDX v1.7."""
    for task, run_type in T4_RUN_TYPE.items():
        suffix = run_type.replace("-", "_")          # model_id / source_scan
        _grouped_bar_figure(
            S, out, f"fig5b_t4_completeness_{suffix}", task, TOOL_ORDER,
            series=[
                Series("cdx_completeness_pct_v16", "completeness v1.6", "dark", err="std"),
                Series("cdx_completeness_pct_v17", "completeness v1.7", "dark", v17=True, err="std"),
                Series("cdx_breadth_pct_v16",      "breadth v1.6",      "light"),
                Series("cdx_breadth_pct_v17",      "breadth v1.7",      "light", v17=True),
            ],
            ylabel="Schema field coverage (%)", subtitle=RUN_TYPE_LABELS[run_type],
            figsize=(6.0, 3.8), fmt=lambda v: f"{v:.2f}",
        )


def _fig5c_t4_ai_completeness(S: dict, out: Path) -> None:
    """Figure 5c — T4: AI-relevant completeness (dark, ±1 SD across inputs) and
    breadth (light), as Figure 5b but over the metadata, components[] and
    vulnerabilities[] fields only.  Hatched = CycloneDX v1.7."""
    for task, run_type in T4_RUN_TYPE.items():
        suffix = run_type.replace("-", "_")          # model_id / source_scan
        _grouped_bar_figure(
            S, out, f"fig5c_t4_ai_completeness_{suffix}", task, TOOL_ORDER,
            series=[
                Series("cdx_ai_completeness_pct_v16", "completeness v1.6", "dark", err="std"),
                Series("cdx_ai_completeness_pct_v17", "completeness v1.7", "dark", v17=True, err="std"),
                Series("cdx_ai_breadth_pct_v16",      "breadth v1.6",      "light"),
                Series("cdx_ai_breadth_pct_v17",      "breadth v1.7",      "light", v17=True),
            ],
            ylabel="AI-relevant field coverage (%)", subtitle=RUN_TYPE_LABELS[run_type],
            figsize=(6.0, 3.8), fmt=lambda v: f"{v:.2f}",
        )

