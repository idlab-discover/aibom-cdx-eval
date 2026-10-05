# Aggregated output and figures

`uv run eval aggregate --run-id <run>` writes to `<root>/aggregated/` (default root `results/`).

## Data: `data/<run>/`

| File | Content |
|---|---|
| `<run>_summary.json` | every metric, per `tool::task` (T1, T2, T3, T4-model-id, T4-source-scan); T3 per scope under `metrics.by_scope` |
| `<run>_report.md` | every headline table in one markdown file |
| `<run>_t1_performance.csv`, `<run>_t2_reliability.csv` | T1 / T2 tables |
| `<run>_t3_scope_{full,python,mixed,config,cloud_api}.{json,csv}` | T3 tables, all targets and per scope |
| `<run>_t4_model_id.csv`, `<run>_t4_source_scan.csv` | T4 tables per run type |

## Figures: `figures/<run>/`

Every figure is written as both PDF and PNG.

| Figure | Run type | Shows |
|---|---|---|
| `fig1_t1_throughput_boxplot` | model ID | T1 throughput (fields / s) |
| `fig2_t1_boxplot` | model ID | T1 generation time |
| `fig3_t2_reliability` | model ID | T2 share of models that succeed, failures by class |
| `fig4a_t3_discovery` | source scan | T3 precision / recall / F1, all targets |
| `fig4b_t3_recall_by_scope` | source scan | T3 recall per target scope, with each tool's documented-support marker |
| `fig5a_t4_conformance_{model_id,source_scan}` | both | CycloneDX conformance pass rate |
| `fig5b_t4_completeness_{model_id,source_scan}` | both | full-BOM completeness and breadth |
| `fig5c_t4_model_completeness_{model_id,source_scan}` | both | model-component completeness |

## Conventions

- **Titles:** none, because the LaTeX caption provides them. The only title is a subtitle naming the run type (for T3, also the scope).
- **One encoding everywhere:**
  - colour = tool
  - shade = metric (dark → light)
  - diagonal hatch = CycloneDX v1.7 (solid = v1.6); hatching means nothing else
- **Legend:** one shared row under the plot, in black/white/grey, explains the shades and hatching for every tool.
- **Error bars:**
  - fig4a shows the 95 % bootstrap CI over targets.
  - fig5b/fig5c show ±1 SD across unique inputs; repeats are averaged per input first.
  - Breadth is one union count per tool and has no error bar.
- **Tools and order:** a figure shows the tools that have data for its task, in `constants.TOOL_ORDER`, labelled with `constants.TOOL_LABELS`. Colours are set in `analysis/graphs.py` `PALETTE`.
