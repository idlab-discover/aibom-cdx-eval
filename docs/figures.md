# Aggregated output and figures

`uv run eval aggregate --run-id <run>` writes to `<root>/aggregated/` (the default root is `results/`).

## Data: `data/<run>/`

| File | Content |
|---|---|
| `<run>_summary.json` | every metric, per `tool::task` (T1, T2, T3, T4-model-id, T4-source-scan); T3 per scope under `metrics.by_scope`; `_scoring.framework_commits` = the scorer version(s) of the `eval score` used |
| `<run>_report.md` | all main tables in one markdown file |
| `<run>_t1_performance.csv`, `<run>_t2_reliability.csv` | T1 and T2 tables |
| `<run>_t3_scope_{full,python,mixed,config,cloud_api}.{json,csv}` | T3 tables, for all targets and per scope |
| `<run>_t3_pairwise.csv` | T3 micro-F1 difference of every tool pair, with a paired 95% bootstrap CI |
| `<run>_t3_loto.csv` | T3 micro F1 of each tool with one target left out |
| `<run>_t4_model_id.csv`, `<run>_t4_source_scan.csv` | T4 tables per run type; every value in % (conformance too; the summary keeps it as a 0-1 pass rate) |

## Figures: `figures/<run>/pdf/` and `figures/<run>/png/`

Every figure is saved as both PDF and PNG.

| Figure | Run type | Shows |
|---|---|---|
| `fig1_t1_generation_time` | model ID | T1 generation time, one value per model (the median of its repeats), with the median Hub requests per run under each box |
| `fig2_t1_throughput` | model ID | T1 throughput (fields per second), one value per model, with the median Hub requests per run under each box |
| `fig3_t2_reliability` | model ID | T2 share of models that succeed, and failures by class |
| `fig4a_t3_discovery` | source scan | T3 precision, recall and F1 over all targets, value on each bar |
| `fig4b_t3_recall_by_scope` | source scan | T3 recall per target scope, with a marker for what each tool's docs claim |
| `fig5a_t4_conformance_{model_id,source_scan}` | both | CycloneDX conformance pass rate |
| `fig5b_t4_completeness_{model_id,source_scan}` | both | full-BOM completeness and breadth |
| `fig5c_t4_ai_completeness_{model_id,source_scan}` | both | AI-relevant completeness and breadth (metadata; ML-model, data and framework components; vulnerabilities that affect them) |

## Rules

- **Figures match the tables:** figures only read `<run>_summary.json`, the same file the tables and the report come from. They pick tools with the same `scoring.aggregate.tools_for`. So a number in a figure is the number in the table (T1 boxes: median and quartiles over unique models).
- **No titles or subtitles:** the LaTeX caption gives the title.
- **One encoding everywhere:**
  - colour = tool
  - shade = metric (dark to light)
  - diagonal hatch = CycloneDX v1.7 (solid = v1.6); hatching means nothing else
- **Legend:** one row under the plot. In the grouped-bar figures (fig4a, fig5a-c) it is one shared black/white/grey legend that explains the shades and hatching for every tool, plus the error bar when the figure has one. fig1/fig2 explain median, mean and Hub requests, fig3 the failure classes, fig4b the tool colours and the claim markers.
- **Error bars:**
  - fig4a has none. Its 95% bootstrap CIs over targets are in the T3 table.
  - fig5b and fig5c show +/-1 SD across unique inputs. Repeats are first averaged per input.
  - Breadth is one count per tool and has no error bar.
- **Tools and order:** a figure shows the tools that have data for its task, in the order of `constants.TOOL_ORDER`, with names from `constants.TOOL_LABELS`. Colours are set in `PALETTE` in `analysis/graphs.py`.
