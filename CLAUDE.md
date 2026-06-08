# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Two systems in one Python codebase:

1. **A meta-analysis pipeline** that turns PDFs of AI-productivity papers into per-study effect sizes (speed log-ratios, quality Hedges' g) keyed to O\*NET work activities and occupations.
2. **A Flask review/analysis web app** (`scripts/review_app.py`) that displays those effect sizes, lets a human edit O\*NET assignments / merge duplicates / delete bad rows, uploads new PDFs (running them through the full pipeline in the background), and runs an O\*NET-imputation analysis on the live table state.

The web app is the primary interface. The CLI pipeline is what backs it.

## Running

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
.venv/bin/python scripts/review_app.py        # http://127.0.0.1:5000
```

CLI pipeline (when not driving via the web app):

```bash
.venv/bin/python scripts/orchestrator.py --papers-dir papers/
.venv/bin/python scripts/orchestrator.py --paper <stem> --stage <name>   # one paper, one stage
.venv/bin/python scripts/build_upload_files.py                            # aggregate to per-O*NET CSVs
```

Stage names (matching `config/settings.yaml`): `extract_quotes`, `structure`, `classify_method`, `classify_outcome`, `map_onet`, `compute_speed`, `compute_quality`, `compute_effects`, `assemble`.

The imputation analysis (`scripts/run_analysis.py`) needs two external files **not in this repo**:
- `~/Downloads/Work Activities.xlsx` — O\*NET Work Activities long-format release
- `~/Documents/nyu/yr2/msft/felten_aioe.csv` — Felten et al. AIOE scores

Paths are at the top of `scripts/run_analysis.py`. Update them if the files live elsewhere.

## Architecture

### Pipeline (per-paper)

The orchestrator runs stages sequentially; each stage is a script in `scripts/0*_*.py` with a `run_one(paper_id, ...)` function and a `__main__` CLI. Stage outputs land in `outputs/<stage>/<paper_id>.json`. **Every stage is idempotent** — re-running skips papers whose output exists unless `--force` is passed.

| Stage | Model | Input | Output |
|---|---|---|---|
| `01a_extract_quotes` | Haiku | PDF | `01a_quotes/*.json` — verbatim quotes by category |
| `01b_structure` | Sonnet | quotes | `01_extraction/*.json` — canonical schema (arms, outcomes, reported_statistics) |
| `02_classify_method` | Sonnet | extraction | `02_method_classification/*.json` — `rct` / `field_experiment` / `natural_experiment` / `observational` / `lab_task` / `within_subject` / `other` |
| `03_classify_outcome` | Sonnet | extraction | `03_outcome_classification/*.json` — primary speed and quality outcomes |
| `04_map_onet` | Sonnet (with prompt caching) | extraction + onet refs | `04_onet_mapping/*.json` — one SOC code or one WA-XX |
| `05a_compute_speed` / `05b_compute_quality` | Sonnet for extraction; Python for arithmetic | extraction + outcomes | `05_effect_sizes/*.{speed,quality}.json` |
| `06_assemble_tables` | — | all above | `outputs/final/{speed_table,quality_table,papers_excluded}.csv` |

Key invariants:
- Stage 2 papers tagged `"other"` are dropped from stages 3+ by the orchestrator.
- Stage 5 LLM calls extract *numbers*; Python computes log ratios / Hedges' g deterministically. Don't move arithmetic into the prompts.
- `reported_percent_change` in stage 5 is the paper's raw signed value. The Python direction-convention flip happens in `05a_compute_speed.compute_speed()`. Don't pre-flip in the prompt.
- Inner double-quotes inside verbatim text must be escaped `\"` in JSON output (the 1b prompt has explicit instructions; if you change the prompt, keep that rule or Haiku will produce invalid JSON on quoted phrases).

### Salvage pass (`scripts/salvage_stage5.py`)

A two-step recovery for papers whose stage 5 produced no effect: first re-run with just the extracted quotes (cheap), then escalate to the full PDF with `cache_control: ephemeral` on the document block so speed and quality calls share the cached PDF. Used after a full pipeline run to recover papers where the initial extraction missed numbers that are actually in the paper.

### Web app (`scripts/review_app.py`)

Single Flask file with embedded HTML/JS/CSS (no build step). Three pages plus an API:

- `/` — the review table. Reads `outputs/final/{speed,quality}_table.csv` + per-paper stage outputs. Edits, deletes, and merges persist to `outputs/review_state.json`. **The CSVs are never mutated by the UI** — `_apply_state()` overlays the edit state on the freshly-read rows on every request. Use the "Export CSVs" button to write the overlaid state back to the CSVs (originals get backed up to `*.pre_review.csv`).
- `/run` — parameter form for the imputation. The metric segmented control, β slider, SOC-major checklist, and activity-weight threshold all drive a live heatmap + bar chart (SVG, rendered client-side). The "+ Upload PDF" button on the home page is also part of this surface.
- `/results/<run_id>` and `/results` — per-run viewer + index. Each run's outputs live in `outputs/analysis_runs/<run_id>/`.

API endpoints worth knowing:
- `POST /api/upload/check` and `POST /api/upload/run` — staged upload + duplicate-check (fuzzy title match via `difflib.SequenceMatcher`) + background pipeline job; status reconciled via `_reconcile_job()` so Flask restarts don't strand jobs.
- `POST /api/run` — feeds the **live review-table state** (not the on-disk CSVs) to `run_analysis.run()` by aggregating via `build_upload_files.collect_from_iters()`. The `data_source` field in the run's `meta` distinguishes `"in_memory"` vs `"csv_files"`.
- `GET /api/heatmap_data` — SOC-major × activity weight matrix + observed-row/col overlays for the chosen metric.
- `POST /api/export` — writes the edited CSVs.

`mapping_type` (occupation vs work_activity) is derived from the `onet_code` shape (`WA-XX` → work_activity, `dd-dddd.dd` → occupation) at aggregation time, so dropdown edits in the UI don't desync the row's destination bucket.

### Imputation (`scripts/run_analysis.py` + `pipeline.py` upstream)

Wraps the bipartite-graph soft-clamped propagation from `pipeline.py` (loaded at module level from `/Users/catherinewu/Documents/nyu/yr2/msft/pipeline.py`). Stages:
- **A/B**: build composite (Importance × Level / 5) matrix, apply row-wise softmax with β.
- **C — manual pruning** (default): drop occupations in the user-deselected SOC majors, drop activities whose summed weight across kept occupations falls below the threshold. Observed nodes are always kept regardless.
- **D**: solve `(L + Ω + Ω_base + εI) x = Ω·y + Ω_base·y_base` for each metric.

Optional `aggregate_to_socmajor=True` collapses the 894 occupations to the 22 SOC major groups before pruning + propagation. When this flag is set, the threshold check multiplies the per-group mean weights by group size, so the pruning semantic matches the bar chart and the occupation-level mode.

The AIOE speed-only baseline is moment-matched to the observed metric's mean/SD using the **full** Felten distribution (not just the overlap subset) per the methodology decisions in `ai_impact_pages_spec.md` upstream.

The matrix build reads a 73k-row xlsx and is cached per-β in `_HEAT_CACHE`.

### Configuration

`config/settings.yaml` is the source of truth for model assignments, max_tokens per stage, retry policy, and which stages run by default. Per-stage model overrides matter — Haiku for the mechanical 1a, Sonnet for everything that requires judgment. Stage 4 uses prompt caching on the O\*NET reference (drops input cost ~3× across the corpus). If you change a prompt, check `prompts/extract.txt` and `prompts/extract_quotes.txt` for the JSON-escaping rule.

`config/onet_activities.json` and `config/onet_occupations.json` are the canonical taxonomy. Don't modify without asking.

### Outputs layout

```
outputs/
├── 01a_quotes/ 01_extraction/ 02_method_classification/ ...   # per-stage per-paper JSON
├── 05_effect_sizes/{paper_id}.{speed,quality}.json
├── final/
│   ├── speed_table.csv quality_table.csv papers_excluded.csv  # one row per (paper, effect)
│   ├── onet_occupations_impact.csv onet_activities_impact.csv # per-O*NET aggregation
│   └── *.pre_review.csv                                       # backups created by /api/export
├── analysis_runs/<run_id>/{occupation_impacts.csv,activity_impacts.csv,run.json}
├── uploads/{jobs.json, staging .pdf}
└── review_state.json                                          # edit overlay for the review table
```

## Things easy to get wrong

- The pipeline subprocess started by `/api/upload/run` uses `start_new_session=True` so it survives a Flask debug-reload (auto-reload kills the parent supervising thread but not the orphaned child orchestrator). On the next status poll, `_reconcile_job()` infers completion from on-disk stage outputs and marks the job done — including the multi-failure-mode cases (theory papers tagged `other`, papers with no primary outcome, parse failures at 1a/1b).
- Running the orchestrator with `--paper foo` and **no** `--stage` runs every stage in `settings.yaml.stages` order for that paper. With `--stage X` it runs just that one stage.
- The `_W_CACHE` / `_HEAT_CACHE` in `run_analysis.py` is keyed by β. Changing the Work Activities.xlsx file content won't be picked up until the Flask process is restarted.
- LLM cost is real (~$0.13/paper full pipeline; ~$0.50/paper full-PDF salvage). The salvage script caches PDFs and runs a cheap text-only pass first; don't rip those out.
