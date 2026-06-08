# AI Impact Meta-Analysis — Review & Imputation App

A Flask web app + the underlying pipeline. Two interfaces:

1. **CLI pipeline** (this README's original content, below): drop PDFs into `papers/`, run `orchestrator.py`, get effect-size tables.
2. **Web app** (`scripts/review_app.py`): review and edit the rows the pipeline produced, upload new PDFs (with a duplicate-check step), and run the O\*NET imputation against the live table.

## Web app

```bash
cd pipeline-run
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
.venv/bin/python scripts/review_app.py
# open http://127.0.0.1:5000
```

Header buttons:
- **+ Upload PDF** — adds a new paper (duplicate-check → confirm → background pipeline run with per-stage progress)
- **Export CSVs** — rewrites `outputs/final/{speed,quality}_table.csv` with edits applied (originals backed up to `*.pre_review.csv`)
- **Run Analysis →** — opens the imputation parameter form; results render at `/results/<run_id>`

The imputation step needs two external files (not in this repo):
- `~/Downloads/Work Activities.xlsx` — O\*NET Work Activities long-format release
- `~/Documents/nyu/yr2/msft/felten_aioe.csv` — Felten et al. AIOE scores (speed-only baseline prior)

Paths are at the top of `scripts/run_analysis.py`.

---

# Pipeline (CLI)

Convert PDFs of AI productivity research papers into standardized effect-size tables keyed to O\*NET work activities and occupations.

## Setup

```bash
# In your working directory (copy from this skill):
cp -r ~/.claude/skills/ai-impact-meta-analysis/* .
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
# Drop PDFs into papers/, then:
.venv/bin/python scripts/orchestrator.py --papers-dir papers/
```

## Stage flow

```
papers/*.pdf
   ↓
1a extract_quotes  →  outputs/01a_quotes/{paper}.json
   ↓
1b structure       →  outputs/01_extraction/{paper}.json
   ↓
2  classify_method →  outputs/02_method_classification/{paper}.json   (drops "other" papers)
   ↓
3  classify_outcome→  outputs/03_outcome_classification/{paper}.json   (validates outcome names)
   ↓
4  map_onet        →  outputs/04_onet_mapping/{paper}.json             (cached system block)
   ↓
5a compute_speed   →  outputs/05_effect_sizes/{paper}.speed.json
5b compute_quality →  outputs/05_effect_sizes/{paper}.quality.json
   ↓
6  assemble        →  outputs/final/{speed,quality,papers_excluded}_table.csv
   ↓
build_upload_files →  outputs/final/onet_{occupations,activities}_impact.csv
```

Models: Haiku for 1a/1b (cheap quote-pulling and structuring); Sonnet for 2-5 (judgment); stage 4 uses prompt caching.

## Common commands

```bash
# Full run
.venv/bin/python scripts/orchestrator.py --papers-dir papers/

# Single stage
.venv/bin/python scripts/orchestrator.py --stage classify_outcome

# Single paper, single stage, force re-run
.venv/bin/python scripts/orchestrator.py --stage map_onet --paper Noy_Zhang_1 --force

# Sanity check on 10 papers across two models (~$2)
.venv/bin/python scripts/orchestrator.py --sensitivity-check

# Aggregate final outputs into per-O*NET impact files
.venv/bin/python scripts/build_upload_files.py

# Regenerate citation keys (after fixing author/title/year extractions)
.venv/bin/python scripts/regen_citation_keys.py
```

## Outputs

Per-paper artifacts (one JSON per stage per paper):
- `outputs/01a_quotes/{paper}.json` — verbatim quoted passages from PDF
- `outputs/01_extraction/{paper}.json` — structured arms / outcomes / reported_statistics
- `outputs/02_method_classification/{paper}.json` — study design + rationale
- `outputs/03_outcome_classification/{paper}.json` — primary speed/quality outcomes + per-outcome operationalization
- `outputs/04_onet_mapping/{paper}.json` — O\*NET code + rationale
- `outputs/05_effect_sizes/{paper}.speed.json` / `.quality.json` — LLM-extracted numbers and Python-computed effect sizes

Audit trail:
- `logs/{stage}.jsonl` — every LLM call's full prompt + response + token counts

Final tables:
- `outputs/final/speed_table.csv` — one row per (paper, speed effect)
- `outputs/final/quality_table.csv` — one row per (paper, quality effect)
- `outputs/final/papers_excluded.csv` — papers that didn't produce an effect, with reason
- `outputs/final/onet_occupations_impact.csv` — aggregated per SOC code
- `outputs/final/onet_activities_impact.csv` — aggregated per work-activity code

## Effect-size conventions

- **Speed**: positive `log_ratio` = AI helps. Direction determined by the LLM's `direction_convention` (`smaller_is_better` for time-based outcomes; `larger_is_better` for throughput). Python applies the appropriate formula.
- **Quality**: positive Hedges' `g` = AI helps. Sign-flipped automatically for `smaller_is_better` outcomes (e.g., error rates).

## Customization

- Change which model handles which stage: edit `config/settings.yaml` → `stage_models`.
- Change concurrency or retry behavior: edit `config/settings.yaml`.
- Swap O\*NET taxonomy: replace `config/onet_activities.json` and `config/onet_occupations.json`.

## Troubleshooting

- **1b parse failure** (occasional Haiku JSON-escaping slip): temporarily switch `stage_models.structure` to Sonnet in `config/settings.yaml`, retry the affected papers with `01b_structure.py --paper {pid} --force`, then switch back.
- **1b output truncation** on a long paper: bump `max_tokens.structure` from 20000 to 24000+.
- **Stage 4 rate-limit (429)** with Sonnet at high concurrency: lower `concurrency` in `settings.yaml` to 3-4.

See [METHODS.md](METHODS.md) (in `reference/`) for full design rationale, schemas, and decision log.
