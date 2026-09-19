# AGENTS.md — orientation for paper-writing agents

Read this first if you are writing up the paper (main text, appendix, figures,
captions) from this repo. It covers where the website is, which results are
current, and which numbers are safe to cite. `CLAUDE.md` covers the code itself
(pipeline stages, Flask app internals). Read it only if you need to change code.

> **Snapshot date: 2026-09-19.** The run IDs and numbers below come from the
> curated run set generated that day. Before citing anything, run the
> "Is this still current?" check at the bottom of this file.

---

## 1. Where things live

| What | Where |
|---|---|
| Paper draft (LaTeX) | `../paper/main.tex` (outside this repo, in the parent `msft/` folder; also `wise.tex`, `informs.tex` venue variants) |
| Imputation methodology write-up | `../AI_Impact_Imputation_Methodology.md` (and `.docx`) |
| Extraction/meta-analysis methods + decision log | `../METHODS.md` |
| Site/methodology decisions (AIOE baseline moment-matching etc.) | `../ai_impact_pages_spec.md` |
| Propagation solver (upstream) | `../pipeline.py`, imported by `scripts/run_analysis.py` |
| Per-study effect sizes | `outputs/final/speed_table.csv`, `outputs/final/quality_table.csv` |
| Excluded papers + reason | `outputs/final/papers_excluded.csv` |
| Curated run manifest (**source of truth for which runs count**) | `config/site_runs.json` (see §3) |
| Per-run digest used by the site's Summary page | `docs/data/summary.json` |
| Per-run outputs | `outputs/analysis_runs/<run_id>/` |
| Source PDFs | `papers/` (105 PDFs) |

---

## 2. The website

**Public, read-only static site:** https://catzwu.github.io/ai-impact-meta-review/
(it 301-redirects to `http://photos.catzwu.com/ai-impact-meta-review/`; cite the
`github.io` URL in the paper.)

- Served by GitHub Pages from `main:/docs`, so it shows **whatever is on `origin/main`**.
- Built by `scripts/build_static_site.py`, which writes `docs/`. Nothing runs on
  the server. All data sits in `docs/data/*.json`.
- Pages: **Studies** (`index.html`, per-study effects with a drawer showing the
  verbatim quotes and O\*NET mapping), **Runs** (`runs.html`), **Results**
  (`results.html?run=<run_id>`; with no `run` it shows the canonical run),
  **Summary** (`summary.html`: headline findings, a sensitivity agreement plot,
  and a LOO τ forest plot), **LOO** (`loo.html?run=<run_id>`),
  **Parameters** (`parameters.html`: the coverage heatmap and an explanation of
  each knob, including β, Ω_ref, Ω_base, thresholds and aggregation), and
  **Transitions** (`transitions.html`).
- As of 2026-09-19 the live site publishes exactly the curated run set from §3,
  so run IDs on the Runs page are safe to cite.

**Local interactive app** (editing, uploads, new runs; you usually don't need it):

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/review_app.py      # http://127.0.0.1:5000
```

Imputation runs also need two files outside the repo: `~/Downloads/Work Activities.xlsx`
and `../felten_aioe.csv`.

**Screenshots of the site for the paper:** take them locally with
`python3 -m http.server -d docs 8000`, then open http://localhost:8000. That way you
capture the version you are describing, not whatever is live.

---

## 3. Which results are current (use these for the appendix)

The **curated run set** is defined by `scripts/generate_site_runs.py` and
recorded in `config/site_runs.json`. It has 24 runs, all computed from one
data snapshot:

- **6 headline runs**: {speed, quality} × {occupation, SOC minor, SOC major}
- **11 sensitivity runs**: one knob changed from the occupation-level headline run
- **7 leave-one-out (LOO) validation runs**

**Where it is:** on `main` (merged 2026-09-19, PR #2). If your local checkout is
on a different branch, read from `origin/main` without switching:

```bash
git fetch origin
git show origin/main:config/site_runs.json
git show origin/main:docs/data/summary.json
git show origin/main:outputs/analysis_runs/<run_id>/occupation_impacts.csv
```

**Data snapshot behind every curated run:** 40 speed observations and 28 quality
observations from 49 distinct studies (40 in the speed table, 27 in the quality
table). They map to 12 occupation codes and 11 work-activity codes. 66 papers are
listed in `papers_excluded.csv`.

**Superseded / exploratory runs (do not cite):** every `outputs/analysis_runs/`
directory **not** listed in `config/site_runs.json`. On `main` these are the
seven June 2026 runs (`20260607_*` to `20260626_*`), which were kept as unique
exploratory runs. Other local branches or checkouts may also hold `20260907_*`
or `20260919_1325*_loo_*` runs. Those were deleted as duplicates of curated runs
on an older data snapshot, so don't cite them either.

### 3a. Default parameters (headline runs)

β = 2.5 (softmax sharpness on Importance × Level / 5 activity weights). Manual
pruning excludes the six physical-work SOC majors (37, 45, 47, 49, 51, 53).
Activity-weight threshold = 10. Ω_ref = 100, σ_ref = 0.1, ε = 1e-6. The AIOE
baseline (Felten et al.) is on for speed only, with Ω_base = 0.5, moment-matched
to the observed mean and SD using the full Felten distribution. Quality has no
baseline. The full parameter dict is in each run's `run.json → params`.

### 3b. Snapshot of results (2026-09-19; verify before citing)

**Headline runs** (`obs` = observed occupation/activity nodes; `kept` = nodes after pruning):

| Key | Run ID | Obs (occ/act) | Kept (occ/act) | Mean occ. estimate |
|---|---|---|---|---|
| speed-occupation (canonical) | `20260919_134544_f33a5b` | 11 / 8 | 607 / 19 | 0.308 log-ratio |
| speed-soc_minor | `20260919_134547_95c0e8` | 9 / 8 | 63 / 19 | 0.284 |
| speed-soc_major | `20260919_134550_e51185` | 7 / 8 | 16 / 19 | 0.246 |
| quality-occupation | `20260919_134553_8b6fb9` | 8 / 8 | 607 / 20 | 0.417 g |
| quality-soc_minor | `20260919_134556_4ef312` | 7 / 8 | 63 / 20 | n/a |
| quality-soc_major | `20260919_134559_d3ef76` | 6 / 8 | 16 / 20 | n/a |

Top imputed activities in the speed-occupation run: Thinking Creatively (0.419,
observed), then Training and Teaching Others, Organizing/Planning/Prioritizing
Work, and Establishing and Maintaining Interpersonal Relationships (all
imputed). The top five for every run are in `summary.json → runs[].top_occ / top_act`.

**Sensitivity**: agreement with the matching headline run (Spearman ρ over shared nodes / top-10 overlap):

| Run | Occupations ρ (top-10) | Activities ρ (top-10) |
|---|---|---|
| speed β=1 | 0.95 (8) | 0.99 (9) |
| speed β=10 | 0.92 (9) | 0.99 (10) |
| speed all 22 SOC kept | 0.99 (10) | 0.96 (9) |
| speed also excl. SOC 11 | 1.00 (10) | 1.00 (10) |
| speed threshold 5 | 0.99 (9) | 1.00 (10) |
| **speed no AIOE baseline** | **0.56 (2)** | **0.59 (7)** |
| quality β=1 | 0.88 (9) | 0.97 (9) |
| **quality β=10** | **0.57 (9)** | 0.79 (8) |
| quality all 22 SOC kept | 1.00 (10) | 0.99 (9) |
| quality also excl. SOC 11 | 1.00 (10) | 0.99 (10) |
| quality threshold 5 | 0.99 (10) | 1.00 (9) |

What this shows: the rankings are robust to pruning and threshold choices. The
speed occupation ranking depends heavily on the AIOE baseline.

**LOO validation**: Kendall τ-b between held-out predictions and observed values, with 95% bootstrap CI (2,000 resamples):

| Run | n | τ-b [95% CI] | Spearman | MAE |
|---|---|---|---|---|
| speed · occupation | 19 | 0.13 [−0.17, 0.43] | 0.21 | 0.167 |
| speed · SOC minor | 17 | **0.40 [0.05, 0.66]** | 0.57 | 0.150 |
| speed · SOC major | 15 | 0.23 [−0.21, 0.62] | 0.29 | 0.129 |
| speed · occupation · no baseline | 19 | 0.04 [−0.38, 0.44] | 0.03 | 0.120 |
| quality · occupation | 16 | **−0.40 [−0.70, −0.05]** | −0.56 | 0.325 |
| quality · SOC minor | 15 | **−0.60 [−0.86, −0.30]** | −0.75 | 0.215 |
| quality · SOC major | 14 | −0.29 [−0.65, 0.17] | −0.41 | 0.236 |

Report these honestly. Only speed at the SOC-minor level recovers held-out order
significantly. **Quality LOO is significantly negative** at the occupation and
SOC-minor levels: held-out quality predictions rank observed effects roughly
backwards. Do not describe the quality imputation as validated. Per-node LOO
detail (actual vs. predicted rank) is in each LOO run's `loocv.csv`. The full
rank statistics (concordance C, footrule, top-K precision vs. chance) are in `loocv.json` and in `run.json → loo_summary.rank`.

---

## 4. Conventions to get right in prose

- **Speed** = log ratio, where positive means AI helps (faster, or more output).
  Convert to percent as `exp(x) − 1`, so 0.308 ≈ +36%. `percent_change_equivalent`
  in `speed_table.csv` is already converted.
- **Quality** = Hedges' g, where positive means AI helps. The sign is flipped for
  outcomes where smaller is better (e.g., error rates).
- An LLM reads the numbers out of each paper. Python computes every effect size
  deterministically (stage 5). Describe it that way, not as "the LLM computed
  effect sizes".
- Models: Haiku extracts quotes (1a). Sonnet handles structuring, classification,
  O\*NET mapping and number extraction. The exact per-stage assignments are in
  `config/settings.yaml`.
- Each study effect maps to **one** O\*NET code: either an occupation (SOC
  `dd-dddd.dd`) or a generalized work activity (`WA-XX`). A human reviewed and
  edited the mappings in the review app. The edits are in
  `outputs/review_state.json`, and the exported CSVs reflect them.
- Imputation = soft-clamped propagation on the occupation↔activity bipartite
  graph: solve `(L + Ω + Ω_base + εI) x = Ω·y + Ω_base·y_base`. See
  `../AI_Impact_Imputation_Methodology.md` for the derivation.
- `posterior_std` in the impact CSVs is large for unobserved occupations: in
  the canonical speed run it is about 0.82–1.03, versus about 0.10 for observed
  nodes. Don't present point estimates for unobserved occupations as precise.

## 5. Output file schemas

- `speed_table.csv`: `citation_key, authors, year, study_design, mapping_type, onet_code, onet_label, log_ratio, log_ratio_variance, percent_change_equivalent, n_human, n_ai, computation_method, confidence, notes`
- `quality_table.csv`: same, but with `hedges_g, variance` in place of the log-ratio columns
- `analysis_runs/<id>/occupation_impacts.csv`: `code, title, observed, n_studies, estimate, posterior_std`
- `analysis_runs/<id>/activity_impacts.csv`: `activity, observed, n_studies, estimate, posterior_std`
- `analysis_runs/<id>/loocv.csv` (LOO runs): `node_type, code, label, actual, predicted, residual, abs_error, posterior_std, actual_rank, loo_rank, rank_delta, …`
- `analysis_runs/<id>/run.json`: params, node counts, `data_source`, and for LOO runs `loo_summary`

## 6. Don'ts

- **Don't run `scripts/generate_site_runs.py`** unless the user asks. It deletes
  the current curated runs and writes new ones with **new run IDs**, which breaks
  every run ID cited in the paper.
- Don't run the extraction pipeline or salvage script. They call paid APIs
  (about $0.13 per paper, or about $0.50 per paper for salvage).
- Don't edit `config/onet_*.json`, `outputs/review_state.json`, or
  `outputs/final/*.csv`. These are the data of record.
- Don't cite runs outside `config/site_runs.json`.
- Paper edits go in `../paper/`, not in this repo.

## 7. Is this still current?

```bash
git fetch origin
git log -1 --format='%ci %h %s' origin/main -- config/site_runs.json   # last regeneration of the run set
git show origin/main:config/site_runs.json | grep canonical              # snapshot: 20260919_134544_f33a5b
```

If `config/site_runs.json` has been regenerated since 2026-09-19, the run IDs and
numbers in §3b are out of date. Re-read them from `site_runs.json` and
`docs/data/summary.json`, and update this file.
