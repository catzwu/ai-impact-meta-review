"""Build a static GitHub Pages site from the review-app data.

Emits into ./docs:
  index.html               — review table + paper drawer (read-only port of INDEX_HTML)
  results.html             — analysis-run viewer; accepts ?run=<id>, defaults to canonical
  summary.html             — overview of the curated run set (findings, agreement, LOO charts)
  runs.html                — index of published runs, grouped (port of RESULTS_INDEX_HTML)
  loo.html                 — leave-one-out viewer; accepts ?run=<id>
  parameters.html          — heatmap + activity bar chart + parameter explanations
  transitions.html         — occupational transitions heatmap (port of TRANSITIONS_HTML)
  .nojekyll
  data/rows.json           — same shape as GET /api/rows, review_state overlay applied
  data/onet.json           — same shape as GET /api/onet
  data/papers/*.json       — one per paper (mirror of GET /api/paper/<id>)
  data/runs/index.json     — { runs: [meta, ...] } (mirror of GET /api/runs)
  data/runs/<id>.json      — full bundle per run (mirror of GET /api/results/<id>)
  data/summary.json        — per-run digest for summary.html (only with config/site_runs.json)
  data/heatmap_speed.json  — precomputed heatmap at canonical β for the parameters page
  data/heatmap_quality.json
  data/transitions.json    — mirror of GET /api/transitions/data
  assets/*.csv             — final CSV downloads (excluding *.pre_review.csv)

Run:
    python3 scripts/build_static_site.py
    python3 -m http.server -d docs 8000    # verify locally

Building the parameters + transitions pages requires the same external files as
the live app (Work Activities.xlsx, felten_aioe.csv, transitions xlsx, pipeline.py).
If any are missing the affected page is skipped with a WARN.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import site_theme  # noqa: E402

OUT =ROOT / "outputs" / "final"
OUTPUTS_DIR = ROOT / "outputs"
CONFIG_DIR = ROOT / "config"
RUNS_DIR = OUTPUTS_DIR / "analysis_runs"
STATE_PATH = OUTPUTS_DIR / "review_state.json"
SITE_RUNS_PATH = CONFIG_DIR / "site_runs.json"

DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
DATA = DOCS / "data"
PAPERS_DATA = DATA / "papers"
RUNS_DATA = DATA / "runs"

REPO_URL = "https://github.com/catzwu/ai-impact-meta-review"
# GitHub Pages custom domain. main() rewrites docs/ from scratch, so it must re-emit
# docs/CNAME itself — otherwise the next rebuild deletes it and Pages drops the domain.
SITE_DOMAIN = "ai-impact.catzwu.com"
CANONICAL_BETA = 2.5


# ---------- helpers (kept in sync with review_app.py) ----------

def _read_json(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _read_csv(p: Path):
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def _load_state() -> dict:
    s = _read_json(STATE_PATH)
    return s if s else {"edits": {}, "deleted": [], "merges": []}


def _resolve_paper_id_map() -> dict[str, str]:
    out = {}
    for p in (OUTPUTS_DIR / "01_extraction").glob("*.json"):
        if p.stem.endswith(".error"):
            continue
        d = _read_json(p)
        if d and d.get("citation_key"):
            out[d["citation_key"]] = p.stem
    return out


def _build_rows(pid_map: dict[str, str]) -> list[dict]:
    rows = []
    for kind, path, value_field, var_field in (
        ("speed", OUT / "speed_table.csv", "log_ratio", "log_ratio_variance"),
        ("quality", OUT / "quality_table.csv", "hedges_g", "variance"),
    ):
        for i, r in enumerate(_read_csv(path)):
            ck = r.get("citation_key", "")
            pid = pid_map.get(ck, "")
            row_id = f"{pid or ck}::{kind}::{i}"
            ext = _read_json(OUTPUTS_DIR / "01_extraction" / f"{pid}.json") if pid else None
            quotes = _read_json(OUTPUTS_DIR / "01a_quotes" / f"{pid}.json") if pid else None
            title = (ext or {}).get("title", "") if ext else ""
            task_desc = ""
            if quotes and isinstance(quotes.get("quotes"), dict):
                qd = quotes["quotes"]
                for key in ("task", "study_design", "outcomes", "arms", "population"):
                    arr = qd.get(key)
                    if arr and isinstance(arr, list) and arr:
                        task_desc = (arr[0] or "")[:300]
                        if task_desc:
                            break
            if not task_desc and ext:
                for s in ext.get("reported_statistics", []):
                    vq = s.get("verbatim_quote")
                    if vq:
                        task_desc = vq[:300]
                        break
            rows.append({
                "row_id": row_id,
                "paper_id": pid,
                "citation_key": ck,
                "citation": site_theme.short_citation((ext or {}).get("authors"), (ext or {}).get("year"), ck),
                "file_name": f"{pid}.pdf" if pid else "",
                "title": title,
                "kind": kind,
                "value": r.get(value_field, ""),
                "variance": r.get(var_field, ""),
                "onet_code": r.get("onet_code", ""),
                "onet_label": r.get("onet_label", ""),
                "mapping_type": r.get("mapping_type", ""),
                "confidence": r.get("confidence", ""),
                "task_description": task_desc,
                "computation_method": r.get("computation_method", ""),
                "notes": r.get("notes", ""),
            })
    return rows


def _apply_state(rows: list[dict], state: dict) -> list[dict]:
    edits = state.get("edits", {})
    deleted = set(state.get("deleted", []))
    merges = state.get("merges", [])
    for m in merges:
        for d in m.get("drop", []):
            deleted.add(d)
    out = []
    for r in rows:
        rid = r["row_id"]
        if rid in deleted:
            continue
        for k, v in edits.get(rid, {}).items():
            r[k] = v
        out.append(r)
    return out


def _paper_bundle(paper_id: str) -> dict:
    return {
        "paper_id": paper_id,
        "01a_quotes": _read_json(OUTPUTS_DIR / "01a_quotes" / f"{paper_id}.json"),
        "01_extraction": _read_json(OUTPUTS_DIR / "01_extraction" / f"{paper_id}.json"),
        "02_method": _read_json(OUTPUTS_DIR / "02_method_classification" / f"{paper_id}.json"),
        "03_outcome": _read_json(OUTPUTS_DIR / "03_outcome_classification" / f"{paper_id}.json"),
        "04_onet": _read_json(OUTPUTS_DIR / "04_onet_mapping" / f"{paper_id}.json"),
        "05_speed": _read_json(OUTPUTS_DIR / "05_effect_sizes" / f"{paper_id}.speed.json"),
        "05_quality": _read_json(OUTPUTS_DIR / "05_effect_sizes" / f"{paper_id}.quality.json"),
    }


def _all_run_dirs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    out = []
    for d in RUNS_DIR.iterdir():
        if not d.is_dir():
            continue
        if (d / "run.json").exists() and (d / "occupation_impacts.csv").exists() and (d / "activity_impacts.csv").exists():
            out.append(d)
    return sorted(out, key=lambda p: p.name, reverse=True)


def _load_manifest() -> dict | None:
    """Curated run set written by scripts/generate_site_runs.py. When present, only
    these runs are published (in manifest order, grouped); otherwise every plain run."""
    m = _read_json(SITE_RUNS_PATH)
    if not m:
        return None
    missing = [e["run_id"] for e in m["runs"] if not (RUNS_DIR / e["run_id"] / "run.json").exists()]
    if missing:
        print(f"WARN: manifest lists missing runs {missing}; re-run scripts/generate_site_runs.py")
        m["runs"] = [e for e in m["runs"] if e["run_id"] not in missing]
    return m


def _agreement(run_id: str, ref_id: str) -> dict | None:
    """Compact rank agreement of a run with its reference run (for runs.html)."""
    try:
        import run_analysis as RA
        c = RA.compare_runs(run_id, ref_id)
    except Exception as e:  # noqa: BLE001
        print(f"WARN: compare {run_id} vs {ref_id} failed: {e}")
        return None
    if not c:
        return None
    out = {"common_level": c["common_level"]}
    for view, v in c["views"].items():
        tk = v["topk_overlap"].get("10")
        out[view] = {"n": v["n"], "spearman": v["metrics"].get("spearman_r"),
                     "kendall": v["metrics"].get("kendall_tau"),
                     "top10": tk["intersection"] if tk else None}
    return out


def _summary_record(entry: dict, bundle: dict) -> dict:
    """Compact per-run digest for summary.html: top effects, agreement, LOO stats."""
    p = bundle.get("params") or {}
    rec = {k: entry[k] for k in ("key", "run_id", "group", "kind", "label")}
    rec.update(metric=p.get("metric"), level=bundle.get("aggregation_level"),
               obs=[bundle.get("n_observed_occ"), bundle.get("n_observed_act")],
               kept=[bundle.get("n_kept_occ"), bundle.get("n_kept_act")])

    def _top(rows, labelcol, n=5):
        vals = []
        for r in rows:
            try:
                est = float(r["estimate"])
            except (TypeError, ValueError):
                continue
            obs = r.get("observed")
            vals.append({"label": str(r.get(labelcol) or ""), "est": round(est, 3),
                         "obs": None if obs in (None, "") else round(float(obs), 3)})
        return sorted(vals, key=lambda v: -v["est"])[:n]

    if bundle.get("type") == "loo":
        s = bundle["loo"]["summary"]
        rk = (s.get("rank") or {}).get("overall") or {}
        worst = sorted(bundle["loo"]["rows"], key=lambda r: -abs(r.get("rank_delta") or 0))[:1]
        rec["loo"] = {"n": s["overall"].get("n"), "mae": s["overall"].get("mae"),
                      "spearman": rk.get("spearman_r"), "tau": rk.get("kendall_tau"), "tau_ci": rk.get("tau_ci"),
                      "worst": [{k: w.get(k) for k in ("label", "actual_rank", "loo_rank")} for w in worst]}
    else:
        occ = bundle["occupation_impacts"]
        ests = [float(r["estimate"]) for r in occ if r.get("estimate") not in (None, "")]
        rec.update(occ_mean=round(sum(ests) / len(ests), 3) if ests else None,
                   top_occ=_top(occ, "title"), top_act=_top(bundle["activity_impacts"], "activity"))
        if bundle.get("agreement"):
            rec["agreement"] = bundle["agreement"]
            rec["agreement_level"] = bundle["agreement"].get("common_level")
    return rec


def _load_run_bundle(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "run.json").read_text())

    def _read_impacts(p: Path) -> list[dict]:
        return [{k: (None if v == "" else v) for k, v in r.items()} for r in _read_csv(p)]

    if (run_dir / "loocv.json").exists():
        meta["type"] = "loo"
        meta["loo"] = json.loads((run_dir / "loocv.json").read_text())
        return meta
    meta["occupation_impacts"] = _read_impacts(run_dir / "occupation_impacts.csv")
    meta["activity_impacts"] = _read_impacts(run_dir / "activity_impacts.csv")
    return meta


# ---------- HTML template plumbing ----------

def _header(active: str) -> str:
    """Shared masthead. `active` is one of 'home', 'summary', 'runs', 'parameters', 'transitions'."""
    items = [
        ("index.html", "Studies", "home"),
        ("summary.html", "Summary", "summary"),
        ("runs.html", "Imputation runs", "runs"),
        ("parameters.html", "Method", "parameters"),
        ("transitions.html", "Transitions", "transitions"),
    ]
    return site_theme.masthead(items, active, home_href="index.html", external=[(REPO_URL, "GitHub")])


_FOOTER = f"""<footer class="site"><div class="wrap">
  AI Impact Meta-Review &middot; effect sizes extracted from the research literature and mapped to O*NET.
  Data and code: <a href="{REPO_URL}" target="_blank" rel="noopener">GitHub</a>.
</div></footer>"""

# The theme stylesheet is injected via __THEME_HEAD__; page styles below hold only page-specific rules.
_STYLE_COMMON = ""


# ---------- index.html (review table + drawer) ----------

_INDEX_STYLE = _STYLE_COMMON + r"""
  .intro { display:grid; grid-template-columns: minmax(0, 1.5fr) minmax(280px, 1fr); gap:32px; align-items:start; padding:34px 0 22px; }
  .intro h1 { margin:0 0 10px; font-size:32px; line-height:1.15; }
  .intro p { margin:0 0 10px; color:var(--ink-2); font-size:15.5px; line-height:1.6; max-width:68ch; }
  .tiles { display:grid; grid-template-columns: 1fr 1fr; gap:10px; }
  .tiles .metric .v { font-family:var(--serif); font-size:28px; font-weight:600; }
  @media (max-width: 860px) { .intro { grid-template-columns: 1fr; } }
  .table-scroll { max-height:none; }
  #table { table-layout: fixed; min-width:1200px; }
  #table th { position:sticky; top:0; z-index:5; }
  td.snippet { white-space:normal; word-break:break-word; color:var(--ink-2); font-size:12.5px; line-height:1.45; }
  td.title { white-space:normal; word-break:break-word; font-family:var(--serif); font-size:14.5px; line-height:1.35; cursor:pointer; color:var(--ink); }
  td.title:hover { color:var(--link); text-decoration:underline; text-underline-offset:2px; }
  td.cite { font-size:13.5px; color:var(--ink); white-space:normal; }
  td.file { font-family:var(--mono); font-size:11px; color:var(--muted); overflow:hidden; text-overflow:ellipsis; }
  td.onet { font-size:12.5px; color:var(--ink); }
  td.onet .code { font-family:var(--mono); font-size:11px; color:var(--muted); margin-right:4px; }
  .downloads { margin:28px 0 0; }
  .downloads ul { margin:0; padding-left:20px; line-height:1.9; }
  .downloads code { color:var(--link); }
"""


_INDEX_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">__THEME_HEAD__<title>AI Impact Meta-Review</title>
<style>__STYLE__</style></head><body>
__HEADER__
<main class="page wrap">
  <section class="intro">
    <div>
      <h1>What does generative AI do to work?</h1>
      <p>This site collects empirical studies of generative AI in the workplace and puts their
        results on a common scale: <b>speed</b> effects as log ratios of task time or output, and <b>quality</b> effects as
        Hedges&rsquo; <i>g</i>. Each effect is linked to an O*NET occupation or work activity, so the evidence can be
        read against the structure of the labor market.</p>
      <p>Browse the coded studies below, and click a title to see what was extracted from the paper and why.
        The <a href="runs.html">imputation runs</a> extend these observations across the O*NET graph; the
        <a href="parameters.html">method</a> page explains how.</p>
    </div>
    <div class="tiles" id="tiles">
      <div class="metric"><div class="k">Studies</div><div class="v" id="tPapers">&ndash;</div></div>
      <div class="metric"><div class="k">Effects coded</div><div class="v" id="tRows">&ndash;</div></div>
      <div class="metric"><div class="k">Speed effects</div><div class="v" id="tSpeed">&ndash;</div></div>
      <div class="metric"><div class="k">Quality effects</div><div class="v" id="tQual">&ndash;</div></div>
    </div>
  </section>
  <h2 class="section-title">Coded studies</h2>
  <div class="toolbar">
    <input type="search" id="search" placeholder="Filter by citation, title, file, or O*NET&hellip;" style="min-width:300px; flex:0 1 420px;"/>
    <span class="page-meta" id="stats"></span>
    <span class="help" style="margin-left:auto;">Click a column header to sort &middot; click a title for details</span>
  </div>
<div class="table-scroll">
<table id="table">
  <colgroup>
    <col style="width:80px"><col style="width:180px"><col style="width:280px">
    <col style="width:80px"><col style="width:100px"><col style="width:300px"><col>
  </colgroup>
  <thead><tr>
    <th data-sort="kind">Kind</th>
    <th data-sort="citation">Citation</th>
    <th data-sort="title">Title</th>
    <th data-sort="value" style="text-align:right">Value</th>
    <th data-sort="confidence">Confidence</th>
    <th>O*NET</th>
    <th>Task snippet</th>
  </tr></thead>
  <tbody id="tbody"></tbody>
</table>
</div>

<div class="downloads card">
  <h2>Download the data</h2>
  <ul>
    <li><a href="assets/speed_table.csv"><code>speed_table.csv</code></a> &mdash; per-paper speed effects (log ratios)</li>
    <li><a href="assets/quality_table.csv"><code>quality_table.csv</code></a> &mdash; per-paper quality effects (Hedges' g)</li>
    <li><a href="assets/onet_activities_impact.csv"><code>onet_activities_impact.csv</code></a> &mdash; observed effects aggregated by O*NET work activity</li>
    <li><a href="assets/onet_occupations_impact.csv"><code>onet_occupations_impact.csv</code></a> &mdash; observed effects aggregated by O*NET occupation</li>
    <li><a href="assets/papers_excluded.csv"><code>papers_excluded.csv</code></a> &mdash; papers dropped by the pipeline, with reasons</li>
  </ul>
</div>
</main>

<div id="backdrop" onclick="closeDrawer()"></div>
<div id="drawer">
  <header>
    <h1 id="drawerTitle">Paper detail</h1>
    <button class="btn btn-quiet" onclick="closeDrawer()">Close &times;</button>
  </header>
  <div id="drawerBody"></div>
</div>

<script>
let ROWS=[], SORT={col:null,dir:1};

function escapeHtml(s){return (s||'').toString().replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function fmtNum(x, places){ if (x===null || x===undefined || x==='') return '—'; const n = parseFloat(x); return isNaN(n) ? x : n.toFixed(places===undefined?3:places); }
function confPill(c){ if(!c) return ''; return `<span class="conf-pill ${c}">${c}</span>`; }

async function load() {
  const r = await (await fetch('data/rows.json')).json();
  ROWS = r.rows;
  const papers = new Set(ROWS.map(x => x.paper_id || x.citation_key));
  document.getElementById('tPapers').textContent = papers.size;
  document.getElementById('tRows').textContent = ROWS.length;
  document.getElementById('tSpeed').textContent = ROWS.filter(x => x.kind === 'speed').length;
  document.getElementById('tQual').textContent = ROWS.filter(x => x.kind === 'quality').length;
  render();
}

function render() {
  const q = (document.getElementById('search').value || '').toLowerCase();
  let rows = ROWS.filter(r =>
    !q || (r.citation+r.citation_key+r.title+r.file_name+r.onet_code+r.onet_label).toLowerCase().includes(q));
  if (SORT.col) {
    rows = [...rows].sort((a,b)=> {
      const x = (a[SORT.col]||'').toString(), y=(b[SORT.col]||'').toString();
      if (SORT.col==='value') return ((parseFloat(x)||0) - (parseFloat(y)||0)) * SORT.dir;
      return x.localeCompare(y) * SORT.dir;
    });
  }
  document.getElementById('stats').textContent = rows.length === ROWS.length ? `${ROWS.length} effects` : `${rows.length} of ${ROWS.length} effects`;
  const tbody = document.getElementById('tbody');
  tbody.innerHTML = rows.map(r => {
    const v = parseFloat(r.value||0);
    const vClass = v > 0.001 ? 'pos' : v < -0.001 ? 'neg' : '';
    const onet = r.onet_code
      ? `<span class="code">${escapeHtml(r.onet_code)}</span>${escapeHtml(r.onet_label||'')}`
      : '<span style="color:var(--muted);">—</span>';
    return `
    <tr data-row="${r.row_id}">
      <td><span class="kind-${r.kind}">${r.kind}</span></td>
      <td class="cite" title="${escapeHtml(r.citation_key)}">${escapeHtml(r.citation||r.citation_key)}</td>
      <td class="title" title="Click to view paper detail" onclick="viewPaper('${r.paper_id}')">${escapeHtml(r.title)}</td>
      <td class="value ${vClass}">${v.toFixed(3)}</td>
      <td>${r.confidence ? `<span class="conf-pill-cell ${r.confidence}">${r.confidence}</span>` : ''}</td>
      <td class="onet">${onet}</td>
      <td class="snippet" title="${escapeHtml(r.task_description||'')}">${escapeHtml(r.task_description||'')}</td>
    </tr>`;
  }).join('');
}

document.getElementById('search').addEventListener('input', render);
document.querySelectorAll('th[data-sort]').forEach(th=>{
  th.addEventListener('click',()=>{
    const c=th.dataset.sort; SORT.dir=(SORT.col===c?-SORT.dir:1); SORT.col=c; render();
  });
});

async function viewPaper(pid) {
  if (!pid) return;
  let d;
  try { d = await (await fetch('data/papers/'+encodeURIComponent(pid)+'.json')).json(); }
  catch (e) { d = {}; }
  document.getElementById('drawerTitle').textContent = pid;
  document.getElementById('drawerBody').innerHTML = renderPaperDetail(pid, d);
  document.getElementById('drawer').classList.add('open');
  document.getElementById('backdrop').classList.add('open');
}
function closeDrawer(){
  document.getElementById('drawer').classList.remove('open');
  document.getElementById('backdrop').classList.remove('open');
}
document.addEventListener('keydown', (e)=>{ if (e.key==='Escape') closeDrawer(); });

function renderEffectCard(kind, eff) {
  if (!eff || !eff.computed) {
    return `<div class="pd-effect empty"><h3>${kind} effect</h3>No ${kind} effect was computed for this paper.</div>`;
  }
  const c = eff.computed, ex = eff.llm_extracted || {};
  const isSpeed = kind === 'speed';
  const headline = isSpeed ? fmtNum(c.log_ratio) : fmtNum(c.hedges_g);
  const pct = isSpeed && c.percent_change_equivalent!=null ? ` (${(c.percent_change_equivalent*100).toFixed(1)}%)` : '';
  const variance = isSpeed ? c.log_ratio_variance : c.variance;
  return `<div class="pd-effect">
    <h3>${kind} effect ${confPill(c.confidence)}</h3>
    <div class="value">${isSpeed?'log ratio':"Hedges' g"} = ${headline}${pct}</div>
    <div class="row"><b>variance</b><span>${fmtNum(variance, 6)}</span></div>
    <div class="row"><b>method</b><span style="font-family:monospace;">${escapeHtml(c.computation_method||'—')}</span></div>
    <div class="row"><b>arms</b><span>${escapeHtml(ex.human_arm_name||'?')} → ${escapeHtml(ex.ai_arm_name||'?')}</span></div>
    <div class="row"><b>n (h / ai)</b><span>${c.n_human||'—'} / ${c.n_ai||'—'}</span></div>
    ${ex.ai_arm_selection_rationale ? `<div class="notes"><b>Arm selection:</b> ${escapeHtml(ex.ai_arm_selection_rationale)}</div>`:''}
    ${c.notes ? `<div class="notes">${escapeHtml(c.notes)}</div>`:''}
  </div>`;
}

function renderQuotes(quotes) {
  if (!quotes || !quotes.quotes) return '';
  const q = quotes.quotes;
  const sections = [];
  for (const key of ['task','study_design','outcomes','arms','population']) {
    const arr = q[key];
    if (!arr || !arr.length) continue;
    const items = arr.map(s => `<div class="pd-quote">${escapeHtml(s)}</div>`).join('');
    sections.push(`<details ${key==='task'?'open':''}><summary><b>${key.replace('_',' ')}</b> <span class="help">(${arr.length})</span></summary>${items}</details>`);
  }
  return sections.length ? `<div class="pd-section"><h3>Verbatim quotes</h3><div class="pd-quotes-block">${sections.join('')}</div></div>` : '';
}

function renderArms(arms) {
  if (!arms || !arms.length) return '';
  const rows = arms.map(a => `<tr>
    <td><b>${escapeHtml(a.arm_name||'')}</b></td>
    <td>${a.n!=null ? a.n : '—'}</td>
    <td>${escapeHtml(a.description||'')}</td>
  </tr>`).join('');
  return `<div class="pd-section"><h3>Arms</h3><table class="pd-table">
    <tr><th style="width:130px;">Name</th><th style="width:50px;">n</th><th>Description</th></tr>
    ${rows}</table></div>`;
}

function renderOutcomes(ext, oc) {
  const outs = (ext && ext.outcomes) || [];
  if (!outs.length) return '';
  const pCats = (oc && oc.per_outcome) || [];
  const catFor = (name) => (pCats.find(x=>x.outcome_name===name)||{}).category || '';
  const primarySpeed = oc && oc.primary_speed_outcome;
  const primaryQual  = oc && oc.primary_quality_outcome;
  const rows = outs.map(o=>{
    const isPS = o.outcome_name===primarySpeed, isPQ = o.outcome_name===primaryQual;
    const star = isPS ? '★ speed' : isPQ ? '★ quality' : '';
    return `<tr><td><b>${escapeHtml(o.outcome_name||'')}</b>${star?` <span class="pill" style="background:var(--warn-bg); color:var(--warn-ink);">${star}</span>`:''}</td>
      <td><span class="badge">${escapeHtml(catFor(o.outcome_name))}</span></td>
      <td>${escapeHtml(o.measurement_unit||'')}</td>
      <td>${escapeHtml(o.description||'')}</td></tr>`;
  }).join('');
  return `<div class="pd-section"><h3>Outcomes</h3><table class="pd-table">
    <tr><th>Name</th><th>Category</th><th>Unit</th><th>Description</th></tr>
    ${rows}</table></div>`;
}

function renderOnet(onet) {
  if (!onet) return '';
  const alts = (onet.alternates||[]).map(a => `<span class="pill" title="${escapeHtml(a.mapping_type||'')}">${escapeHtml(a.onet_code)} — ${escapeHtml(a.onet_label)}</span>`).join('');
  return `<div class="pd-section"><h3>O*NET mapping</h3>
    <div><span class="pill" style="background:var(--accent-soft); color:var(--accent); font-family:var(--mono); border-radius:3px;">${escapeHtml(onet.onet_code||'?')}</span>
      <b>${escapeHtml(onet.onet_label||'')}</b> ${confPill(onet.mapping_confidence)}
      <span class="help">(${escapeHtml((onet.mapping_type||'').replace('_',' '))})</span></div>
    ${onet.rationale ? `<div class="pd-onet-rationale">${escapeHtml(onet.rationale)}</div>`:''}
    ${alts ? `<div class="pd-onet-alt"><b>Alternates:</b><br>${alts}</div>`:''}
  </div>`;
}

function renderStats(stats) {
  if (!stats || !stats.length) return '';
  const items = stats.slice(0, 30).map(s => {
    const m = s.means_and_sds || {};
    const numParts = [];
    for (const [arm, v] of Object.entries(m)) {
      if (v && (v.mean!=null || v.sd!=null)) {
        const bits = [];
        if (v.mean!=null) bits.push(`mean=${v.mean}`);
        if (v.sd!=null) bits.push(`SD=${v.sd}`);
        if (v.se!=null) bits.push(`SE=${v.se}`);
        numParts.push(`${arm}: ${bits.join(', ')}`);
      }
    }
    const extras = [];
    if (s.effect_in_outcome_units) extras.push(`effect: ${s.effect_in_outcome_units}`);
    if (s.regression_coefficient) extras.push(`coef: ${s.regression_coefficient}`);
    if (s.test_statistic) extras.push(`test: ${s.test_statistic}`);
    if (s.p_value) extras.push(`p: ${s.p_value}`);
    if (s.confidence_interval) extras.push(`CI: ${s.confidence_interval}`);
    return `<div class="pd-stat">
      <div class="head"><b>${escapeHtml(s.outcome_name||'?')}</b><span class="help">${escapeHtml(s.arm_comparison||'')}</span></div>
      ${numParts.length ? `<div class="nums">${escapeHtml(numParts.join('  •  '))}</div>`:''}
      ${extras.length ? `<div class="nums">${escapeHtml(extras.join('  •  '))}</div>`:''}
      ${s.verbatim_quote ? `<div class="vq">"${escapeHtml(s.verbatim_quote)}"</div>`:''}
    </div>`;
  }).join('');
  const more = stats.length>30 ? `<div class="help">… ${stats.length-30} more</div>` : '';
  return `<div class="pd-section"><h3>Reported statistics (${stats.length})</h3>${items}${more}</div>`;
}

function renderPaperDetail(pid, d) {
  const ext = d['01_extraction'] || {};
  const meth = d['02_method'] || {};
  const oc = d['03_outcome'] || {};
  const onet = d['04_onet'] || {};
  const speed = d['05_speed'], quality = d['05_quality'];
  const quotes = d['01a_quotes'] || null;
  return `
    <div class="pd-meta">
      <h2>${escapeHtml(ext.title||pid)}</h2>
      <div class="authors">${escapeHtml((ext.authors||[]).join('; '))}${ext.year?` (${ext.year})`:''}${ext.venue?` · <i>${escapeHtml(ext.venue)}</i>`:''}</div>
      <div class="ids">
        <span class="pill">${escapeHtml(ext.citation_key||'')}</span>
        <span class="pill">${escapeHtml(pid)}.pdf</span>
      </div>
    </div>
    <div class="pd-effects">
      ${renderEffectCard('speed', speed)}
      ${renderEffectCard('quality', quality)}
    </div>
    ${meth.classification ? `<div class="pd-section"><h3>Method classification</h3>
      <div><b>${escapeHtml(meth.classification.replace(/_/g,' '))}</b> ${confPill(meth.confidence)}</div>
      ${meth.rationale ? `<div style="color:var(--ink-2); font-size:13.5px; margin-top:6px; line-height:1.55;">${escapeHtml(meth.rationale)}</div>`:''}
    </div>`:''}
    ${renderOnet(onet)}
    ${renderArms(ext.arms)}
    ${renderOutcomes(ext, oc)}
    ${renderStats(ext.reported_statistics)}
    ${renderQuotes(quotes)}
    <details class="pd-raw"><summary>Raw JSON (all stages)</summary>
      <pre>${escapeHtml(JSON.stringify(d, null, 2))}</pre>
    </details>
  `;
}

load();
</script>
</body></html>
"""


# ---------- results.html (per-run viewer) ----------

_RESULTS_STYLE = _STYLE_COMMON + r"""
  .bar { display:inline-block; height:6px; vertical-align:middle; border-radius:2px; }
  .bar.pos { background:var(--pos); } .bar.neg { background:var(--neg); }
  #t th { cursor:pointer; user-select:none; }
  .params { font-family:var(--mono); font-size:12px; color:var(--ink-2); line-height:1.7; }
"""


_RESULTS_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">__THEME_HEAD__<title>Run results &mdash; AI Impact Meta-Review</title>
<style>__STYLE__</style></head><body>
__HEADER__
<main class="page wrap">
  <section class="page-head">
    <div>
      <div class="crumbs"><a href="runs.html">Imputation runs</a> / <span id="crumbId" class="mono"></span></div>
      <h1>Imputed impacts</h1>
      <p class="lede">Estimated effects for every occupation and work activity after propagating the observed studies across
        the O*NET graph. <span class="badge observed">Observed</span> rows are anchored by at least one study; the rest are imputed.</p>
      <p class="page-meta" id="stats" style="margin:8px 0 0;"></p>
    </div>
    <div class="page-actions"><button class="btn" onclick="exportCsv()">Download this view (CSV)</button></div>
  </section>
  <details class="card"><summary>Run parameters</summary><div class="params" id="params">Loading&hellip;</div></details>
  <div id="warn"></div>
  <div class="toolbar">
    <div class="seg" id="viewSeg">
      <button data-v="occ" class="active">Occupations</button>
      <button data-v="act">Activities</button>
    </div>
    <label>Posterior SD &le; <input type="number" id="stdFilter" value="5" step="0.05" style="width:70px;"/></label>
    <label><input type="checkbox" id="onlyObserved"/> Observed only</label>
    <input type="search" id="search" placeholder="Search title or code&hellip;" style="margin-left:auto; min-width:240px;"/>
  </div>
  <table id="t" class="data"><thead id="th"></thead><tbody id="tb"></tbody></table>
</main>
<script>
const DEFAULT_RUN_ID = "__DEFAULT_RUN_ID__";
let DATA = null, VIEW='occ', SORT={col:'estimate', dir:-1};

function fmtNum(x, p){ if (x===null||x===undefined||x==='') return '—'; const n=parseFloat(x); return isNaN(n)?x:n.toFixed(p===undefined?3:p); }
function escapeHtml(s){return (s||'').toString().replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}

async function load(){
  const params = new URLSearchParams(location.search);
  const runId = params.get('run') || DEFAULT_RUN_ID;
  if (!runId) { document.querySelector('main').innerHTML = '<div class="err" style="margin-top:30px;">No canonical run is available. See <a href="runs.html">all runs</a>.</div>'; return; }
  const r = await fetch('data/runs/' + encodeURIComponent(runId) + '.json');
  if (!r.ok) { document.querySelector('main').innerHTML = '<div class="err" style="margin-top:30px;">Unknown run: ' + escapeHtml(runId) + '. See <a href="runs.html">all runs</a>.</div>'; return; }
  DATA = await r.json();
  const p = DATA.params;
  document.getElementById('crumbId').textContent = DATA.run_id;
  document.getElementById('stats').textContent =
    `Metric: ${p.metric} · β = ${p.beta} · Ω_ref = ${p.omega_ref}` +
    (DATA.baseline_active ? ` · AIOE baseline Ω_b = ${p.omega_base}` : '') +
    ` · ${DATA.n_kept_occ} occupations and ${DATA.n_kept_act} activities kept`;
  document.getElementById('params').innerHTML =
    `<b>Run:</b> ${escapeHtml(DATA.run_id)} · <b>${p.metric}</b> · β=${p.beta} · agg=${escapeHtml(p.aggregation_level||'-')} · ` +
    `Ω_ref=${p.omega_ref}${DATA.baseline_active?` · baseline Ω_b=${p.omega_base}`:''} · ` +
    `n_obs occ/act = ${DATA.n_observed_occ}/${DATA.n_observed_act} · kept occ/act = ${DATA.n_kept_occ}/${DATA.n_kept_act} · ` +
    `data_source=${escapeHtml(DATA.data_source||'-')}`;
  const un = (DATA.unmatched_occ_codes||[]).concat(DATA.unmatched_act_labels||[]);
  if (un.length) document.getElementById('warn').innerHTML =
    `<div class="callout warn" style="margin-bottom:14px;">
      ${un.length} observation(s) could not be matched to O*NET and were skipped: ${escapeHtml(un.slice(0,10).join(', '))}${un.length>10?'…':''}
    </div>`;
  render();
}

document.querySelectorAll('#viewSeg button').forEach(b=>{
  b.onclick=()=>{document.querySelectorAll('#viewSeg button').forEach(x=>x.classList.remove('active')); b.classList.add('active'); VIEW=b.dataset.v; render();};
});
document.getElementById('stdFilter').addEventListener('input', render);
document.getElementById('onlyObserved').addEventListener('change', render);
document.getElementById('search').addEventListener('input', render);

function render(){
  if (!DATA) return;
  const rows = (VIEW==='occ' ? DATA.occupation_impacts : DATA.activity_impacts).slice();
  const stdMax = parseFloat(document.getElementById('stdFilter').value) || 999;
  const onlyObs = document.getElementById('onlyObserved').checked;
  const q = (document.getElementById('search').value||'').toLowerCase();
  let filtered = rows.filter(r => {
    if (parseFloat(r.posterior_std) > stdMax) return false;
    if (onlyObs && (r.observed===null || r.observed===undefined || isNaN(parseFloat(r.observed)))) return false;
    const blob = (VIEW==='occ' ? (r.code+' '+r.title) : r.activity).toString().toLowerCase();
    return !q || blob.includes(q);
  });
  filtered.sort((a,b)=>{
    let x=a[SORT.col], y=b[SORT.col];
    if (typeof x==='string' || typeof y==='string') return ((x||'')+'').localeCompare((y||'')+'') * SORT.dir;
    return ((parseFloat(x)||0) - (parseFloat(y)||0)) * SORT.dir;
  });

  const maxAbs = Math.max(...filtered.map(r=>Math.abs(parseFloat(r.estimate)||0)), 0.1);
  const colName = VIEW==='occ' ? 'title' : 'activity';
  const cols = VIEW==='occ'
    ? ['code', colName, 'observed', 'aioe_baseline', 'estimate', 'posterior_std', 'n_studies']
    : [colName, 'observed', 'estimate', 'posterior_std', 'n_studies'];

  document.getElementById('th').innerHTML = '<tr>' + cols.map(c => {
    const labels = {code:'SOC code', title:'Occupation', activity:'Work activity', observed:'Observed', aioe_baseline:'AIOE', estimate:'Estimate', posterior_std:'Posterior SD', n_studies:'Studies'};
    const num = ['aioe_baseline','estimate','posterior_std','n_studies'].includes(c) ? ' style="text-align:right"' : '';
    return `<th${num} onclick="sortBy('${c}')">${labels[c]||c}${SORT.col===c?(SORT.dir<0?' ↓':' ↑'):''}</th>`;
  }).join('') + '<th>Effect</th></tr>';

  document.getElementById('tb').innerHTML = filtered.map(r=>{
    const isObs = !(r.observed===null||r.observed===undefined||r.observed===''||isNaN(parseFloat(r.observed)));
    const e = parseFloat(r.estimate)||0;
    const eClass = e > 0.001 ? 'pos' : e < -0.001 ? 'neg' : '';
    const barWidth = Math.min(100, Math.abs(e)/maxAbs * 100);
    const bar = `<span class="bar ${eClass}" style="width:${barWidth}px;"></span>`;
    return '<tr>' + cols.map(c => {
      if (c==='observed') return `<td>${isObs ? `<span class="badge observed">${fmtNum(r.observed)}</span>` : '<span class="badge imputed">imputed</span>'}</td>`;
      if (c==='estimate') return `<td class="num ${eClass}">${fmtNum(r.estimate)}</td>`;
      if (c==='posterior_std') return `<td class="num">${fmtNum(r.posterior_std)}</td>`;
      if (c==='aioe_baseline') return `<td class="num">${r.aioe_baseline===undefined||r.aioe_baseline===null?'':fmtNum(r.aioe_baseline)}</td>`;
      if (c==='n_studies') return `<td class="num">${r.n_studies===null||r.n_studies===undefined||r.n_studies===''?'':parseInt(r.n_studies)}</td>`;
      if (c==='code') return `<td class="mono" style="font-size:12px; color:var(--ink-2);">${escapeHtml(r[c]||'')}</td>`;
      return `<td>${escapeHtml(r[c]||'')}</td>`;
    }).join('') + `<td>${bar}</td></tr>`;
  }).join('');
}

function sortBy(col){ SORT.dir = SORT.col===col ? -SORT.dir : -1; SORT.col=col; render(); }

function exportCsv() {
  const rows = (VIEW==='occ' ? DATA.occupation_impacts : DATA.activity_impacts);
  const keys = Object.keys(rows[0]||{});
  const csv = [keys.join(',')].concat(rows.map(r => keys.map(k => JSON.stringify(r[k]??'')).join(','))).join('\n');
  const blob = new Blob([csv], {type:'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${DATA.run_id}_${VIEW}.csv`; a.click();
}

load();
</script>
</body></html>
"""


# ---------- runs.html (all runs index) ----------

_RUNS_STYLE = _STYLE_COMMON + r"""
  :root { --page-w: 1120px; }
  tr.clickable { cursor:pointer; }
  td.id { font-family:var(--mono); font-size:11.5px; color:var(--ink-2); white-space:nowrap; }
  th.num, td.num { text-align:right; white-space:nowrap; }
  .group { margin-top:30px; }
  .group h2 { margin:0 0 4px; }
  .group .blurb { color:var(--ink-2); margin:0 0 12px; max-width:80ch; }
  .verdict { display:inline-block; padding:1px 8px; border-radius:10px; font-size:11.5px; font-weight:600; white-space:nowrap; }
  .verdict.good { background:var(--ok-bg); color:var(--ok-ink); } .verdict.bad { background:var(--err-bg); color:var(--err-ink); }
  .verdict.neutral { background:var(--surface-2); color:var(--ink-2); }
  .agree { display:inline-block; height:6px; border-radius:2px; background:var(--accent); vertical-align:middle; margin-left:6px; opacity:.7; }
"""

_RUNS_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">__THEME_HEAD__<title>Imputation runs &mdash; AI Impact Meta-Review</title>
<style>__STYLE__</style></head><body>
__HEADER__
<main class="page wrap">
  <section class="page-head">
    <div>
      <h1>Imputation runs</h1>
      <p class="lede" id="intro">Each run propagates the observed effects across the O*NET graph under one combination of metric, &beta;,
        pruning, and baseline settings. Click a row to see its results; the <a href="summary.html">summary</a> charts
        findings across the whole set, and the <a href="parameters.html">method</a> page explains each setting.</p>
    </div>
    <span class="page-meta" id="stats"></span>
  </section>
  <div id="groups"></div>
</main>
<script>
function escapeHtml(s){return String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
const f = (x,p) => x==null ? '—' : (+x).toFixed(p==null?2:p);
const LV = {occupation:'occupation', soc_minor:'SOC minor', soc_major:'SOC major'};
const href = r => (r.type==='loo' ? 'loo.html' : 'results.html') + '?run=' + encodeURIComponent(r.run_id);
const pill = m => `<span class="pill ${m}">${escapeHtml(m)}</span>`;
const lvl = r => LV[(r.params||{}).aggregation_level] || (r.params||{}).aggregation_level || 'occupation';
const row = (r, cells) => `<tr class="clickable" onclick="location.href='${href(r)}'">${cells.join('')}</tr>`;
const agreeCell = a => a ? `<td class="num">${f(a.spearman)}<span class="agree" style="width:${Math.max(0,a.spearman||0)*40}px"></span></td><td class="num">${a.top10==null?'—':a.top10+'/10'}</td>` : '<td class="num">—</td><td class="num">—</td>';
function verdict(rk){
  const ci = rk && rk.tau_ci; if (!ci) return '<span class="verdict neutral">n/a</span>';
  if (ci[0] > 0) return '<span class="verdict good">recovers order</span>';
  if (ci[1] < 0) return '<span class="verdict bad">reverses order</span>';
  return '<span class="verdict neutral">no clear signal</span>';
}

function table(head, body){ return `<table class="data"><thead><tr>${head.map(h=>`<th${h[1]?' class="num"':''}>${h[0]}</th>`).join('')}</tr></thead><tbody>${body}</tbody></table>`; }

function renderGroup(g, runs){
  let html = `<section class="group"><h2>${escapeHtml(g.title)}</h2><p class="blurb">${escapeHtml(g.blurb||'')}</p>`;
  if (g.id === 'validation') {
    html += table([['Run'],['Metric'],['Setting'],['Folds',1],['Kendall τ-b',1],['95% CI',1],['Spearman ρ',1],['MAE',1],['Verdict']],
      runs.map(r => { const s=r.loo_summary||{}, rk=(s.rank||{}).overall||{}, o=s.overall||{};
        return row(r, [`<td class="id">${escapeHtml(r.run_id)}</td>`, `<td>${pill(r.params.metric)}</td>`,
          `<td>${escapeHtml((r.label||'').replace(/^LOO · (Speed|Quality) · /,''))}</td>`, `<td class="num">${o.n??'—'}</td>`,
          `<td class="num">${f(rk.kendall_tau)}</td>`, `<td class="num">${rk.tau_ci?`[${f(rk.tau_ci[0])}, ${f(rk.tau_ci[1])}]`:'—'}</td>`,
          `<td class="num">${f(rk.spearman_r)}</td>`, `<td class="num">${f(o.mae,3)}</td>`, `<td>${verdict(rk)}</td>`]); }).join(''));
  } else if (g.id === 'sensitivity') {
    html += table([['Run'],['Metric'],['Change vs headline'],['Kept occ/act',1],['Occ ρ',1],['Occ top-10',1],['Act ρ',1],['Act top-10',1]],
      runs.map(r => { const a=r.agreement||{};
        return row(r, [`<td class="id">${escapeHtml(r.run_id)}</td>`, `<td>${pill(r.params.metric)}</td>`,
          `<td>${escapeHtml((r.label||'').replace(/^(Speed|Quality) · /,''))}</td>`, `<td class="num">${r.n_kept_occ} / ${r.n_kept_act}</td>`,
          agreeCell(a.occupation), agreeCell(a.activity)]); }).join(''));
  } else {
    html += table([['Run'],['Metric'],['Aggregation'],['β',1],['Baseline'],['Observed occ/act',1],['Kept occ/act',1],['Occ ρ vs occupation run',1]],
      runs.map(r => { const p=r.params||{}, a=(r.agreement||{}).occupation;
        return row(r, [`<td class="id">${escapeHtml(r.run_id)}</td>`, `<td>${pill(p.metric)}</td>`, `<td>${escapeHtml(lvl(r))}</td>`,
          `<td class="num">${p.beta}</td>`, `<td>${r.baseline_active?`AIOE, Ω<sub>b</sub>=${p.omega_base}`:'—'}</td>`,
          `<td class="num">${r.n_observed_occ} / ${r.n_observed_act}</td>`, `<td class="num">${r.n_kept_occ} / ${r.n_kept_act}</td>`,
          `<td class="num">${a?f(a.spearman):'reference'}</td>`]); }).join(''));
  }
  return html + '</section>';
}

fetch('data/runs/index.json').then(r=>r.json()).then(d=>{
  const el = document.getElementById('groups');
  if (!d.runs.length) { el.innerHTML='<p style="color:var(--muted);">No runs published.</p>'; return; }
  document.getElementById('stats').textContent = `${d.runs.length} runs`;
  if (d.observations) {
    const o = d.observations;
    document.getElementById('intro').innerHTML +=
      ` All runs share one data snapshot: ${o.speed} speed + ${o.quality} quality effect rows, aggregated to ` +
      `${o.occ_codes} observed occupation and ${o.act_codes} activity codes.`;
  }
  const groups = (d.groups && d.groups.length) ? d.groups : [{id:'all', title:'All runs', blurb:''}];
  el.innerHTML = groups.map(g => {
    const runs = d.runs.filter(r => g.id==='all' || r.group===g.id);
    return runs.length ? renderGroup(g, runs) : '';
  }).join('');
});
</script>
</body></html>
"""


# ---------- loo.html (leave-one-out viewer; port of review_app.LOO_HTML) ----------

_LOO_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Leave-one-out CV · AI Impact Meta-Review</title>
__THEME_HEAD__
<style>
  .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap:10px; }
  .subheads { display:flex; gap:24px; margin-top:12px; color:var(--muted); font-size:13px; flex-wrap:wrap; }
  .subheads span { color:var(--ink) !important; font-family:var(--mono) !important; font-size:12.5px; }
  .card table { border-top:1.5px solid var(--ink); border-bottom:1.5px solid var(--ink); }
  .card table th { cursor:pointer; user-select:none; }
  .card .toolbar { border:0; padding:0; background:transparent; }
  svg.scatter { display:block; background:var(--surface); }
  svg.scatter .axis { stroke:#9a968d; stroke-width:1; }
  svg.scatter .grid { stroke:#eeebe4; stroke-width:1; }
  svg.scatter .tick { font-family:var(--sans); font-size:11px; fill:#787c84; }
  svg.scatter .diag { stroke:#1d1f23; stroke-width:1; stroke-dasharray:4 3; }
  svg.scatter circle { stroke:#fff; stroke-width:1; }
  svg.scatter circle.occ { fill:#2e67a8; opacity:0.8; }
  svg.scatter circle.act { fill:#c0762b; opacity:0.8; }
  svg.scatter .band { fill:#23406a; opacity:0.06; }
  .note ol { margin:4px 0 0 18px; padding:0; }
  .charts { display:flex; flex-wrap:wrap; gap:24px; }
  .charts > div { flex:1 1 420px; min-width:0; }
  .charts h3 { margin:0 0 6px; font-size:15px; }
</style></head><body>
__HEADER__
<main class="page wrap">
  <section class="page-head">
    <div>
      <div class="crumbs"><a href="runs.html">Imputation runs</a> / <span class="mono" id="crumbId"></span></div>
      <h1>Leave-one-out cross-validation</h1>
      <p class="lede">Each observed effect is held out in turn and re-predicted from the rest of the graph. This tests whether the graph recovers held-out values and their ordering.</p>
    </div>
    <div class="page-actions"><span class="page-meta" id="stats">…</span></div>
  </section>
  <div class="card">
    <h2>Run parameters</h2>
    <div id="paramsLine" class="mono" style="font-size:12.5px; color:var(--ink-2);">…</div>
  </div>

  <div class="card">
    <h2>Overall accuracy</h2>
    <div class="grid" id="overallGrid"></div>
    <div class="subheads">
      <div>Occupations: <span id="occSummary" style="color:#333; font-family:ui-monospace,Menlo,monospace;">…</span></div>
      <div>Activities:  <span id="actSummary" style="color:#333; font-family:ui-monospace,Menlo,monospace;">…</span></div>
    </div>
  </div>

  <div class="card">
    <h2>Rank recovery (held-out)</h2>
    <div class="help" style="margin-bottom:12px;">
      Does the graph put held-out nodes in the right <b>order</b>? Each node is ranked by its held-out prediction and compared
      with its rank by actual value. Rank statistics ignore uniform shrinkage, so they test ordering, not magnitude.
    </div>
    <div class="grid" id="rankGrid"></div>
    <div class="subheads">
      <div>Occupations (ranked within type): <span id="occRank" style="color:#333; font-family:ui-monospace,Menlo,monospace;">…</span></div>
      <div>Activities (ranked within type): <span id="actRank" style="color:#333; font-family:ui-monospace,Menlo,monospace;">…</span></div>
    </div>
    <div class="note" id="rankNote"></div>
  </div>

  <div class="card">
    <h2>Held-out predictions</h2>
    <div class="charts">
      <div>
        <h3>Value: actual vs predicted</h3>
        <div id="scatterWrap"></div>
        <div class="help" style="margin-top:6px;">Dashed line: perfect prediction (y = x).</div>
      </div>
      <div>
        <h3>Rank: actual rank vs held-out rank</h3>
        <div id="rankScatterWrap"></div>
        <div class="help" style="margin-top:6px;">Rank 1 is the largest effect. Shaded band: within &plusmn;2 ranks.</div>
      </div>
    </div>
    <div class="chart-legend"><span><span class="sw" style="background:#2e67a8; border-radius:50%; width:10px;"></span>Occupation held out</span><span><span class="sw" style="background:#c0762b; border-radius:50%; width:10px;"></span>Activity held out</span></div>
  </div>

  <div class="card">
    <h2>Per-observation table</h2>
    <div class="toolbar">
      <div class="seg" id="filterSeg">
        <button data-f="all" class="active">All</button>
        <button data-f="occupation">Occupations</button>
        <button data-f="activity">Activities</button>
      </div>
      <input type="search" id="search" placeholder="Search label or code…" style="min-width:240px;"/>
      <button class="btn" onclick="exportCsv()" style="margin-left:auto;">Download CSV</button>
    </div>
    <table><thead id="th"></thead><tbody id="tb"></tbody></table>
  </div>
</main>
<script>
const RUN_ID = new URLSearchParams(location.search).get('run') || '';
let DATA = null, FILTER='all', SORT={col:'abs_rank_delta', dir:-1};

function fmt(x,p){ if(x==null) return '—'; const n=parseFloat(x); return isNaN(n) ? x : n.toFixed(p==null?3:p); }
function escapeHtml(s){return (s||'').toString().replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}

async function load() {
  document.getElementById('crumbId').textContent = RUN_ID;
  const resp = await fetch('data/runs/' + encodeURIComponent(RUN_ID) + '.json');
  if (!resp.ok) { document.querySelector('main').innerHTML = '<div class="err" style="margin-top:30px;">Unknown run: ' + escapeHtml(RUN_ID) + '. See <a href="runs.html">Imputation runs</a>.</div>'; return; }
  DATA = await resp.json();
  if (!DATA.loo) { location.replace('results.html?run=' + encodeURIComponent(RUN_ID)); return; }
  if (DATA.label) document.querySelector('.page-head h1').textContent = DATA.label;
  const p = DATA.params;
  document.getElementById('stats').textContent =
    `${p.metric.toUpperCase()} · β=${p.beta} · Ω_ref=${p.omega_ref}` +
    (DATA.baseline_active ? ` · baseline Ω_b=${p.omega_base}` : '') +
    ` · ${DATA.n_folds} folds`;
  document.getElementById('paramsLine').textContent =
    `metric=${p.metric} · β=${p.beta} · Ω_ref=${p.omega_ref} · ` +
    `agg=${p.aggregation_level} · manual_prune=${p.manual_prune} · ` +
    `weight_threshold=${p.activity_weight_threshold} · excluded=[${(p.excluded_soc_majors||[]).join(',')}]` +
    (DATA.baseline_active ? ` · baseline Ω_b=${p.omega_base}` : '');

  const s = DATA.loo.summary;
  const grid = document.getElementById('overallGrid');
  const o = s.overall || {};
  const cells = [
    ['n folds', o.n],
    ['MAE', o.mae!=null ? fmt(o.mae, 4) : '—'],
    ['RMSE', o.rmse!=null ? fmt(o.rmse, 4) : '—'],
    ['bias', o.bias!=null ? fmt(o.bias, 4) : '—'],
    ['R²', o.r2!=null ? fmt(o.r2, 3) : '—'],
    ['Pearson r', o.pearson_r!=null ? fmt(o.pearson_r, 3) : '—'],
    ['Spearman ρ', o.spearman_r!=null ? fmt(o.spearman_r, 3) : '—'],
  ];
  grid.innerHTML = cells.map(([k,v]) => `<div class="metric"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');
  const sub = (obj) => {
    if (!obj || !obj.n) return 'no folds';
    return `n=${obj.n} · MAE=${fmt(obj.mae,3)} · R²=${obj.r2!=null?fmt(obj.r2,3):'—'} · ρ=${obj.spearman_r!=null?fmt(obj.spearman_r,3):'—'}`;
  };
  document.getElementById('occSummary').textContent = sub(s.occupation);
  document.getElementById('actSummary').textContent = sub(s.activity);

  renderRankCard();
  const rows = DATA.loo.rows;
  renderScatter('scatterWrap', rows, 'actual', 'predicted', {xLabel:'actual (held-out)', yLabel:'predicted'});
  if (rows.length && rows[0].loo_rank != null) {
    renderScatter('rankScatterWrap', rows, 'actual_rank', 'loo_rank',
      {xLabel:'actual rank', yLabel:'held-out rank', rank:true, band:2});
  } else {
    document.getElementById('rankScatterWrap').innerHTML = '<div style="color:#888;">No rank data.</div>';
  }
  renderTable();
}

function pfmt(p){ if(p==null) return ''; return p < 0.001 ? 'p<0.001' : 'p=' + p.toFixed(3); }

function renderRankCard() {
  const R = (DATA.loo.summary||{}).rank;
  const grid = document.getElementById('rankGrid');
  if (!R || !R.overall || !R.overall.n) { grid.innerHTML = '<div style="color:#888;">Not enough folds.</div>'; return; }
  const o = R.overall;
  const tk = o.topk || {};
  const cell = (k, v, sub) => `<div class="metric"><div class="k">${k}</div><div class="v">${v}</div>${sub?`<div class="sub">${sub}</div>`:''}</div>`;
  const cells = [
    cell("Kendall τ-b", o.kendall_tau!=null?fmt(o.kendall_tau,3):'—', pfmt(o.kendall_p)),
    cell("τ 95% CI", o.tau_ci ? `[${fmt(o.tau_ci[0],2)}, ${fmt(o.tau_ci[1],2)}]` : '—', 'bootstrap over folds'),
    cell("Concordance C", o.concordance_c!=null?fmt(o.concordance_c,3):'—', 'P(pair ordered right) · chance 0.5'),
    cell("Spearman ρ", o.spearman_r!=null?fmt(o.spearman_r,3):'—', pfmt(o.spearman_p)),
    cell("Mean |Δrank|", o.mean_abs_rank_delta!=null?fmt(o.mean_abs_rank_delta,2):'—',
         o.mean_abs_rank_delta_chance!=null?`chance ${fmt(o.mean_abs_rank_delta_chance,2)}`:''),
    cell("Footrule (norm.)", o.footrule_norm!=null?fmt(o.footrule_norm,3):'—', '0 = identical · 1 = max'),
  ];
  for (const K of ['3','5','10']) {
    if (tk[K]) cells.push(cell(`Top-${K} precision`, `${tk[K].hits}/${tk[K].K}`, `chance ${fmt(tk[K].chance,2)}`));
  }
  grid.innerHTML = cells.join('');
  const sub = (x) => {
    if (!x || !x.n) return 'no folds';
    if (x.n < 3) return `n=${x.n} · too few folds`;
    const t5 = (x.topk||{})['5'] || (x.topk||{})['3'];
    return `n=${x.n} · τ=${x.kendall_tau!=null?fmt(x.kendall_tau,3):'—'} ${pfmt(x.kendall_p)} · C=${x.concordance_c!=null?fmt(x.concordance_c,3):'—'}` +
           ` · |Δrank|=${fmt(x.mean_abs_rank_delta,2)} (chance ${fmt(x.mean_abs_rank_delta_chance,2)})` +
           (t5 ? ` · top-${t5.K} ${t5.hits}/${t5.K}` : '');
  };
  document.getElementById('occRank').textContent = sub(R.occupation);
  document.getElementById('actRank').textContent = sub(R.activity);
  const m = R.method || {};
  document.getElementById('rankNote').innerHTML =
    (R.backfilled ? '<div style="color:var(--warn-ink);">Rank stats computed on load for this older run.</div>' : '') +
    escapeHtml(m.description||'') +
    (m.citations && m.citations.length ? '<ol>' + m.citations.map(c=>`<li>${escapeHtml(c)}</li>`).join('') + '</ol>' : '');
}

function renderScatter(containerId, rows, xKey, yKey, opts) {
  opts = opts || {};
  const pad = {l:52, r:16, t:14, b:40};
  const W = 520, H = 360;
  const xs = rows.map(r=>+r[xKey]), ys = rows.map(r=>+r[yKey]);
  let xL, xH;
  if (opts.rank) {
    xL = 0.5; xH = Math.max(...xs, ...ys, 1) + 0.5;
  } else {
    const lo = Math.min(...xs, ...ys, 0), hi = Math.max(...xs, ...ys, 0.001);
    const range = hi - lo || 1; xL = lo - range*0.05; xH = hi + range*0.05;
  }
  // Rank axes are reversed so rank 1 (largest effect) sits top-right.
  const fx = v => opts.rank ? (xH - v) / (xH - xL) : (v - xL) / (xH - xL);
  const sx = v => pad.l + fx(v) * (W - pad.l - pad.r);
  const sy = v => H - pad.b - fx(v) * (H - pad.t - pad.b);
  function ticks(a,b,n){
    const r=b-a||1, step=Math.pow(10,Math.floor(Math.log10(r/n))); const err=n*step/r;
    let m=1; if(err<=0.15)m=10; else if(err<=0.35)m=5; else if(err<=0.75)m=2;
    const s=Math.max(opts.rank?1:0, m*step); const t0=Math.ceil(a/s)*s; const out=[];
    for(let v=t0; v<=b+1e-9; v+=s) out.push(Math.round(v/s)*s);
    return out;
  }
  const xt = ticks(xL, xH, 6);
  const tf = t => opts.rank ? String(Math.round(t)) : t.toFixed(2);
  const gridLines = xt.map(t=>{
    const x=sx(t), y=sy(t);
    return `<line class="grid" x1="${x}" y1="${pad.t}" x2="${x}" y2="${H-pad.b}"/>` +
           `<line class="grid" x1="${pad.l}" y1="${y}" x2="${W-pad.r}" y2="${y}"/>`;
  }).join('');
  const xTicks = xt.map(t=>`<text class="tick" x="${sx(t)}" y="${H-pad.b+14}" text-anchor="middle">${tf(t)}</text>`).join('');
  const yTicks = xt.map(t=>`<text class="tick" x="${pad.l-6}" y="${sy(t)+3}" text-anchor="end">${tf(t)}</text>`).join('');
  let band = '';
  if (opts.band) {
    const k = opts.band;
    const pts = [[xL, xL+k],[xH, xH+k],[xH, xH-k],[xL, xL-k]]
      .map(([x,y]) => `${sx(x)},${sy(y)}`).join(' ');
    band = `<clipPath id="${containerId}_clip"><rect x="${pad.l}" y="${pad.t}" width="${W-pad.l-pad.r}" height="${H-pad.t-pad.b}"/></clipPath>` +
           `<polygon class="band" clip-path="url(#${containerId}_clip)" points="${pts}"/>`;
  }
  const diag = `<line class="diag" x1="${sx(xL)}" y1="${sy(xL)}" x2="${sx(xH)}" y2="${sy(xH)}"/>`;
  const pts = rows.map(r => {
    const cls = r.node_type === 'occupation' ? 'occ' : 'act';
    const title = opts.rank
      ? `${r.label}: actual rank ${fmt(r[xKey],1)}, held-out rank ${fmt(r[yKey],1)} (Δ ${fmt(r.rank_delta,1)})`
      : `${r.label}: actual ${fmt(r.actual,3)}, predicted ${fmt(r.predicted,3)}, resid ${fmt(r.residual,3)}`;
    return `<circle class="${cls}" cx="${sx(+r[xKey])}" cy="${sy(+r[yKey])}" r="4"><title>${escapeHtml(title)}</title></circle>`;
  }).join('');
  const xAxis = `<line class="axis" x1="${pad.l}" y1="${H-pad.b}" x2="${W-pad.r}" y2="${H-pad.b}"/>`;
  const yAxis = `<line class="axis" x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${H-pad.b}"/>`;
  const xLabel = `<text class="tick" x="${(pad.l+W-pad.r)/2}" y="${H-6}" text-anchor="middle">${opts.xLabel||xKey}</text>`;
  const yLabel = `<text class="tick" x="${-H/2}" y="14" text-anchor="middle" transform="rotate(-90)">${opts.yLabel||yKey}</text>`;
  document.getElementById(containerId).innerHTML =
    `<svg class="scatter" viewBox="0 0 ${W} ${H}" style="width:100%; max-width:${W}px; height:auto;">
      ${gridLines}${band}${diag}${pts}${xAxis}${yAxis}${xTicks}${yTicks}${xLabel}${yLabel}
    </svg>`;
}

document.getElementById('search').addEventListener('input', renderTable);
document.querySelectorAll('#filterSeg button').forEach(b=>{
  b.onclick=()=>{document.querySelectorAll('#filterSeg button').forEach(x=>x.classList.remove('active')); b.classList.add('active'); FILTER=b.dataset.f; renderTable();};
});

function renderTable() {
  const q = (document.getElementById('search').value||'').toLowerCase();
  let rows = DATA.loo.rows.map(r => Object.assign({}, r,
    {abs_rank_delta: r.rank_delta != null ? Math.abs(r.rank_delta) : null}));
  if (FILTER !== 'all') rows = rows.filter(r => r.node_type === FILTER);
  if (q) rows = rows.filter(r => ((r.label||'')+' '+(r.code||'')).toLowerCase().includes(q));
  rows.sort((a,b)=>{
    let x=a[SORT.col], y=b[SORT.col];
    if (typeof x==='string' || typeof y==='string') return ((x||'')+'').localeCompare((y||'')+'') * SORT.dir;
    if (x == null && y == null) return 0;
    if (x == null) return 1;
    if (y == null) return -1;
    return ((parseFloat(x)||0) - (parseFloat(y)||0)) * SORT.dir;
  });
  const cols = ['node_type','code','label','actual','predicted','residual','abs_error','posterior_std',
                'actual_rank','loo_rank','rank_delta','abs_rank_delta'];
  const labels = {node_type:'Type', code:'Code', label:'Label', actual:'Actual', predicted:'Predicted',
                  residual:'Residual', abs_error:'|error|', posterior_std:'Post. std',
                  actual_rank:'Actual rank', loo_rank:'Held-out rank', rank_delta:'Δrank', abs_rank_delta:'|Δrank|'};
  document.getElementById('th').innerHTML = '<tr>' + cols.map(c =>
    `<th onclick="sortBy('${c}')">${labels[c]}${SORT.col===c?(SORT.dir<0?' ↓':' ↑'):''}</th>`).join('') + '</tr>';
  document.getElementById('tb').innerHTML = rows.map(r => {
    const rc = parseFloat(r.residual)||0;
    const rcls = rc > 0.001 ? 'pos' : rc < -0.001 ? 'neg' : '';
    return '<tr>' +
      `<td>${r.node_type}</td>` +
      `<td class="mono" style="font-size:11.5px; color:var(--muted);">${escapeHtml(r.code||'')}</td>` +
      `<td>${escapeHtml(r.label||'')}</td>` +
      `<td class="num">${fmt(r.actual,3)}</td>` +
      `<td class="num">${fmt(r.predicted,3)}</td>` +
      `<td class="num ${rcls}">${fmt(r.residual,3)}</td>` +
      `<td class="num">${fmt(r.abs_error,3)}</td>` +
      `<td class="num">${fmt(r.posterior_std,3)}</td>` +
      `<td class="num">${fmt(r.actual_rank,1)}</td>` +
      `<td class="num">${fmt(r.loo_rank,1)}</td>` +
      `<td class="num">${fmt(r.rank_delta,1)}</td>` +
      `<td class="num">${fmt(r.abs_rank_delta,1)}</td>` +
      '</tr>';
  }).join('');
}

function sortBy(col){ SORT.dir = SORT.col===col ? -SORT.dir : -1; SORT.col=col; renderTable(); }

function exportCsv() {
  const rows = DATA.loo.rows;
  const keys = Object.keys(rows[0]||{});
  const csv = [keys.join(',')].concat(rows.map(r => keys.map(k => JSON.stringify(r[k]??'')).join(','))).join('\n');
  const blob = new Blob([csv], {type:'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${RUN_ID}_loocv.csv`; a.click();
}

load();
</script></body></html>
"""


# ---------- summary.html (curated-run overview: findings, agreement and LOO charts) ----------

_SUMMARY_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">__THEME_HEAD__<title>Run summary &mdash; AI Impact Meta-Review</title>
<style>
:root{
  --page-w:1120px;
  --grid:var(--rule-soft); --good:var(--ok-ink); --good-bg:var(--ok-bg);
  --bad:var(--err-ink); --bad-bg:var(--err-bg); --neutral-bg:var(--surface-2);
}
.sum section{display:flex;flex-direction:column;gap:14px;margin-top:34px}
.sum section.page-head{display:flex;flex-direction:row;margin-top:0}
.sum h2{margin:0}
.sum h3{font-size:12px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin:0;font-weight:600;font-family:var(--sans)}
.sum p{margin:0;max-width:72ch;color:var(--ink-2)}
.num{font-family:var(--mono);font-variant-numeric:tabular-nums}
.facts{display:flex;flex-wrap:wrap;gap:8px 26px;margin-top:10px;font-size:13px;color:var(--ink-2)}
.facts b{color:var(--ink);font-family:var(--mono);font-weight:500}
.findings{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}
.finding{background:var(--surface);border:1px solid var(--rule);border-radius:var(--radius);padding:14px 16px;display:flex;flex-direction:column;gap:8px}
.finding strong{font-family:var(--serif);font-size:16px;font-weight:600;line-height:1.3}
.finding p{font-size:13.5px}
.chip{display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:600;padding:2px 8px;border-radius:999px;width:fit-content;letter-spacing:.02em}
.chip.good{background:var(--good-bg);color:var(--good)} .chip.bad{background:var(--bad-bg);color:var(--bad)} .chip.neutral{background:var(--neutral-bg);color:var(--ink-2)}
.key{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}
.key.speed{background:var(--speed)} .key.quality{background:var(--quality)}
.panel{background:var(--surface);border:1px solid var(--rule);border-radius:var(--radius);padding:16px 18px}
.scroll{overflow-x:auto}
.sum .panel table{border-collapse:collapse;width:100%;font-size:13px}
.sum .panel th{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);font-weight:600;text-align:left;padding:6px 10px;border-bottom:1px solid var(--rule);white-space:nowrap;background:transparent}
.sum .panel td{padding:7px 10px;border-bottom:1px solid var(--grid);vertical-align:top}
.sum .panel tr:last-child td{border-bottom:0}
.sum td.num,.sum th.num{text-align:right;white-space:nowrap}
.sum tr.link{cursor:pointer} .sum tr.link:hover td{background:var(--hover)}
.rid{font-family:var(--mono);font-size:11px;color:var(--muted)}
.sum .seg{display:inline-flex;border:1px solid var(--rule);border-radius:var(--radius);overflow:hidden;background:var(--surface)}
.sum .seg button{font:inherit;font-size:13px;padding:5px 14px;border:0;background:transparent;color:var(--ink-2);cursor:pointer}
.sum .seg button[aria-pressed="true"]{background:var(--accent);color:#fff}
.sum .seg button:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
.levels{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:12px}
.level{background:var(--surface);border:1px solid var(--rule);border-radius:var(--radius);padding:14px 16px;display:flex;flex-direction:column;gap:10px}
.level header{display:flex;justify-content:space-between;align-items:baseline;gap:8px;flex-wrap:wrap}
.level header strong{font-size:15px}
.meta{font-size:12px;color:var(--muted)}
ol.top{margin:0;padding:0;list-style:none;display:flex;flex-direction:column;gap:3px;font-size:13px}
ol.top li{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:10px;align-items:baseline}
ol.top li span:first-child{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.obs{font-size:10px;font-weight:600;color:var(--good);margin-left:6px;letter-spacing:.04em}
.chart svg{display:block;width:100%;height:auto}
.chart text{fill:var(--ink-2);font-family:var(--sans);font-size:12px}
.chart .tick{fill:var(--muted);font-size:11px;font-family:var(--mono)}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:12px;color:var(--ink-2);align-items:center}
.legend svg{vertical-align:middle;margin-right:4px}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--paper);font-size:12px;padding:6px 9px;border-radius:5px;max-width:320px;line-height:1.4;z-index:5}
ul.plain{margin:0;padding-left:18px;color:var(--ink-2);font-size:13.5px;display:flex;flex-direction:column;gap:4px;max-width:80ch}
</style></head><body>
__HEADER__
<main class="page wrap sum">
  <section class="page-head">
    <div>
      <h1>What the published runs say</h1>
      <p class="lede">These runs are the ones published on this site, all computed from one snapshot of the review table so they're directly comparable. They cover headline estimates at three aggregation levels, one-knob sensitivity checks against the occupation-level headline, and leave-one-out validation of rank recovery. Full tables are on <a href="runs.html">Imputation runs</a>.</p>
      <div class="facts" id="facts"></div>
    </div>
    <span class="page-meta" id="stats"></span>
  </section>
  <section>
    <h2>Findings</h2>
    <div class="findings" id="findings"></div>
  </section>

  <section>
    <h2>Headline estimates</h2>
    <p>Default parameters: β = 2.5, SOC majors 37/45/47/49/51/53 pruned, activity threshold 10, and the AIOE baseline for speed only. Lists show the five largest estimated effects. Speed is a log-ratio (higher = faster with AI); quality is Hedges' g. <span class="obs" style="margin:0">OBS</span> marks an observed (not imputed) node.</p>
    <div><div class="seg" role="group" aria-label="Metric" id="metricSeg">
      <button type="button" id="m-speed" data-m="speed" aria-pressed="true">Speed</button>
      <button type="button" id="m-quality" data-m="quality" aria-pressed="false">Quality</button>
    </div></div>
    <div class="levels" id="levels"></div>
  </section>

  <section>
    <h2>Does the ranking survive a changed knob?</h2>
    <p>Each row changes one setting and compares the resulting estimates with the matching occupation-level headline run. The comparison uses Spearman ρ over the nodes both runs share, with coarser levels rolled up before joining. ρ near 1 means that choice doesn't change the ordering.</p>
    <div class="panel">
      <div class="legend" style="margin-bottom:8px">
        <span><svg width="12" height="12"><circle cx="6" cy="6" r="5" fill="var(--ink-2)"/></svg>occupations</span>
        <span><svg width="12" height="12"><rect x="1.5" y="1.5" width="9" height="9" transform="rotate(45 6 6)" fill="none" stroke="var(--ink-2)" stroke-width="2"/></svg>activities</span>
        <span><span class="key speed"></span>speed</span><span><span class="key quality"></span>quality</span>
      </div>
      <div class="chart scroll" id="sensChart"></div>
    </div>
    <div class="panel scroll"><table id="sensTable"></table></div>
  </section>

  <section>
    <h2>Can the graph recover held-out effects?</h2>
    <p>Each observed node is held out and re-imputed. Kendall τ-b compares the order of held-out predictions with the order of the actual values. The bars are bootstrap 95% CIs over folds. An interval entirely right of 0 means the order is recovered; entirely left of 0 means it's reversed.</p>
    <div class="panel"><div class="chart scroll" id="looChart"></div></div>
    <div class="panel scroll"><table id="looTable"></table></div>
  </section>

  <section>
    <h2>How these runs were chosen</h2>
    <ul class="plain">
      <li><b>Headline:</b> both metrics at occupation, SOC-minor and SOC-major level, with default parameters.</li>
      <li><b>Sensitivity:</b> each run changes one knob from the occupation-level headline: β (1, 10), SOC pruning (keep all 22 majors; also drop Management, SOC 11), activity threshold (5), and, for speed only, removing the AIOE baseline. The baseline is speed-only, so a quality run without it would be identical to the headline.</li>
      <li><b>Validation:</b> leave-one-out at each level for both metrics, plus speed without the baseline to test whether the AIOE prior helps recover held-out effects.</li>
      <li>The set is defined in <span class="num">scripts/generate_site_runs.py</span>, which regenerates it from the live review table.</li>
    </ul>
  </section>
</main>
<div id="tip" hidden></div>

<script>
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const f = (x, p=2) => x == null ? '—' : (+x).toFixed(p);
const LV = {occupation:'Occupation', soc_minor:'SOC minor group', soc_major:'SOC major group'};
const LVs = {occupation:'occupation', soc_minor:'SOC minor', soc_major:'SOC major'};
const verdict = l => !l.tau_ci ? ['neutral','n/a'] : l.tau_ci[0] > 0 ? ['good','recovers order'] : l.tau_ci[1] < 0 ? ['bad','reverses order'] : ['neutral','no clear signal'];
const go = (page, id) => `onclick="location.href='${page}?run=${encodeURIComponent(id)}'"`;
let S, R;

function findings(){
  const out = [];
  const ag = k => R[k] && R[k].agreement && R[k].agreement.occupation;
  const prune = ['speed-allsoc','speed-excl11','speed-thr5','quality-allsoc','quality-excl11','quality-thr5'].map(ag).filter(Boolean);
  if (prune.length) {
    const lo = Math.min(...prune.map(a => a.spearman));
    out.push(lo >= 0.9
      ? ['good','Robust','Pruning choices don’t move the ranking', `Keeping every SOC major, also excluding Management (11), or lowering the activity threshold to 5 all give ρ ≥ ${f(lo)} against the headline run for both metrics.`]
      : ['neutral','Sensitive','Pruning choices shift the ranking', `The lowest agreement among the pruning variants is ρ = ${f(lo)} against the headline run.`]);
  }
  const nb = ag('speed-nobase');
  if (nb) out.push(nb.spearman < 0.8
    ? ['neutral','Driver','The speed ranking mostly comes from AIOE', `Without the AIOE baseline, speed occupation ρ falls to ${f(nb.spearman)} and only ${nb.top10}/10 of the top occupations stay the same. With so few observed occupations, the prior shapes most of the imputed ordering.`]
    : ['good','Robust','The speed ranking holds without AIOE', `Dropping the AIOE baseline still gives ρ = ${f(nb.spearman)} against the headline run.`]);
  for (const m of ['speed','quality']) {
    const runs = ['occupation','soc_minor','soc_major'].map(l => R[`loo-${m}-${l}`]).filter(Boolean);
    if (!runs.length) continue;
    const vs = runs.map(r => [r, verdict(r.loo)[0]]);
    const txt = runs.map(r => `${LVs[r.level] || r.level} τ = ${f(r.loo.tau)} [${f(r.loo.tau_ci[0])}, ${f(r.loo.tau_ci[1])}]`).join('; ');
    const M = m[0].toUpperCase() + m.slice(1);
    if (vs.some(v => v[1] === 'bad'))
      out.push(['bad','Warning',`${M} predictions reverse the true order at some levels`, `Held-out ${m}: ${txt}. At least one CI is below 0, so treat imputed ${m} rankings as unvalidated.`]);
    else if (vs.some(v => v[1] === 'good'))
      out.push(['neutral','Weak signal',`${M} recovery is positive but thin`, `Held-out ${m}: ${txt}. Only levels whose CI excludes 0 show reliable recovery.`]);
    else
      out.push(['neutral','No signal',`${M} recovery is indistinguishable from chance`, `Held-out ${m}: ${txt}.`]);
  }
  $('findings').innerHTML = out.map(([c,t,h,b]) => `<div class="finding"><span class="chip ${c}">${c==='good'?'✓':c==='bad'?'!':'•'} ${t}</span><strong>${h}</strong><p>${b}</p></div>`).join('');
}

function renderLevels(metric){
  $('levels').innerHTML = ['occupation','soc_minor','soc_major'].map(lv => {
    const r = R[`${metric}-${lv}`]; if (!r) return '';
    const li = arr => arr.map(t => `<li><span title="${esc(t.label)}">${esc(t.label)}${t.obs!=null?'<span class="obs">OBS</span>':''}</span><span class="num">${f(t.est,3)}</span></li>`).join('');
    const ag = r.agreement && r.agreement.occupation ? `<span class="meta">ρ vs occupation run: ${f(r.agreement.occupation.spearman)}</span>` : `<span class="meta">reference run</span>`;
    return `<article class="level">
      <header><strong><span class="key ${metric}"></span>${LV[lv]}</strong><a class="rid" href="results.html?run=${encodeURIComponent(r.run_id)}">${r.run_id}</a></header>
      <div class="meta">observed ${r.obs[0]} / ${r.obs[1]} · kept ${r.kept[0]} occ / ${r.kept[1]} act · mean est. <span class="num">${f(r.occ_mean,3)}</span></div>
      <h3>Top occupations</h3><ol class="top">${li(r.top_occ)}</ol>
      <h3>Top activities</h3><ol class="top">${li(r.top_act)}</ol>
      ${ag}
    </article>`;
  }).join('');
}
document.querySelectorAll('#metricSeg button').forEach(b => b.onclick = () => {
  document.querySelectorAll('#metricSeg button').forEach(x => x.setAttribute('aria-pressed', x===b));
  renderLevels(b.dataset.m);
});

const tip = $('tip');
function bindTips(root){
  root.querySelectorAll('[data-tip]').forEach(el => {
    el.addEventListener('mousemove', e => { tip.hidden=false; tip.innerHTML=el.dataset.tip; tip.style.left=Math.min(e.clientX+14, innerWidth-330)+'px'; tip.style.top=(e.clientY+14)+'px'; });
    el.addEventListener('mouseleave', () => tip.hidden=true);
  });
}

function renderSens(){
  const sens = S.runs.filter(r => r.group==='sensitivity' && r.agreement).sort((a,b) => (a.metric>b.metric?-1:a.metric<b.metric?1:0));
  const rowH=30, padL=300, padR=20, padT=24, W=880, H=padT+sens.length*rowH+30;
  const x0=0.2, x1=1.0, sx = v => padL + (Math.max(v,x0)-x0)/(x1-x0)*(W-padL-padR);
  let g='';
  [0.2,0.4,0.6,0.8,1.0].forEach(t => { g += `<line x1="${sx(t)}" x2="${sx(t)}" y1="${padT-6}" y2="${H-26}" stroke="var(--grid)"/><text class="tick" x="${sx(t)}" y="${H-10}" text-anchor="middle">${t.toFixed(1)}</text>`; });
  g += `<text class="tick" x="${padL}" y="12">Spearman ρ vs headline occupation run →</text>`;
  sens.forEach((r,i) => {
    const y = padT + i*rowH + rowH/2, col = `var(--${r.metric})`;
    if (i>0 && sens[i-1].metric!==r.metric) g += `<line x1="0" x2="${W}" y1="${y-rowH/2}" y2="${y-rowH/2}" stroke="var(--rule)"/>`;
    g += `<text x="0" y="${y+4}"><tspan fill="${col}">●</tspan> ${esc(r.label.replace(/^(Speed|Quality) · /,''))}</text>`;
    const oc=r.agreement.occupation, ac=r.agreement.activity;
    if (oc && ac) g += `<line x1="${sx(oc.spearman)}" x2="${sx(ac.spearman)}" y1="${y}" y2="${y}" stroke="${col}" stroke-opacity=".35" stroke-width="2"/>`;
    if (ac){ const cx=sx(ac.spearman); g += `<rect x="${cx-5}" y="${y-5}" width="10" height="10" transform="rotate(45 ${cx} ${y})" fill="var(--surface)" stroke="${col}" stroke-width="2" data-tip="${esc(r.label)}<br>activities: ρ = ${f(ac.spearman,3)} (n=${ac.n}) · top-10 overlap ${ac.top10??'—'}/10"/>`; }
    if (oc) g += `<circle cx="${sx(oc.spearman)}" cy="${y}" r="6" fill="${col}" stroke="var(--surface)" stroke-width="2" data-tip="${esc(r.label)}<br>occupations: ρ = ${f(oc.spearman,3)} (n=${oc.n}) · top-10 overlap ${oc.top10??'—'}/10"/>`;
  });
  $('sensChart').innerHTML = `<svg viewBox="0 0 ${W} ${H}" style="min-width:640px" role="img" aria-label="Spearman agreement of each sensitivity run with its headline run">${g}</svg>`;
  bindTips($('sensChart'));
  $('sensTable').innerHTML = `<thead><tr><th>Run</th><th>Change</th><th class="num">Kept occ/act</th><th class="num">Occ ρ</th><th class="num">Occ top-10</th><th class="num">Act ρ</th><th class="num">Act top-10</th></tr></thead><tbody>` +
    sens.map(r => { const oc=r.agreement.occupation||{}, ac=r.agreement.activity||{};
      return `<tr class="link" ${go('results.html', r.run_id)}><td class="rid">${r.run_id}</td><td><span class="key ${r.metric}"></span>${esc(r.label)}</td><td class="num">${r.kept[0]} / ${r.kept[1]}</td><td class="num">${f(oc.spearman)}</td><td class="num">${oc.top10??'—'}/10</td><td class="num">${f(ac.spearman)}</td><td class="num">${ac.top10??'—'}/10</td></tr>`; }).join('') + '</tbody>';
}

function renderLoo(){
  const loo = S.runs.filter(r => r.kind==='loo' && r.loo && r.loo.tau_ci);
  const rowH=34, padL=300, padR=20, padT=24, W=880, H=padT+loo.length*rowH+30;
  const sx = v => padL + (v+1)/2*(W-padL-padR);
  let g='';
  [-1,-0.5,0,0.5,1].forEach(t => { g += `<line x1="${sx(t)}" x2="${sx(t)}" y1="${padT-6}" y2="${H-26}" stroke="${t===0?'var(--muted)':'var(--grid)'}" ${t===0?'stroke-dasharray="3 3"':''}/><text class="tick" x="${sx(t)}" y="${H-10}" text-anchor="middle">${t}</text>`; });
  g += `<text class="tick" x="${sx(-1)}" y="12">← reversed</text><text class="tick" x="${sx(1)}" y="12" text-anchor="end">recovered →</text>`;
  loo.forEach((r,i) => {
    const l=r.loo, y=padT+i*rowH+rowH/2, col=`var(--${r.metric})`;
    g += `<text x="0" y="${y+4}">${esc(r.label.replace(/^LOO · /,''))}</text>`;
    g += `<line x1="${sx(l.tau_ci[0])}" x2="${sx(l.tau_ci[1])}" y1="${y}" y2="${y}" stroke="${col}" stroke-width="2.5" stroke-linecap="round"/>`;
    g += `<circle cx="${sx(l.tau)}" cy="${y}" r="7" fill="${col}" stroke="var(--surface)" stroke-width="2" data-tip="${esc(r.label)}<br>τ-b = ${f(l.tau,3)} · 95% CI [${f(l.tau_ci[0])}, ${f(l.tau_ci[1])}]<br>Spearman ρ = ${f(l.spearman,3)} · ${l.n} folds · MAE ${f(l.mae,3)}"/>`;
  });
  $('looChart').innerHTML = `<svg viewBox="0 0 ${W} ${H}" style="min-width:640px" role="img" aria-label="Kendall tau with 95% confidence intervals for each LOO run">${g}</svg>`;
  bindTips($('looChart'));
  $('looTable').innerHTML = `<thead><tr><th>Run</th><th>Setting</th><th class="num">Folds</th><th class="num">τ-b</th><th class="num">95% CI</th><th class="num">ρ</th><th class="num">MAE</th><th>Verdict</th><th>Largest rank miss</th></tr></thead><tbody>` +
    loo.map(r => { const l=r.loo, [c,v]=verdict(l), w=(l.worst||[])[0];
      return `<tr class="link" ${go('loo.html', r.run_id)}><td class="rid">${r.run_id}</td><td><span class="key ${r.metric}"></span>${esc(r.label.replace(/^LOO · /,''))}</td><td class="num">${l.n}</td><td class="num">${f(l.tau)}</td><td class="num">[${f(l.tau_ci[0])}, ${f(l.tau_ci[1])}]</td><td class="num">${f(l.spearman)}</td><td class="num">${f(l.mae,3)}</td><td><span class="chip ${c}">${v}</span></td><td style="font-size:12px">${w ? `${esc(w.label)} <span class="meta">(rank ${f(w.actual_rank,0)} → ${f(w.loo_rank,0)})</span>` : '—'}</td></tr>`; }).join('') + '</tbody>';
}

fetch('data/summary.json').then(r => r.json()).then(d => {
  S = d; R = Object.fromEntries(S.runs.map(r => [r.key, r]));
  const o = S.observations || {};
  const n = g => S.runs.filter(r => r.group===g).length;
  $('facts').innerHTML = [
    `<span><b>${o.speed}</b> speed + <b>${o.quality}</b> quality effect rows</span>`,
    `<span>→ <b>${o.occ_codes}</b> occupation · <b>${o.act_codes}</b> activity codes</span>`,
    `<span><b>${S.runs.length}</b> runs: ${n('headline')} headline · ${n('sensitivity')} sensitivity · ${n('validation')} LOO</span>`,
  ].join('');
  const st = $('stats'); if (st) st.textContent = `${S.runs.length} curated runs`;
  findings(); renderLevels('speed'); renderSens(); renderLoo();
}).catch(() => { document.querySelector('main.sum').innerHTML = '<p>No run summary published. Build the site with a run manifest (scripts/generate_site_runs.py).</p>'; });
</script>
</body></html>
"""


# ---------- parameters.html (heatmap + bar chart + explanations, no submit) ----------

_PARAMS_STYLE = _STYLE_COMMON + r"""
  :root { --page-w: 1080px; }
  .paramgrid { display:grid; grid-template-columns: 210px 1fr; gap:10px 22px; align-items:baseline; margin-top:6px; }
  .paramgrid .k { font-weight:500; color:var(--ink); font-family:var(--mono); font-size:13px; }
  .paramgrid .v { color:var(--ink-2); font-size:14px; line-height:1.6; }
  .paramgrid .v b { color:var(--ink); font-family:var(--mono); font-weight:500; font-size:13px; }
  .threshold-row { display:flex; gap:14px; margin:18px 0 8px; align-items:baseline; flex-wrap:wrap; }
  .threshold-row input[type=number] { width:90px; }
  .summary-line { margin-top:12px; padding:10px 14px; background:var(--accent-soft); border-radius:4px; font-size:13.5px; color:var(--ink); }
  #heatmap, #barchart { overflow:auto; border:1px solid var(--rule-soft); border-radius:4px; padding:8px; background:var(--surface); }
  @media (max-width: 720px) { .paramgrid { grid-template-columns: 1fr; gap:2px 0; } .paramgrid .v { margin-bottom:10px; } }
"""


_PARAMS_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">__THEME_HEAD__<title>Method &mdash; AI Impact Meta-Review</title>
<style>__STYLE__</style></head><body>
__HEADER__
<main class="page wrap">
  <section class="page-head">
    <div>
      <h1>Method: from studies to O*NET-wide estimates</h1>
      <p class="lede">The imputation model spreads observed effects from the occupations and activities that were studied to
        those that weren&rsquo;t, following how strongly each occupation relies on each work activity.
        This page shows the inputs to that model at the canonical setting (&beta; = __BETA__) and explains every parameter.</p>
    </div>
  </section>
  <div class="callout" style="margin-bottom:18px;">
    This is a read-only view. To launch runs with other settings, run the review app locally
    (<code>python scripts/review_app.py</code>) and open <code>/run</code>.
  </div>

  <div class="card">
    <h2>Metric</h2>
    <div class="seg" id="metricSeg">
      <button data-m="speed" class="active">speed</button>
      <button data-m="quality">quality</button>
    </div>
    <div class="help" style="margin-top:8px;">A run targets one metric at a time. Switching changes which activities and
      SOC groups are flagged as observed in the charts below.</div>
  </div>

  <div class="card">
    <h2>Stage B: how occupations load on work activities</h2>
    <div class="help" style="margin-bottom:10px;">
      Mean Stage-B weight per SOC major group &times; O*NET work activity. Weights come from applying a
      row-wise softmax with temperature &beta; to each occupation's (Importance &times; Level / 5) profile, then
      averaging within each SOC-major group. Red markers and bold labels flag activities and groups with an observed
      effect for this metric; shaded rows are excluded. Click a row label or checkbox to toggle its inclusion.
    </div>
    <div id="heatmap"></div>

    <div class="threshold-row">
      <label for="weightThreshold" style="font-weight:600;">Activity weight threshold</label>
      <input type="number" id="weightThreshold" value="10" step="0.5" min="0"/>
      <div class="help">Activities whose summed weight (over kept occupations) is below this are dropped.
        Observed activities are always kept.</div>
    </div>
    <div id="barchart"></div>
    <div id="pruneSummary" class="summary-line"></div>
  </div>

  <div class="card">
    <h2>What each parameter does</h2>
    <div class="paramgrid">
      <div class="k">metric</div>
      <div class="v">Which effect-size column feeds the observations. <b>speed</b> = mean log-ratio;
        <b>quality</b> = mean Hedges' g. Two independent runs.</div>

      <div class="k">&beta; (specificity)</div>
      <div class="v">Softmax temperature for Stage B. Higher &beta; means each occupation concentrates on
        fewer activities. Default <b>2.5</b> yields roughly 7 effective activities/occupation.</div>

      <div class="k">aggregation level</div>
      <div class="v"><b>occupation</b> (default, ~894 O*NET occupations), <b>soc_minor</b> (~92 3-digit groups
        like <code>13-2000 Financial Specialists</code>), or <b>soc_major</b> (22 2-digit majors). In aggregated
        modes, the composite weight matrix is the per-group row-mean; observations are IV-weighted (or simple-mean
        fallback) within the group; the AIOE baseline is averaged within the group.</div>

      <div class="k">excluded_soc_majors</div>
      <div class="v">SOC 2-digit prefixes to drop entirely. Default excludes the physical majors
        (37, 45, 47, 49, 51, 53) plus 11 &amp; anything the user has toggled off.</div>

      <div class="k">activity_weight_threshold</div>
      <div class="v">See the bar chart above &mdash; activities whose <i>summed</i> weight across the kept
        occupations falls below this are pruned. Observed activities are always retained regardless.</div>

      <div class="k">&Omega;<sub>ref</sub></div>
      <div class="v">Stage D observation-trust precision. Higher pulls estimates harder toward the observed
        values. Default <b>100</b>.</div>

      <div class="k">use_baseline / &Omega;<sub>base</sub></div>
      <div class="v"><b>speed only.</b> When on, mixes in the Felten et al. AIOE score
        (moment-matched to the observed metric's mean and SD) as a per-occupation soft anchor. &Omega;<sub>base</sub>
        controls the strength: <b>0.1</b> ≈ tiebreaker, <b>0.5</b> ≈ balanced, <b>1.0</b> ≈ AIOE-led,
        <b>5.0</b> ≈ AIOE-replaces-graph.</div>

      <div class="k">&sigma;<sub>ref</sub></div>
      <div class="v">Stage D SE floor / scale (default <b>0.1</b>). Effectively converts a paper's SE into
        an inverse-variance weight relative to &Omega;<sub>ref</sub>.</div>

      <div class="k">&epsilon; (regularizer)</div>
      <div class="v">Ridge added to the diagonal of the linear system to keep it invertible (default
        <b>10<sup>-6</sup></b>).</div>
    </div>
  </div>
</main>

<script>
function escapeHtml(s){return (s||'').toString().replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}

let METRIC='speed';
let HEAT = null;
let EXCLUDED = new Set();

document.querySelectorAll('#metricSeg button').forEach(b=>{
  b.onclick=()=>{
    document.querySelectorAll('#metricSeg button').forEach(x=>x.classList.remove('active'));
    b.classList.add('active'); METRIC=b.dataset.m;
    loadHeatmap();
  };
});
document.getElementById('weightThreshold').addEventListener('input', renderBarChart);

async function loadHeatmap() {
  const r = await fetch('data/heatmap_' + METRIC + '.json');
  if (!r.ok) { document.getElementById('heatmap').innerHTML = '<div class="err">Missing heatmap data for '+METRIC+'.</div>'; return; }
  HEAT = await r.json();
  if (EXCLUDED.size === 0) EXCLUDED = new Set(HEAT.default_excluded_socs);
  renderCharts();
}

function vColor(v, vmax) {
  const stops = [[68,1,84],[59,82,139],[33,144,141],[93,200,99],[253,231,37]];
  const t = Math.max(0, Math.min(1, v / vmax));
  const idx = t * (stops.length-1);
  const i = Math.floor(idx), f = idx - i;
  const a = stops[i], b = stops[Math.min(i+1, stops.length-1)];
  return `rgb(${Math.round(a[0]+(b[0]-a[0])*f)},${Math.round(a[1]+(b[1]-a[1])*f)},${Math.round(a[2]+(b[2]-a[2])*f)})`;
}

function renderHeatmap() {
  if (!HEAT) return;
  const rowOrder = HEAT.row_order, colOrder = HEAT.col_order;
  const rows = rowOrder.map(i => HEAT.soc_groups[i]);
  const cols = colOrder.map(j => HEAT.activities[j]);
  const W = HEAT.weights;
  const cell = 18;
  const labelW = 240, topPad = 12, bottomLabelH = 200, padR = 30;
  const labelH = topPad;
  const width  = labelW + cols.length * cell + padR;
  const height = labelH + rows.length * cell + bottomLabelH + 16;
  const vals = []; W.forEach(row => row.forEach(v => { if (v > 0) vals.push(v); }));
  vals.sort((a,b)=>a-b); const vmax = vals[Math.floor(vals.length*0.99)] || 1;

  let cellsSvg = '';
  for (let ri = 0; ri < rows.length; ri++) {
    const realRow = rowOrder[ri];
    for (let ci = 0; ci < cols.length; ci++) {
      const v = W[realRow][colOrder[ci]];
      const x = labelW + ci * cell, y = labelH + ri * cell;
      cellsSvg += `<rect x="${x}" y="${y}" width="${cell}" height="${cell}" fill="${vColor(v, vmax)}"/>`;
      if (v >= 0.06) {
        const col = (v < vmax * 0.4) ? '#fff' : '#222';
        cellsSvg += `<text x="${x+cell/2}" y="${y+cell/2+3}" text-anchor="middle" font-size="8" fill="${col}">${v.toFixed(2)}</text>`;
      }
    }
  }
  let overlaySvg = '';
  rows.forEach((r, ri) => {
    if (EXCLUDED.has(r.code)) {
      overlaySvg += `<rect x="${labelW}" y="${labelH+ri*cell}" width="${cols.length*cell}" height="${cell}" fill="rgba(244,241,234,0.85)"/>`;
    }
  });
  let lineSvg = '';
  cols.forEach((c, ci) => {
    if (c.observed) {
      const x = labelW + ci*cell + cell/2;
      lineSvg += `<rect x="${x-cell/2+2}" y="${labelH-7}" width="${cell-4}" height="5" rx="1" fill="#d1492e"/>`;
    }
  });
  rows.forEach((r, ri) => {
    if (r.observed && !EXCLUDED.has(r.code)) {
      const y = labelH + ri*cell + cell/2;
      lineSvg += `<rect x="${labelW-6}" y="${y-cell/2+2}" width="5" height="${cell-4}" rx="1" fill="#d1492e"/>`;
    }
  });
  const rowLabels = rows.map((r, ri) => {
    const y = labelH + ri*cell;
    const fade = EXCLUDED.has(r.code) ? 'color:#a9a59c;' : '';
    const bold = r.observed ? 'font-weight:600;' : '';
    return `<div style="position:absolute; left:0; top:${y}px; height:${cell}px; width:${labelW-6}px;
                       display:flex; align-items:center; gap:5px; font-size:11px; ${fade}${bold} cursor:pointer;"
                 onclick="toggleSoc('${r.code}')">
              <input type="checkbox" ${EXCLUDED.has(r.code)?'':'checked'} onclick="event.stopPropagation(); toggleSoc('${r.code}');" style="margin:0 4px;">
              <span style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${escapeHtml(r.name)} (n=${r.n})">
                ${escapeHtml(r.name)} <span style="color:var(--muted); font-weight:400;">(n=${r.n})</span>
              </span>
            </div>`;
  }).join('');
  const colLabelY = labelH + rows.length * cell + 6;
  const colLabels = cols.map((c, ci) => {
    const x = labelW + ci*cell + cell/2;
    const fill = c.observed ? '#1d1f23' : '#5b5f67';
    const fw = c.observed ? '600' : '400';
    return `<text x="${x}" y="${colLabelY}" font-size="10" fill="${fill}" font-weight="${fw}"
                  transform="rotate(60 ${x} ${colLabelY})" text-anchor="start">${escapeHtml(c.name)}</text>`;
  }).join('');

  document.getElementById('heatmap').innerHTML = `
    <div style="position:relative; width:${width}px;">
      ${rowLabels}
      <svg width="${width}" height="${height}" style="display:block;">
        ${cellsSvg}${overlaySvg}${lineSvg}${colLabels}
      </svg>
      <div class="chart-legend">
        <span><span class="sw" style="background:#d1492e; width:10px; height:5px;"></span>Marker above a column / left of a row, with bold label: has an observed effect for this metric</span>
        <span><span class="sw" style="background:#f4f1ea; border:1px solid #e2ddd2;"></span>Excluded</span>
        <span>Weight <span class="sw" style="width:70px; margin:0 4px; background:linear-gradient(90deg, rgb(68,1,84), rgb(59,82,139), rgb(33,144,141), rgb(93,200,99), rgb(253,231,37));"></span>low &rarr; high</span>
      </div>
    </div>`;
}

function renderBarChart() {
  if (!HEAT) return;
  const threshold = parseFloat(document.getElementById('weightThreshold').value) || 10;
  const acts0 = HEAT.activities.map((a, j) => {
    let s = 0;
    HEAT.soc_groups.forEach((g, i) => {
      if (!EXCLUDED.has(g.code)) s += HEAT.weights[i][j] * g.n;
    });
    return {...a, j, total: s};
  });
  const acts = acts0.slice().sort((x, y) => y.total - x.total);
  const sums = acts.map(a => a.total);
  const vmax = Math.max(...sums, 1);
  const barH = 14;
  const labelW = 240, plotW = 480, padT = 12;
  const height = padT + acts.length * barH + 16;

  let bars = '';
  acts.forEach((a, i) => {
    const w = (sums[i] / vmax) * plotW;
    const y = padT + i * barH;
    const below = sums[i] < threshold;
    const fill = a.observed ? '#23406a' : '#a9b4c2';
    const opacity = below ? 0.35 : 1.0;
    bars += `<rect x="${labelW}" y="${y+1}" width="${w}" height="${barH-3}" fill="${fill}" opacity="${opacity}"/>`;
    bars += `<text x="${labelW + w + 4}" y="${y+barH-3}" font-size="9" fill="${below?'#a9a59c':'#474b53'}">${sums[i].toFixed(1)}</text>`;
    const tcol = below ? '#a9a59c' : '#1d1f23';
    bars += `<text x="${labelW-4}" y="${y+barH-3}" font-size="10" fill="${tcol}" text-anchor="end" font-weight="${a.observed?'600':'400'}">${escapeHtml(a.name)}</text>`;
  });
  const tx = labelW + (threshold / vmax) * plotW;
  bars += `<line x1="${tx}" y1="${padT}" x2="${tx}" y2="${padT + acts.length*barH}" stroke="#1d1f23" stroke-width="1.5" stroke-dasharray="4 3"/>`;
  bars += `<text x="${tx + 3}" y="${padT - 2}" font-size="10" fill="#1d1f23">threshold = ${threshold.toFixed(1)}</text>`;

  document.getElementById('barchart').innerHTML = `
    <svg width="${labelW + plotW + 70}" height="${height}" style="display:block;">${bars}</svg>
    <div class="chart-legend">
      <span><span class="sw" style="background:#23406a;"></span>Observed activity (always kept)</span>
      <span><span class="sw" style="background:#a9b4c2;"></span>Other activity</span>
      <span>Faded bars fall left of the dashed threshold and are dropped.</span>
    </div>`;

  let occTotal = 0, occIncluded = 0;
  HEAT.soc_groups.forEach(g => {
    occTotal += g.n;
    if (!EXCLUDED.has(g.code)) occIncluded += g.n;
  });
  const actTotal = HEAT.activities.length;
  const actIncluded = acts.filter(a => a.observed || a.total >= threshold).length;
  const socIncluded = HEAT.soc_groups.length - EXCLUDED.size;
  document.getElementById('pruneSummary').innerHTML =
    `<b>Included:</b> ${occIncluded.toLocaleString()} of ${occTotal.toLocaleString()} occupations ` +
    `(${socIncluded} of ${HEAT.soc_groups.length} SOC major groups) · ` +
    `${actIncluded} of ${actTotal} activities`;
}

function renderCharts() { renderHeatmap(); renderBarChart(); }

function toggleSoc(code) {
  if (EXCLUDED.has(code)) EXCLUDED.delete(code); else EXCLUDED.add(code);
  renderCharts();
}

loadHeatmap();
</script>
</body></html>
"""


# ---------- transitions.html (occupational transitions heatmap) ----------

_TRANSITIONS_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">__THEME_HEAD__<title>Occupational transitions &mdash; AI Impact Meta-Review</title>
<style>
  #main { display:grid; grid-template-columns: minmax(0,1fr) 460px; gap:18px; padding:0 28px 32px; align-items:start; }
  @media (max-width: 1100px) { #main { grid-template-columns: minmax(0,1fr); } }
  .panel { background:var(--surface); border:1px solid var(--rule); border-radius:var(--radius); padding:16px 18px; }
  .panel h2 { margin:0 0 10px; font-size:17px; }
  .heatmap-wrap { position:relative; overflow:auto; max-height:78vh; }
  .hm-grid { display:grid; grid-template-columns: var(--label-w) auto; grid-template-rows: var(--label-h) auto; gap:0; }
  .hm-corner { background:#fff; }
  canvas { display:block; cursor:crosshair; }
  .axis { font-size:10px; color:#333; user-select:none; position:relative; }
  .axis.left .lbl {
    height: var(--cell); width: var(--label-w); box-sizing: border-box;
    padding: 0 6px; text-align: right; white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis; line-height: var(--cell); border-bottom: 1px solid transparent; cursor: pointer;
  }
  .axis.top .lbl {
    width: var(--cell); height: var(--label-h); box-sizing: border-box; overflow: hidden;
    position: relative; border-right: 1px solid transparent; cursor: pointer;
  }
  .axis.top .lbl span {
    position: absolute; left: 0; right: 0; bottom: 4px;
    writing-mode: vertical-rl; transform: rotate(180deg);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    max-height: calc(var(--label-h) - 8px);
    line-height: var(--cell); font-size: 10px;
  }
  .axis .lbl.major-end { border-bottom-color: rgba(60,60,60,0.5); }
  .axis.top .lbl.major-end { border-bottom-color: transparent; border-right-color: rgba(60,60,60,0.5); }
  .axis .lbl:hover { background:var(--hover); }
  .axis .lbl.selected { background:var(--accent-soft); font-weight:600; }
  .legend { display:flex; align-items:center; gap:8px; margin-top:12px; font-size:12px; color:var(--muted); flex-wrap:wrap; }
  .legend-bar { height:10px; width:160px; background: linear-gradient(to right, #fff, #fde0a0, #ef7838, #6e1a08); border:1px solid var(--rule); border-radius:2px; }
  .tooltip { position:fixed; background:var(--ink); color:#fff; padding:7px 11px; border-radius:4px; font-size:12px; pointer-events:none; z-index:1000; display:none; max-width:320px; line-height:1.45; box-shadow:0 6px 18px rgba(0,0,0,0.18); }
  .drill h3 { margin:6px 0 4px; font-size:18px; line-height:1.3; }
  .drill h2 { font-size:15px !important; margin-top:18px !important; }
  .drill .sub { color:var(--muted); font-size:13px; margin-bottom:8px; }
  .drill .est-line { display:flex; gap:8px; align-items:baseline; font-size:13px; margin-bottom:10px; color:var(--ink-2); }
  .drill .est-line .v { font-family:var(--mono); font-weight:500; color:var(--ink); font-size:15px; }
  .drill .crumb { font-size:13px; color:var(--muted); margin-bottom:8px; }
  .drill .crumb a { color:var(--link); cursor:pointer; }
  table.tbl { width:100%; border-collapse:collapse; font-size:13px; }
  table.tbl th { padding:5px 6px; color:var(--muted); font-weight:500; font-size:12px; border-bottom:1px solid var(--rule); background:transparent; }
  table.tbl td { padding:5px 6px; border-bottom:1px solid var(--rule-soft); vertical-align:top; font-size:13px; }
  table.tbl td.num { text-align:right; font-family:var(--mono); font-size:12px; }
  table.tbl tr.row-click { cursor:pointer; }
  table.tbl tr.row-click:hover { background:var(--hover); }
  .est-pos { color:var(--pos); } .est-neg { color:var(--neg); } .est-na { color:var(--muted); }
  .empty { color:var(--muted); font-size:14px; padding:24px 8px; text-align:center; font-style:italic; font-family:var(--serif); }
  .search { display:flex; gap:8px; margin-bottom:10px; position: relative; }
  .search input { flex:1; }
  .typeahead { position:absolute; background:var(--surface); border:1px solid var(--rule); border-radius:5px; max-height:260px; overflow:auto; z-index:50; left:0; right:90px; top:40px; box-shadow:0 10px 30px rgba(29,31,35,0.14); display:none; }
  .typeahead .item { padding:6px 10px; font-size:13px; cursor:pointer; border-bottom:1px solid var(--rule-soft); }
  .typeahead .item:hover { background:var(--accent-soft); }
  .typeahead .item .code { color:var(--muted); font-size:11px; font-family:var(--mono); }
</style></head><body>
__HEADER__
<div class="wrap wide" style="padding:0 28px;">
  <section class="page-head">
    <div>
      <h1>Occupational transitions</h1>
      <p class="lede">How often workers move from one occupation to another. Rows are source SOC minor groups and columns are
        destinations; each row sums to one.</p>
    </div>
    <span class="page-meta" id="stats">Loading&hellip;</span>
  </section>
</div>
<div id="main">
  <div class="panel">
    <h2>Transition shares by SOC minor group</h2>
    <div class="search results">
      <input id="search" placeholder="Search occupation (title or SOC code)&hellip;" autocomplete="off"/>
      <button id="clearBtn" class="btn">Clear</button>
      <div class="typeahead" id="typeahead"></div>
    </div>
    <div class="heatmap-wrap" id="hmWrap">
      <div class="hm-grid" id="hmGrid">
        <div class="hm-corner"></div>
        <div class="axis top" id="axisTop"></div>
        <div class="axis left" id="axisLeft"></div>
        <canvas id="hm"></canvas>
      </div>
    </div>
    <div class="legend">
      <span>0</span><div class="legend-bar"></div><span>max share (square-root scale)</span>
      <span style="margin-left:14px;">Click a row label or cell to drill into a SOC minor group; click an occupation within it for outgoing &amp; incoming transitions.</span>
    </div>
  </div>
  <div class="panel drill" id="drill">
    <div class="empty">Click a minor group on the heatmap, or search an occupation above.</div>
  </div>
</div>
<div class="tooltip" id="tip"></div>
<script>
const CELL = 14;
const LABEL_W = 240;
const LABEL_H = 240;
let DATA = null;
let MINOR_IDX = {};
let SEL_MINOR = -1;
let SEL_OCC = -1;

async function load() {
  const r = await fetch('data/transitions.json');
  if (!r.ok) { document.body.innerHTML = '<p style="padding:30px;">Missing data/transitions.json &mdash; the site was built without the transitions xlsx. Rebuild locally with that file present.</p>'; return; }
  DATA = await r.json();
  DATA.minors.forEach((m,i)=>{ MINOR_IDX[m.code]=i; });
  document.getElementById('stats').textContent =
    `${DATA.minors.length} SOC minor groups · ${DATA.occs.length} non-physical occupations · augmentation from run ${DATA.run_id||'(none)'}`;
  drawHeatmap();
  renderAxes();
}

function colorRamp(t) {
  t = Math.max(0, Math.min(1, t));
  if (t < 0.001) return [255,255,255];
  const stops = [
    [1.000, 1.000, 1.000],
    [0.992, 0.878, 0.627],
    [0.937, 0.470, 0.220],
    [0.431, 0.102, 0.031],
  ];
  const s = t * (stops.length - 1);
  const i = Math.floor(s);
  const f = s - i;
  const a = stops[i], b = stops[Math.min(i+1, stops.length-1)];
  return [
    Math.round((a[0] + (b[0]-a[0])*f) * 255),
    Math.round((a[1] + (b[1]-a[1])*f) * 255),
    Math.round((a[2] + (b[2]-a[2])*f) * 255),
  ];
}

function drawHeatmap(highlight=-1) {
  const M = DATA.minor_matrix;
  const n = DATA.minors.length;
  const cv = document.getElementById('hm');
  const W = n * CELL, H = n * CELL;
  cv.width = W; cv.height = H;
  cv.style.width = W + 'px'; cv.style.height = H + 'px';
  const ctx = cv.getContext('2d');
  ctx.fillStyle = '#fafafa';
  ctx.fillRect(0,0,W,H);
  let maxV = 0;
  for (const row of M) for (const v of row) if (v>maxV) maxV=v;
  for (let i=0;i<n;i++) {
    for (let j=0;j<n;j++) {
      const v = M[i][j];
      if (v <= 0) continue;
      const t = Math.sqrt(v / maxV);
      const [r,g,b] = colorRamp(t);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(j*CELL, i*CELL, CELL, CELL);
    }
  }
  ctx.strokeStyle = 'rgba(60,60,60,0.45)';
  ctx.lineWidth = 1;
  for (let i=1;i<n;i++) {
    if (DATA.minors[i].major !== DATA.minors[i-1].major) {
      ctx.beginPath();
      ctx.moveTo(0, i*CELL+0.5); ctx.lineTo(W, i*CELL+0.5);
      ctx.moveTo(i*CELL+0.5, 0); ctx.lineTo(i*CELL+0.5, H);
      ctx.stroke();
    }
  }
  if (highlight >= 0) {
    ctx.strokeStyle = '#06c';
    ctx.lineWidth = 1.8;
    ctx.strokeRect(0.5, highlight*CELL+0.5, W-1, CELL-1);
  }
}

function renderAxes() {
  const n = DATA.minors.length;
  const left = document.getElementById('axisLeft');
  const top = document.getElementById('axisTop');
  const root = document.getElementById('hmGrid');
  root.style.setProperty('--cell', CELL + 'px');
  root.style.setProperty('--label-w', LABEL_W + 'px');
  root.style.setProperty('--label-h', LABEL_H + 'px');
  left.style.display = 'flex';
  left.style.flexDirection = 'column';
  left.style.width = LABEL_W + 'px';
  left.style.height = (n * CELL) + 'px';
  top.style.display = 'flex';
  top.style.flexDirection = 'row';
  top.style.height = LABEL_H + 'px';
  top.style.width = (n * CELL) + 'px';
  top.style.alignItems = 'flex-end';
  left.innerHTML = '';
  top.innerHTML = '';
  for (let i=0;i<n;i++) {
    const m = DATA.minors[i];
    const isEnd = (i+1<n && DATA.minors[i+1].major !== m.major);
    const label = `${m.code.slice(0,4)} · ${m.title}`;
    const tip = `${m.code} — ${m.title} (${m.n_occs} occs · ${socMajorLabel(m.major)})`;
    const l = document.createElement('div');
    l.className = 'lbl' + (isEnd ? ' major-end' : '');
    l.textContent = label;
    l.title = tip;
    l.onclick = () => showMinor(i);
    l.dataset.idx = i;
    left.appendChild(l);
    const t = document.createElement('div');
    t.className = 'lbl' + (isEnd ? ' major-end' : '');
    const inner = document.createElement('span');
    inner.textContent = label;
    t.appendChild(inner);
    t.title = tip;
    t.onclick = () => showMinor(i);
    t.dataset.idx = i;
    top.appendChild(t);
  }
}

function estFmt(soc6) {
  const e = DATA.estimates[soc6];
  if (e === undefined) return '<span class="est-na">—</span>';
  const cls = e >= 0 ? 'est-pos' : 'est-neg';
  return `<span class="${cls}">${e.toFixed(3)}</span>`;
}

function socMajorLabel(major) { return DATA.soc_names[major] || major; }

function minorLabel(code) {
  const m = DATA.minors[MINOR_IDX[code]];
  return m ? `${code.slice(0,4)} · ${m.title}` : code;
}

function showMinor(mIdx) {
  SEL_MINOR = mIdx; SEL_OCC = -1;
  const m = DATA.minors[mIdx];
  const occIdxs = DATA.minor_to_occs[m.code] || [];
  const sorted = occIdxs.slice().sort((a,b) => (DATA.totals[b]||0) - (DATA.totals[a]||0));
  const occRows = sorted.map(i => {
    const o = DATA.occs[i];
    return `<tr class="row-click" data-occ="${i}">
      <td>${escapeHtml(o.title)}<div style="color:#888;font-size:11px;">${o.code}</div></td>
      <td class="num">${(DATA.totals[i]||0).toLocaleString(undefined,{maximumFractionDigits:0})}</td>
      <td class="num">${estFmt(o.code)}</td>
    </tr>`;
  }).join('');
  const row = DATA.minor_matrix[mIdx];
  const outRanked = row.map((v,j)=>[j,v]).filter(([,v])=>v>0).sort((a,b)=>b[1]-a[1]).slice(0,10);
  const incRanked = DATA.minor_matrix.map((r,i)=>[i, r[mIdx]]).filter(([,v])=>v>0).sort((a,b)=>b[1]-a[1]).slice(0,10);
  const drill = document.getElementById('drill');
  drill.innerHTML = `
    <h3>${escapeHtml(m.title)}</h3>
    <div class="sub">SOC minor group ${m.code.slice(0,4)} · ${socMajorLabel(m.major)} · ${m.n_occs} occupations</div>
    <h2 style="margin-top:6px;">Occupations in this group</h2>
    <table class="tbl"><thead><tr><th>Occupation</th><th class="num">obs. transitions</th><th class="num">augment.</th></tr></thead>
    <tbody>${occRows || '<tr><td colspan="3" class="empty">No occupations.</td></tr>'}</tbody></table>
    <h2 style="margin-top:14px;">Top outgoing minor groups</h2>
    ${minorTransTable(outRanked, 'target')}
    <h2 style="margin-top:14px;">Top incoming minor groups</h2>
    ${minorTransTable(incRanked, 'source')}
  `;
  drill.querySelectorAll('tr[data-occ]').forEach(tr => { tr.onclick = () => showOcc(parseInt(tr.dataset.occ)); });
  drill.querySelectorAll('tr[data-minor]').forEach(tr => { tr.onclick = () => showMinor(parseInt(tr.dataset.minor)); });
  drawHeatmap(mIdx);
}

function minorTransTable(arr, dirLabel) {
  if (!arr.length) return '<div class="empty">No transitions.</div>';
  const rows = arr.map(([idx, share]) => {
    const m = DATA.minors[idx];
    return `<tr class="row-click" data-minor="${idx}">
      <td>${escapeHtml(m.title)}<div style="color:#888;font-size:11px;">${m.code.slice(0,4)} · ${escapeHtml(socMajorLabel(m.major))}</div></td>
      <td class="num">${(share*100).toFixed(2)}%</td>
    </tr>`;
  }).join('');
  return `<table class="tbl"><thead><tr>
    <th>${dirLabel === 'target' ? 'Next minor group' : 'Previous minor group'}</th>
    <th class="num">share</th>
  </tr></thead><tbody>${rows}</tbody></table>`;
}

function showOcc(i) {
  SEL_OCC = i;
  const occ = DATA.occs[i];
  const total = DATA.totals[i] || 0;
  const out = DATA.rows[i].slice(0, 15);
  const inc = DATA.incoming[i].slice(0, 15);
  const drill = document.getElementById('drill');
  drill.innerHTML = `
    <div class="crumb">
      <a id="backToMinor">← back to ${escapeHtml(minorLabel(occ.minor))}</a>
    </div>
    <h3>${escapeHtml(occ.title)}</h3>
    <div class="sub">SOC ${occ.code} · ${socMajorLabel(occ.major)} · ${total.toLocaleString(undefined,{maximumFractionDigits:0})} observed transitions</div>
    <div class="est-line">Augmentation estimate: <span class="v">${estFmt(occ.code)}</span></div>
    <h2 style="margin-top:6px;">Top outgoing (next occupations)</h2>
    ${occTable(out, 'target')}
    <h2 style="margin-top:14px;">Top incoming (previous occupations)</h2>
    ${occTable(inc, 'source')}
  `;
  document.getElementById('backToMinor').onclick = () => showMinor(MINOR_IDX[occ.minor]);
  drill.querySelectorAll('tr[data-occ]').forEach(tr => { tr.onclick = () => showOcc(parseInt(tr.dataset.occ)); });
  drawHeatmap(MINOR_IDX[occ.minor]);
}

function occTable(arr, dirLabel) {
  if (!arr.length) return '<div class="empty">No transitions.</div>';
  const rows = arr.map(([idx, share]) => {
    const o = DATA.occs[idx];
    return `<tr class="row-click" data-occ="${idx}">
      <td>${escapeHtml(o.title)}<div style="color:#888;font-size:11px;">${o.code} · ${escapeHtml(socMajorLabel(o.major))}</div></td>
      <td class="num">${(share*100).toFixed(2)}%</td>
      <td class="num">${estFmt(o.code)}</td>
    </tr>`;
  }).join('');
  return `<table class="tbl"><thead><tr>
    <th>${dirLabel === 'target' ? 'Next occupation' : 'Previous occupation'}</th>
    <th class="num">share</th><th class="num">augment.</th>
  </tr></thead><tbody>${rows}</tbody></table>`;
}

function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

document.addEventListener('DOMContentLoaded', () => {
  const cv = document.getElementById('hm');
  const tip = document.getElementById('tip');
  cv.addEventListener('mousemove', e => {
    if (!DATA) return;
    const rect = cv.getBoundingClientRect();
    const x = e.clientX - rect.left, y = e.clientY - rect.top;
    const i = Math.floor(y / CELL), j = Math.floor(x / CELL);
    const n = DATA.minors.length;
    if (i<0||j<0||i>=n||j>=n) { tip.style.display='none'; return; }
    const share = DATA.minor_matrix[i][j];
    const so = DATA.minors[i], to = DATA.minors[j];
    tip.innerHTML =
      `<b>${escapeHtml(so.title)}</b> <span style="opacity:.7">(${so.code.slice(0,4)})</span><br>
       → <b>${escapeHtml(to.title)}</b> <span style="opacity:.7">(${to.code.slice(0,4)})</span><br>
       ${(share*100).toFixed(2)}% of transitions from this minor group`;
    tip.style.left = (e.clientX + 14) + 'px';
    tip.style.top = (e.clientY + 14) + 'px';
    tip.style.display = 'block';
  });
  cv.addEventListener('mouseleave', () => { tip.style.display='none'; });
  cv.addEventListener('click', e => {
    if (!DATA) return;
    const rect = cv.getBoundingClientRect();
    const y = e.clientY - rect.top;
    const i = Math.floor(y / CELL);
    if (i>=0 && i<DATA.minors.length) showMinor(i);
  });

  const search = document.getElementById('search');
  const ta = document.getElementById('typeahead');
  function refreshTypeahead() {
    const q = search.value.trim().toLowerCase();
    if (!q) { ta.style.display='none'; return; }
    const hits = [];
    for (let i=0;i<DATA.occs.length && hits.length<25;i++) {
      const o = DATA.occs[i];
      if (o.code.toLowerCase().includes(q) || o.title.toLowerCase().includes(q)) hits.push(i);
    }
    if (!hits.length) { ta.style.display='none'; return; }
    ta.innerHTML = hits.map(i => {
      const o = DATA.occs[i];
      return `<div class="item" data-occ="${i}">${escapeHtml(o.title)} <span class="code">${o.code}</span></div>`;
    }).join('');
    ta.style.display = 'block';
    ta.querySelectorAll('.item').forEach(el => {
      el.onclick = () => {
        const idx = parseInt(el.dataset.occ);
        search.value = '';
        ta.style.display='none';
        showOcc(idx);
      };
    });
  }
  search.addEventListener('input', refreshTypeahead);
  search.addEventListener('focus', refreshTypeahead);
  document.addEventListener('click', (e) => {
    if (!ta.contains(e.target) && e.target !== search) ta.style.display='none';
  });
  document.getElementById('clearBtn').onclick = () => {
    search.value = '';
    ta.style.display='none';
    document.getElementById('drill').innerHTML =
      '<div class="empty">Click a minor group on the heatmap, or search an occupation above.</div>';
    drawHeatmap();
  };
  load();
});
</script></body></html>
"""


# ---------- render helpers ----------

def _render(template: str, replacements: dict[str, str]) -> str:
    out = template
    for k, v in replacements.items():
        out = out.replace(k, v)
    return site_theme.apply(out, "").replace("</body></html>", _FOOTER + "\n</body></html>")


# ---------- main ----------

def main():
    if DOCS.exists():
        shutil.rmtree(DOCS)
    DOCS.mkdir(parents=True)
    ASSETS.mkdir(parents=True)
    DATA.mkdir(parents=True)
    PAPERS_DATA.mkdir(parents=True)
    RUNS_DATA.mkdir(parents=True)

    (DOCS / ".nojekyll").write_text("")
    (DOCS / "CNAME").write_text(SITE_DOMAIN + "\n")

    # 1. Rows (review-state overlay applied)
    state = _load_state()
    pid_map = _resolve_paper_id_map()
    rows = _apply_state(_build_rows(pid_map), state)
    (DATA / "rows.json").write_text(json.dumps({"rows": rows}))
    print(f"wrote rows.json ({len(rows)} rows after overlay)")

    # 2. O*NET reference
    onet = {
        "activities": _read_json(CONFIG_DIR / "onet_activities.json") or [],
        "occupations": _read_json(CONFIG_DIR / "onet_occupations.json") or [],
    }
    (DATA / "onet.json").write_text(json.dumps(onet))
    print(f"wrote onet.json ({len(onet['activities'])} activities, {len(onet['occupations'])} occupations)")

    # 3. Per-paper bundles
    paper_ids = sorted({r["paper_id"] for r in rows if r["paper_id"]})
    for pid in paper_ids:
        (PAPERS_DATA / f"{pid}.json").write_text(json.dumps(_paper_bundle(pid)))
    print(f"wrote {len(paper_ids)} per-paper bundles into data/papers/")

    # 4. Runs (per-run + index). The curated manifest decides what's published.
    manifest = _load_manifest()
    if manifest:
        entries = {e["run_id"]: e for e in manifest["runs"]}
        run_dirs = [RUNS_DIR / rid for rid in entries]
    else:
        entries = {}
        run_dirs = _all_run_dirs()
    all_run_metas = []
    summary_runs = []
    for d in run_dirs:
        bundle = _load_run_bundle(d)
        e = entries.get(d.name)
        if e:
            bundle.update(label=e["label"], group=e["group"], reference_run_id=e.get("reference_run_id"))
            if e.get("reference_run_id"):
                bundle["agreement"] = _agreement(d.name, e["reference_run_id"])
            summary_runs.append(_summary_record(e, bundle))
        (RUNS_DATA / f"{d.name}.json").write_text(json.dumps(bundle))
        meta_only = {k: v for k, v in bundle.items() if k not in ("occupation_impacts", "activity_impacts", "loo")}
        if bundle.get("type") == "loo":
            meta_only["loo_summary"] = bundle["loo"].get("summary")
        all_run_metas.append(meta_only)
    canonical_run_id = (manifest or {}).get("canonical") or next(
        (m["run_id"] for m in all_run_metas if m.get("type") != "loo"), "")
    (RUNS_DATA / "index.json").write_text(json.dumps({
        "runs": all_run_metas,
        "groups": (manifest or {}).get("groups", []),
        "observations": (manifest or {}).get("observations"),
        "canonical": canonical_run_id,
    }))
    print(f"wrote {len(run_dirs)} run bundles into data/runs/ (canonical: {canonical_run_id or 'none'}"
          f"{', curated' if manifest else ''})")
    if manifest:
        (DATA / "summary.json").write_text(json.dumps({
            "observations": manifest.get("observations"), "runs": summary_runs}))
        print(f"wrote summary.json ({len(summary_runs)} runs)")

    # 5. Heatmap precompute (needs run_analysis + Work Activities.xlsx)
    try:
        import run_analysis as RA
        for metric in ("speed", "quality"):
            try:
                data = RA.heatmap_data(CANONICAL_BETA, metric)
                (DATA / f"heatmap_{metric}.json").write_text(json.dumps(data))
                print(f"wrote heatmap_{metric}.json")
            except Exception as e:  # noqa: BLE001
                print(f"WARN: heatmap_{metric} failed: {e}")
    except Exception as e:  # noqa: BLE001
        print(f"WARN: cannot import run_analysis ({e}); parameters.html heatmap will be empty")

    # 6. Transitions (needs the transitions xlsx)
    try:
        import transitions as TR
        try:
            (DATA / "transitions.json").write_text(json.dumps(TR.get_data()))
            print("wrote transitions.json")
        except Exception as e:  # noqa: BLE001
            print(f"WARN: transitions.get_data failed: {e}")
    except Exception as e:  # noqa: BLE001
        print(f"WARN: cannot import transitions ({e}); transitions.html will show a placeholder")

    # 7. CSV downloads
    if OUT.exists():
        for p in OUT.glob("*.csv"):
            if p.name.endswith(".pre_review.csv"):
                continue
            shutil.copy2(p, ASSETS / p.name)
        print(f"copied {len(list(ASSETS.glob('*.csv')))} CSVs to assets/")

    # 8. HTML pages
    pages = [
        ("index.html",       _INDEX_HTML,       {"__STYLE__": _INDEX_STYLE,  "__HEADER__": _header("home")}),
        ("results.html",     _RESULTS_HTML,     {"__STYLE__": _RESULTS_STYLE, "__HEADER__": _header("runs"),
                                                  "__DEFAULT_RUN_ID__": canonical_run_id}),
        ("summary.html",     _SUMMARY_HTML,     {"__HEADER__": _header("summary")}),
        ("runs.html",        _RUNS_HTML,        {"__STYLE__": _RUNS_STYLE,   "__HEADER__": _header("runs")}),
        ("loo.html",         _LOO_HTML,         {"__HEADER__": _header("runs")}),
        ("parameters.html",  _PARAMS_HTML,      {"__STYLE__": _PARAMS_STYLE, "__HEADER__": _header("parameters"),
                                                  "__BETA__": str(CANONICAL_BETA)}),
        ("transitions.html", _TRANSITIONS_HTML, {"__HEADER__": _header("transitions")}),
    ]
    for name, template, repl in pages:
        (DOCS / name).write_text(_render(template, repl))
    print(f"wrote {len(pages)} HTML pages into {DOCS}")


if __name__ == "__main__":
    main()
