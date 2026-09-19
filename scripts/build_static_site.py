"""Build a static GitHub Pages site from the review-app data.

Emits into ./docs:
  index.html               — review table + paper drawer (read-only port of INDEX_HTML)
  results.html             — analysis-run viewer; accepts ?run=<id>, defaults to canonical
  runs.html                — index of all runs (port of RESULTS_INDEX_HTML)
  parameters.html          — heatmap + activity bar chart + parameter explanations
  transitions.html         — occupational transitions heatmap (port of TRANSITIONS_HTML)
  .nojekyll
  data/rows.json           — same shape as GET /api/rows, review_state overlay applied
  data/onet.json           — same shape as GET /api/onet
  data/papers/*.json       — one per paper (mirror of GET /api/paper/<id>)
  data/runs/index.json     — { runs: [meta, ...] } (mirror of GET /api/runs)
  data/runs/<id>.json      — full bundle per run (mirror of GET /api/results/<id>)
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

DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
DATA = DOCS / "data"
PAPERS_DATA = DATA / "papers"
RUNS_DATA = DATA / "runs"

REPO_URL = "https://github.com/catzwu/ai-impact-meta-review"
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


def _load_run_bundle(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "run.json").read_text())

    def _read_impacts(p: Path) -> list[dict]:
        return [{k: (None if v == "" else v) for k, v in r.items()} for r in _read_csv(p)]

    meta["occupation_impacts"] = _read_impacts(run_dir / "occupation_impacts.csv")
    meta["activity_impacts"] = _read_impacts(run_dir / "activity_impacts.csv")
    return meta


# ---------- HTML template plumbing ----------

def _header(active: str) -> str:
    """Shared masthead. `active` is one of 'home', 'runs', 'parameters', 'transitions'."""
    items = [
        ("index.html", "Studies", "home"),
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
  #table { table-layout: fixed; min-width:1350px; }
  #table th { position:sticky; top:0; z-index:5; }
  td.snippet { white-space:normal; word-break:break-word; color:var(--ink-2); font-size:12.5px; line-height:1.45; }
  td.title { white-space:normal; word-break:break-word; font-family:var(--serif); font-size:14.5px; line-height:1.35; cursor:pointer; color:var(--ink); }
  td.title:hover { color:var(--link); text-decoration:underline; text-underline-offset:2px; }
  td.cite { font-family:var(--mono); font-size:11.5px; color:var(--ink-2); overflow:hidden; text-overflow:ellipsis; }
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
    <col style="width:76px"><col style="width:190px"><col style="width:290px"><col style="width:160px">
    <col style="width:80px"><col style="width:96px"><col style="width:300px"><col style="width:340px">
  </colgroup>
  <thead><tr>
    <th data-sort="kind">Kind</th>
    <th data-sort="citation_key">Citation</th>
    <th data-sort="title">Title</th>
    <th data-sort="file_name">File</th>
    <th data-sort="value">Value</th>
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
    !q || (r.citation_key+r.title+r.file_name+r.onet_code+r.onet_label).toLowerCase().includes(q));
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
      <td class="cite">${escapeHtml(r.citation_key)}</td>
      <td class="title" title="Click to view paper detail" onclick="viewPaper('${r.paper_id}')">${escapeHtml(r.title)}</td>
      <td class="file" title="${escapeHtml(r.file_name)}">${escapeHtml(r.file_name)}</td>
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
  td.id { font-family:var(--mono); font-size:11.5px; color:var(--ink-2); }
"""

_RUNS_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">__THEME_HEAD__<title>All runs &mdash; AI Impact Meta-Review</title>
<style>__STYLE__</style></head><body>
__HEADER__
<main class="page wrap">
  <section class="page-head">
    <div>
      <h1>Imputation runs</h1>
      <p class="lede">Each run propagates the observed effects across the O*NET graph under one combination of metric, &beta;,
        pruning, and baseline settings. Click a row to see its occupation- and activity-level estimates;
        the <a href="parameters.html">method</a> page explains each setting.</p>
    </div>
    <span class="page-meta" id="stats"></span>
  </section>
  <table id="t" class="data"><thead><tr>
    <th>Run ID</th><th>Started (UTC)</th><th>Metric</th><th>&beta;</th><th>Aggregation</th>
    <th>&Omega;<sub>ref</sub></th><th>Baseline</th>
    <th style="text-align:right">Observed (occ/act)</th><th style="text-align:right">Kept (occ/act)</th>
  </tr></thead><tbody id="tb"></tbody></table>
</main>
<script>
function escapeHtml(s){return String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
fetch('data/runs/index.json').then(r=>r.json()).then(d=>{
  const tb = document.getElementById('tb');
  if (!d.runs.length) { tb.innerHTML='<tr><td colspan="9" style="text-align:center; padding:28px; color:var(--muted);">No runs on record.</td></tr>'; return; }
  document.getElementById('stats').textContent = `${d.runs.length} runs`;
  tb.innerHTML = d.runs.map(r=>{
    const p = r.params||{};
    return `<tr class="clickable" onclick="location.href='results.html?run='+encodeURIComponent('${r.run_id}')">
      <td class="id">${escapeHtml(r.run_id)}</td>
      <td>${escapeHtml((r.started_utc||'').replace('T',' ').slice(0,19))}</td>
      <td><span class="pill ${p.metric}">${escapeHtml(p.metric||'')}</span></td>
      <td class="num">${p.beta}</td>
      <td>${escapeHtml((p.aggregation_level||'occupation').replace('_',' '))}</td>
      <td class="num">${p.omega_ref}</td>
      <td>${r.baseline_active ? `AIOE, Ω<sub>b</sub>=${p.omega_base}` : '—'}</td>
      <td class="num">${r.n_observed_occ} / ${r.n_observed_act}</td>
      <td class="num">${r.n_kept_occ} / ${r.n_kept_act}</td>
    </tr>`;
  }).join('');
});
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

    # 4. Runs (per-run + index)
    run_dirs = _all_run_dirs()
    all_run_metas = []
    for d in run_dirs:
        bundle = _load_run_bundle(d)
        (RUNS_DATA / f"{d.name}.json").write_text(json.dumps(bundle))
        meta_only = {k: v for k, v in bundle.items() if k not in ("occupation_impacts", "activity_impacts")}
        all_run_metas.append(meta_only)
    (RUNS_DATA / "index.json").write_text(json.dumps({"runs": all_run_metas}))
    canonical_run_id = run_dirs[0].name if run_dirs else ""
    print(f"wrote {len(run_dirs)} run bundles into data/runs/ (canonical: {canonical_run_id or 'none'})")

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
        ("runs.html",        _RUNS_HTML,        {"__STYLE__": _RUNS_STYLE,   "__HEADER__": _header("runs")}),
        ("parameters.html",  _PARAMS_HTML,      {"__STYLE__": _PARAMS_STYLE, "__HEADER__": _header("parameters"),
                                                  "__BETA__": str(CANONICAL_BETA)}),
        ("transitions.html", _TRANSITIONS_HTML, {"__HEADER__": _header("transitions")}),
    ]
    for name, template, repl in pages:
        (DOCS / name).write_text(_render(template, repl))
    print(f"wrote {len(pages)} HTML pages into {DOCS}")


if __name__ == "__main__":
    main()
