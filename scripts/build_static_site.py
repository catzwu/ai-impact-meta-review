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

OUT = ROOT / "outputs" / "final"
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
    """Shared top nav. `active` is one of 'home', 'runs', 'parameters', 'transitions'."""
    items = [
        ("index.html", "Home", "home"),
        ("runs.html", "All runs", "runs"),
        ("parameters.html", "Parameters", "parameters"),
        ("transitions.html", "Transitions", "transitions"),
    ]
    links = []
    for href, label, key in items:
        style = ' style="text-decoration:underline;"' if key == active else ""
        links.append(f'<a href="{href}"{style}>{label}</a>')
    links.append(f'<a class="gh" href="{REPO_URL}" target="_blank" rel="noopener">GitHub &#8599;</a>')
    return f"""
<header class="site">
  <h1>AI Impact Meta-Review</h1>
  <span class="stats" id="stats"></span>
  <nav>{"".join(links)}</nav>
</header>
"""


_STYLE_COMMON = r"""
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin:0; font-size:13px; color:#222; background:#fff; }
  header.site { background:#222; color:#fff; padding:10px 16px; display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
  header.site h1 { margin:0; font-size:16px; }
  header.site nav { display:flex; gap:12px; align-items:center; margin-left:auto; }
  header.site nav a { color:#9cf; text-decoration:none; font-size:13px; }
  header.site nav a:hover { text-decoration:underline; }
  header.site nav a.gh { color:#fff; background:#333; padding:4px 10px; border-radius:4px; }
  header.site nav a.gh:hover { background:#444; text-decoration:none; }
  header.site .stats { color:#bbb; font-size:12px; }
"""


# ---------- index.html (review table + drawer) ----------

_INDEX_STYLE = _STYLE_COMMON + r"""
  table { border-collapse: collapse; table-layout: fixed; width: 100%; }
  th, td { padding:8px 10px; border-bottom:1px solid #eee; vertical-align: top; text-align: left; overflow:hidden; text-overflow:ellipsis; }
  th { background:#f7f7f9; position:sticky; top:0; cursor:pointer; user-select:none; font-size:11px; text-transform:uppercase;
       font-weight:600; color:#555; letter-spacing:0.04em; border-bottom:2px solid #ddd; z-index:5; }
  tbody tr:nth-child(even) { background:#fafafa; }
  tbody tr:hover { background:#eef6ff; }
  td.snippet { white-space: normal; word-break: break-word; color:#555; font-size:12px; line-height:1.4; }
  td.title { white-space: normal; word-break: break-word; font-weight:500; font-size:13px; line-height:1.3; cursor:pointer; color:#0a3d7a; }
  td.title:hover { text-decoration: underline; }
  td.value { font-family: ui-monospace, Menlo, monospace; text-align: right; white-space: nowrap; font-weight:500; }
  td.value.pos { color:#0a6c2c; }
  td.value.neg { color:#b32020; }
  td.cite { font-family: ui-monospace, Menlo, monospace; font-size:11px; color:#555; }
  td.file { font-family: ui-monospace, Menlo, monospace; font-size:11px; color:#888; }
  td.onet { font-size:12px; color:#333; }
  td.onet .code { font-family: monospace; color:#888; margin-right:4px; }
  .kind-speed   { background:#e6f3ff; color:#0463a3; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:600; }
  .kind-quality { background:#ffe6f0; color:#a0286c; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:600; }
  .conf-pill-cell { display:inline-block; padding:1px 7px; border-radius:10px; font-size:10px; text-transform:uppercase; font-weight:600; }
  .conf-pill-cell.high   { background:#d4f4d4; color:#0a6c2c; }
  .conf-pill-cell.medium { background:#fff0c0; color:#8a6900; }
  .conf-pill-cell.low    { background:#ffd6d6; color:#a00; }

  #backdrop { position:fixed; inset:0; background:rgba(0,0,0,0.25); opacity:0; pointer-events:none; transition:opacity 0.2s; z-index:90; }
  #backdrop.open { opacity:1; pointer-events:auto; }
  #drawer { position:fixed; top:0; right:0; width:60%; min-width:560px; height:100%; background:#fff; box-shadow:-3px 0 12px rgba(0,0,0,0.2);
            transform:translateX(100%); transition:transform 0.2s; overflow:auto; z-index:100; }
  #drawer.open { transform:translateX(0); }
  #drawer > header { background:#333; color:#fff; padding:10px 16px; display:flex; gap:12px; align-items:center; justify-content:space-between; }
  #drawer > header h1 { margin:0; font-size:15px; font-family:monospace; }
  #drawer > header button { background:#a00; color:#fff; border:0; padding:5px 12px; border-radius:3px; cursor:pointer; }

  .pd-meta { padding: 14px 18px; background:#fafafa; border-bottom:1px solid #ddd; }
  .pd-meta h2 { margin: 0 0 4px; font-size:18px; line-height:1.25; }
  .pd-meta .authors { color:#444; font-size:12px; margin-bottom:4px; }
  .pd-meta .ids { color:#666; font-size:11px; }
  .pd-meta .ids .pill { display:inline-block; background:#eee; padding:1px 6px; border-radius:3px; margin-right:6px; font-family:monospace; }
  .pd-effects { display:flex; gap:10px; padding: 12px 18px; border-bottom:1px solid #eee; }
  .pd-effect { flex:1; border:1px solid #ddd; border-radius:6px; padding:10px; }
  .pd-effect.empty { color:#999; border-style:dashed; }
  .pd-effect h3 { margin:0 0 6px; font-size:13px; display:flex; justify-content:space-between; }
  .pd-effect .value { font-family:monospace; font-size:18px; }
  .pd-effect .row { display:flex; justify-content:space-between; font-size:11px; color:#555; margin-top:3px; }
  .pd-effect .row b { color:#222; font-weight:500; }
  .pd-effect .notes { margin-top:6px; font-size:11px; color:#777; line-height:1.4; }
  .pd-section { padding: 4px 18px 12px; border-bottom:1px solid #f0f0f0; }
  .pd-section h3 { margin:14px 0 6px; font-size:13px; text-transform:uppercase; color:#444; letter-spacing:0.04em; }
  .pd-table { width:100%; border-collapse:collapse; font-size:12px; }
  .pd-table th, .pd-table td { padding:4px 8px; border-bottom:1px solid #eee; text-align:left; vertical-align:top; }
  .pd-table th { background:#f5f5f5; font-weight:500; font-size:11px; color:#555; }
  .pd-onet-rationale { background:#f7faff; border-left:3px solid #69a; padding:8px 12px; margin-top:4px; font-size:12px; line-height:1.4; }
  .pd-onet-alt { font-size:11px; color:#555; margin-top:4px; }
  .pd-onet-alt .pill { display:inline-block; background:#eef; padding:1px 6px; margin:2px 4px 2px 0; border-radius:3px; font-family:monospace; }
  .pd-quote { background:#fffce0; border-left:3px solid #c90; padding:6px 10px; margin:4px 0; font-size:12px; line-height:1.4; font-style:italic; }
  .pd-quotes-block { max-height:260px; overflow-y:auto; }
  .pd-stat { border:1px solid #eee; border-radius:4px; padding:6px 10px; margin:4px 0; font-size:11px; }
  .pd-stat .head { display:flex; justify-content:space-between; font-size:12px; margin-bottom:2px; }
  .pd-stat .head b { color:#222; }
  .pd-stat .nums { font-family:monospace; color:#063; margin:2px 0; }
  .pd-stat .vq { font-size:11px; color:#666; font-style:italic; }
  details.pd-raw { margin: 0 18px 14px; }
  details.pd-raw summary { cursor:pointer; padding:4px 0; font-size:12px; color:#666; }
  details.pd-raw pre { background:#f7f7f7; padding:8px; font-size:10px; overflow:auto; max-height:300px; border-radius:3px; }
  .conf-pill { padding:1px 6px; border-radius:3px; font-size:10px; text-transform:uppercase; }
  .conf-pill.high   { background:#d4f4d4; color:#070; }
  .conf-pill.medium { background:#fff0c0; color:#a60; }
  .conf-pill.low    { background:#ffd6d6; color:#a00; }
  .badge { display:inline-block; padding:1px 5px; font-size:10px; border-radius:2px; background:#eee; margin-left:4px; }

  .downloads { margin: 18px 20px; padding: 14px 18px; background:#f7f7f9; border:1px solid #eee; border-radius:6px; }
  .downloads h2 { margin: 0 0 8px; font-size:13px; text-transform:uppercase; color:#555; letter-spacing:0.04em; }
  .downloads ul { margin:0; padding-left:20px; font-size:12px; }
  .downloads a { color:#0a3d7a; }
"""


_INDEX_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>AI Impact Meta-Review</title>
<style>__STYLE__</style></head><body>
__HEADER__
<div style="padding:10px 16px; background:#fafafa; border-bottom:1px solid #eee; display:flex; gap:14px; align-items:center;">
  <input type="text" id="search" placeholder="filter by citation, title, file, O*NET&hellip;" style="padding:5px 8px; border:1px solid #ccc; border-radius:4px; min-width:280px;"/>
  <span style="color:#888; font-size:12px;">Click a title to open the paper drawer.</span>
</div>
<table id="table">
  <colgroup>
    <col style="width:70px"><col style="width:200px"><col style="width:280px"><col style="width:200px">
    <col style="width:80px"><col style="width:60px"><col style="width:320px"><col style="width:340px">
  </colgroup>
  <thead><tr>
    <th data-sort="kind">Kind</th>
    <th data-sort="citation_key">Citation</th>
    <th data-sort="title">Title</th>
    <th data-sort="file_name">File</th>
    <th data-sort="value">Value</th>
    <th data-sort="confidence">Conf</th>
    <th>O*NET</th>
    <th>Task snippet</th>
  </tr></thead>
  <tbody id="tbody"></tbody>
</table>

<div class="downloads">
  <h2>Downloads</h2>
  <ul>
    <li><a href="assets/speed_table.csv">speed_table.csv</a> &mdash; per-paper speed effects (log ratios)</li>
    <li><a href="assets/quality_table.csv">quality_table.csv</a> &mdash; per-paper quality effects (Hedges' g)</li>
    <li><a href="assets/onet_activities_impact.csv">onet_activities_impact.csv</a> &mdash; observed effects aggregated by O*NET work activity</li>
    <li><a href="assets/onet_occupations_impact.csv">onet_occupations_impact.csv</a> &mdash; observed effects aggregated by O*NET occupation</li>
    <li><a href="assets/papers_excluded.csv">papers_excluded.csv</a> &mdash; papers dropped by the pipeline, with reasons</li>
  </ul>
</div>

<div id="backdrop" onclick="closeDrawer()"></div>
<div id="drawer">
  <header>
    <h1 id="drawerTitle">Paper detail</h1>
    <button onclick="closeDrawer()">Close</button>
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
  document.getElementById('stats').textContent = `${rows.length} rows (of ${ROWS.length} total)`;
  const tbody = document.getElementById('tbody');
  tbody.innerHTML = rows.map(r => {
    const v = parseFloat(r.value||0);
    const vClass = v > 0.001 ? 'pos' : v < -0.001 ? 'neg' : '';
    const onet = r.onet_code
      ? `<span class="code">${escapeHtml(r.onet_code)}</span>${escapeHtml(r.onet_label||'')}`
      : '<span style="color:#aaa;">—</span>';
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
  if (!pid) return alert('No paper_id resolved for this row.');
  let d;
  try { d = await (await fetch('data/papers/'+encodeURIComponent(pid)+'.json')).json(); }
  catch (e) { return alert('Missing paper detail for '+pid); }
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
    return `<div class="pd-effect empty"><h3>${kind} effect</h3>No effect computed for this paper.</div>`;
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
    sections.push(`<details ${key==='task'?'open':''}><summary><b>${key}</b> (${arr.length})</summary>${items}</details>`);
  }
  return sections.length ? `<div class="pd-section"><h3>Verbatim quotes (from 01a)</h3><div class="pd-quotes-block">${sections.join('')}</div></div>` : '';
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
    return `<tr><td><b>${escapeHtml(o.outcome_name||'')}</b>${star?` <span style="color:#a60;font-size:11px;">${star}</span>`:''}</td>
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
    <div><span class="pill" style="background:#cef; padding:2px 8px; font-family:monospace; border-radius:3px;">${escapeHtml(onet.onet_code||'?')}</span>
      <b>${escapeHtml(onet.onet_label||'')}</b> ${confPill(onet.mapping_confidence)}
      <span style="color:#999;font-size:11px;">(${escapeHtml(onet.mapping_type||'')})</span></div>
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
      <div class="head"><b>${escapeHtml(s.outcome_name||'?')}</b><span style="color:#666;">${escapeHtml(s.arm_comparison||'')}</span></div>
      ${numParts.length ? `<div class="nums">${escapeHtml(numParts.join('  •  '))}</div>`:''}
      ${extras.length ? `<div class="nums">${escapeHtml(extras.join('  •  '))}</div>`:''}
      ${s.verbatim_quote ? `<div class="vq">"${escapeHtml(s.verbatim_quote)}"</div>`:''}
    </div>`;
  }).join('');
  const more = stats.length>30 ? `<div style="color:#999;font-size:11px;">… ${stats.length-30} more</div>` : '';
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
      <div><b>${escapeHtml(meth.classification)}</b> ${confPill(meth.confidence)}</div>
      ${meth.rationale ? `<div style="color:#555; font-size:12px; margin-top:4px;">${escapeHtml(meth.rationale)}</div>`:''}
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
  main { padding: 16px 24px; background:#f5f5f7; min-height: calc(100vh - 50px); }
  .toolbar { display:flex; gap:14px; align-items:center; background:#fff; padding:10px 14px; border-radius:6px; margin-bottom:12px; box-shadow:0 1px 3px rgba(0,0,0,0.05); flex-wrap: wrap; }
  .toolbar label { font-size:12px; color:#555; }
  .toolbar input, .toolbar select { padding:4px 7px; border:1px solid #ccc; border-radius:3px; font-size:12px; }
  .toolbar .seg { display:inline-flex; border:1px solid #ccc; border-radius:4px; overflow:hidden; }
  .toolbar .seg button { padding:4px 11px; background:#fff; border:0; cursor:pointer; color:#555; font-size:12px; }
  .toolbar .seg button.active { background:#5b3aa6; color:#fff; }
  table { width:100%; background:#fff; border-collapse:collapse; border-radius:6px; overflow:hidden; box-shadow:0 1px 3px rgba(0,0,0,0.05); }
  th, td { padding:6px 10px; text-align:left; border-bottom:1px solid #eee; font-size:12px; }
  th { background:#f7f7f9; font-size:10px; text-transform:uppercase; color:#555; cursor:pointer; user-select:none; letter-spacing:0.04em; }
  tbody tr:nth-child(even) { background:#fafafa; }
  tbody tr:hover { background:#eef6ff; }
  td.num { font-family:ui-monospace,Menlo,monospace; text-align:right; }
  td.num.pos { color:#0a6c2c; } td.num.neg { color:#b32020; }
  .bar { display:inline-block; height:6px; vertical-align:middle; border-radius:2px; }
  .bar.pos { background:#0a6c2c; } .bar.neg { background:#b32020; }
  .badge { display:inline-block; padding:1px 7px; border-radius:10px; font-size:10px; font-weight:600; text-transform:uppercase; }
  .badge.observed { background:#d4f4d4; color:#0a6c2c; } .badge.imputed { background:#eee; color:#555; }
  .params { background:#fff; padding:10px 14px; border-radius:6px; margin-bottom:12px; box-shadow:0 1px 3px rgba(0,0,0,0.05); font-size:12px; color:#555; }
  .params b { color:#222; font-family: monospace; }
"""


_RESULTS_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Run results &mdash; AI Impact Meta-Review</title>
<style>__STYLE__</style></head><body>
__HEADER__
<main>
  <div class="params" id="params">Loading&hellip;</div>
  <div class="toolbar">
    <div class="seg" id="viewSeg">
      <button data-v="occ" class="active">Occupations</button>
      <button data-v="act">Activities</button>
    </div>
    <label>std &le; <input type="number" id="stdFilter" value="5" step="0.05" style="width:60px;"/></label>
    <label><input type="checkbox" id="onlyObserved"/> only observed</label>
    <label>search <input type="text" id="search" placeholder="title/code&hellip;"/></label>
    <button onclick="exportCsv()" style="margin-left:auto; padding:4px 10px; cursor:pointer;">Export view as CSV</button>
  </div>
  <div id="warn"></div>
  <table id="t"><thead id="th"></thead><tbody id="tb"></tbody></table>
</main>
<script>
const DEFAULT_RUN_ID = "__DEFAULT_RUN_ID__";
let DATA = null, VIEW='occ', SORT={col:'estimate', dir:-1};

function fmtNum(x, p){ if (x===null||x===undefined||x==='') return '—'; const n=parseFloat(x); return isNaN(n)?x:n.toFixed(p===undefined?3:p); }
function escapeHtml(s){return (s||'').toString().replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}

async function load(){
  const params = new URLSearchParams(location.search);
  const runId = params.get('run') || DEFAULT_RUN_ID;
  if (!runId) { document.body.innerHTML = '<p style="padding:30px;">No canonical run available. See <a href="runs.html">All runs</a>.</p>'; return; }
  const r = await fetch('data/runs/' + encodeURIComponent(runId) + '.json');
  if (!r.ok) { document.body.innerHTML = '<p style="padding:30px;">Unknown run: ' + escapeHtml(runId) + '. See <a href="runs.html">All runs</a>.</p>'; return; }
  DATA = await r.json();
  const p = DATA.params;
  document.getElementById('stats').textContent =
    `${p.metric.toUpperCase()} · β=${p.beta} · Ω_ref=${p.omega_ref}` +
    (DATA.baseline_active ? ` · baseline Ω_b=${p.omega_base}` : '') +
    ` · kept ${DATA.n_kept_occ}/${DATA.n_kept_act}`;
  document.getElementById('params').innerHTML =
    `<b>Run:</b> ${escapeHtml(DATA.run_id)} · <b>${p.metric}</b> · β=${p.beta} · agg=${escapeHtml(p.aggregation_level||'-')} · ` +
    `Ω_ref=${p.omega_ref}${DATA.baseline_active?` · baseline Ω_b=${p.omega_base}`:''} · ` +
    `n_obs occ/act = ${DATA.n_observed_occ}/${DATA.n_observed_act} · kept occ/act = ${DATA.n_kept_occ}/${DATA.n_kept_act} · ` +
    `data_source=${escapeHtml(DATA.data_source||'-')}`;
  const un = (DATA.unmatched_occ_codes||[]).concat(DATA.unmatched_act_labels||[]);
  if (un.length) document.getElementById('warn').innerHTML =
    `<div style="background:#fff8e0; border:1px solid #d2b048; padding:8px 12px; border-radius:4px; margin-bottom:10px; font-size:12px;">
      ${un.length} unmatched observation(s) skipped: ${un.slice(0,10).join(', ')}${un.length>10?'…':''}
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
    const labels = {code:'Code', title:'Title', activity:'Activity', observed:'Observed', aioe_baseline:'AIOE', estimate:'Estimate', posterior_std:'Std', n_studies:'n'};
    return `<th onclick="sortBy('${c}')">${labels[c]||c}${SORT.col===c?(SORT.dir<0?' ↓':' ↑'):''}</th>`;
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
  main { max-width:1100px; margin:24px auto; padding:0 18px; }
  .intro { color:#555; font-size:13px; margin-bottom:14px; }
  table { width:100%; background:#fff; border-collapse:collapse; border-radius:8px; overflow:hidden; box-shadow:0 1px 4px rgba(0,0,0,0.06); font-size:13px; }
  th, td { padding:9px 12px; text-align:left; border-bottom:1px solid #eee; }
  th { background:#f7f7f9; font-size:11px; text-transform:uppercase; color:#555; letter-spacing:0.04em; }
  tr.clickable:hover { background:#f0f6ff; cursor:pointer; }
  .pill { display:inline-block; padding:1px 8px; border-radius:10px; font-size:11px; font-weight:600; }
  .pill.speed { background:#e6f3ff; color:#0463a3; } .pill.quality { background:#ffe6f0; color:#a0286c; }
  td.num { font-variant-numeric: tabular-nums; font-family: ui-monospace, Menlo, monospace; }
"""

_RUNS_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>All runs &mdash; AI Impact Meta-Review</title>
<style>__STYLE__</style></head><body>
__HEADER__
<main>
  <p class="intro">
    All analysis runs on record. Each row records a single (metric, &beta;, prune, baseline) combination.
    Click a row to open its per-occupation / per-activity results.
    See <a href="parameters.html">Parameters</a> for what these knobs do.
  </p>
  <table id="t"><thead><tr>
    <th>Run ID</th><th>Started (UTC)</th><th>Metric</th><th>&beta;</th><th>Aggregation</th>
    <th>&Omega;<sub>ref</sub></th><th>Baseline</th>
    <th>Obs (occ/act)</th><th>Kept (occ/act)</th>
  </tr></thead><tbody id="tb"></tbody></table>
</main>
<script>
function escapeHtml(s){return String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
fetch('data/runs/index.json').then(r=>r.json()).then(d=>{
  const tb = document.getElementById('tb');
  if (!d.runs.length) { tb.innerHTML='<tr><td colspan="9" style="text-align:center; padding:20px; color:#888;">No runs on record.</td></tr>'; return; }
  document.getElementById('stats').textContent = `${d.runs.length} runs`;
  tb.innerHTML = d.runs.map(r=>{
    const p = r.params||{};
    return `<tr class="clickable" onclick="location.href='results.html?run='+encodeURIComponent('${r.run_id}')">
      <td style="font-family:monospace; font-size:11px;">${escapeHtml(r.run_id)}</td>
      <td>${escapeHtml((r.started_utc||'').replace('T',' ').slice(0,19))}</td>
      <td><span class="pill ${p.metric}">${escapeHtml(p.metric||'')}</span></td>
      <td class="num">${p.beta}</td>
      <td>${escapeHtml(p.aggregation_level||'occupation')}</td>
      <td class="num">${p.omega_ref}</td>
      <td>${r.baseline_active ? `yes (Ω<sub>b</sub>=${p.omega_base})` : '—'}</td>
      <td class="num">${r.n_observed_occ}/${r.n_observed_act}</td>
      <td class="num">${r.n_kept_occ}/${r.n_kept_act}</td>
    </tr>`;
  }).join('');
});
</script>
</body></html>
"""


# ---------- parameters.html (heatmap + bar chart + explanations, no submit) ----------

_PARAMS_STYLE = _STYLE_COMMON + r"""
  main { max-width:1000px; margin: 20px auto; padding: 0 18px; }
  .card { background:#fff; border-radius:8px; box-shadow:0 1px 4px rgba(0,0,0,0.06); padding:20px 24px; margin-bottom:18px; }
  h2 { font-size:14px; text-transform:uppercase; letter-spacing:0.04em; color:#666; margin: 0 0 10px; }
  .callout { padding: 12px 18px; background:#fff8e0; border:1px solid #d2b048; border-radius:6px; margin-bottom:18px; font-size:13px; color:#5a4200; line-height:1.5; }
  .callout code { background:#fff; padding:1px 5px; border-radius:3px; font-size:12px; }
  .seg { display:inline-flex; border:1px solid #ccc; border-radius:6px; overflow:hidden; }
  .seg button { padding:6px 16px; background:#fff; border:0; cursor:pointer; font-size:13px; color:#555; }
  .seg button.active { background:#5b3aa6; color:#fff; }
  .help { color:#666; font-size:12px; line-height:1.5; }
  .paramgrid { display:grid; grid-template-columns: 200px 1fr; gap: 6px 18px; align-items:baseline; margin-top: 4px; }
  .paramgrid .k { font-weight:500; color:#333; font-family: ui-monospace, Menlo, monospace; font-size:13px; }
  .paramgrid .v { color:#555; font-size:12px; line-height:1.5; }
  .paramgrid .v b { color:#222; font-family: ui-monospace, Menlo, monospace; }
  .threshold-row { display:flex; gap:14px; margin: 12px 0 6px; align-items:baseline; }
  .threshold-row input[type=number] { padding:5px 7px; border:1px solid #ccc; border-radius:4px; font-size:13px; width:80px; }
"""


_PARAMS_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Parameters &mdash; AI Impact Meta-Review</title>
<style>__STYLE__</style></head><body>
__HEADER__
<main>
  <div class="callout">
    <b>Read-only view.</b> This page visualises the inputs to the imputation model at the canonical
    setting (&beta; = __BETA__). To launch new runs with different parameters, run the review app locally
    (<code>python scripts/review_app.py</code>) and open <code>/run</code>.
  </div>

  <div class="card">
    <h2>Metric</h2>
    <div class="seg" id="metricSeg">
      <button data-m="speed" class="active">speed</button>
      <button data-m="quality">quality</button>
    </div>
    <div class="help" style="margin-top:6px;">A run targets one metric at a time. Switching swaps the observation
      overlays (green vertical lines = observed activities, cyan horizontal lines = SOC majors with at least one observed
      occupation).</div>
  </div>

  <div class="card">
    <h2>Stage B → Coverage heatmap</h2>
    <div class="help" style="margin-bottom:8px;">
      Mean Stage-B weight per SOC major group &times; O*NET work activity. Weights come from applying a
      row-wise softmax with temperature &beta; to each occupation's (Importance &times; Level / 5) profile, then
      averaging within each SOC-major group. Cells are the average of a group's occupations. Click a row label
      or checkbox to toggle its inclusion.
    </div>
    <div id="heatmap" style="overflow:auto;"></div>

    <div class="threshold-row">
      <label style="font-weight:500;">Activity weight threshold</label>
      <input type="number" id="weightThreshold" value="10" step="0.5" min="0"/>
      <div class="help">Activities whose summed weight (over kept occupations) is below this are dropped.
        Observed activities are always kept.</div>
    </div>
    <div id="barchart" style="overflow:auto;"></div>
    <div id="pruneSummary" style="margin-top:10px; padding:8px 12px; background:#f0f4f8; border-radius:4px; font-size:13px; color:#333;"></div>
  </div>

  <div class="card">
    <h2>What every parameter does</h2>
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
  if (!r.ok) { document.getElementById('heatmap').innerHTML = '<div style="color:#a00;">Missing heatmap data for '+METRIC+'.</div>'; return; }
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
      overlaySvg += `<rect x="${labelW}" y="${labelH+ri*cell}" width="${cols.length*cell}" height="${cell}" fill="rgba(180,180,180,0.55)"/>`;
    }
  });
  let lineSvg = '';
  cols.forEach((c, ci) => {
    if (c.observed) {
      const x = labelW + ci*cell + cell/2;
      lineSvg += `<line x1="${x}" y1="${labelH}" x2="${x}" y2="${labelH+rows.length*cell}" stroke="#39FF14" stroke-width="2"/>`;
    }
  });
  rows.forEach((r, ri) => {
    if (r.observed && !EXCLUDED.has(r.code)) {
      const y = labelH + ri*cell + cell/2;
      lineSvg += `<line x1="${labelW}" y1="${y}" x2="${labelW+cols.length*cell}" y2="${y}" stroke="#00E5FF" stroke-width="2"/>`;
    }
  });
  const rowLabels = rows.map((r, ri) => {
    const y = labelH + ri*cell;
    const fade = EXCLUDED.has(r.code) ? 'color:#aaa;' : '';
    const bold = r.observed ? 'font-weight:600; color:#0077aa;' : '';
    return `<div style="position:absolute; left:0; top:${y}px; height:${cell}px; width:${labelW-6}px;
                       display:flex; align-items:center; gap:5px; font-size:11px; ${fade}${bold} cursor:pointer;"
                 onclick="toggleSoc('${r.code}')">
              <input type="checkbox" ${EXCLUDED.has(r.code)?'':'checked'} onclick="event.stopPropagation(); toggleSoc('${r.code}');" style="margin:0 4px;">
              <span style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${escapeHtml(r.name)} (n=${r.n})">
                ${escapeHtml(r.name)} <span style="color:#999; font-weight:400;">(n=${r.n})</span>
              </span>
            </div>`;
  }).join('');
  const colLabelY = labelH + rows.length * cell + 6;
  const colLabels = cols.map((c, ci) => {
    const x = labelW + ci*cell + cell/2;
    const fill = c.observed ? '#1a9850' : '#444';
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
      <div style="display:flex; gap:18px; font-size:11px; color:#666; margin-top:6px;">
        <span><span style="display:inline-block; width:10px; height:2px; background:#39FF14; vertical-align:middle;"></span> observed activity</span>
        <span><span style="display:inline-block; width:10px; height:2px; background:#00E5FF; vertical-align:middle;"></span> SOC with observed occupation</span>
        <span><span style="display:inline-block; width:10px; height:10px; background:rgba(180,180,180,0.55); vertical-align:middle;"></span> deselected</span>
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
    const fill = a.observed ? '#0a6c2c' : '#4C72B0';
    const opacity = below ? 0.35 : 1.0;
    bars += `<rect x="${labelW}" y="${y+1}" width="${w}" height="${barH-3}" fill="${fill}" opacity="${opacity}"/>`;
    bars += `<text x="${labelW + w + 4}" y="${y+barH-3}" font-size="9" fill="${below?'#999':'#333'}">${sums[i].toFixed(1)}</text>`;
    const tcol = a.observed ? '#0a6c2c' : '#333';
    bars += `<text x="${labelW-4}" y="${y+barH-3}" font-size="10" fill="${tcol}" text-anchor="end" font-weight="${a.observed?'600':'400'}">${escapeHtml(a.name)}</text>`;
  });
  const tx = labelW + (threshold / vmax) * plotW;
  bars += `<line x1="${tx}" y1="${padT}" x2="${tx}" y2="${padT + acts.length*barH}" stroke="#a00" stroke-width="1.5" stroke-dasharray="4 3"/>`;
  bars += `<text x="${tx + 3}" y="${padT - 2}" font-size="10" fill="#a00">threshold = ${threshold.toFixed(1)}</text>`;

  document.getElementById('barchart').innerHTML = `
    <svg width="${labelW + plotW + 70}" height="${height}" style="display:block;">${bars}</svg>
    <div style="font-size:11px; color:#666; margin-top:4px;">
      Bars below the red dashed line are dropped. Green label/bar = observed (given).
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
    `<b>Included in analysis:</b> ${occIncluded.toLocaleString()} of ${occTotal.toLocaleString()} occupations ` +
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
<html><head><meta charset="utf-8"><title>Occupational transitions &mdash; AI Impact Meta-Review</title>
<style>
  __STYLE_COMMON__
  #main { display:grid; grid-template-columns: minmax(0,1fr) 460px; gap:14px; padding:14px; align-items:start; background:#fafafa; }
  .panel { background:#fff; border:1px solid #e5e5e5; border-radius:6px; padding:12px; }
  .panel h2 { margin:0 0 8px; font-size:13px; font-weight:600; color:#444; text-transform:uppercase; letter-spacing:0.04em; }
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
    position: absolute; left: 50%; bottom: 4px; transform-origin: 0 0;
    transform: rotate(-90deg) translateY(50%); white-space: nowrap;
    line-height: var(--cell); font-size: 10px;
  }
  .axis .lbl.major-end { border-bottom-color: rgba(60,60,60,0.5); }
  .axis.top .lbl.major-end { border-bottom-color: transparent; border-right-color: rgba(60,60,60,0.5); }
  .axis .lbl:hover { background:#eef5ff; }
  .axis .lbl.selected { background:#dbe9ff; }
  .legend { display:flex; align-items:center; gap:8px; margin-top:10px; font-size:11px; color:#666; flex-wrap:wrap; }
  .legend-bar { height:10px; width:160px; background: linear-gradient(to right, #fff, #fde0a0, #ef7838, #6e1a08); border:1px solid #ccc; }
  .tooltip { position:fixed; background:#222; color:#fff; padding:6px 10px; border-radius:4px; font-size:11px; pointer-events:none; z-index:1000; display:none; max-width:300px; line-height:1.4; }
  .drill h3 { margin:6px 0 4px; font-size:14px; }
  .drill .sub { color:#666; font-size:12px; margin-bottom:6px; }
  .drill .est-line { display:flex; gap:8px; align-items:baseline; font-size:12px; margin-bottom:10px; }
  .drill .est-line .v { font-weight:600; color:#222; font-size:14px; }
  .drill .crumb { font-size:12px; color:#666; margin-bottom:8px; }
  .drill .crumb a { color:#06c; cursor:pointer; }
  table.tbl { width:100%; border-collapse:collapse; font-size:12px; }
  table.tbl th { text-align:left; padding:4px 6px; color:#666; font-weight:500; border-bottom:1px solid #ddd; background:#fafafa; }
  table.tbl td { padding:4px 6px; border-bottom:1px solid #f0f0f0; vertical-align:top; }
  table.tbl td.num { text-align:right; font-variant-numeric:tabular-nums; }
  table.tbl tr.row-click { cursor:pointer; }
  table.tbl tr.row-click:hover { background:#f5f9ff; }
  .est-pos { color:#0a6f2a; } .est-neg { color:#a00; } .est-na { color:#aaa; }
  .empty { color:#888; font-size:12px; padding:8px; text-align:center; }
  .search { display:flex; gap:8px; margin-bottom:8px; position: relative; }
  .search input { flex:1; padding:6px 10px; border:1px solid #ccc; border-radius:4px; font-size:13px; }
  .typeahead { position:absolute; background:#fff; border:1px solid #ccc; border-radius:4px; max-height:240px; overflow:auto; z-index:50; left:0; right:120px; top:38px; box-shadow:0 4px 12px rgba(0,0,0,0.08); display:none; }
  .typeahead .item { padding:5px 9px; font-size:12px; cursor:pointer; border-bottom:1px solid #f0f0f0; }
  .typeahead .item:hover { background:#f5f9ff; }
  .typeahead .item .code { color:#888; font-size:11px; }
</style></head><body>
__HEADER__
<div id="main">
  <div class="panel">
    <h2>Transition probabilities by SOC minor group (source → target, row-conditional)</h2>
    <div class="search results">
      <input id="search" placeholder="Search occupation (title or SOC code)&hellip;" autocomplete="off"/>
      <button id="clearBtn" style="background:#eee;border:0;padding:6px 12px;border-radius:4px;cursor:pointer;">Clear</button>
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
      <span>0</span><div class="legend-bar"></div><span>max share (sqrt-scaled)</span>
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
    return out


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
        ("transitions.html", _TRANSITIONS_HTML, {"__STYLE_COMMON__": _STYLE_COMMON, "__HEADER__": _header("transitions")}),
    ]
    for name, template, repl in pages:
        (DOCS / name).write_text(_render(template, repl))
    print(f"wrote {len(pages)} HTML pages into {DOCS}")


if __name__ == "__main__":
    main()
