"""Review tool for the meta-analysis output tables.

Run:
    cd pipeline-run
    .venv/bin/pip install flask
    .venv/bin/python scripts/review_app.py
    # open http://127.0.0.1:5000

Reads from outputs/final/{speed,quality}_table.csv + per-paper stage outputs.
All edits persist to outputs/review_state.json. An "Export" button rewrites
outputs/final/{speed,quality}_table.csv with the edits applied.
"""
from __future__ import annotations

import csv
import difflib
import json
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from common import OUTPUTS_DIR, REPO_ROOT  # noqa: E402
import run_analysis as RA  # noqa: E402
import transitions as TR  # noqa: E402
import site_theme  # noqa: E402
import importlib.util as _ilu
_buf_spec = _ilu.spec_from_file_location("build_upload_files", SCRIPTS_DIR / "build_upload_files.py")
BUF = _ilu.module_from_spec(_buf_spec); _buf_spec.loader.exec_module(BUF)

PAPERS_DIR = REPO_ROOT / "papers"
UPLOAD_DIR = OUTPUTS_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
JOBS_FILE = UPLOAD_DIR / "jobs.json"
_jobs_lock = threading.Lock()


# ---------- upload helpers ----------

def _sanitize(name: str) -> str:
    base = re.sub(r"[^\w.\-]+", "_", name.strip())
    return base[:160] or "upload"


def _quick_extract_title(pdf_path: Path) -> tuple[str, str]:
    """Cheap title/header heuristic via pypdf. Returns (title_guess, first_page_text)."""
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover
        return "", ""
    try:
        r = PdfReader(str(pdf_path))
        meta_title = (r.metadata.title if r.metadata else None) or ""
        first = (r.pages[0].extract_text() or "") if r.pages else ""
    except Exception:
        return "", ""
    if meta_title and len(meta_title) > 8:
        return meta_title.strip(), first
    lines = [ln.strip() for ln in first.splitlines() if ln.strip()][:25]
    candidates = [ln for ln in lines if 12 <= len(ln) <= 220 and not ln.lower().startswith(("abstract", "doi"))]
    return (candidates[0] if candidates else (lines[0] if lines else "")), first


def _norm_title(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _dup_candidates(title: str, top_k: int = 5, threshold: float = 0.55) -> list[dict]:
    """Score all existing extractions against the new title."""
    if not title:
        return []
    nt = _norm_title(title)
    if not nt:
        return []
    out = []
    for p in (OUTPUTS_DIR / "01_extraction").glob("*.json"):
        if p.stem.endswith(".error"):
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        other_title = d.get("title", "")
        score = difflib.SequenceMatcher(None, nt, _norm_title(other_title)).ratio()
        if score >= threshold:
            out.append({
                "paper_id": p.stem,
                "citation_key": d.get("citation_key"),
                "title": other_title,
                "authors": (d.get("authors") or [])[:4],
                "year": d.get("year"),
                "score": round(score, 3),
            })
    out.sort(key=lambda x: -x["score"])
    return out[:top_k]


def _load_jobs() -> dict:
    if not JOBS_FILE.exists():
        return {}
    try:
        return json.loads(JOBS_FILE.read_text())
    except Exception:
        return {}


def _save_jobs(d: dict) -> None:
    with _jobs_lock:
        JOBS_FILE.write_text(json.dumps(d, indent=2))


def _stage_status(paper_id: str) -> dict:
    """Snapshot which pipeline stages have produced output for a paper."""
    stages = [
        ("extract_quotes", OUTPUTS_DIR / "01a_quotes" / f"{paper_id}.json"),
        ("structure",      OUTPUTS_DIR / "01_extraction" / f"{paper_id}.json"),
        ("classify_method",   OUTPUTS_DIR / "02_method_classification" / f"{paper_id}.json"),
        ("classify_outcome",  OUTPUTS_DIR / "03_outcome_classification" / f"{paper_id}.json"),
        ("map_onet",          OUTPUTS_DIR / "04_onet_mapping" / f"{paper_id}.json"),
        ("compute_speed",     OUTPUTS_DIR / "05_effect_sizes" / f"{paper_id}.speed.json"),
        ("compute_quality",   OUTPUTS_DIR / "05_effect_sizes" / f"{paper_id}.quality.json"),
    ]
    return {name: p.exists() for name, p in stages}


def _run_pipeline_for_paper(paper_id: str, job_id: str) -> None:
    """Launch one orchestrator subprocess that runs ALL stages for this paper, then
    assemble + build_upload_files. Update job state. The subprocess is started with
    its own process group so it survives the parent if Flask is restarted —
    self-healing in _stage_status() will then reconcile the job on the next status call.
    """
    jobs = _load_jobs()
    job = jobs.get(job_id, {})
    job["state"] = "running"
    job["started"] = time.time()
    jobs[job_id] = job
    _save_jobs(jobs)
    log_path = REPO_ROOT / "logs" / f"upload_{paper_id}.log"
    log_path.parent.mkdir(exist_ok=True)
    venv_py = REPO_ROOT / ".venv" / "bin" / "python"
    try:
        with open(log_path, "w") as lf:
            # All stages for this paper in one subprocess (orchestrator runs every stage in settings.stages)
            rc = subprocess.run(
                [str(venv_py), "scripts/orchestrator.py", "--paper", paper_id],
                cwd=str(REPO_ROOT), stdout=lf, stderr=subprocess.STDOUT,
                start_new_session=True,
            ).returncode
            if rc != 0:
                raise RuntimeError(f"orchestrator returned {rc}")
            subprocess.run([str(venv_py), "scripts/orchestrator.py", "--stage", "assemble"],
                           cwd=str(REPO_ROOT), stdout=lf, stderr=subprocess.STDOUT, check=True,
                           start_new_session=True)
            subprocess.run([str(venv_py), "scripts/build_upload_files.py"],
                           cwd=str(REPO_ROOT), stdout=lf, stderr=subprocess.STDOUT, check=True,
                           start_new_session=True)
        jobs = _load_jobs(); j = jobs.get(job_id, {})
        j["state"] = "done"; j["finished"] = time.time()
        jobs[job_id] = j; _save_jobs(jobs)
    except Exception as e:  # noqa: BLE001
        jobs = _load_jobs(); j = jobs.get(job_id, {})
        j["state"] = "failed"; j["error"] = str(e); j["finished"] = time.time()
        jobs[job_id] = j; _save_jobs(jobs)


def _classify_outcome_summary(paper_id: str) -> dict:
    """Inspect this paper's stage outputs and produce a human-readable summary
    of what happened: how many rows landed in the table (and why not, if zero)."""
    summary = {"speed_added": False, "quality_added": False, "reasons": []}
    method = _read_json(OUTPUTS_DIR / "02_method_classification" / f"{paper_id}.json")
    onet = _read_json(OUTPUTS_DIR / "04_onet_mapping" / f"{paper_id}.json")
    if method and method.get("classification") == "other":
        summary["reasons"].append(
            f"Classified as 'other' (not an in-scope study design): "
            f"{(method.get('rationale') or '').strip()[:240]}"
        )
        return summary
    outcome = _read_json(OUTPUTS_DIR / "03_outcome_classification" / f"{paper_id}.json")
    has_primary_speed = bool(outcome and outcome.get("primary_speed_outcome"))
    has_primary_qual  = bool(outcome and outcome.get("primary_quality_outcome"))
    if outcome and not has_primary_speed and not has_primary_qual:
        summary["reasons"].append("No primary speed or quality outcome could be identified for this paper.")
        return summary
    for kind in ("speed", "quality"):
        eff = _read_json(OUTPUTS_DIR / "05_effect_sizes" / f"{paper_id}.{kind}.json")
        primary = outcome and outcome.get(f"primary_{kind}_outcome")
        if not primary:
            continue
        if not eff:
            summary["reasons"].append(f"{kind}: stage 5 produced no output (primary outcome was '{primary}').")
            continue
        c = eff.get("computed") or {}
        method_used = c.get("computation_method")
        if not method_used:
            note = (c.get("notes") or "").strip()
            summary["reasons"].append(f"{kind}: effect not computable — {note[:240] or 'no usable numbers'}")
            continue
        if not (onet and onet.get("onet_code")):
            summary["reasons"].append(f"{kind}: computed but no O*NET mapping available.")
            continue
        summary[f"{kind}_added"] = True
        val = c.get("hedges_g") if kind == "quality" else c.get("log_ratio")
        summary["reasons"].append(
            f"{kind}: added → {kind=='quality' and 'g' or 'log_ratio'}={val:.3f}, "
            f"O*NET {onet.get('onet_code')} ({onet.get('onet_label','')[:40]})"
        )
    return summary


def _reconcile_job(job: dict) -> dict:
    """If a job is 'running' but its end state can be inferred from the outputs,
    promote it to done/failed. Handles: theory papers (method=other → done with note),
    papers with no primary outcomes, parse failures at 1a/1b, and normal completion."""
    if job.get("state") not in ("running", "queued"):
        return job
    pid = job["paper_id"]

    # 1a parse fail
    if (OUTPUTS_DIR / "01a_quotes" / f"{pid}.error.json").exists():
        return _finalize_job(job, "failed", error="stage 1a (extract_quotes) JSON parse failed")
    # 1b parse fail
    if (OUTPUTS_DIR / "01_extraction" / f"{pid}.error.json").exists():
        return _finalize_job(job, "failed", error="stage 1b (structure) JSON parse failed — paper extraction unusable")

    method = _read_json(OUTPUTS_DIR / "02_method_classification" / f"{pid}.json")
    if not method:
        return job  # stages 1+ still in flight
    # Theory / out-of-scope paper: done, with explanation.
    if method.get("classification") == "other":
        return _finalize_job(job, "done", outcome=_classify_outcome_summary(pid))

    outcome = _read_json(OUTPUTS_DIR / "03_outcome_classification" / f"{pid}.json")
    if not outcome:
        return job
    has_primary_speed = bool(outcome.get("primary_speed_outcome"))
    has_primary_qual  = bool(outcome.get("primary_quality_outcome"))

    # No primary outcomes at all → orchestrator will not produce stage 5 files; done.
    if not has_primary_speed and not has_primary_qual:
        return _finalize_job(job, "done", outcome=_classify_outcome_summary(pid))

    # Stage 5 done? Wait for whichever primary outcomes exist.
    need_speed = has_primary_speed and not (OUTPUTS_DIR / "05_effect_sizes" / f"{pid}.speed.json").exists()
    need_qual  = has_primary_qual  and not (OUTPUTS_DIR / "05_effect_sizes" / f"{pid}.quality.json").exists()
    if need_speed or need_qual:
        return job
    return _finalize_job(job, "done", outcome=_classify_outcome_summary(pid))


def _finalize_job(job: dict, state: str, *, error: str | None = None, outcome: dict | None = None) -> dict:
    jobs = _load_jobs()
    j = jobs.get(job["job_id"], job)
    j["state"] = state
    j["finished"] = time.time()
    j["_reconciled"] = True
    if error:
        j["error"] = error
    if outcome is not None:
        j["outcome"] = outcome
    jobs[j["job_id"]] = j
    _save_jobs(jobs)
    return j

CONFIG_DIR = REPO_ROOT / "config"
FINAL_DIR = OUTPUTS_DIR / "final"
STATE_PATH = OUTPUTS_DIR / "review_state.json"

app = Flask(__name__, static_folder=None)


# ---------- data loading ----------

def _read_json(p: Path):
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
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


def _save_state(state: dict) -> None:
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def _resolve_paper_id(citation_key: str) -> str | None:
    """Look up the underlying paper_id (filename stem) for a citation_key."""
    for p in (OUTPUTS_DIR / "01_extraction").glob("*.json"):
        if p.stem.endswith(".error"):
            continue
        d = _read_json(p)
        if d and d.get("citation_key") == citation_key:
            return p.stem
    return None


def _build_rows():
    """Merge speed_table + quality_table into a single list with extra metadata."""
    rows = []
    for kind, path, value_field, var_field in (
        ("speed", FINAL_DIR / "speed_table.csv", "log_ratio", "log_ratio_variance"),
        ("quality", FINAL_DIR / "quality_table.csv", "hedges_g", "variance"),
    ):
        for i, r in enumerate(_read_csv(path)):
            ck = r.get("citation_key", "")
            pid = _resolve_paper_id(ck) or ""
            row_id = f"{pid or ck}::{kind}::{i}"
            ext = _read_json(OUTPUTS_DIR / "01_extraction" / f"{pid}.json") if pid else None
            quotes = _read_json(OUTPUTS_DIR / "01a_quotes" / f"{pid}.json") if pid else None
            title = (ext or {}).get("title", "") if ext else ""
            # Prefer a verbatim quote from the paper text (01a_quotes), not an AI rationale.
            task_desc = ""
            if quotes and isinstance(quotes.get("quotes"), dict):
                qd = quotes["quotes"]
                for key in ("task", "study_design", "outcomes", "arms", "population"):
                    arr = qd.get(key)
                    if arr and isinstance(arr, list) and arr:
                        task_desc = (arr[0] or "")[:300]
                        if task_desc:
                            break
            # Fallback: verbatim_quote attached to any reported_statistics row in 01_extraction
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
                "raw": r,
            })
    return rows


def _apply_state(rows, state):
    """Apply edits, deletes, merges. Returns the displayable rows."""
    edits = state.get("edits", {})
    deleted = set(state.get("deleted", []))
    merges = state.get("merges", [])  # list of {keep: row_id, drop: [row_ids]}

    merged_into = {}
    for m in merges:
        for d in m.get("drop", []):
            merged_into[d] = m.get("keep")
            deleted.add(d)

    for r in rows:
        rid = r["row_id"]
        r["_deleted"] = rid in deleted
        r["_merged_into"] = merged_into.get(rid)
        for k, v in edits.get(rid, {}).items():
            r[k] = v
    return rows


# ---------- routes ----------

@app.route("/")
def index():
    return INDEX_HTML


@app.route("/api/rows")
def api_rows():
    state = _load_state()
    rows = _apply_state(_build_rows(), state)
    return jsonify({"rows": rows, "state": state})


@app.route("/api/onet")
def api_onet():
    acts = _read_json(CONFIG_DIR / "onet_activities.json") or []
    occs = _read_json(CONFIG_DIR / "onet_occupations.json") or []
    return jsonify({"activities": acts, "occupations": occs})


@app.route("/api/paper/<path:paper_id>")
def api_paper(paper_id):
    """Full per-paper drilldown."""
    return jsonify({
        "paper_id": paper_id,
        "01a_quotes": _read_json(OUTPUTS_DIR / "01a_quotes" / f"{paper_id}.json"),
        "01_extraction": _read_json(OUTPUTS_DIR / "01_extraction" / f"{paper_id}.json"),
        "02_method": _read_json(OUTPUTS_DIR / "02_method_classification" / f"{paper_id}.json"),
        "03_outcome": _read_json(OUTPUTS_DIR / "03_outcome_classification" / f"{paper_id}.json"),
        "04_onet": _read_json(OUTPUTS_DIR / "04_onet_mapping" / f"{paper_id}.json"),
        "05_speed": _read_json(OUTPUTS_DIR / "05_effect_sizes" / f"{paper_id}.speed.json"),
        "05_quality": _read_json(OUTPUTS_DIR / "05_effect_sizes" / f"{paper_id}.quality.json"),
    })


@app.route("/api/pdf/<path:paper_id>")
def api_pdf(paper_id):
    return send_from_directory(REPO_ROOT / "papers", f"{paper_id}.pdf")


@app.route("/api/row/<path:row_id>", methods=["POST"])
def api_update_row(row_id):
    body = request.json or {}
    state = _load_state()
    state.setdefault("edits", {}).setdefault(row_id, {}).update(body)
    _save_state(state)
    return jsonify({"ok": True})


@app.route("/api/row/<path:row_id>/delete", methods=["POST"])
def api_delete_row(row_id):
    state = _load_state()
    state.setdefault("deleted", [])
    if row_id not in state["deleted"]:
        state["deleted"].append(row_id)
    _save_state(state)
    return jsonify({"ok": True})


@app.route("/api/row/<path:row_id>/undelete", methods=["POST"])
def api_undelete_row(row_id):
    state = _load_state()
    state["deleted"] = [x for x in state.get("deleted", []) if x != row_id]
    _save_state(state)
    return jsonify({"ok": True})


@app.route("/api/merge", methods=["POST"])
def api_merge():
    body = request.json or {}
    keep = body.get("keep")
    drop = body.get("drop", [])
    if not keep or not drop:
        return jsonify({"ok": False, "error": "need keep + drop"}), 400
    state = _load_state()
    state.setdefault("merges", []).append({"keep": keep, "drop": drop})
    _save_state(state)
    return jsonify({"ok": True})


@app.route("/api/export", methods=["POST"])
def api_export():
    """Rewrite speed_table.csv and quality_table.csv with edits applied; drop deleted/merged rows."""
    state = _load_state()
    rows = _apply_state(_build_rows(), state)
    written = {"speed": 0, "quality": 0}
    for kind, path in (("speed", FINAL_DIR / "speed_table.csv"),
                       ("quality", FINAL_DIR / "quality_table.csv")):
        # backup
        if path.exists():
            (path.parent / f"{path.stem}.pre_review.csv").write_bytes(path.read_bytes())
        keep_rows = [r for r in rows if r["kind"] == kind and not r["_deleted"]]
        if not keep_rows:
            continue
        # union of raw keys + any edited fields we surfaced
        fieldnames = list(keep_rows[0]["raw"].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in keep_rows:
                merged = dict(r["raw"])
                # apply edits to raw columns where applicable
                for k in ("onet_code", "onet_label", "confidence", "notes"):
                    if k in merged:
                        merged[k] = r.get(k, merged[k])
                # Re-derive mapping_type from the (possibly edited) onet_code
                code = (merged.get("onet_code") or "").strip()
                if code.startswith("WA-"):
                    merged["mapping_type"] = "work_activity"
                elif code:
                    merged["mapping_type"] = "occupation"
                w.writerow(merged)
                written[kind] += 1
    return jsonify({"ok": True, "written": written})


# ---------- upload routes ----------

@app.route("/api/upload/check", methods=["POST"])
def api_upload_check():
    f = request.files.get("pdf")
    if not f or not f.filename:
        return jsonify({"ok": False, "error": "no file"}), 400
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"ok": False, "error": "must be a .pdf"}), 400
    raw_stem = Path(f.filename).stem
    sanitized_stem = _sanitize(raw_stem)
    staged_id = f"{int(time.time())}_{uuid.uuid4().hex[:6]}_{sanitized_stem}"
    staged_pdf = UPLOAD_DIR / f"{staged_id}.pdf"
    f.save(staged_pdf)
    title, _first = _quick_extract_title(staged_pdf)
    candidates = _dup_candidates(title)
    target_paper_id = sanitized_stem
    if (PAPERS_DIR / f"{target_paper_id}.pdf").exists():
        target_paper_id = f"{sanitized_stem}_{int(time.time())}"
    return jsonify({
        "ok": True,
        "staged_id": staged_id,
        "extracted_title": title,
        "target_paper_id": target_paper_id,
        "duplicate_candidates": candidates,
    })


@app.route("/api/upload/run", methods=["POST"])
def api_upload_run():
    body = request.json or {}
    staged_id = body.get("staged_id")
    target_paper_id = body.get("target_paper_id") or staged_id
    if not staged_id:
        return jsonify({"ok": False, "error": "staged_id required"}), 400
    staged_pdf = UPLOAD_DIR / f"{staged_id}.pdf"
    if not staged_pdf.exists():
        return jsonify({"ok": False, "error": "staged file not found"}), 404
    target_paper_id = _sanitize(target_paper_id)
    final_pdf = PAPERS_DIR / f"{target_paper_id}.pdf"
    if final_pdf.exists():
        target_paper_id = f"{target_paper_id}_{int(time.time())}"
        final_pdf = PAPERS_DIR / f"{target_paper_id}.pdf"
    staged_pdf.replace(final_pdf)
    job_id = uuid.uuid4().hex[:10]
    jobs = _load_jobs()
    jobs[job_id] = {"job_id": job_id, "paper_id": target_paper_id, "state": "queued", "queued": time.time()}
    _save_jobs(jobs)
    t = threading.Thread(target=_run_pipeline_for_paper, args=(target_paper_id, job_id), daemon=True)
    t.start()
    return jsonify({"ok": True, "job_id": job_id, "paper_id": target_paper_id})


@app.route("/api/upload/status/<job_id>")
def api_upload_status(job_id):
    jobs = _load_jobs()
    job = jobs.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "unknown job"}), 404
    job = _reconcile_job(job)
    stages = _stage_status(job["paper_id"])
    return jsonify({"ok": True, "job": job, "stages": stages})


@app.route("/api/upload/jobs")
def api_upload_jobs():
    """List all jobs (newest first) with reconciled state."""
    jobs = _load_jobs()
    out = []
    for j in jobs.values():
        out.append(_reconcile_job(j))
    out.sort(key=lambda j: -(j.get("queued") or 0))
    return jsonify({"ok": True, "jobs": out})


# ---------- analysis routes ----------

@app.route("/run")
def page_run():
    return RUN_HTML


@app.route("/transitions")
def page_transitions():
    return TRANSITIONS_HTML


@app.route("/api/transitions/data")
def api_transitions_data():
    return jsonify(TR.get_data())


@app.route("/results")
def page_results_index():
    return RESULTS_INDEX_HTML


@app.route("/results/<run_id>")
def page_results(run_id):
    return RESULTS_HTML.replace("__RUN_ID__", run_id)


@app.route("/api/run", methods=["POST"])
def api_run():
    body = request.json or {}
    try:
        params = RA.RunParams(
            metric=body.get("metric", "speed"),
            beta=float(body.get("beta", 2.5)),
            aggregation_level=str(body.get("aggregation_level", "occupation")),
            aggregate_to_socmajor=bool(body.get("aggregate_to_socmajor", False)),
            manual_prune=bool(body.get("manual_prune", True)),
            excluded_soc_majors=list(body.get("excluded_soc_majors", RA.DEFAULT_EXCLUDED_SOCS)),
            activity_weight_threshold=float(body.get("activity_weight_threshold", 10.0)),
            alpha=float(body.get("alpha", 0.7)),
            hops=int(body.get("hops", 4)),
            c_occ=float(body.get("c_occ", 1.0)),
            prune_activities=bool(body.get("prune_activities", False)),
            c_act=float(body.get("c_act", 1.5)),
            omega_ref=float(body.get("omega_ref", 100.0)),
            sigma_ref=float(body.get("sigma_ref", 0.1)),
            eps=float(body.get("eps", 1e-6)),
            use_baseline=bool(body.get("use_baseline", True)),
            omega_base=float(body.get("omega_base", 0.5)),
        )
    except Exception as e:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"bad params: {e}"}), 400
    if params.metric not in ("speed", "quality"):
        return jsonify({"ok": False, "error": "metric must be 'speed' or 'quality'"}), 400
    # Build observation rows from the LIVE review table (with edits/deletes/merges applied),
    # so the user doesn't have to re-export CSVs before running.
    state = _load_state()
    rows = _apply_state(_build_rows(), state)
    speed_raw, quality_raw = [], []
    for r in rows:
        if r["_deleted"]:
            continue
        # apply per-row onet edits onto the raw row before aggregating
        raw = dict(r["raw"])
        raw["onet_code"] = r.get("onet_code", raw.get("onet_code", ""))
        raw["onet_label"] = r.get("onet_label", raw.get("onet_label", ""))
        # Derive mapping_type from the code shape so dropdown edits don't desync it.
        code = (raw.get("onet_code") or "").strip()
        if code.startswith("WA-"):
            raw["mapping_type"] = "work_activity"
        elif code:
            raw["mapping_type"] = "occupation"
        (speed_raw if r["kind"] == "speed" else quality_raw).append(raw)
    occ_rows, act_rows = BUF.collect_from_iters(speed_raw, quality_raw)
    meta = RA.run(params, occ_rows=occ_rows, act_rows=act_rows)
    meta["n_observations_used"] = {"speed": len(speed_raw), "quality": len(quality_raw),
                                    "occ_codes": len(occ_rows), "act_codes": len(act_rows)}
    return jsonify({"ok": True, "run_id": meta["run_id"], "meta": meta})


@app.route("/api/runs")
def api_runs():
    return jsonify({"runs": RA.list_runs()})


def _observations_from_state():
    """Aggregate the LIVE review-table state (edits/deletes/merges applied) into
    the occ_rows/act_rows shape run_analysis expects. Shared by /api/run, /api/loo,
    and /api/heatmap_data."""
    state = _load_state()
    rows = _apply_state(_build_rows(), state)
    speed_raw, quality_raw = [], []
    for r in rows:
        if r["_deleted"]:
            continue
        raw = dict(r["raw"])
        raw["onet_code"] = r.get("onet_code", raw.get("onet_code", ""))
        raw["onet_label"] = r.get("onet_label", raw.get("onet_label", ""))
        code = (raw.get("onet_code") or "").strip()
        if code.startswith("WA-"):
            raw["mapping_type"] = "work_activity"
        elif code:
            raw["mapping_type"] = "occupation"
        (speed_raw if r["kind"] == "speed" else quality_raw).append(raw)
    occ_rows, act_rows = BUF.collect_from_iters(speed_raw, quality_raw)
    return occ_rows, act_rows, len(speed_raw), len(quality_raw)


@app.route("/api/loo", methods=["POST"])
def api_loo():
    """Leave-one-out cross-validation using the LIVE table state."""
    body = request.json or {}
    try:
        params = RA.RunParams(
            metric=body.get("metric", "speed"),
            beta=float(body.get("beta", 2.5)),
            aggregation_level=str(body.get("aggregation_level", "occupation")),
            aggregate_to_socmajor=bool(body.get("aggregate_to_socmajor", False)),
            manual_prune=bool(body.get("manual_prune", True)),
            excluded_soc_majors=list(body.get("excluded_soc_majors", RA.DEFAULT_EXCLUDED_SOCS)),
            activity_weight_threshold=float(body.get("activity_weight_threshold", 10.0)),
            alpha=float(body.get("alpha", 0.7)),
            hops=int(body.get("hops", 4)),
            c_occ=float(body.get("c_occ", 1.0)),
            prune_activities=bool(body.get("prune_activities", False)),
            c_act=float(body.get("c_act", 1.5)),
            omega_ref=float(body.get("omega_ref", 100.0)),
            sigma_ref=float(body.get("sigma_ref", 0.1)),
            eps=float(body.get("eps", 1e-6)),
            use_baseline=bool(body.get("use_baseline", True)),
            omega_base=float(body.get("omega_base", 0.5)),
        )
    except Exception as e:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"bad params: {e}"}), 400
    if params.metric not in ("speed", "quality"):
        return jsonify({"ok": False, "error": "metric must be 'speed' or 'quality'"}), 400
    occ_rows, act_rows, n_speed, n_qual = _observations_from_state()
    meta = RA.run_loo(params, occ_rows=occ_rows, act_rows=act_rows)
    meta["n_observations_used"] = {"speed": n_speed, "quality": n_qual,
                                    "occ_codes": len(occ_rows), "act_codes": len(act_rows)}
    return jsonify({"ok": True, "run_id": meta["run_id"], "meta": meta})


@app.route("/loo/<run_id>")
def page_loo(run_id):
    return LOO_HTML.replace("__RUN_ID__", run_id)


@app.route("/api/loo/<run_id>")
def api_loo_get(run_id):
    d = RA.load_loo(run_id)
    if not d:
        return jsonify({"ok": False, "error": "not a LOO run or not found"}), 404
    return jsonify({"ok": True, "data": d})


@app.route("/compare")
def page_compare():
    return COMPARE_HTML


@app.route("/api/compare")
def api_compare():
    a = request.args.get("a"); b = request.args.get("b")
    if not a or not b:
        return jsonify({"ok": False, "error": "need a= and b= run ids"}), 400
    d = RA.compare_runs(a, b)
    if d is None:
        return jsonify({"ok": False, "error": "one or both runs not found"}), 404
    return jsonify({"ok": True, "data": d})


@app.route("/api/heatmap_data")
def api_heatmap_data():
    """SOC-major × activity weight matrix + observed overlays for the given metric.
    Pulled from the LIVE review table (so observed overlays reflect deletes/edits)."""
    try:
        metric = request.args.get("metric", "speed")
        beta = float(request.args.get("beta", 2.5))
    except Exception as e:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(e)}), 400
    state = _load_state()
    rows = _apply_state(_build_rows(), state)
    speed_raw, quality_raw = [], []
    for r in rows:
        if r["_deleted"]:
            continue
        raw = dict(r["raw"])
        raw["onet_code"] = r.get("onet_code", raw.get("onet_code", ""))
        raw["onet_label"] = r.get("onet_label", raw.get("onet_label", ""))
        code = (raw.get("onet_code") or "").strip()
        if code.startswith("WA-"):
            raw["mapping_type"] = "work_activity"
        elif code:
            raw["mapping_type"] = "occupation"
        (speed_raw if r["kind"] == "speed" else quality_raw).append(raw)
    occ_rows, act_rows = BUF.collect_from_iters(speed_raw, quality_raw)
    data = RA.heatmap_data(beta, metric, occ_rows=occ_rows, act_rows=act_rows)
    return jsonify({"ok": True, **data})


@app.route("/api/results/<run_id>")
def api_results(run_id):
    d = RA.load_run(run_id)
    if not d:
        return jsonify({"ok": False, "error": "not found"}), 404
    return jsonify({"ok": True, "data": d})


# ---------- frontend ----------

INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Studies · AI Impact Meta-Review</title>
__THEME_HEAD__
<style>
  /* review table */
  #table { table-layout: fixed; min-width:1300px; }
  #table th, #table td { overflow:hidden; text-overflow:ellipsis; }
  #table th { position:sticky; top:0; z-index:5; }
  th .resizer { position:absolute; right:-3px; top:0; width:8px; height:100%; cursor:col-resize; user-select:none; z-index:2; }
  th .resizer:hover, th .resizer:active { background:var(--accent-soft); }
  tr.deleted { opacity:0.5; background:var(--err-bg) !important; }
  tr.deleted td.title { text-decoration:line-through; }
  tr.merged-into { background:var(--warn-bg) !important; }
  td.snippet { white-space:normal; word-break:break-word; color:var(--ink-2); font-size:12.5px; line-height:1.45; }
  td.title { white-space:normal; word-break:break-word; font-family:var(--serif); font-size:14.5px; line-height:1.35; cursor:pointer; color:var(--ink); }
  td.title:hover { color:var(--link); text-decoration:underline; text-underline-offset:2px; }
  td.cite { font-size:13.5px; color:var(--ink); white-space:normal; }
  td.file { font-family:var(--mono); font-size:11px; color:var(--muted); }
  /* searchable onet picker */
  .onet-picker { position:relative; max-width:320px; }
  .onet-picker .display { border:1px solid var(--rule); padding:4px 8px; font-size:12.5px; cursor:pointer; background:var(--surface); border-radius:4px; min-height:18px; line-height:1.4; }
  .onet-picker .display:hover { border-color:var(--accent); }
  .onet-picker .display.empty { color:var(--muted); font-style:italic; }
  #onetPanel { position:absolute; background:var(--surface); border:1px solid var(--rule); border-radius:6px; box-shadow:0 10px 30px rgba(29,31,35,0.16); z-index:200;
               width:400px; max-height:420px; display:none; overflow:hidden; }
  #onetPanel.open { display:block; }
  #onetPanel input.search { width:calc(100% - 16px); margin:8px; }
  #onetPanel .list { max-height:340px; overflow-y:auto; font-size:13px; }
  #onetPanel .group-hdr { background:var(--surface-2); padding:5px 10px; font-weight:600; font-size:12px; color:var(--ink-2); position:sticky; top:0; }
  #onetPanel .item { padding:5px 10px; cursor:pointer; }
  #onetPanel .item:hover, #onetPanel .item.active { background:var(--accent-soft); }
  #onetPanel .code { font-family:var(--mono); color:var(--muted); font-size:11px; margin-right:6px; }
  #onetPanel .clear { padding:7px 10px; color:var(--neg); cursor:pointer; border-top:1px solid var(--rule); font-size:12px; }
  .pd-meta a { margin-left:auto; }
  .stage-list { border:1px solid var(--rule); border-radius:5px; padding:8px 14px; margin-top:10px; }
  .stage-list .st { display:flex; gap:10px; font-size:13px; padding:2px 0; color:var(--muted); font-family:var(--mono); }
  .stage-list .st.done { color:var(--pos); }
  .dup { border:1px solid var(--rule); border-radius:5px; padding:10px 14px; margin-bottom:8px; display:flex; justify-content:space-between; align-items:flex-start; gap:12px; }
  .dup .t { font-family:var(--serif); font-size:15px; }
  .dup .m { font-size:12px; color:var(--muted); margin-top:2px; }
</style></head><body>
__MASTHEAD__
<div class="wrap wide">
  <section class="page-head">
    <div>
      <h1>Coded studies</h1>
      <p class="lede">One row per extracted effect. Click a title to read the paper's extraction and evidence.
        Edits, merges, and deletions are saved as an overlay; the source CSVs are only changed by <i>Export CSVs</i>.</p>
    </div>
    <div class="page-actions">
      <button id="jobsPill" class="btn btn-warn" style="display:none;" onclick="showJobsList()"></button>
      <button id="uploadBtn" class="btn btn-primary">Upload paper (PDF)</button>
    </div>
  </section>
  <div class="toolbar">
    <input type="search" id="search" placeholder="Filter by citation, title, file, or O*NET&hellip;" style="min-width:320px; flex:0 1 420px;"/>
    <span class="page-meta" id="stats"></span>
    <span style="flex:1"></span>
    <span class="help">Selected rows:</span>
    <button id="mergeBtn" class="btn">Merge</button>
    <button id="deleteBtn" class="btn btn-danger">Delete</button>
    <button id="restoreBtn" class="btn" title="Undo delete for the selected rows">Restore</button>
    <button id="exportBtn" class="btn" title="Write the edited tables back to outputs/final/*.csv">Export CSVs</button>
  </div>
  <input type="file" id="uploadFile" accept="application/pdf" style="display:none;"/>
<div id="uploadModal" class="modal-shade" style="display:none;">
  <div class="modal">
    <div class="modal-head">
      <h2>Upload a paper</h2>
      <button class="btn btn-quiet" onclick="closeUpload()" aria-label="Close">Close</button>
    </div>
    <div id="uploadBody" class="modal-body"></div>
  </div>
</div>
<div class="table-scroll">
<table id="table">
  <colgroup>
    <col style="width:36px"><col style="width:80px"><col style="width:180px"><col style="width:280px">
    <col style="width:80px"><col style="width:100px"><col style="width:300px"><col>
  </colgroup>
  <thead><tr>
    <th><input type="checkbox" id="selAll"/><div class="resizer"></div></th>
    <th data-sort="kind">Kind<div class="resizer"></div></th>
    <th data-sort="citation">Citation<div class="resizer"></div></th>
    <th data-sort="title">Title<div class="resizer"></div></th>
    <th data-sort="value" style="text-align:right">Value<div class="resizer"></div></th>
    <th data-sort="confidence">Confidence<div class="resizer"></div></th>
    <th>O*NET<div class="resizer"></div></th>
    <th>Task snippet<div class="resizer"></div></th>
  </tr></thead>
  <tbody id="tbody"></tbody>
</table>
</div>
</div>
<div id="backdrop" onclick="closeDrawer()"></div>
<div id="drawer">
  <header>
    <h1 id="drawerTitle">Paper detail</h1>
    <button class="btn btn-quiet" onclick="closeDrawer()">Close &times;</button>
  </header>
  <div id="drawerBody"></div>
</div>
<div id="onetPanel">
  <input type="text" class="search" id="onetSearch" placeholder="search activities & occupations..." />
  <div class="list" id="onetList"></div>
  <div class="clear" id="onetClear">Clear selection (none)</div>
</div>
<script>
let ROWS=[], ONET={activities:[],occupations:[]}, SORT={col:null,dir:1};
let PICKER = { rowId:null, target:null };

async function load() {
  const [rResp, oResp] = await Promise.all([fetch('/api/rows'), fetch('/api/onet')]);
  const r = await rResp.json(); ONET = await oResp.json();
  ROWS = r.rows; render();
}

function onetDisplay(code, label) {
  if (!code) return '<div class="display empty">Not mapped. Click to set</div>';
  return `<div class="display">${escapeHtml(code)} — ${escapeHtml(label||'')}</div>`;
}

function openPicker(rowId, targetEl) {
  PICKER.rowId = rowId;
  PICKER.target = targetEl;
  const panel = document.getElementById('onetPanel');
  const rect = targetEl.getBoundingClientRect();
  panel.style.left = (rect.left + window.scrollX) + 'px';
  panel.style.top  = (rect.bottom + window.scrollY + 2) + 'px';
  document.getElementById('onetSearch').value = '';
  renderPickerList('');
  panel.classList.add('open');
  setTimeout(()=>document.getElementById('onetSearch').focus(), 30);
}
function closePicker() {
  document.getElementById('onetPanel').classList.remove('open');
  PICKER.rowId = null; PICKER.target = null;
}
function renderPickerList(q) {
  q = (q||'').toLowerCase().trim();
  const match = (x) => !q || (x.code+' '+x.label+' '+(x.job_family||'')).toLowerCase().includes(q);
  const acts = ONET.activities.filter(match);
  const occs = ONET.occupations.filter(match);
  const fmt = (x) => `<div class="item" data-code="${x.code}" data-label="${escapeHtml(x.label)}">
      <span class="code">${x.code}</span>${escapeHtml(x.label)}${x.job_family?` <span style="color:#999;font-size:10px">(${escapeHtml(x.job_family)})</span>`:''}
    </div>`;
  document.getElementById('onetList').innerHTML =
    `<div class="group-hdr">Work Activities (${acts.length})</div>` +
    (acts.length ? acts.map(fmt).join('') : '<div style="padding:5px 10px;color:#999">no matches</div>') +
    `<div class="group-hdr">Occupations (${occs.length})</div>` +
    (occs.length ? occs.map(fmt).join('') : '<div style="padding:5px 10px;color:#999">no matches</div>');
}
document.addEventListener('click', (e)=>{
  // close picker if click outside it AND outside the originating display
  const panel = document.getElementById('onetPanel');
  if (!panel.classList.contains('open')) return;
  if (panel.contains(e.target)) return;
  if (PICKER.target && PICKER.target.contains(e.target)) return;
  closePicker();
});
document.getElementById('onetSearch').addEventListener('input', (e)=> renderPickerList(e.target.value));
document.getElementById('onetList').addEventListener('click', (e)=>{
  const item = e.target.closest('.item'); if (!item) return;
  const code = item.dataset.code, label = item.dataset.label;
  applyPicked(code, label);
});
document.getElementById('onetClear').addEventListener('click', ()=> applyPicked('', ''));

async function applyPicked(code, label) {
  if (!PICKER.rowId) return closePicker();
  await updateOnet(PICKER.rowId, code, label);
  const row = ROWS.find(r=>r.row_id===PICKER.rowId);
  if (row && PICKER.target) PICKER.target.innerHTML = onetDisplay(code, label);
  closePicker();
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
  const live = rows.filter(r => !r._deleted).length;
  document.getElementById('stats').textContent = `${rows.length} rows · ${live} live · ${rows.length-live} dropped`;
  const tbody = document.getElementById('tbody');
  tbody.innerHTML = rows.map(r => {
    const v = parseFloat(r.value||0);
    const vClass = v > 0.001 ? 'pos' : v < -0.001 ? 'neg' : '';
    return `
    <tr class="${r._deleted?'deleted':''}${r._merged_into?' merged-into':''}" data-row="${r.row_id}">
      <td><input type="checkbox" class="sel"/></td>
      <td><span class="kind-${r.kind}">${r.kind}</span>${r._merged_into?'<span class="badge">merged</span>':''}</td>
      <td class="cite" title="${escapeHtml(r.citation_key)}">${escapeHtml(r.citation||r.citation_key)}</td>
      <td class="title" title="Click to view paper detail" onclick="viewPaper('${r.paper_id}','${r.row_id}')">${escapeHtml(r.title)}</td>
      <td class="value ${vClass}">${v.toFixed(3)}</td>
      <td>${r.confidence ? `<span class="conf-pill-cell ${r.confidence}">${r.confidence}</span>` : ''}</td>
      <td>
        <div class="onet-picker" onclick="openPicker('${r.row_id}', this)">
          ${onetDisplay(r.onet_code, r.onet_label)}
        </div>
      </td>
      <td class="snippet" title="${escapeHtml(r.task_description||'')}">${escapeHtml(r.task_description||'')}</td>
    </tr>`;
  }).join('');
}

function escapeHtml(s){return (s||'').toString().replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}

document.getElementById('search').addEventListener('input', render);
document.querySelectorAll('th[data-sort]').forEach(th=>{
  th.addEventListener('click',()=>{
    const c=th.dataset.sort; SORT.dir=(SORT.col===c?-SORT.dir:1); SORT.col=c; render();
  });
});

document.getElementById('selAll').addEventListener('change', e=>{
  document.querySelectorAll('input.sel').forEach(cb=>cb.checked=e.target.checked);
});

async function updateOnet(rowId, code, label) {
  await fetch('/api/row/'+encodeURIComponent(rowId), {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({onet_code: code, onet_label: label||''})});
  const row = ROWS.find(r=>r.row_id===rowId);
  if (row){ row.onet_code=code; row.onet_label=label||''; }
}

async function deleteRow(id){ await fetch('/api/row/'+encodeURIComponent(id)+'/delete',{method:'POST'}); await load(); }
async function undeleteRow(id){ await fetch('/api/row/'+encodeURIComponent(id)+'/undelete',{method:'POST'}); await load(); }

document.getElementById('deleteBtn').addEventListener('click', async ()=>{
  const ids = selectedIds(); if (!ids.length) return;
  if (!confirm(`Delete ${ids.length} rows?`)) return;
  for (const id of ids) await fetch('/api/row/'+encodeURIComponent(id)+'/delete',{method:'POST'});
  await load();
});
document.getElementById('restoreBtn').addEventListener('click', async ()=>{
  const ids = selectedIds(); if (!ids.length) return;
  for (const id of ids) await fetch('/api/row/'+encodeURIComponent(id)+'/undelete',{method:'POST'});
  await load();
});
document.getElementById('mergeBtn').addEventListener('click', async ()=>{
  const ids = selectedIds();
  if (ids.length<2) return alert('Select 2+ rows to merge');
  const keep = prompt('Which row_id to KEEP? (others get dropped). Selected:\\n'+ids.join('\\n'), ids[0]);
  if (!keep || !ids.includes(keep)) return;
  const drop = ids.filter(i=>i!==keep);
  await fetch('/api/merge',{method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({keep, drop})});
  await load();
});
document.getElementById('exportBtn').addEventListener('click', async ()=>{
  if (!confirm('Overwrite outputs/final/{speed,quality}_table.csv with current edits? (originals backed up to *.pre_review.csv)')) return;
  const r = await (await fetch('/api/export',{method:'POST'})).json();
  alert('Exported: ' + JSON.stringify(r.written));
});

function selectedIds(){
  return [...document.querySelectorAll('tbody tr')].filter(tr=>tr.querySelector('input.sel').checked).map(tr=>tr.dataset.row);
}

async function viewPaper(pid, rowId) {
  if (!pid) return alert('No paper_id resolved for this row.');
  const d = await (await fetch('/api/paper/'+encodeURIComponent(pid))).json();
  document.getElementById('drawerTitle').textContent = pid;
  document.getElementById('drawerBody').innerHTML = renderPaperDetail(pid, d);
  // wire up the inner search-quote toggles (none yet, but room to grow)
  document.getElementById('drawer').classList.add('open');
  document.getElementById('backdrop').classList.add('open');
}
function closeDrawer(){
  document.getElementById('drawer').classList.remove('open');
  document.getElementById('backdrop').classList.remove('open');
}
document.addEventListener('keydown', (e)=>{ if (e.key==='Escape') closeDrawer(); });

function fmtNum(x, places){ if (x===null || x===undefined || x==='') return '—'; const n = parseFloat(x); return isNaN(n) ? x : n.toFixed(places===undefined?3:places); }
function confPill(c){ if(!c) return ''; return `<span class="conf-pill ${c}">${c}</span>`; }

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
  return `<div class="pd-section"><h3>O*NET mapping (pipeline's choice)</h3>
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
  const quotes = d['01a_quotes'] || null; // backend doesn't return this yet; we'll add it
  return `
    <div class="pd-meta">
      <h2>${escapeHtml(ext.title||pid)}</h2>
      <div class="authors">${escapeHtml((ext.authors||[]).join('; '))}${ext.year?` (${ext.year})`:''}${ext.venue?` · <i>${escapeHtml(ext.venue)}</i>`:''}</div>
      <div class="ids">
        <span class="pill">${escapeHtml(ext.citation_key||'')}</span>
        <span class="pill">${escapeHtml(pid)}.pdf</span>
        <a href="/api/pdf/${encodeURIComponent(pid)}" target="_blank">Open PDF →</a>
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

// ---------- upload flow ----------
let UPLOAD = {staged:null};
let POLL_TIMER = null;          // active setTimeout id
let POLL_JOB = null;            // job_id we're currently polling for
let UPLOAD_OPEN = false;        // modal visibility tracker
let ACTIVE_JOBS = JSON.parse(localStorage.getItem('activeJobs')||'[]'); // array of {job_id,paper_id}

function updateActiveJobs(jobs) {
  ACTIVE_JOBS = jobs;
  try { localStorage.setItem('activeJobs', JSON.stringify(jobs)); } catch(_){}
  refreshJobsPill();
}
function refreshJobsPill() {
  const pill = document.getElementById('jobsPill');
  if (!ACTIVE_JOBS.length) { pill.style.display = 'none'; return; }
  pill.style.display = '';
  pill.textContent = `${ACTIVE_JOBS.length} pipeline job${ACTIVE_JOBS.length>1?'s':''} running`;
}

function openUpload(html){
  document.getElementById('uploadBody').innerHTML = html;
  if (!UPLOAD_OPEN) {
    document.getElementById('uploadModal').style.display = 'flex';
    UPLOAD_OPEN = true;
  }
}
function closeUpload(){
  document.getElementById('uploadModal').style.display = 'none';
  UPLOAD_OPEN = false;
  if (POLL_TIMER) { clearTimeout(POLL_TIMER); POLL_TIMER = null; POLL_JOB = null; }
}

document.getElementById('uploadBtn').onclick = ()=> document.getElementById('uploadFile').click();
document.getElementById('uploadFile').onchange = async (e)=>{
  const file = e.target.files[0]; if (!file) return;
  openUpload(`<div class="help">Uploading <b>${escapeHtml(file.name)}</b> and checking for duplicates…</div>`);
  const fd = new FormData(); fd.append('pdf', file);
  let r;
  try {
    r = await (await fetch('/api/upload/check', {method:'POST', body: fd})).json();
  } catch (err) {
    openUpload(`<div class="err">Upload failed: ${escapeHtml(err.message)}</div>`); return;
  }
  if (!r.ok) { openUpload(`<div class="err">${escapeHtml(r.error)}</div>`); return; }
  UPLOAD.staged = r;
  e.target.value = ''; // reset for next upload
  renderDupCheck(r);
};

function renderDupCheck(r) {
  const dups = r.duplicate_candidates || [];
  let dupBlock;
  if (!dups.length) {
    dupBlock = `<div class="callout ok" style="margin:14px 0;">No likely duplicates found in the corpus.</div>`;
  } else {
    dupBlock = `<div class="callout warn" style="margin:14px 0 10px;">
      Found ${dups.length} possible duplicate${dups.length>1?'s':''} already in the corpus. Please check before running the pipeline.
    </div>` + dups.map((d, i) => `
      <div class="dup">
        <div style="flex:1;">
          <div class="t">${escapeHtml(d.title)}</div>
          <div class="m">
            <span class="mono">${escapeHtml(d.citation_key||'')}</span> ·
            ${escapeHtml((d.authors||[]).join('; '))}${d.year?` (${d.year})`:''} ·
            <b>${Math.round(d.score*100)}% title match</b> with <span class="mono">${escapeHtml(d.paper_id)}</span>
          </div>
        </div>
        <button onclick="viewPaper('${d.paper_id}','dup')" class="row-btn">View existing</button>
      </div>`).join('');
  }
  openUpload(`
    <div class="help">Detected title</div>
    <div style="font-family:var(--serif); font-size:17px; line-height:1.3;">${escapeHtml(r.extracted_title || '(none — title heuristic failed)')}</div>
    <div class="help" style="margin-top:4px;">Will be saved as <code>${escapeHtml(r.target_paper_id)}.pdf</code></div>
    ${dupBlock}
    <div style="display:flex; gap:10px; margin-top:18px; justify-content:flex-end;">
      <button class="btn" onclick="closeUpload()">Cancel and discard</button>
      <button class="btn btn-primary" onclick="confirmUploadRun()">
        ${dups.length ? 'Add anyway and run pipeline' : 'Run pipeline'}
      </button>
    </div>
    <p class="help" style="margin:12px 0 0;">The full pipeline takes a few minutes and costs roughly $0.13 in API calls.</p>
  `);
}

async function confirmUploadRun() {
  const r = UPLOAD.staged; if (!r) return;
  openUpload(`<div class="help">Starting pipeline for <code>${escapeHtml(r.target_paper_id)}</code>…</div>`);
  const res = await (await fetch('/api/upload/run', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({staged_id: r.staged_id, target_paper_id: r.target_paper_id})})).json();
  if (!res.ok) { openUpload(`<div class="err">${escapeHtml(res.error)}</div>`); return; }
  // remember this job so we can re-attach later
  const list = ACTIVE_JOBS.filter(j=>j.job_id !== res.job_id);
  list.unshift({job_id: res.job_id, paper_id: r.target_paper_id, started: Date.now()});
  updateActiveJobs(list);
  pollUpload(res.job_id, r.target_paper_id);
}

async function showJobsList() {
  // Fetch fresh job list and either reattach to a single running job or show a chooser.
  const r = await (await fetch('/api/upload/jobs')).json();
  const jobs = (r.jobs||[]);
  const running = jobs.filter(j => j.state==='running' || j.state==='queued');
  // sync localStorage with reality
  updateActiveJobs(running.map(j=>({job_id:j.job_id, paper_id:j.paper_id, started: (j.started||0)*1000})));
  if (running.length === 1) {
    pollUpload(running[0].job_id, running[0].paper_id);
    return;
  }
  // Multi-job picker
  const items = jobs.slice(0,15).map(j => `
    <div class="dup" style="align-items:center;">
      <div>
        <div class="mono">${escapeHtml(j.paper_id)}</div>
        <div class="m">${j.state} · job ${escapeHtml(j.job_id)}</div>
      </div>
      <button class="row-btn" onclick="pollUpload('${j.job_id}','${escapeHtml(j.paper_id)}')">Open</button>
    </div>`).join('');
  openUpload(`<div class="help" style="margin-bottom:8px;">Pipeline jobs (most recent 15)</div>${items || '<div class="help">No jobs yet.</div>'}`);
}

const STAGE_NAMES = ['extract_quotes','structure','classify_method','classify_outcome','map_onet','compute_speed','compute_quality'];

async function pollUpload(jobId, paperId) {
  POLL_JOB = jobId;
  // Ensure modal is open the first time the user explicitly opened a job
  if (!UPLOAD_OPEN) { document.getElementById('uploadModal').style.display='flex'; UPLOAD_OPEN=true; }
  const tick = async ()=>{
    // If user navigated away or switched to another job, stop polling for this one
    if (POLL_JOB !== jobId || !UPLOAD_OPEN) return;
    const r = await (await fetch('/api/upload/status/'+encodeURIComponent(jobId))).json();
    if (!r.ok) {
      if (UPLOAD_OPEN) openUpload(`<div class="err">${escapeHtml(r.error)}</div>`);
      return;
    }
    const state = r.job.state;
    const done = STAGE_NAMES.filter(s => r.stages[s]).length;
    const stageList = STAGE_NAMES.map(s =>
      `<div class="st ${r.stages[s]?'done':''}"><span style="width:14px;">${r.stages[s]?'✓':'·'}</span><span>${s}</span></div>`).join('');
    let footer = '';
    if (state === 'done') {
      const o = r.job.outcome || {};
      const added = (o.speed_added || o.quality_added);
      const reasons = (o.reasons||[]).map(s=>`<li>${escapeHtml(s)}</li>`).join('');
      const headline = added
        ? 'Pipeline complete. New row(s) were added to the table.'
        : 'Pipeline complete, but nothing was added to the table.';
      footer = `<div class="callout ${added?'ok':'warn'}" style="margin-top:12px;">
        <div><b>${headline}</b></div>
        ${reasons ? `<ul style="margin:6px 0 0; padding-left:20px;">${reasons}</ul>` : ''}
        <div style="margin-top:8px;">
          <button class="row-btn" onclick="closeUpload(); load();">${added?'Refresh table':'Back to table'}</button>
          <button class="row-btn" onclick="viewPaper('${paperId}','new')">View paper detail</button>
        </div></div>`;
      updateActiveJobs(ACTIVE_JOBS.filter(j=>j.job_id !== jobId));
    } else if (state === 'failed') {
      footer = `<div class="err">
        Pipeline failed: ${escapeHtml(r.job.error||'')}<br><span style="font-size:12px;">See <code>logs/upload_${escapeHtml(paperId)}.log</code></span></div>`;
      updateActiveJobs(ACTIVE_JOBS.filter(j=>j.job_id !== jobId));
    } else {
      footer = `<div class="help" style="margin-top:10px;">Running: ${done} of ${STAGE_NAMES.length} stages complete. Closing this window will not stop the job.</div>`;
    }
    if (UPLOAD_OPEN && POLL_JOB === jobId) {
      openUpload(`
        <div>Processing <code>${escapeHtml(paperId)}</code> <span class="pill">${state}</span></div>
        <div class="stage-list">${stageList}</div>
        ${footer}
      `);
    }
    if (state !== 'done' && state !== 'failed') {
      POLL_TIMER = setTimeout(tick, 2000);
    }
  };
  tick();
}

// On page load: sync active jobs from server and show pill
(async function syncJobsAtLoad() {
  try {
    const r = await (await fetch('/api/upload/jobs')).json();
    const running = (r.jobs||[]).filter(j => j.state==='running' || j.state==='queued');
    updateActiveJobs(running.map(j=>({job_id:j.job_id, paper_id:j.paper_id, started: (j.started||0)*1000})));
  } catch(_){}
})();

// column resizing — measure widths from <th>, write to <col>; swallow click so sort doesn't fire
(function initResizers(){
  const cols = document.querySelectorAll('#table colgroup col');
  const ths  = document.querySelectorAll('#table thead th');
  // restore saved widths first
  ths.forEach((th, i)=>{
    try { const w = localStorage.getItem('colw2_'+i); if (w && cols[i]) cols[i].style.width = w; } catch(_){}
  });
  document.querySelectorAll('#table th .resizer').forEach((r, i)=>{
    let dragging = false;
    r.addEventListener('mousedown', (e)=>{
      e.preventDefault(); e.stopPropagation();
      dragging = true;
      const col = cols[i], th = ths[i]; if (!col || !th) return;
      const startX = e.pageX;
      const startW = th.getBoundingClientRect().width;
      document.body.style.cursor = 'col-resize';
      const move = (ev)=>{
        const w = Math.max(30, Math.round(startW + (ev.pageX - startX)));
        col.style.width = w + 'px';
      };
      const up = ()=>{
        document.removeEventListener('mousemove', move);
        document.removeEventListener('mouseup', up);
        document.body.style.cursor = '';
        try{ localStorage.setItem('colw2_'+i, col.style.width); }catch(_){}
        setTimeout(()=>{ dragging=false; }, 0);
      };
      document.addEventListener('mousemove', move);
      document.addEventListener('mouseup', up);
    });
    // swallow the click so it doesn't reach the th sort handler
    r.addEventListener('click', (e)=>{ e.stopPropagation(); e.preventDefault(); });
  });
})();

load();
</script>
</body></html>
"""


RUN_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Run analysis · AI Impact Meta-Review</title>
__THEME_HEAD__
<style>
  :root { --page-w: 1080px; }
  .row { display:flex; gap:18px; margin-bottom:14px; align-items:baseline; }
  .row label { width:200px; flex:none; font-size:14px; color:var(--ink); font-weight:500; }
  .row .ctl { flex:1; }
  .row .ctl input[type=number], .row .ctl input[type=text], .row .ctl select { width:130px; }
  .row .help { margin-top:3px; }
  .actions { display:flex; gap:10px; align-items:center; margin-top:18px; }
  .status { font-size:13px; color:var(--ink-2); }
  .step { display:inline-flex; align-items:center; justify-content:center; width:22px; height:22px; border-radius:50%;
          background:var(--accent-soft); color:var(--accent); font-family:var(--sans); font-size:12px; font-weight:600; margin-right:8px; vertical-align:2px; }
  .inset { margin-top:16px; padding:14px 16px; border:1px solid var(--rule); border-radius:5px; background:var(--surface-2); }
  .inset .t { font-weight:600; margin-bottom:8px; }
  .summary-line { margin-top:12px; padding:10px 14px; background:var(--accent-soft); border-radius:4px; font-size:13.5px; color:var(--ink); }
  #heatmap, #barchart { overflow:auto; border:1px solid var(--rule-soft); border-radius:4px; padding:8px; background:var(--surface); }
</style></head><body>
__MASTHEAD__
<main class="page wrap">
  <section class="page-head">
    <div>
      <h1>Run an imputation analysis</h1>
      <p class="lede">Propagate the observed study effects across the O*NET occupation&ndash;activity graph to estimate impacts where no study exists.
        Runs use the live state of the studies table, including your edits.</p>
    </div>
  </section>
  <div id="liveCount" class="callout" style="margin:0 0 18px;">Checking current table…</div>

  <div class="card">
    <h2><span class="step">1</span>Outcome metric</h2>
    <div class="seg" id="metricSeg">
      <button data-m="speed" class="active">speed</button>
      <button data-m="quality">quality</button>
    </div>
    <div class="help" style="margin-top:8px;">A run targets one metric at a time. Switching swaps the observation columns consumed by the pipeline.</div>
  </div>

  <div class="card">
    <h2><span class="step">2</span>Scope: which parts of O*NET to include</h2>
    <p class="help" style="margin-top:0;">
      Each cell is the mean Stage-B weight of an activity within a SOC major group. Red markers and bold labels flag activities and SOC major groups
      with an observed effect for this metric. Shaded rows are excluded from the analysis.
      Click a row label or its checkbox to include or exclude it.
    </p>
    <div id="heatmap"></div>

    <div style="display:flex; gap:14px; margin:18px 0 8px; align-items:baseline; flex-wrap:wrap;">
      <label for="weightThreshold" style="font-weight:600;">Activity weight threshold</label>
      <input type="number" id="weightThreshold" value="10" step="0.5" min="0" style="width:90px;"/>
      <div class="help">Activities whose summed weight across the included occupations falls below this are dropped.</div>
    </div>
    <div id="barchart"></div>
    <div id="pruneSummary" class="summary-line"></div>

    <div class="inset">
      <div class="t">Unit of analysis</div>
      <div id="aggLevelSeg" class="seg">
        <button type="button" data-val="occupation" class="agg-seg active">Occupation (~894)</button>
        <button type="button" data-val="soc_minor" class="agg-seg">SOC minor group (~92)</button>
        <button type="button" data-val="soc_major" class="agg-seg">SOC major group (22)</button>
      </div>
      <input type="hidden" id="aggregationLevel" value="occupation"/>
      <div class="help" style="margin-top:8px;">
        Occupation (default): runs over the ~894 individual O*NET occupations.
        SOC minor: collapses to ~92 3-digit minor groups (e.g. 13-2000 Financial Specialists).
        SOC major: collapses to the 22 2-digit major groups.
        In aggregated modes, W is the per-group row-mean; observations are IV-weighted (or simple-mean fallback) within the group; AIOE baseline is averaged within the group.
      </div>
    </div>
  </div>

  <details class="card">
    <summary>Advanced parameters <span class="help" style="font-family:var(--sans); font-weight:400;">&nbsp;optional; defaults follow the methodology spec</span></summary>
    <div class="row">
      <label>Specificity (β)</label>
      <div class="ctl">
        <input type="number" id="beta" value="2.5" step="0.1" min="0.1"/>
        <div class="help">Stage B softmax temperature. Higher = each occupation concentrates on fewer activities. Default 2.5 ≈ 7 effective activities/occupation.</div>
      </div>
    </div>
    <div class="row">
      <label>Observation trust (Ω<sub>ref</sub>)</label>
      <div class="ctl">
        <input type="number" id="omega_ref" value="100" step="10"/>
        <div class="help">Stage D — reference precision; higher pulls estimates harder toward observed values.</div>
      </div>
    </div>
    <div id="baselineCard">
      <div class="row" style="margin-top:14px;">
        <label>AIOE baseline (speed only)</label>
        <div class="ctl">
          <input type="checkbox" id="use_baseline" checked/>
          <span class="help">Felten et al. AIOE, moment-matched to the observed mean and SD</span>
        </div>
      </div>
      <div class="row">
        <label>Baseline strength (Ω<sub>base</sub>)</label>
        <div class="ctl">
          <input type="number" id="omega_base" value="0.5" step="0.1" min="0"/>
          <div class="help">0.1 ≈ tiebreaker · 0.5 ≈ balanced · 1.0 ≈ AIOE-led · 5.0 ≈ AIOE-replaces-graph.</div>
        </div>
      </div>
    </div>
    <div class="row" style="margin-top:14px;"><label>σ<sub>ref</sub></label><div class="ctl"><input type="number" id="sigma_ref" value="0.1" step="0.01"/></div></div>
    <div class="row"><label>ε (regularizer)</label><div class="ctl"><input type="number" id="eps" value="0.000001" step="0.000001"/></div></div>
    <!-- legacy network-prune knobs, hidden (path retained for reproducibility) -->
    <input type="hidden" id="alpha" value="0.7"/>
    <input type="hidden" id="hops" value="4"/>
    <input type="hidden" id="c_occ" value="1.0"/>
    <input type="hidden" id="c_act" value="1.5"/>
    <input type="hidden" id="prune_activities" value="false"/>
  </details>

  <div class="card">
    <h2><span class="step">3</span>Run</h2>
    <div class="seg" id="runTypeSeg" style="margin-bottom:8px;">
      <button data-t="full" class="active">Full run</button>
      <button data-t="loo">Leave-one-out CV</button>
    </div>
    <div class="help" id="runTypeHelp">
      A full run produces occupation + activity impact estimates.
      LOO CV holds out each observation one at a time, re-solves, and reports how well the graph recovers the held-out value (MAE / RMSE / R²) and, more importantly, its rank among the observed nodes (Kendall τ-b, concordance, top-K precision).
    </div>
    <div class="actions">
      <button class="primary" id="runBtn">Run analysis</button>
      <button class="secondary" onclick="resetDefaults()">Reset to defaults</button>
      <span class="status" id="status"></span>
    </div>
    <div id="errBox"></div>
  </div>
</main>
<script>
function escapeHtml(s){return (s||'').toString().replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}

// Show live count of speed/quality rows that will feed the run (after deletes/merges)
(async function showLiveCount(){
  try {
    const r = await (await fetch('/api/rows')).json();
    const rows = (r.rows||[]).filter(x => !x._deleted);
    const speed = rows.filter(x => x.kind==='speed').length;
    const qual  = rows.filter(x => x.kind==='quality').length;
    const withOnet = rows.filter(x => x.onet_code).length;
    document.getElementById('liveCount').innerHTML =
      `The current table has <b>${speed}</b> speed and <b>${qual}</b> quality effects (deleted and merged rows excluded); <b>${withOnet}</b> have an O*NET code.`;
  } catch (e) {
    document.getElementById('liveCount').textContent = 'Could not load the current table.';
  }
})();

let METRIC='speed';
document.querySelectorAll('#metricSeg button').forEach(b=>{
  b.onclick=()=>{
    document.querySelectorAll('#metricSeg button').forEach(x=>x.classList.remove('active'));
    b.classList.add('active'); METRIC=b.dataset.m;
    document.getElementById('baselineCard').style.opacity = METRIC==='speed' ? 1 : 0.4;
    document.getElementById('use_baseline').disabled = METRIC!=='speed';
    document.getElementById('omega_base').disabled = METRIC!=='speed';
    loadHeatmap();  // observed overlays depend on metric
  };
});
document.querySelectorAll('#aggLevelSeg .agg-seg').forEach(b=>{
  b.onclick = () => {
    document.querySelectorAll('#aggLevelSeg .agg-seg').forEach(x=>x.classList.remove('active'));
    b.classList.add('active');
    document.getElementById('aggregationLevel').value = b.dataset.val;
  };
});
function resetDefaults() {
  document.getElementById('beta').value=2.5;
  document.getElementById('omega_ref').value=100;
  document.getElementById('use_baseline').checked=true;
  document.getElementById('omega_base').value=0.5;
  document.getElementById('sigma_ref').value=0.1;
  document.getElementById('eps').value=0.000001;
  document.getElementById('weightThreshold').value=10;
  EXCLUDED = new Set(HEAT ? HEAT.default_excluded_socs : []);
  renderCharts();
}

// ---------- Heatmap + bar chart ----------
let HEAT = null;
let EXCLUDED = new Set();

async function loadHeatmap() {
  const beta = parseFloat(document.getElementById('beta').value) || 2.5;
  const url = `/api/heatmap_data?metric=${METRIC}&beta=${beta}`;
  const r = await (await fetch(url)).json();
  if (!r.ok) { document.getElementById('heatmap').innerHTML = `<div class="err">${escapeHtml(r.error)}</div>`; return; }
  HEAT = r;
  if (EXCLUDED.size === 0) EXCLUDED = new Set(r.default_excluded_socs);
  renderCharts();
}

function vColor(v, vmax) {
  // viridis-ish ramp
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
  // vmax = 99th percentile of non-zero
  const vals = []; W.forEach(row => row.forEach(v => { if (v > 0) vals.push(v); }));
  vals.sort((a,b)=>a-b); const vmax = vals[Math.floor(vals.length*0.99)] || 1;

  // build cells
  let cellsSvg = '';
  for (let ri = 0; ri < rows.length; ri++) {
    const r = rows[ri];
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
  // gray overlay for deselected SOC rows
  let overlaySvg = '';
  rows.forEach((r, ri) => {
    if (EXCLUDED.has(r.code)) {
      overlaySvg += `<rect x="${labelW}" y="${labelH+ri*cell}" width="${cols.length*cell}" height="${cell}" fill="rgba(244,241,234,0.85)"/>`;
    }
  });
  // observed overlays
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
  // row labels with checkboxes (HTML overlay because checkboxes in SVG are clunky)
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
  // column labels rotated downward beneath the grid
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
  // Compute each activity's summed weight across NON-excluded SOC majors' occupations
  // total = Σ_g (W_g[j] * n_g)  for g not excluded
  const acts0 = HEAT.activities.map((a, j) => {
    let s = 0;
    HEAT.soc_groups.forEach((g, i) => {
      if (!EXCLUDED.has(g.code)) s += HEAT.weights[i][j] * g.n;
    });
    return {...a, j, total: s};
  });
  // Sort by total weight, descending
  const acts = acts0.slice().sort((x, y) => y.total - x.total);
  const sums = acts.map(a => a.total);
  const vmax = Math.max(...sums, 1);
  const barH = 14;
  const labelW = 240, plotW = 480, padL = 8, padT = 12;
  const height = padT + acts.length * barH + 16;

  let bars = '';
  acts.forEach((a, i) => {
    const w = (sums[i] / vmax) * plotW;
    const y = padT + i * barH;
    const below = sums[i] < threshold;
    const fill = a.observed ? '#23406a' : '#a9b4c2';   // navy = observed; gray-blue = other
    const opacity = below ? 0.35 : 1.0;
    bars += `<rect x="${labelW}" y="${y+1}" width="${w}" height="${barH-3}" fill="${fill}" opacity="${opacity}"/>`;
    bars += `<text x="${labelW + w + 4}" y="${y+barH-3}" font-size="9" fill="${below?'#a9a59c':'#474b53'}">${sums[i].toFixed(1)}</text>`;
    const tcol = below ? '#a9a59c' : '#1d1f23';
    bars += `<text x="${labelW-4}" y="${y+barH-3}" font-size="10" fill="${tcol}" text-anchor="end" font-weight="${a.observed?'600':'400'}">${escapeHtml(a.name)}</text>`;
  });
  // threshold line
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

  // Bottom summary: how many occupations and activities will feed the analysis
  let occTotal = 0, occIncluded = 0;
  HEAT.soc_groups.forEach(g => {
    occTotal += g.n;
    if (!EXCLUDED.has(g.code)) occIncluded += g.n;
  });
  const actTotal = HEAT.activities.length;
  // observed activities are always kept even if below threshold
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

document.getElementById('weightThreshold').addEventListener('input', renderBarChart);
document.getElementById('beta').addEventListener('change', loadHeatmap);
loadHeatmap();
let RUN_TYPE = 'full';
document.querySelectorAll('#runTypeSeg button').forEach(b=>{
  b.onclick = ()=>{
    document.querySelectorAll('#runTypeSeg button').forEach(x=>x.classList.remove('active'));
    b.classList.add('active'); RUN_TYPE = b.dataset.t;
    document.getElementById('runBtn').textContent = RUN_TYPE==='loo' ? 'Run LOO cross-validation' : 'Run analysis';
  };
});

document.getElementById('runBtn').onclick = async ()=>{
  const btn = document.getElementById('runBtn');
  const status = document.getElementById('status');
  const err = document.getElementById('errBox');
  btn.disabled = true;
  status.textContent = RUN_TYPE==='loo' ? 'Running leave-one-out CV (one solve per observation, about 1–2 min)…' : 'Running (about 30 s)…';
  err.innerHTML = '';
  const payload = {
    metric: METRIC,
    beta: parseFloat(document.getElementById('beta').value),
    aggregation_level: document.getElementById('aggregationLevel').value,
    manual_prune: true,
    excluded_soc_majors: Array.from(EXCLUDED),
    activity_weight_threshold: parseFloat(document.getElementById('weightThreshold').value),
    omega_ref: parseFloat(document.getElementById('omega_ref').value),
    use_baseline: document.getElementById('use_baseline').checked,
    omega_base: parseFloat(document.getElementById('omega_base').value),
    sigma_ref: parseFloat(document.getElementById('sigma_ref').value),
    eps: parseFloat(document.getElementById('eps').value),
  };
  const endpoint = RUN_TYPE==='loo' ? '/api/loo' : '/api/run';
  try {
    const r = await (await fetch(endpoint, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)})).json();
    if (!r.ok) throw new Error(r.error||'unknown');
    status.textContent = 'Done. Opening results…';
    location.href = (RUN_TYPE==='loo' ? '/loo/' : '/results/') + encodeURIComponent(r.run_id);
  } catch(e) {
    err.innerHTML = '<div class="err">'+escapeHtml(e.message)+'</div>';
    status.textContent=''; btn.disabled=false;
  }
};
</script>
</body></html>
"""


RESULTS_INDEX_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Past runs · AI Impact Meta-Review</title>
__THEME_HEAD__
<style>
  :root { --page-w: 1120px; }
  #t tbody tr { cursor:pointer; }
  #t td.id { font-family:var(--mono); font-size:11.5px; color:var(--ink-2); }
</style></head><body>
__MASTHEAD__
<main class="page wrap">
  <section class="page-head">
    <div>
      <h1>Past analysis runs</h1>
      <p class="lede">Every imputation run is stored with its parameters and outputs. Click a row to open it.
        To compare two full runs, tick their boxes and choose <i>Compare</i>. Leave-one-out runs (<span class="tag loo">LOO</span>)
        open a cross-validation report and can&rsquo;t be compared.</p>
    </div>
    <div class="page-actions">
      <button id="cmpBtn" class="btn" disabled>Compare selected (0/2)</button>
      <a class="btn btn-primary" href="/run">New run</a>
    </div>
  </section>
  <table id="t" class="data"><thead><tr>
    <th style="width:28px;"></th>
    <th>Run ID</th><th>Started (UTC)</th><th>Type</th><th>Metric</th><th>&beta;</th><th>&Omega;<sub>ref</sub></th><th>Baseline</th>
    <th>Observed (occ/act)</th><th>Kept (occ/act)</th>
  </tr></thead><tbody id="tb"></tbody></table>
</main>
<script>
const SELECTED = new Set();
function refreshCompareBtn() {
  const btn = document.getElementById('cmpBtn');
  const n = SELECTED.size;
  btn.textContent = `Compare selected (${n}/2)`;
  btn.disabled = n !== 2;
  btn.classList.toggle('btn-primary', n === 2);
}
function toggleSel(id, checked) {
  if (checked) SELECTED.add(id); else SELECTED.delete(id);
  refreshCompareBtn();
}
document.getElementById('cmpBtn').onclick = () => {
  if (SELECTED.size !== 2) return;
  const [a, b] = [...SELECTED];
  location.href = `/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`;
};
fetch('/api/runs').then(r=>r.json()).then(d=>{
  const tb = document.getElementById('tb');
  if (!d.runs.length) { tb.innerHTML='<tr><td colspan="10" style="text-align:center; padding:28px; color:var(--muted);">No runs yet. <a href="/run">Start one</a>.</td></tr>'; return; }
  tb.innerHTML = d.runs.map(r=>{
    const p = r.params||{};
    const isLoo = r.type === 'loo';
    const typeTag = isLoo
      ? '<span class="tag loo">LOO</span>'
      : '<span class="help">Full</span>';
    const href = isLoo ? '/loo/' + encodeURIComponent(r.run_id) : '/results/' + encodeURIComponent(r.run_id);
    const cb = isLoo
      ? '<span title="LOO runs cannot be compared" style="color:var(--rule);">—</span>'
      : `<input type="checkbox" onclick="event.stopPropagation(); toggleSel('${r.run_id}', this.checked);"/>`;
    return `<tr onclick="if (event.target.tagName!=='INPUT') location.href='${href}'">
      <td onclick="event.stopPropagation();">${cb}</td>
      <td class="id">${r.run_id}</td>
      <td>${(r.started_utc||'').replace('T',' ').slice(0,19)}</td>
      <td>${typeTag}</td>
      <td><span class="pill ${p.metric}">${p.metric||''}</span></td>
      <td class="num">${p.beta}</td><td class="num">${p.omega_ref}</td>
      <td>${r.baseline_active ? `AIOE, Ω<sub>b</sub>=${p.omega_base}` : '—'}</td>
      <td class="num">${r.n_observed_occ} / ${r.n_observed_act}</td>
      <td class="num">${r.n_kept_occ} / ${r.n_kept_act}</td>
    </tr>`;
  }).join('');
});
</script></body></html>
"""


RESULTS_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Run results · AI Impact Meta-Review</title>
__THEME_HEAD__
<style>
  .bar { display:inline-block; height:6px; vertical-align:middle; border-radius:2px; }
  .bar.pos { background:var(--pos); } .bar.neg { background:var(--neg); }
  #t th { cursor:pointer; user-select:none; }
  .chart-panel .chart-sub { margin:-4px 0 12px; }
  .chart-svg { width:100%; display:block; overflow:visible; }
  .chart-svg .lbl { font-family:var(--sans); font-size:12px; fill:#1d1f23; }
  .chart-svg .val { font-family:var(--mono); font-size:10.5px; fill:#474b53; }
  .chart-svg .axis { stroke:#9a968d; stroke-width:1; }
  .chart-svg .tick { font-family:var(--sans); font-size:11px; fill:#787c84; }
  .chart-svg .grid { stroke:#eeebe4; stroke-width:1; }
  .chart-svg .bar.observed { fill:#23406a; } .chart-svg .bar.imputed { fill:#b9c1cc; }
  .chart-svg .bar.observed.neg { fill:#9c2f2f; } .chart-svg .bar.imputed.neg { fill:#e0bdb8; }
  .chart-svg g:hover .bar { opacity:0.8; }
  .sw.observed { background:#23406a; } .sw.imputed { background:#b9c1cc; }
</style></head><body>
__MASTHEAD__
<main class="page wrap">
  <section class="page-head">
    <div>
      <div class="crumbs"><a href="/results">Past runs</a> / <span class="mono">__RUN_ID__</span></div>
      <h1>Run results</h1>
      <p class="page-meta" id="stats" style="margin:6px 0 0;">…</p>
    </div>
    <div class="page-actions">
      <button class="btn" onclick="exportCsv()">Download this view (CSV)</button>
      <a class="btn" href="/run">New run</a>
    </div>
  </section>
  <div id="warn"></div>
  <div class="card chart-panel">
    <h2 id="chartTitle">…</h2>
    <div class="help chart-sub" id="chartSub"></div>
    <div id="chartWrap"></div>
    <div class="chart-legend"><span><span class="sw observed"></span>Observed in at least one study</span><span><span class="sw imputed"></span>Imputed</span><span>Darker red / lighter red: negative estimates</span></div>
  </div>
  <div class="toolbar">
    <div class="seg" id="viewSeg">
      <button data-v="occ" class="active">Occupations</button>
      <button data-v="act">Activities</button>
    </div>
    <label>Posterior SD &le; <input type="number" id="stdFilter" value="5" step="0.05" style="width:70px;"/></label>
    <label><input type="checkbox" id="onlyObserved"/> Observed only</label>
    <input type="search" id="search" placeholder="Search title or code…" style="margin-left:auto; min-width:240px;"/>
  </div>
  <table id="t" class="data"><thead id="th"></thead><tbody id="tb"></tbody></table>
</main>
<script>
const RUN_ID = "__RUN_ID__";
let DATA = null, VIEW='occ', SORT={col:'estimate', dir:-1};

const SOC_NAMES = {
  "11":"Management","13":"Business & Financial","15":"Computer & Math",
  "17":"Architecture & Engineering","19":"Life, Physical, Social Science",
  "21":"Community & Social Service","23":"Legal","25":"Education",
  "27":"Arts, Design, Media","29":"Healthcare Practitioners","31":"Healthcare Support",
  "33":"Protective Service","35":"Food Prep & Serving","37":"Building & Grounds Cleaning",
  "39":"Personal Care & Service","41":"Sales","43":"Office & Admin Support",
  "45":"Farming, Fishing, Forestry","47":"Construction & Extraction",
  "49":"Installation, Maint, Repair","51":"Production","53":"Transportation & Material Moving"
};

function fmtNum(x, p){ if (x===null||x===undefined||x==='') return '—'; const n=parseFloat(x); return isNaN(n)?x:n.toFixed(p===undefined?3:p); }
function escapeHtml(s){return (s||'').toString().replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function hasObs(r){ return !(r.observed===null||r.observed===undefined||r.observed===''||isNaN(parseFloat(r.observed))); }
function mean(xs){ return xs.length ? xs.reduce((a,b)=>a+b,0)/xs.length : 0; }

async function load(){
  const r = await (await fetch('/api/results/'+encodeURIComponent(RUN_ID))).json();
  if (!r.ok) { document.querySelector('main').innerHTML = '<div class="err" style="margin-top:30px;">'+escapeHtml(r.error)+'</div>'; return; }
  DATA = r.data;
  const p = DATA.params;
  document.getElementById('stats').textContent =
    `Metric: ${p.metric} · β = ${p.beta} · Ω_ref = ${p.omega_ref}` +
    (DATA.baseline_active ? ` · AIOE baseline Ω_b = ${p.omega_base}` : '') +
    ` · kept ${DATA.n_kept_occ} occupations, ${DATA.n_kept_act} activities`;
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
    if (r.posterior_std > stdMax) return false;
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
  }).join('') + (VIEW==='occ'?'<th>Effect</th>':'<th>Effect</th>') + '</tr>';

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
      if (c==='aioe_baseline') return `<td class="num">${r.aioe_baseline===undefined?'':fmtNum(r.aioe_baseline)}</td>`;
      if (c==='n_studies') return `<td class="num">${r.n_studies===null||r.n_studies===undefined||r.n_studies===''?'':parseInt(r.n_studies)}</td>`;
      if (c==='code') return `<td class="mono" style="font-size:12px; color:var(--ink-2);">${escapeHtml(r[c]||'')}</td>`;
      return `<td>${escapeHtml(r[c]||'')}</td>`;
    }).join('') + `<td>${bar}</td></tr>`;
  }).join('');

  renderChart();
}

function buildChartRows(){
  const isOcc = VIEW === 'occ';
  const agg = (DATA.params && DATA.params.aggregation_level) || 'occupation';
  const alreadyAgg = DATA.socmajor_aggregated || agg === 'soc_major' || agg === 'soc_minor';
  if (isOcc && !alreadyAgg) {
    // Aggregate the 894 occupation-level estimates into 22 SOC major groups
    // so the chart is legible. Each group's estimate is the mean over its
    // member occupations; observed value is the mean over observed members
    // (imputed if none are observed).
    const groups = {};
    for (const r of DATA.occupation_impacts) {
      const g = (r.code||'').split('-')[0];
      if (!SOC_NAMES[g]) continue;
      if (!groups[g]) groups[g] = {est:[], obs:[]};
      const e = parseFloat(r.estimate);
      if (!isNaN(e)) groups[g].est.push(e);
      if (hasObs(r)) groups[g].obs.push(parseFloat(r.observed));
    }
    return Object.entries(groups).map(([code, g]) => ({
      label: SOC_NAMES[code],
      code, estimate: mean(g.est),
      observed: g.obs.length ? mean(g.obs) : null,
      isObserved: g.obs.length > 0,
      n_obs: g.obs.length,
    }));
  }
  if (isOcc) {
    return DATA.occupation_impacts.map(r => ({
      label: r.title || r.code, code: r.code,
      estimate: parseFloat(r.estimate)||0,
      observed: hasObs(r) ? parseFloat(r.observed) : null,
      isObserved: hasObs(r),
    }));
  }
  return DATA.activity_impacts.map(r => ({
    label: r.activity, code: null,
    estimate: parseFloat(r.estimate)||0,
    observed: hasObs(r) ? parseFloat(r.observed) : null,
    isObserved: hasObs(r),
  }));
}

function renderChart(){
  const rows = buildChartRows().slice().sort((a,b) => b.estimate - a.estimate);
  const metric = (DATA.params.metric||'speed').toLowerCase();
  const metricNoun = metric === 'quality' ? 'quality gain (Hedges\\' g)' : 'speed gain (log points)';
  const isOcc = VIEW === 'occ';
  const agg = (DATA.params && DATA.params.aggregation_level) || 'occupation';
  const alreadyAgg = DATA.socmajor_aggregated || agg === 'soc_major' || agg === 'soc_minor';
  const title = isOcc
    ? (alreadyAgg ? 'By SOC major group' : 'By SOC major group (mean over occupations)')
    : 'By work activity';
  document.getElementById('chartTitle').textContent = `Estimated ${metricNoun} — ${title}`;
  document.getElementById('chartSub').textContent = isOcc && !alreadyAgg
    ? `Occupation-level results averaged into ${rows.length} SOC major groups for display; observed = mean over group members with an observed value.`
    : `${rows.length} rows, sorted by estimate.`;

  // Layout
  const rowH = 22;
  const labelW = 340;
  const rightPad = 60;
  const topPad = 8;
  const bottomPad = 30;
  const chartW = 640; // px reserved for bars+axis
  const totalW = labelW + chartW + rightPad;
  const totalH = topPad + rows.length * rowH + bottomPad;

  const maxV = Math.max(0.1, ...rows.map(r => Math.max(0, r.estimate)));
  const minV = Math.min(0, ...rows.map(r => r.estimate));
  const x0 = labelW - minV / (maxV - minV) * chartW;  // x for value=0
  const xScale = v => labelW + (v - minV) / (maxV - minV) * chartW;

  // Nice ticks
  function niceTicks(lo, hi, n){
    const range = hi - lo || 1;
    const step = Math.pow(10, Math.floor(Math.log10(range/n)));
    const err = n * step / range;
    let mult = 1;
    if (err <= 0.15) mult = 10;
    else if (err <= 0.35) mult = 5;
    else if (err <= 0.75) mult = 2;
    const s = mult * step;
    const t0 = Math.ceil(lo / s) * s;
    const out = [];
    for (let v = t0; v <= hi + 1e-9; v += s) out.push(Math.round(v/s)*s);
    return out;
  }
  const ticks = niceTicks(minV, maxV, 5);

  const bars = rows.map((r, i) => {
    const y = topPad + i * rowH;
    const cls = (r.isObserved ? 'observed' : 'imputed') + (r.estimate < 0 ? ' neg' : '');
    const xVal = xScale(r.estimate);
    const xZero = xScale(0);
    const x = Math.min(xVal, xZero);
    const w = Math.abs(xVal - xZero);
    const label = escapeHtml(r.label);
    const val = fmtNum(r.estimate);
    const valX = r.estimate >= 0 ? xVal + 4 : xVal - 4;
    const valAnchor = r.estimate >= 0 ? 'start' : 'end';
    const title = `${r.label}: ${val}${r.isObserved ? ` (observed${r.n_obs?`, n=${r.n_obs}`:''})` : ' (imputed)'}`;
    return `
      <g>
        <title>${escapeHtml(title)}</title>
        <text class="lbl" x="${labelW - 8}" y="${y + rowH/2 + 4}" text-anchor="end">${label}</text>
        <rect class="bar ${cls}" x="${x}" y="${y + 3}" width="${Math.max(0.5, w)}" height="${rowH - 6}" rx="1.5"/>
        <text class="val" x="${valX}" y="${y + rowH/2 + 4}" text-anchor="${valAnchor}">${val}</text>
      </g>`;
  }).join('');

  const axisY = topPad + rows.length * rowH + 2;
  const gridlines = ticks.map(t => {
    const x = xScale(t);
    return `<line class="grid" x1="${x}" y1="${topPad}" x2="${x}" y2="${axisY}"/>`;
  }).join('');
  const tickLbls = ticks.map(t => {
    const x = xScale(t);
    return `<text class="tick" x="${x}" y="${axisY + 14}" text-anchor="middle">${t.toFixed(2)}</text>`;
  }).join('');
  const zeroLine = `<line class="axis" x1="${xScale(0)}" y1="${topPad}" x2="${xScale(0)}" y2="${axisY}"/>`;
  const axisLine = `<line class="axis" x1="${labelW}" y1="${axisY}" x2="${labelW + chartW}" y2="${axisY}"/>`;
  const xLbl = `<text class="tick" x="${labelW + chartW/2}" y="${axisY + 26}" text-anchor="middle">Estimated ${metricNoun}</text>`;

  document.getElementById('chartWrap').innerHTML =
    `<svg class="chart-svg" viewBox="0 0 ${totalW} ${totalH + 10}" preserveAspectRatio="xMinYMin meet">
      ${gridlines}${bars}${zeroLine}${axisLine}${tickLbls}${xLbl}
    </svg>`;
}

function sortBy(col){ SORT.dir = SORT.col===col ? -SORT.dir : -1; SORT.col=col; render(); }

function exportCsv() {
  const rows = (VIEW==='occ' ? DATA.occupation_impacts : DATA.activity_impacts);
  const keys = Object.keys(rows[0]||{});
  const csv = [keys.join(',')].concat(rows.map(r => keys.map(k => JSON.stringify(r[k]??'')).join(','))).join('\\n');
  const blob = new Blob([csv], {type:'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${RUN_ID}_${VIEW}.csv`; a.click();
}

load();
</script></body></html>
"""


TRANSITIONS_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Occupational transitions · AI Impact Meta-Review</title>
__THEME_HEAD__
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
  /* Left axis: fixed-height rows so they line up with cells; truncate overflow. */
  .axis.left .lbl {
    height: var(--cell);
    width: var(--label-w);
    box-sizing: border-box;
    padding: 0 6px 0 6px;
    text-align: right;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    line-height: var(--cell);
    border-bottom: 1px solid transparent;
    cursor: pointer;
  }
  /* Top axis: fixed-width columns; rotated text. Use a wrapper so the rotated
     text doesn't blow up the column width. */
  .axis.top .lbl {
    width: var(--cell);
    height: var(--label-h);
    box-sizing: border-box;
    overflow: hidden;
    position: relative;
    border-right: 1px solid transparent;
    cursor: pointer;
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
  .est-pos { color:var(--pos); }
  .est-neg { color:var(--neg); }
  .est-na { color:var(--muted); }
  .empty { color:var(--muted); font-size:14px; padding:24px 8px; text-align:center; font-style:italic; font-family:var(--serif); }
  .search { display:flex; gap:8px; margin-bottom:10px; }
  .search input { flex:1; }
  .results { position:relative; }
  .typeahead { position:absolute; background:var(--surface); border:1px solid var(--rule); border-radius:5px; max-height:260px; overflow:auto; z-index:50; left:0; right:90px; top:40px; box-shadow:0 10px 30px rgba(29,31,35,0.14); display:none; }
  .typeahead .item { padding:6px 10px; font-size:13px; cursor:pointer; border-bottom:1px solid var(--rule-soft); }
  .typeahead .item:hover { background:var(--accent-soft); }
  .typeahead .item .code { color:var(--muted); font-size:11px; font-family:var(--mono); }
</style></head><body>
__MASTHEAD__
<div class="wrap wide" style="padding:0 28px;">
  <section class="page-head">
    <div>
      <h1>Occupational transitions</h1>
      <p class="lede">How often workers move from one occupation to another. Rows are source SOC minor groups and columns are
        destinations; each row sums to one.</p>
    </div>
    <span class="page-meta" id="metaTxt">Loading…</span>
  </section>
</div>
<div id="main">
  <div class="panel">
    <h2>Transition shares by SOC minor group</h2>
    <div class="search results">
      <input id="search" placeholder="Search occupation (title or SOC code)…" autocomplete="off"/>
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
const CELL = 14;            // px per minor-group cell
const LABEL_W = 240;        // left label column
const LABEL_H = 240;        // top label row
let DATA = null;
let MINOR_IDX = {};         // minor code -> index
let SEL_MINOR = -1;
let SEL_OCC = -1;

async function load() {
  const r = await fetch('/api/transitions/data');
  DATA = await r.json();
  DATA.minors.forEach((m,i)=>{ MINOR_IDX[m.code]=i; });
  document.getElementById('metaTxt').textContent =
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
  // Per-cell fill with sqrt scale.
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
  // SOC major group dividers.
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
  // Highlight selected row.
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
  // Expose CELL / LABEL sizes as CSS vars so the .lbl rules pick them up.
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

// ---- Drill: minor group panel ----
function showMinor(mIdx) {
  SEL_MINOR = mIdx; SEL_OCC = -1;
  const m = DATA.minors[mIdx];
  const occIdxs = DATA.minor_to_occs[m.code] || [];
  // Sort member occupations by their source total_obs desc.
  const sorted = occIdxs.slice().sort((a,b) => (DATA.totals[b]||0) - (DATA.totals[a]||0));
  const occRows = sorted.map(i => {
    const o = DATA.occs[i];
    return `<tr class="row-click" data-occ="${i}">
      <td>${escapeHtml(o.title)}<div style="color:var(--muted);font-size:11px;" class="mono">${o.code}</div></td>
      <td class="num">${(DATA.totals[i]||0).toLocaleString(undefined,{maximumFractionDigits:0})}</td>
      <td class="num">${estFmt(o.code)}</td>
    </tr>`;
  }).join('');
  // Top outgoing & incoming at the MINOR level.
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
  drill.querySelectorAll('tr[data-occ]').forEach(tr => {
    tr.onclick = () => showOcc(parseInt(tr.dataset.occ));
  });
  drill.querySelectorAll('tr[data-minor]').forEach(tr => {
    tr.onclick = () => showMinor(parseInt(tr.dataset.minor));
  });
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

// ---- Drill: occupation panel ----
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
  document.getElementById('backToMinor').onclick =
    () => showMinor(MINOR_IDX[occ.minor]);
  drill.querySelectorAll('tr[data-occ]').forEach(tr => {
    tr.onclick = () => showOcc(parseInt(tr.dataset.occ));
  });
  // Highlight the minor group on the heatmap.
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

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

// Heatmap hover + click.
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

  // Search typeahead over occupations.
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


LOO_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Leave-one-out CV · AI Impact Meta-Review</title>
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
__MASTHEAD__
<main class="page wrap">
  <section class="page-head">
    <div>
      <div class="crumbs"><a href="/results">Past runs</a> / <span class="mono">__RUN_ID__</span></div>
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
const RUN_ID = "__RUN_ID__";
let DATA = null, FILTER='all', SORT={col:'abs_rank_delta', dir:-1};

function fmt(x,p){ if(x==null) return '—'; const n=parseFloat(x); return isNaN(n) ? x : n.toFixed(p==null?3:p); }
function escapeHtml(s){return (s||'').toString().replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}

async function load() {
  const r = await (await fetch('/api/loo/'+encodeURIComponent(RUN_ID))).json();
  if (!r.ok) { document.querySelector('main').innerHTML='<div class="err" style="margin-top:30px;">'+escapeHtml(r.error)+'</div>'; return; }
  DATA = r.data;
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


COMPARE_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Compare runs · AI Impact Meta-Review</title>
__THEME_HEAD__
<style>
  .rr { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
  .rr .col { background:var(--surface-2); border-radius:5px; padding:10px 14px; font-size:13px; }
  .rr .col .id { font-family:var(--mono); font-size:11.5px; color:var(--muted); }
  .rr .col .kv { color:var(--ink-2); margin-top:4px; line-height:1.55; }
  .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap:10px; }
  .card h2 + .seg { margin-bottom:8px; }
  .card table { border-top:1.5px solid var(--ink); border-bottom:1.5px solid var(--ink); }
  .card table th { cursor:pointer; user-select:none; }
  .card table.topk-table { border:0; background:transparent !important; }
  .topk-table td, .topk-table th { padding:4px 12px; font-family:var(--mono); font-size:12px; }
  svg.scatter { display:block; background:var(--surface); }
  svg.scatter .axis { stroke:#9a968d; stroke-width:1; }
  svg.scatter .grid { stroke:#eeebe4; stroke-width:1; }
  svg.scatter .tick { font-family:var(--sans); font-size:11px; fill:#787c84; }
  svg.scatter .diag { stroke:#1d1f23; stroke-width:1; stroke-dasharray:4 3; }
  svg.scatter circle { fill:#2e67a8; opacity:0.6; stroke:#fff; stroke-width:1; }
</style></head><body>
__MASTHEAD__
<div class="wrap">
  <section class="page-head">
    <div>
      <div class="crumbs"><a href="/results">Past runs</a> / compare</div>
      <h1>Compare two runs</h1>
      <p class="lede">How much do the imputed rankings change between two parameter settings?</p>
    </div>
    <span class="page-meta" id="stats">Loading…</span>
  </section>
</div>
<main id="body" class="page wrap"><div class="card help">Loading…</div></main>
<script>
const qs = new URLSearchParams(location.search);
const A = qs.get('a'), B = qs.get('b');

function fmt(x, p){ if (x==null) return '—'; const n=parseFloat(x); return isNaN(n)?x:n.toFixed(p==null?3:p); }
function fmtP(p){ if (p==null || isNaN(parseFloat(p))) return ''; const n=parseFloat(p); return n < 0.001 ? 'p < 0.001' : 'p = ' + n.toFixed(3); }
function escapeHtml(s){return (s||'').toString().replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}

let DATA=null, VIEW='occupation', SORT={col:'rank_delta', dir:-1};

async function load() {
  if (!A || !B) { document.getElementById('body').innerHTML='<div class="err">Two run IDs are needed (?a=…&amp;b=…). Pick them on the <a href="/results">past runs</a> page.</div>'; return; }
  const r = await (await fetch(`/api/compare?a=${encodeURIComponent(A)}&b=${encodeURIComponent(B)}`)).json();
  if (!r.ok) { document.getElementById('body').innerHTML='<div class="err">'+escapeHtml(r.error||'error')+'</div>'; return; }
  DATA = r.data;
  document.getElementById('stats').textContent = `A=${A.slice(-6)} · B=${B.slice(-6)}`;
  render();
}

function paramsLine(m) {
  const p = m.params||{};
  return `${p.metric||'?'} · β=${p.beta} · Ω_ref=${p.omega_ref}` +
    (m.baseline_active ? ` · Ω_b=${p.omega_base}` : '') +
    ` · agg=${p.aggregation_level||'occupation'}` +
    ` · kept ${m.n_kept_occ}/${m.n_kept_act}`;
}

function render() {
  const body = document.getElementById('body');
  const views = DATA.views || {};
  const viewList = Object.keys(views);
  if (!viewList.length) { body.innerHTML='<div class="card">No comparable outputs.</div>'; return; }
  if (!views[VIEW]) VIEW = viewList[0];
  const v = views[VIEW];
  const m = v.metrics || {};
  const levelBanner = DATA.levels_differ
    ? `<div class="callout warn" style="margin-top:10px;">
         ⚠ Aggregation levels differ: A is <b>${escapeHtml(DATA.level_a)}</b>, B is <b>${escapeHtml(DATA.level_b)}</b>.
         Occupations from the finer-grained run are rolled up to <b>${escapeHtml(DATA.common_level)}</b>
         (mean estimate within each ${escapeHtml(DATA.common_level)} group) before comparing.
       </div>`
    : `<div class="help" style="margin-top:8px;">Both runs use aggregation level <b>${escapeHtml(DATA.common_level)}</b>.</div>`;
  body.innerHTML = `
    <div class="card">
      <h2>Runs being compared</h2>
      <div class="rr">
        <div class="col"><div class="id">A · ${escapeHtml(DATA.a.run_id)}</div><div class="kv">${escapeHtml(paramsLine(DATA.a))}</div></div>
        <div class="col"><div class="id">B · ${escapeHtml(DATA.b.run_id)}</div><div class="kv">${escapeHtml(paramsLine(DATA.b))}</div></div>
      </div>
      ${levelBanner}
    </div>

    <div class="card">
      <h2>View</h2>
      <div class="seg" id="viewSeg">
        ${viewList.map(k => `<button data-v="${k}" class="${k===VIEW?'active':''}">${k}s (${views[k].n})</button>`).join('')}
      </div>

      <h2 style="margin-top:14px;">Ranking agreement (${VIEW}s)</h2>
      <div class="grid">
        <div class="metric"><div class="k">n compared</div><div class="v">${m.n||0}</div></div>
        <div class="metric"><div class="k">Spearman ρ</div><div class="v">${m.spearman_r!=null?fmt(m.spearman_r,3):'—'}</div><div class="p">${fmtP(m.spearman_p)}</div></div>
        <div class="metric"><div class="k">Kendall τ</div><div class="v">${m.kendall_tau!=null?fmt(m.kendall_tau,3):'—'}</div><div class="p">${fmtP(m.kendall_p)}</div></div>
        <div class="metric"><div class="k">Pearson r</div><div class="v">${m.pearson_r!=null?fmt(m.pearson_r,3):'—'}</div><div class="p">${fmtP(m.pearson_p)}</div></div>
      </div>

      <h2 style="margin-top:14px;">Top-K overlap (Jaccard on top-K sets)</h2>
      <table class="topk-table" style="width:auto;">
        <thead><tr><th>K</th><th>intersection</th><th>Jaccard</th></tr></thead>
        <tbody>
          ${Object.entries(v.topk_overlap||{}).map(([k,o]) =>
            `<tr><td>${k}</td><td>${o.intersection}</td><td>${fmt(o.jaccard,3)}</td></tr>`).join('') || '<tr><td colspan="3">—</td></tr>'}
        </tbody>
      </table>
    </div>

    <div class="card">
      <h2>Estimate A vs estimate B (${VIEW}s)</h2>
      <div id="scatterWrap"></div>
      <div class="help" style="margin-top:6px;">Dashed line: identical estimates (y = x).</div>
    </div>

    <div class="card">
      <h2>Row-by-row comparison</h2>
      <input type="search" id="search" placeholder="Search label or code…" style="margin-bottom:10px; width:280px;"/>
      <table><thead id="th"></thead><tbody id="tb"></tbody></table>
    </div>
  `;
  document.querySelectorAll('#viewSeg button').forEach(b=>{
    b.onclick = ()=>{ VIEW=b.dataset.v; render(); };
  });
  document.getElementById('search').addEventListener('input', renderTable);
  renderScatter();
  renderTable();
}

function renderScatter() {
  const v = DATA.views[VIEW];
  const rows = v.rows;
  const pad = {l:60, r:20, t:16, b:44};
  const W = 620, H = 380;
  const xs = rows.map(r=>parseFloat(r.estimate_a));
  const ys = rows.map(r=>parseFloat(r.estimate_b));
  const lo = Math.min(...xs, ...ys, 0);
  const hi = Math.max(...xs, ...ys, 0.001);
  const range = hi - lo || 1;
  const xL = lo - range*0.05, xH = hi + range*0.05;
  const sx = v => pad.l + (v - xL) / (xH - xL) * (W - pad.l - pad.r);
  const sy = v => H - pad.b - (v - xL) / (xH - xL) * (H - pad.t - pad.b);
  function ticks(a,b,n){
    const r=b-a||1, step=Math.pow(10,Math.floor(Math.log10(r/n))); const err=n*step/r;
    let m=1; if(err<=0.15)m=10; else if(err<=0.35)m=5; else if(err<=0.75)m=2;
    const s=m*step; const t0=Math.ceil(a/s)*s; const out=[];
    for(let v=t0; v<=b+1e-9; v+=s) out.push(Math.round(v/s)*s);
    return out;
  }
  const xt = ticks(xL, xH, 6);
  const gridLines = xt.map(t=>{
    const x=sx(t), y=sy(t);
    return `<line class="grid" x1="${x}" y1="${pad.t}" x2="${x}" y2="${H-pad.b}"/>` +
           `<line class="grid" x1="${pad.l}" y1="${y}" x2="${W-pad.r}" y2="${y}"/>`;
  }).join('');
  const xTicks = xt.map(t=>`<text class="tick" x="${sx(t)}" y="${H-pad.b+14}" text-anchor="middle">${t.toFixed(2)}</text>`).join('');
  const yTicks = xt.map(t=>`<text class="tick" x="${pad.l-6}" y="${sy(t)+3}" text-anchor="end">${t.toFixed(2)}</text>`).join('');
  const diag = `<line class="diag" x1="${sx(xL)}" y1="${sy(xL)}" x2="${sx(xH)}" y2="${sy(xH)}"/>`;
  const label = (r) => (v.labelcol === v.keycol) ? (r[v.keycol]||'') : (r[v.labelcol]||r[v.keycol]||'');
  const pts = rows.map(r => {
    const title = `${label(r)} — A ${fmt(r.estimate_a,3)}, B ${fmt(r.estimate_b,3)}, Δ ${fmt(r.delta,3)}`;
    return `<circle cx="${sx(parseFloat(r.estimate_a))}" cy="${sy(parseFloat(r.estimate_b))}" r="4"><title>${escapeHtml(title)}</title></circle>`;
  }).join('');
  const xAxis = `<line class="axis" x1="${pad.l}" y1="${H-pad.b}" x2="${W-pad.r}" y2="${H-pad.b}"/>`;
  const yAxis = `<line class="axis" x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${H-pad.b}"/>`;
  const xLabel = `<text class="tick" x="${(pad.l+W-pad.r)/2}" y="${H-6}" text-anchor="middle">estimate A</text>`;
  const yLabel = `<text class="tick" x="${-H/2}" y="14" text-anchor="middle" transform="rotate(-90)">estimate B</text>`;
  document.getElementById('scatterWrap').innerHTML =
    `<svg class="scatter" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
      ${gridLines}${diag}${pts}${xAxis}${yAxis}${xTicks}${yTicks}${xLabel}${yLabel}
    </svg>`;
}

function renderTable() {
  const v = DATA.views[VIEW];
  const q = (document.getElementById('search').value||'').toLowerCase();
  let rows = v.rows.slice();
  if (q) {
    rows = rows.filter(r => {
      const blob = (r[v.labelcol]||'') + ' ' + (r[v.keycol]||'');
      return blob.toLowerCase().includes(q);
    });
  }
  rows.sort((a,b) => {
    let x=a[SORT.col], y=b[SORT.col];
    if (typeof x==='string' || typeof y==='string') return ((x||'')+'').localeCompare((y||'')+'') * SORT.dir;
    return ((parseFloat(x)||0) - (parseFloat(y)||0)) * SORT.dir;
  });
  const keycol = v.keycol, labelcol = v.labelcol;
  const cols = keycol === labelcol
    ? [keycol, 'estimate_a', 'rank_a', 'estimate_b', 'rank_b', 'delta', 'rank_delta']
    : [keycol, labelcol, 'estimate_a', 'rank_a', 'estimate_b', 'rank_b', 'delta', 'rank_delta'];
  const labels = {code:'Code', title:'Title', activity:'Activity',
                  estimate_a:'Est. A', rank_a:'Rank A', estimate_b:'Est. B', rank_b:'Rank B',
                  delta:'Δ (A−B)', rank_delta:'ΔRank'};
  document.getElementById('th').innerHTML = '<tr>' + cols.map(c =>
    `<th onclick="sortBy('${c}')">${labels[c]||c}${SORT.col===c?(SORT.dir<0?' ↓':' ↑'):''}</th>`).join('') + '</tr>';
  document.getElementById('tb').innerHTML = rows.map(r => {
    const dd = parseFloat(r.delta)||0;
    const dcls = dd > 0.001 ? 'pos' : dd < -0.001 ? 'neg' : '';
    return '<tr>' + cols.map(c => {
      if (c === keycol) return `<td style="font-family:ui-monospace,Menlo,monospace; font-size:11px;">${escapeHtml(r[c]||'')}</td>`;
      if (c === labelcol) return `<td>${escapeHtml(r[c]||'')}</td>`;
      if (c === 'delta') return `<td class="num ${dcls}">${fmt(r[c],3)}</td>`;
      if (c === 'rank_delta') {
        const rd = parseFloat(r[c])||0;
        return `<td class="num ${rd>0?'pos':rd<0?'neg':''}">${rd>=0?'+':''}${fmt(r[c],1)}</td>`;
      }
      if (c.startsWith('rank')) return `<td class="num">${fmt(r[c],1)}</td>`;
      return `<td class="num">${fmt(r[c],3)}</td>`;
    }).join('') + '</tr>';
  }).join('');
}

function sortBy(col){ SORT.dir = SORT.col===col ? -SORT.dir : -1; SORT.col=col; renderTable(); }

load();
</script></body></html>
"""


# ---------- shared theme: fill __THEME_HEAD__ / __MASTHEAD__ in every page ----------

_NAV = [("/", "Studies", "studies"), ("/run", "Run analysis", "run"),
        ("/results", "Past runs", "runs"), ("/transitions", "Transitions", "transitions")]


def _themed(html: str, active: str) -> str:
    return site_theme.apply(html, site_theme.masthead(_NAV, active, home_href="/"))


INDEX_HTML = _themed(INDEX_HTML, "studies")
RUN_HTML = _themed(RUN_HTML, "run")
RESULTS_INDEX_HTML = _themed(RESULTS_INDEX_HTML, "runs")
RESULTS_HTML = _themed(RESULTS_HTML, "runs")
LOO_HTML = _themed(LOO_HTML, "runs")
COMPARE_HTML = _themed(COMPARE_HTML, "runs")
TRANSITIONS_HTML = _themed(TRANSITIONS_HTML, "transitions")


if __name__ == "__main__":
    import os
    app.run(debug=True, port=int(os.environ.get("PORT", 5050)))
