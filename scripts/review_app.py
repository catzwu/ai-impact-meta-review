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
<html><head><meta charset="utf-8"><title>Meta-Analysis Review</title>
<style>
  body { font-family: -apple-system, sans-serif; margin: 0; font-size: 13px; }
  header { background:#222; color:#fff; padding:10px 16px; display:flex; gap:16px; align-items:center; }
  header h1 { margin:0; font-size:16px; }
  header .stats { color:#bbb; font-size:12px; }
  header button { background:#0a7; color:#fff; border:0; padding:6px 12px; border-radius:4px; cursor:pointer; }
  header button.secondary { background:#444; }
  header input[type=text] { padding:5px 8px; border-radius:4px; border:0; min-width:200px; }
  table { border-collapse: collapse; table-layout: fixed; width: 100%; }
  th, td { padding:8px 10px; border-bottom:1px solid #eee; vertical-align: top; text-align: left; overflow:hidden; text-overflow:ellipsis; }
  th { background:#f7f7f9; position:sticky; top:0; cursor:pointer; user-select:none; font-size:11px; text-transform:uppercase;
       font-weight:600; color:#555; letter-spacing:0.04em; position: relative; border-bottom:2px solid #ddd; }
  th .resizer { position:absolute; right:-3px; top:0; width:8px; height:100%; cursor:col-resize; user-select:none; z-index:2; }
  th .resizer:hover { background:#aac; }
  th .resizer:active { background:#88a; }
  tbody tr:nth-child(even) { background:#fafafa; }
  tbody tr:hover { background:#eef6ff; }
  tr.deleted { opacity:0.5; background:#fee !important; }
  tr.merged-into { background:#fffce0 !important; }
  td.snippet { white-space: normal; word-break: break-word; color:#555; font-size:12px; line-height:1.4; }
  td.title { white-space: normal; word-break: break-word; font-weight:500; font-size:13px; line-height:1.3; cursor:pointer; color:#0a3d7a; }
  td.title:hover { text-decoration: underline; }
  td.value { font-family: ui-monospace, Menlo, monospace; text-align: right; white-space: nowrap; font-weight:500; }
  td.value.pos { color:#0a6c2c; }
  td.value.neg { color:#b32020; }
  td.cite { font-family: ui-monospace, Menlo, monospace; font-size:11px; color:#555; }
  td.file { font-family: ui-monospace, Menlo, monospace; font-size:11px; color:#888; }
  /* searchable onet picker */
  .onet-picker { position:relative; max-width:320px; }
  .onet-picker .display { border:1px solid #ccc; padding:4px 6px; font-size:12px; cursor:pointer; background:#fff; border-radius:3px; min-height:18px; }
  .onet-picker .display.empty { color:#999; }
  #onetPanel { position:absolute; background:#fff; border:1px solid #888; box-shadow:0 4px 12px rgba(0,0,0,0.18); z-index:200;
               width:380px; max-height:400px; display:none; }
  #onetPanel.open { display:block; }
  #onetPanel input.search { width:calc(100% - 12px); margin:6px; padding:5px 8px; box-sizing:border-box; font-size:12px; border:1px solid #ccc; border-radius:3px; }
  #onetPanel .list { max-height:340px; overflow-y:auto; font-size:12px; }
  #onetPanel .group-hdr { background:#eef; padding:4px 8px; font-weight:bold; font-size:11px; text-transform:uppercase; color:#336; position:sticky; top:0; }
  #onetPanel .item { padding:5px 10px; cursor:pointer; }
  #onetPanel .item:hover, #onetPanel .item.active { background:#cef; }
  #onetPanel .code { font-family:monospace; color:#888; font-size:11px; margin-right:6px; }
  #onetPanel .clear { padding:5px 10px; color:#a00; cursor:pointer; border-top:1px solid #eee; font-size:11px; }
  .kind-speed   { background:#e6f3ff; color:#0463a3; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:600; }
  .kind-quality { background:#ffe6f0; color:#a0286c; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:600; }
  .conf-pill-cell { display:inline-block; padding:1px 7px; border-radius:10px; font-size:10px; text-transform:uppercase; font-weight:600; }
  .conf-pill-cell.high   { background:#d4f4d4; color:#0a6c2c; }
  .conf-pill-cell.medium { background:#fff0c0; color:#8a6900; }
  .conf-pill-cell.low    { background:#ffd6d6; color:#a00; }
  button.row-btn { padding:3px 10px; font-size:11px; cursor:pointer; margin-right:4px; background:#fff; border:1px solid #ccc; border-radius:3px; color:#333; }
  button.row-btn:hover { background:#f0f0f0; }
  button.row-btn.danger { color:#a00; border-color:#e0b0b0; }
  button.row-btn.danger:hover { background:#fee; }
  /* drawer */
  #backdrop { position:fixed; inset:0; background:rgba(0,0,0,0.25); opacity:0; pointer-events:none;
              transition:opacity 0.2s; z-index:90; }
  #backdrop.open { opacity:1; pointer-events:auto; }
  #drawer { position:fixed; top:0; right:0; width:60%; min-width:560px; height:100%; background:#fff; box-shadow:-3px 0 12px rgba(0,0,0,0.2);
            transform:translateX(100%); transition:transform 0.2s; overflow:auto; z-index:100; }
  #drawer.open { transform:translateX(0); }
  #drawer header { background:#333; }
  #drawer .close { background:#a00; }
  /* paper detail sections */
  .pd-meta { padding: 14px 18px; background:#fafafa; border-bottom:1px solid #ddd; }
  .pd-meta h2 { margin: 0 0 4px; font-size:18px; line-height:1.25; }
  .pd-meta .authors { color:#444; font-size:12px; margin-bottom:4px; }
  .pd-meta .ids { color:#666; font-size:11px; }
  .pd-meta .ids .pill { display:inline-block; background:#eee; padding:1px 6px; border-radius:3px; margin-right:6px; font-family:monospace; }
  .pd-meta a { color:#06c; margin-right:14px; }
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
  .pd-quotes-block { max-height:240px; overflow-y:auto; }
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
</style></head><body>
<header>
  <h1>Meta-Analysis Review</h1>
  <span class="stats" id="stats"></span>
  <input type="text" id="search" placeholder="filter by citation, title, file..." />
  <button id="mergeBtn" class="secondary">Merge selected</button>
  <button id="deleteBtn" class="secondary">Delete selected</button>
  <button id="exportBtn" class="secondary">Export CSVs</button>
  <button id="uploadBtn" style="background:#0a7;">+ Upload PDF</button>
  <button id="jobsPill" style="display:none; background:#d2b048; color:#222;" onclick="showJobsList()"></button>
  <a href="/transitions" style="text-decoration:none;"><button style="background:#2a7a8a;">Transitions →</button></a>
  <a href="/run" style="text-decoration:none;"><button style="background:#5b3aa6;">Run Analysis →</button></a>
  <input type="file" id="uploadFile" accept="application/pdf" style="display:none;"/>
</header>
<div id="uploadModal" style="display:none; position:fixed; inset:0; background:rgba(0,0,0,0.4); z-index:300; align-items:flex-start; justify-content:center; padding-top:60px;">
  <div style="background:#fff; border-radius:8px; width:min(720px, 92vw); max-height:80vh; overflow:auto; box-shadow:0 8px 32px rgba(0,0,0,0.3);">
    <div style="padding:14px 20px; border-bottom:1px solid #eee; display:flex; justify-content:space-between; align-items:center;">
      <h2 style="margin:0; font-size:15px;">Upload PDF</h2>
      <button onclick="closeUpload()" style="background:#a00; color:#fff; border:0; padding:4px 12px; border-radius:3px; cursor:pointer;">Close</button>
    </div>
    <div id="uploadBody" style="padding:18px 22px;"></div>
  </div>
</div>
<table id="table">
  <colgroup>
    <col style="width:32px"><col style="width:70px"><col style="width:200px"><col style="width:240px">
    <col style="width:200px"><col style="width:70px"><col style="width:60px"><col style="width:320px">
    <col style="width:340px"><col style="width:80px">
  </colgroup>
  <thead><tr>
    <th><input type="checkbox" id="selAll"/><div class="resizer"></div></th>
    <th data-sort="kind">Kind<div class="resizer"></div></th>
    <th data-sort="citation_key">Citation<div class="resizer"></div></th>
    <th data-sort="title">Title<div class="resizer"></div></th>
    <th data-sort="file_name">File<div class="resizer"></div></th>
    <th data-sort="value">Value<div class="resizer"></div></th>
    <th data-sort="confidence">Conf<div class="resizer"></div></th>
    <th>O*NET<div class="resizer"></div></th>
    <th>Task snippet<div class="resizer"></div></th>
    <th>Actions<div class="resizer"></div></th>
  </tr></thead>
  <tbody id="tbody"></tbody>
</table>
<div id="backdrop" onclick="closeDrawer()"></div>
<div id="drawer">
  <header style="display:flex; justify-content:space-between;">
    <h1 id="drawerTitle">Paper detail</h1>
    <button class="close" onclick="closeDrawer()">Close</button>
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
  if (!code) return '<div class="display empty">(none — click to set)</div>';
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
    !q || (r.citation_key+r.title+r.file_name+r.onet_code+r.onet_label).toLowerCase().includes(q));
  if (SORT.col) {
    rows = [...rows].sort((a,b)=> {
      const x = (a[SORT.col]||'').toString(), y=(b[SORT.col]||'').toString();
      if (SORT.col==='value') return (parseFloat(x)||0 - parseFloat(y)||0) * SORT.dir;
      return x.localeCompare(y) * SORT.dir;
    });
  }
  const live = rows.filter(r => !r._deleted).length;
  document.getElementById('stats').textContent = `${rows.length} rows (${live} live, ${rows.length-live} dropped)`;
  const tbody = document.getElementById('tbody');
  tbody.innerHTML = rows.map(r => {
    const v = parseFloat(r.value||0);
    const vClass = v > 0.001 ? 'pos' : v < -0.001 ? 'neg' : '';
    return `
    <tr class="${r._deleted?'deleted':''}${r._merged_into?' merged-into':''}" data-row="${r.row_id}">
      <td><input type="checkbox" class="sel"/></td>
      <td><span class="kind-${r.kind}">${r.kind}</span>${r._merged_into?'<span class="badge">merged</span>':''}</td>
      <td class="cite">${escapeHtml(r.citation_key)}</td>
      <td class="title" title="Click to view paper detail" onclick="viewPaper('${r.paper_id}','${r.row_id}')">${escapeHtml(r.title)}</td>
      <td class="file" title="${escapeHtml(r.file_name)}">${escapeHtml(r.file_name)}</td>
      <td class="value ${vClass}">${v.toFixed(3)}</td>
      <td>${r.confidence ? `<span class="conf-pill-cell ${r.confidence}">${r.confidence}</span>` : ''}</td>
      <td>
        <div class="onet-picker" onclick="openPicker('${r.row_id}', this)">
          ${onetDisplay(r.onet_code, r.onet_label)}
        </div>
      </td>
      <td class="snippet" title="${escapeHtml(r.task_description||'')}">${escapeHtml(r.task_description||'')}</td>
      <td>
        ${r._deleted
          ? `<button class="row-btn" onclick="undeleteRow('${r.row_id}')">Restore</button>`
          : `<button class="row-btn danger" onclick="deleteRow('${r.row_id}')">Delete</button>`}
      </td>
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
  return `<div class="pd-section"><h3>O*NET mapping (pipeline's choice)</h3>
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
  pill.textContent = `⚙ ${ACTIVE_JOBS.length} running`;
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
  openUpload(`<div style="color:#666;">Uploading <b>${escapeHtml(file.name)}</b> and checking for duplicates…</div>`);
  const fd = new FormData(); fd.append('pdf', file);
  let r;
  try {
    r = await (await fetch('/api/upload/check', {method:'POST', body: fd})).json();
  } catch (err) {
    openUpload(`<div style="color:#a00;">Upload failed: ${escapeHtml(err.message)}</div>`); return;
  }
  if (!r.ok) { openUpload(`<div style="color:#a00;">${escapeHtml(r.error)}</div>`); return; }
  UPLOAD.staged = r;
  e.target.value = ''; // reset for next upload
  renderDupCheck(r);
};

function renderDupCheck(r) {
  const dups = r.duplicate_candidates || [];
  let dupBlock;
  if (!dups.length) {
    dupBlock = `<div style="background:#e6f4ea; color:#0a6c2c; padding:10px 14px; border-radius:4px; margin:14px 0; font-size:13px;">
      ✓ No likely duplicates found in the corpus.
    </div>`;
  } else {
    dupBlock = `<div style="background:#fff4d6; color:#7a5800; padding:10px 14px; border-radius:4px; margin:14px 0 8px; font-size:13px;">
      ⚠ Found ${dups.length} possible duplicate${dups.length>1?'s':''}. Review before running.
    </div>` + dups.map((d, i) => `
      <div style="border:1px solid #ddd; border-radius:4px; padding:10px 12px; margin-bottom:8px;">
        <div style="display:flex; justify-content:space-between; align-items:start; gap:10px;">
          <div style="flex:1;">
            <div style="font-weight:500; font-size:13px;">${escapeHtml(d.title)}</div>
            <div style="font-size:11px; color:#666; margin-top:2px;">
              <span style="font-family:monospace;">${escapeHtml(d.citation_key||'')}</span> ·
              ${escapeHtml((d.authors||[]).join('; '))}${d.year?` (${d.year})`:''} ·
              <b>${Math.round(d.score*100)}% match</b> with <span style="font-family:monospace;">${escapeHtml(d.paper_id)}</span>
            </div>
          </div>
          <button onclick="viewPaper('${d.paper_id}','dup')" class="row-btn">View existing</button>
        </div>
      </div>`).join('');
  }
  openUpload(`
    <div><b>Title detected:</b> ${escapeHtml(r.extracted_title || '(none — title heuristic failed)')}</div>
    <div style="font-size:11px; color:#888; margin-top:2px;">Will be saved as <code>${escapeHtml(r.target_paper_id)}.pdf</code></div>
    ${dupBlock}
    <div style="display:flex; gap:10px; margin-top:14px;">
      <button class="row-btn" onclick="closeUpload()">Cancel (discard upload)</button>
      <button onclick="confirmUploadRun()" style="background:#0a7; color:#fff; border:0; padding:6px 16px; border-radius:3px; cursor:pointer;">
        ${dups.length ? 'Add anyway and run pipeline' : 'Run pipeline'}
      </button>
    </div>
  `);
}

async function confirmUploadRun() {
  const r = UPLOAD.staged; if (!r) return;
  openUpload(`<div style="color:#555;">Starting pipeline for <code>${escapeHtml(r.target_paper_id)}</code>…</div>`);
  const res = await (await fetch('/api/upload/run', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({staged_id: r.staged_id, target_paper_id: r.target_paper_id})})).json();
  if (!res.ok) { openUpload(`<div style="color:#a00;">${escapeHtml(res.error)}</div>`); return; }
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
    <div style="border:1px solid #ddd; border-radius:4px; padding:8px 12px; margin-bottom:6px; display:flex; justify-content:space-between; align-items:center;">
      <div>
        <div style="font-family:monospace; font-size:12px;">${escapeHtml(j.paper_id)}</div>
        <div style="font-size:11px; color:#666;">state: <b>${j.state}</b> · job ${escapeHtml(j.job_id)}</div>
      </div>
      <button class="row-btn" onclick="pollUpload('${j.job_id}','${escapeHtml(j.paper_id)}')">Open</button>
    </div>`).join('');
  openUpload(`<div><b>Pipeline jobs</b> (last 15)</div>${items || '<div style="color:#888;">No jobs yet.</div>'}`);
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
      if (UPLOAD_OPEN) openUpload(`<div style="color:#a00;">${escapeHtml(r.error)}</div>`);
      return;
    }
    const state = r.job.state;
    const done = STAGE_NAMES.filter(s => r.stages[s]).length;
    const stageList = STAGE_NAMES.map(s =>
      `<div style="display:flex; gap:8px; font-size:12px; padding:2px 0;">
         <span style="width:14px;">${r.stages[s]?'✓':'·'}</span>
         <span style="color:${r.stages[s]?'#0a6c2c':'#888'};">${s}</span>
       </div>`).join('');
    let footer = '';
    if (state === 'done') {
      const o = r.job.outcome || {};
      const added = (o.speed_added || o.quality_added);
      const reasons = (o.reasons||[]).map(s=>`<li>${escapeHtml(s)}</li>`).join('');
      const bg = added ? '#e6f4ea' : '#fff8e0';
      const fg = added ? '#0a6c2c' : '#7a5800';
      const headline = added
        ? '✓ Pipeline complete — new row(s) added to the table.'
        : '⚠ Pipeline complete, but nothing was added to the table.';
      footer = `<div style="background:${bg}; color:${fg}; padding:10px 14px; border-radius:4px; margin-top:10px;">
        <div><b>${headline}</b></div>
        ${reasons ? `<ul style="margin:6px 0 0; padding-left:20px;">${reasons}</ul>` : ''}
        <div style="margin-top:8px;">
          <button class="row-btn" onclick="closeUpload(); load();">${added?'Refresh table':'Back to table'}</button>
          <button class="row-btn" onclick="viewPaper('${paperId}','new')">View paper detail</button>
        </div></div>`;
      updateActiveJobs(ACTIVE_JOBS.filter(j=>j.job_id !== jobId));
    } else if (state === 'failed') {
      footer = `<div style="background:#fee; color:#a00; padding:10px 14px; border-radius:4px; margin-top:10px;">
        ✗ Pipeline failed: ${escapeHtml(r.job.error||'')}<br><span style="font-size:11px;">See logs/upload_${escapeHtml(paperId)}.log</span></div>`;
      updateActiveJobs(ACTIVE_JOBS.filter(j=>j.job_id !== jobId));
    } else {
      footer = `<div style="color:#888; font-size:12px; margin-top:8px;">running… (${done}/${STAGE_NAMES.length} stages) — closing this window will not stop the job</div>`;
    }
    if (UPLOAD_OPEN && POLL_JOB === jobId) {
      openUpload(`
        <div><b>Processing</b> <code>${escapeHtml(paperId)}</code> — state: <b>${state}</b></div>
        <div style="margin-top:10px; border:1px solid #eee; border-radius:4px; padding:8px 12px;">${stageList}</div>
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
    try { const w = localStorage.getItem('colw_'+i); if (w && cols[i]) cols[i].style.width = w; } catch(_){}
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
        try{ localStorage.setItem('colw_'+i, col.style.width); }catch(_){}
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


RUN_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Run Analysis</title>
<style>
  body { font-family:-apple-system,sans-serif; margin:0; background:#f5f5f7; }
  header { background:#222; color:#fff; padding:10px 16px; display:flex; gap:14px; align-items:center; }
  header h1 { margin:0; font-size:16px; }
  header a { color:#9cf; text-decoration:none; font-size:13px; }
  main { max-width:780px; margin: 24px auto; padding: 0 18px; }
  .card { background:#fff; border-radius:8px; box-shadow:0 1px 4px rgba(0,0,0,0.06); padding:20px 24px; margin-bottom:18px; }
  h2 { font-size:14px; text-transform:uppercase; letter-spacing:0.04em; color:#666; margin: 0 0 12px; }
  .row { display:flex; gap:14px; margin-bottom:10px; align-items:baseline; }
  .row label { width:170px; font-size:13px; color:#333; font-weight:500; }
  .row .ctl { flex:1; }
  .row .ctl input[type=number], .row .ctl input[type=text], .row .ctl select { padding:5px 8px; border:1px solid #ccc; border-radius:4px; font-size:13px; width:120px; }
  .row .help { color:#888; font-size:11px; margin-top:2px; }
  .seg { display:inline-flex; border:1px solid #ccc; border-radius:6px; overflow:hidden; }
  .seg button { padding:6px 16px; background:#fff; border:0; cursor:pointer; font-size:13px; color:#555; }
  .seg button.active { background:#5b3aa6; color:#fff; }
  details { margin-top:14px; }
  details summary { cursor:pointer; font-size:12px; color:#666; padding:4px 0; }
  .actions { display:flex; gap:10px; align-items:center; margin-top:16px; }
  button.primary { background:#5b3aa6; color:#fff; border:0; padding:8px 22px; border-radius:4px; cursor:pointer; font-size:14px; }
  button.primary:disabled { background:#aaa; cursor:wait; }
  button.secondary { background:#eee; color:#333; border:0; padding:8px 14px; border-radius:4px; cursor:pointer; font-size:13px; }
  .status { font-size:13px; color:#555; }
  .err { background:#ffe4e4; border:1px solid #f99; color:#a00; padding:8px 12px; border-radius:4px; margin-top:10px; font-size:13px; }
</style></head><body>
<header>
  <h1>Run Analysis</h1>
  <a href="/">← back to review</a>
  <a href="/results" style="margin-left:auto;">Past runs →</a>
</header>
<main>
  <div id="liveCount" style="color:#666; font-size:12px; margin: 0 0 12px 4px;">checking current table…</div>

  <div class="card">
    <h2>Metric</h2>
    <div class="seg" id="metricSeg">
      <button data-m="speed" class="active">speed</button>
      <button data-m="quality">quality</button>
    </div>
    <div class="help" style="margin-top:6px; color:#888; font-size:12px;">A run targets one metric at a time. Switching swaps the observation columns consumed by the pipeline.</div>
  </div>

  <div class="card">
    <h2>Pruning: O*NET coverage</h2>
    <div style="font-size:12px; color:#666; line-height:1.5; margin-bottom:8px;">
      Heatmap of mean Stage-B weight per SOC major group × activity. Green vertical lines = observed activities for this metric.
      Cyan horizontal lines = SOC majors containing an observed occupation. Translucent gray = deselected (excluded from analysis).
      Click a row label or its checkbox to toggle.
    </div>
    <div id="heatmap" style="overflow:auto;"></div>

    <div style="display:flex; gap:18px; margin:14px 0 6px; align-items:baseline;">
      <label style="font-weight:500;">Activity weight threshold</label>
      <input type="number" id="weightThreshold" value="10" step="0.5" min="0" style="width:80px; padding:5px 7px;"/>
      <div class="help" style="color:#888; font-size:12px;">activities whose summed weight (over selected occupations) is below this are dropped</div>
    </div>
    <div id="barchart" style="overflow:auto;"></div>
    <div id="pruneSummary" style="margin-top:10px; padding:8px 12px; background:#f0f4f8; border-radius:4px; font-size:13px; color:#333;"></div>

    <div style="margin-top:14px; padding:10px 12px; border:1px solid #e0e0e0; border-radius:4px; background:#fafafa;">
      <div style="font-size:13px; font-weight:500; margin-bottom:6px;">Analysis unit</div>
      <div id="aggLevelSeg" style="display:inline-flex; border:1px solid #ccc; border-radius:4px; overflow:hidden; font-size:12px;">
        <button type="button" data-val="occupation" class="agg-seg active" style="padding:6px 12px; border:0; background:#06c; color:#fff; cursor:pointer;">Occupation (~894)</button>
        <button type="button" data-val="soc_minor" class="agg-seg" style="padding:6px 12px; border:0; background:#fff; color:#333; cursor:pointer; border-left:1px solid #ccc;">SOC minor (~92, XX-X000)</button>
        <button type="button" data-val="soc_major" class="agg-seg" style="padding:6px 12px; border:0; background:#fff; color:#333; cursor:pointer; border-left:1px solid #ccc;">SOC major (22)</button>
      </div>
      <input type="hidden" id="aggregationLevel" value="occupation"/>
      <div style="margin-top:6px; color:#888; font-size:12px;">
        Occupation (default): runs over the ~894 individual O*NET occupations.
        SOC minor: collapses to ~92 3-digit minor groups (e.g. 13-2000 Financial Specialists).
        SOC major: collapses to the 22 2-digit major groups.
        In aggregated modes, W is the per-group row-mean; observations are IV-weighted (or simple-mean fallback) within the group; AIOE baseline is averaged within the group.
      </div>
    </div>
  </div>

  <details class="card">
    <summary>Advanced parameters</summary>
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
          <span style="font-size:12px; color:#888;">Felten et al. AIOE moment-matched to observed mean/SD</span>
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
      `Current table: <b>${speed}</b> speed · <b>${qual}</b> quality (live, deleted/merged rows excluded) · <b>${withOnet}</b> have an O*NET code`;
  } catch (e) {
    document.getElementById('liveCount').textContent = 'could not load live count';
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
    document.querySelectorAll('#aggLevelSeg .agg-seg').forEach(x=>{
      x.classList.remove('active');
      x.style.background = '#fff'; x.style.color = '#333';
    });
    b.classList.add('active');
    b.style.background = '#06c'; b.style.color = '#fff';
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
  if (!r.ok) { document.getElementById('heatmap').innerHTML = `<div style="color:#a00;">${r.error}</div>`; return; }
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
      overlaySvg += `<rect x="${labelW}" y="${labelH+ri*cell}" width="${cols.length*cell}" height="${cell}" fill="rgba(180,180,180,0.55)"/>`;
    }
  });
  // observed overlays
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
  // row labels with checkboxes (HTML overlay because checkboxes in SVG are clunky)
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
  // column labels rotated downward beneath the grid
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
        <span><span style="display:inline-block; width:10px; height:10px; background:rgba(180,180,180,0.55); vertical-align:middle;"></span> deselected (excluded)</span>
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
    const fill = a.observed ? '#0a6c2c' : '#4C72B0';   // green = given/observed; blue = other
    const opacity = below ? 0.35 : 1.0;
    bars += `<rect x="${labelW}" y="${y+1}" width="${w}" height="${barH-3}" fill="${fill}" opacity="${opacity}"/>`;
    bars += `<text x="${labelW + w + 4}" y="${y+barH-3}" font-size="9" fill="${below?'#999':'#333'}">${sums[i].toFixed(1)}</text>`;
    const tcol = a.observed ? '#0a6c2c' : '#333';
    bars += `<text x="${labelW-4}" y="${y+barH-3}" font-size="10" fill="${tcol}" text-anchor="end" font-weight="${a.observed?'600':'400'}">${escapeHtml(a.name)}</text>`;
  });
  // threshold line
  const tx = labelW + (threshold / vmax) * plotW;
  bars += `<line x1="${tx}" y1="${padT}" x2="${tx}" y2="${padT + acts.length*barH}" stroke="#a00" stroke-width="1.5" stroke-dasharray="4 3"/>`;
  bars += `<text x="${tx + 3}" y="${padT - 2}" font-size="10" fill="#a00">threshold = ${threshold.toFixed(1)}</text>`;

  document.getElementById('barchart').innerHTML = `
    <svg width="${labelW + plotW + 70}" height="${height}" style="display:block;">${bars}</svg>
    <div style="font-size:11px; color:#666; margin-top:4px;">
      Bars below the red dashed line are dropped. Green label/bar = observed (given).
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
    `<b>Included in analysis:</b> ${occIncluded.toLocaleString()} of ${occTotal.toLocaleString()} occupations ` +
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
document.getElementById('runBtn').onclick = async ()=>{
  const btn = document.getElementById('runBtn');
  const status = document.getElementById('status');
  const err = document.getElementById('errBox');
  btn.disabled = true; status.textContent='running… (≈30s)'; err.innerHTML='';
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
  try {
    const r = await (await fetch('/api/run', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)})).json();
    if (!r.ok) throw new Error(r.error||'unknown');
    status.textContent = 'done — redirecting…';
    location.href = '/results/' + encodeURIComponent(r.run_id);
  } catch(e) {
    err.innerHTML = '<div class="err">'+e.message+'</div>';
    status.textContent=''; btn.disabled=false;
  }
};
</script>
</body></html>
"""


RESULTS_INDEX_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Past runs</title>
<style>
  body { font-family:-apple-system,sans-serif; margin:0; background:#f5f5f7; }
  header { background:#222; color:#fff; padding:10px 16px; display:flex; gap:14px; align-items:center; }
  header h1 { margin:0; font-size:16px; }
  header a { color:#9cf; text-decoration:none; font-size:13px; }
  main { max-width:1000px; margin:24px auto; padding:0 18px; }
  table { width:100%; background:#fff; border-collapse:collapse; border-radius:8px; overflow:hidden; box-shadow:0 1px 4px rgba(0,0,0,0.06); font-size:13px; }
  th, td { padding:9px 12px; text-align:left; border-bottom:1px solid #eee; }
  th { background:#f7f7f9; font-size:11px; text-transform:uppercase; color:#555; letter-spacing:0.04em; }
  tr:hover { background:#f0f6ff; cursor:pointer; }
  .pill { display:inline-block; padding:1px 8px; border-radius:10px; font-size:11px; font-weight:600; }
  .pill.speed { background:#e6f3ff; color:#0463a3; } .pill.quality { background:#ffe6f0; color:#a0286c; }
  a { color:#5b3aa6; text-decoration:none; }
</style></head><body>
<header><h1>Past analysis runs</h1><a href="/">← review</a><a href="/run">+ new run</a></header>
<main>
  <table id="t"><thead><tr>
    <th>Run ID</th><th>Started</th><th>Metric</th><th>β</th><th>Ω_ref</th><th>Baseline</th>
    <th>Obs (occ/act)</th><th>Kept (occ/act)</th>
  </tr></thead><tbody id="tb"></tbody></table>
</main>
<script>
fetch('/api/runs').then(r=>r.json()).then(d=>{
  const tb = document.getElementById('tb');
  if (!d.runs.length) { tb.innerHTML='<tr><td colspan="8" style="text-align:center; padding:20px; color:#888;">No runs yet. <a href="/run">Start one</a>.</td></tr>'; return; }
  tb.innerHTML = d.runs.map(r=>{
    const p = r.params||{};
    return `<tr onclick="location.href='/results/${encodeURIComponent(r.run_id)}'">
      <td style="font-family:monospace; font-size:11px;">${r.run_id}</td>
      <td>${(r.started_utc||'').replace('T',' ').slice(0,19)}</td>
      <td><span class="pill ${p.metric}">${p.metric||''}</span></td>
      <td>${p.beta}</td><td>${p.omega_ref}</td>
      <td>${r.baseline_active ? `yes (Ω<sub>b</sub>=${p.omega_base})` : '—'}</td>
      <td>${r.n_observed_occ}/${r.n_observed_act}</td>
      <td>${r.n_kept_occ}/${r.n_kept_act}</td>
    </tr>`;
  }).join('');
});
</script></body></html>
"""


RESULTS_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Run results</title>
<style>
  body { font-family:-apple-system,sans-serif; margin:0; background:#f5f5f7; font-size:13px; }
  header { background:#222; color:#fff; padding:10px 16px; display:flex; gap:14px; align-items:center; }
  header h1 { margin:0; font-size:16px; }
  header a { color:#9cf; text-decoration:none; font-size:13px; }
  header .stats { color:#bbb; font-size:11px; margin-left:auto; }
  main { padding: 16px 24px; }
  .toolbar { display:flex; gap:14px; align-items:center; background:#fff; padding:10px 14px; border-radius:6px; margin-bottom:12px; box-shadow:0 1px 3px rgba(0,0,0,0.05); }
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
  .help { color:#888; font-size:11px; }
</style></head><body>
<header>
  <h1>Run results</h1><a href="/results">← all runs</a><a href="/run">+ new</a>
  <span class="stats" id="stats">…</span>
</header>
<main>
  <div class="toolbar">
    <div class="seg" id="viewSeg">
      <button data-v="occ" class="active">Occupations</button>
      <button data-v="act">Activities</button>
    </div>
    <label>std ≤ <input type="number" id="stdFilter" value="5" step="0.05" style="width:60px;"/></label>
    <label><input type="checkbox" id="onlyObserved"/> only observed</label>
    <label>search <input type="text" id="search" placeholder="title/code…"/></label>
    <button onclick="exportCsv()" style="margin-left:auto; padding:4px 10px; cursor:pointer;">Export view as CSV</button>
  </div>
  <div id="warn"></div>
  <table id="t"><thead id="th"></thead><tbody id="tb"></tbody></table>
</main>
<script>
const RUN_ID = "__RUN_ID__";
let DATA = null, VIEW='occ', SORT={col:'estimate', dir:-1};

function fmtNum(x, p){ if (x===null||x===undefined||x==='') return '—'; const n=parseFloat(x); return isNaN(n)?x:n.toFixed(p===undefined?3:p); }
function escapeHtml(s){return (s||'').toString().replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}

async function load(){
  const r = await (await fetch('/api/results/'+encodeURIComponent(RUN_ID))).json();
  if (!r.ok) { document.body.innerHTML = '<p style="padding:30px;">'+r.error+'</p>'; return; }
  DATA = r.data;
  const p = DATA.params;
  document.getElementById('stats').textContent =
    `${p.metric.toUpperCase()} · β=${p.beta} · Ω_ref=${p.omega_ref}` +
    (DATA.baseline_active ? ` · baseline Ω_b=${p.omega_base}` : '') +
    ` · kept ${DATA.n_kept_occ}/${DATA.n_kept_act}`;
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
    const labels = {code:'Code', title:'Title', activity:'Activity', observed:'Observed', aioe_baseline:'AIOE', estimate:'Estimate', posterior_std:'Std', n_studies:'n'};
    return `<th onclick="sortBy('${c}')">${labels[c]||c}${SORT.col===c?(SORT.dir<0?' ↓':' ↑'):''}</th>`;
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
      return `<td>${escapeHtml(r[c]||'')}</td>`;
    }).join('') + `<td>${bar}</td></tr>`;
  }).join('');
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


TRANSITIONS_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Occupational transitions</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; margin:0; padding:0; background:#fafafa; color:#222; }
  header { display:flex; align-items:center; gap:14px; padding:10px 18px; border-bottom:1px solid #ddd; background:#fff; position:sticky; top:0; z-index:10; }
  header h1 { margin:0; font-size:16px; font-weight:600; }
  header a { color:#06c; text-decoration:none; font-size:13px; }
  header a:hover { text-decoration:underline; }
  .meta { color:#666; font-size:12px; margin-left:auto; }
  #main { display:grid; grid-template-columns: minmax(0,1fr) 460px; gap:14px; padding:14px; align-items:start; }
  .panel { background:#fff; border:1px solid #e5e5e5; border-radius:6px; padding:12px; }
  .panel h2 { margin:0 0 8px; font-size:13px; font-weight:600; color:#444; text-transform:uppercase; letter-spacing:0.04em; }
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
    position: absolute;
    left: 50%;
    bottom: 4px;
    transform-origin: 0 0;
    transform: rotate(-90deg) translateY(50%);
    white-space: nowrap;
    line-height: var(--cell);
    font-size: 10px;
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
  .est-pos { color:#0a6f2a; }
  .est-neg { color:#a00; }
  .est-na { color:#aaa; }
  .empty { color:#888; font-size:12px; padding:8px; text-align:center; }
  .search { display:flex; gap:8px; margin-bottom:8px; }
  .search input { flex:1; padding:6px 10px; border:1px solid #ccc; border-radius:4px; font-size:13px; }
  .results { position:relative; }
  .typeahead { position:absolute; background:#fff; border:1px solid #ccc; border-radius:4px; max-height:240px; overflow:auto; z-index:50; left:0; right:120px; top:38px; box-shadow:0 4px 12px rgba(0,0,0,0.08); display:none; }
  .typeahead .item { padding:5px 9px; font-size:12px; cursor:pointer; border-bottom:1px solid #f0f0f0; }
  .typeahead .item:hover { background:#f5f9ff; }
  .typeahead .item .code { color:#888; font-size:11px; }
</style></head><body>
<header>
  <h1>Occupational transitions</h1>
  <a href="/">← review</a>
  <a href="/run">run analysis</a>
  <a href="/results">runs</a>
  <span class="meta" id="metaTxt">loading…</span>
</header>
<div id="main">
  <div class="panel">
    <h2>Transition probabilities by SOC minor group (source → target, row-conditional)</h2>
    <div class="search results">
      <input id="search" placeholder="Search occupation (title or SOC code)…" autocomplete="off"/>
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
      <td>${escapeHtml(o.title)}<div style="color:#888;font-size:11px;">${o.code}</div></td>
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


if __name__ == "__main__":
    app.run(debug=True, port=5000)
