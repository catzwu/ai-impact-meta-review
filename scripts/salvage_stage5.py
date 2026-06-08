"""Stage 5 salvage pass — two-step with prompt caching.

Step 1 (cheap): re-send the stage-1b extracted quotes + failure reason to Sonnet and
                ask it to find any missed numbers WITHOUT the PDF.
Step 2 (PDF):   if step 1 still produces an unconvertible record, send the full PDF
                with prompt caching enabled. Salvage targets are grouped by paper_id
                so speed and quality re-use the same cached PDF block (≈10x cheaper
                read for the second call within 5-min TTL).

A successful step-1 skips step 2. A paper whose step-1 says arm_pair_resolvable=false
on grounds that need PDF verification still escalates to step 2.
"""
from __future__ import annotations

import argparse
import base64
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from common import (  # noqa: E402
    CostTracker, LOG_DIR, OUTPUTS_DIR, REPO_ROOT, call_llm, get_client, get_logger,
    load_prompt, load_settings, parse_json_response, read_json, write_json,
)

log = get_logger("salvage_stage5")
PAPERS_DIR = REPO_ROOT / "papers"
EXCLUDED = OUTPUTS_DIR / "final" / "papers_excluded.csv"
EFFECTS_DIR = OUTPUTS_DIR / "05_effect_sizes"


def _load_compute_modules():
    def _load(name, fname):
        spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / fname)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return _load("compute_speed", "05a_compute_speed.py"), _load("compute_quality", "05b_compute_quality.py")


def _read_pdf_b64(paper_id: str) -> str:
    pdf = PAPERS_DIR / f"{paper_id}.pdf"
    return base64.standard_b64encode(pdf.read_bytes()).decode("ascii")


def _failures_to_salvage() -> list[tuple[str, str, str]]:
    out = []
    with open(EXCLUDED) as f:
        for r in csv.DictReader(f):
            pid, stage, reason = r["paper_id"], r["stage_failed"], r["reason"]
            if stage == "05_speed":
                out.append((pid, "speed", reason))
            elif stage == "05_quality":
                out.append((pid, "quality", reason))
            elif stage == "05_effect_sizes":
                if "speed" in reason or "no speed or quality primary outcome" in reason:
                    out.append((pid, "speed", reason))
                if "quality" in reason or "no speed or quality primary outcome" in reason:
                    out.append((pid, "quality", reason))
    return out


SALVAGE_HEADER_TXT = {
    "speed": """SALVAGE pass for a meta-analysis pipeline. A prior pass failed to produce SPEED
effect-size numbers for this paper. Look more carefully at the extracted statistics
and find any numbers the prior pass missed. Output JSON conforming to the schema
below. If after careful re-read the paper truly does not report needed numbers,
set arm_pair_resolvable=false and explain.""",
    "quality": """SALVAGE pass. A prior pass failed to produce QUALITY (Hedges' g) numbers.
Look harder at the per-arm n, SDs, t/F stats, paired-design diffs, regression
coefs + SE. Output JSON conforming to the schema below. If data truly aren't
present, set arm_pair_resolvable=false and explain.""",
}

SALVAGE_HEADER_PDF = {
    "speed": """SALVAGE pass with FULL PDF. A prior text-only pass also failed. Re-read the PDF
end-to-end (tables, appendix, supplements) for the SPEED outcome and pull any
numbers the earlier passes missed. Output JSON per schema below. If the paper
truly does not fit the human vs human+AI speed framework, set
arm_pair_resolvable=false and explain.""",
    "quality": """SALVAGE pass with FULL PDF. A prior text-only pass also failed. Re-read the PDF
for QUALITY stats: per-arm n, SDs, t/F, paired diffs, reported d/g, regression
coefs + SE. Output JSON per schema below.""",
}


def _build_text_user(kind: str, prev_numbers, prev_stats, primary_outcome, failure_reason) -> str:
    schema = load_prompt(f"compute_{kind}.txt")
    ctx = {
        "previous_failure_reason": failure_reason,
        "previously_extracted_numbers": prev_numbers,
        "previously_extracted_primary_outcome": primary_outcome,
        "previously_extracted_reported_statistics": prev_stats,
    }
    return (
        SALVAGE_HEADER_TXT[kind]
        + "\n\nPRIOR-PASS CONTEXT:\n"
        + json.dumps(ctx, indent=2, default=str)
        + "\n\nSCHEMA (your output MUST conform):\n\n"
        + schema
    )


def _build_pdf_user(kind: str, prev_numbers, failure_reason, text_attempt_notes) -> list[dict]:
    """User content list: PDF (with cache_control) + targeted text prompt."""
    schema = load_prompt(f"compute_{kind}.txt")
    ctx = {
        "previous_failure_reason": failure_reason,
        "text_only_pass_also_failed_with_notes": text_attempt_notes,
        "text_only_pass_numbers": prev_numbers,
    }
    return [
        SALVAGE_HEADER_PDF[kind]
        + "\n\nPRIOR-PASS CONTEXT:\n"
        + json.dumps(ctx, indent=2, default=str)
        + "\n\nSCHEMA:\n\n"
        + schema
    ]


def _stage_1_artifacts(paper_id: str, kind: str):
    """Return (citation_key, primary_outcome_obj, reported_stats_for_outcome)."""
    ext_p = OUTPUTS_DIR / "01_extraction" / f"{paper_id}.json"
    oc_p = OUTPUTS_DIR / "03_outcome_classification" / f"{paper_id}.json"
    if not ext_p.exists():
        return None, None, None
    ext = read_json(ext_p)
    citation_key = ext.get("citation_key")
    primary_obj, prev_stats = None, None
    if oc_p.exists():
        oc = read_json(oc_p)
        primary_name = oc.get(f"primary_{kind}_outcome")
        if primary_name:
            primary_obj = next((o for o in ext.get("outcomes", []) if o.get("outcome_name") == primary_name), None)
            prev_stats = [r for r in ext.get("reported_statistics", []) if r.get("outcome_name") == primary_name]
    return citation_key, primary_obj, prev_stats


def _is_unconvertible(computed: dict) -> bool:
    m = computed.get("computation_method")
    return m in (None, "unconvertible")


def _call_text_pass(paper_id: str, kind: str, failure_reason: str, *, client, settings, cost):
    citation_key, primary_obj, prev_stats = _stage_1_artifacts(paper_id, kind)
    out_path = EFFECTS_DIR / f"{paper_id}.{kind}.json"
    prev_numbers = None
    if out_path.exists():
        try:
            prev_numbers = read_json(out_path).get("llm_extracted")
        except Exception:  # noqa: BLE001
            pass

    user = _build_text_user(kind, prev_numbers, prev_stats, primary_obj, failure_reason)
    model = settings.get("stage_models", {}).get(f"compute_{kind}", settings["model"])
    res = call_llm(
        client=client, model=model,
        system="You output only valid JSON. Do not perform arithmetic; only extract numbers.",
        user=user,
        max_tokens=settings["max_tokens"][f"compute_{kind}"] * 2,
        temperature=0.0,
        retry_cfg=settings["retry"],
        log_path=LOG_DIR / f"salvage_{kind}_text.jsonl",
        stage=f"salvage_text_{kind}", paper_id=paper_id,
    )
    cost.add(res.model, res.input_tokens, res.output_tokens)
    try:
        numbers = parse_json_response(res.text)
    except Exception as e:  # noqa: BLE001
        log.error("step1 parse fail %s.%s: %s", paper_id, kind, e)
        return None, None, None
    return numbers, citation_key, None


def _call_pdf_pass(paper_id: str, kind: str, failure_reason: str, text_attempt_notes,
                    *, client, settings, cost, cache_pdf: bool):
    out_path = EFFECTS_DIR / f"{paper_id}.{kind}.json"
    prev_numbers = None
    if out_path.exists():
        try:
            prev_numbers = read_json(out_path).get("llm_extracted")
        except Exception:  # noqa: BLE001
            pass

    pdf_b64 = _read_pdf_b64(paper_id)
    pdf_block = {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_b64},
    }
    if cache_pdf:
        pdf_block["cache_control"] = {"type": "ephemeral"}

    user_text = _build_pdf_user(kind, prev_numbers, failure_reason, text_attempt_notes)[0]
    user_content = [pdf_block, {"type": "text", "text": user_text}]

    model = settings.get("stage_models", {}).get(f"compute_{kind}", settings["model"])
    res = call_llm(
        client=client, model=model,
        system="You output only valid JSON. Do not perform arithmetic; only extract numbers.",
        user=user_content,
        max_tokens=settings["max_tokens"][f"compute_{kind}"] * 2,
        temperature=0.0,
        retry_cfg=settings["retry"],
        log_path=LOG_DIR / f"salvage_{kind}_pdf.jsonl",
        stage=f"salvage_pdf_{kind}", paper_id=paper_id,
    )
    cost.add(res.model, res.input_tokens, res.output_tokens)
    try:
        return parse_json_response(res.text)
    except Exception as e:  # noqa: BLE001
        log.error("step2 parse fail %s.%s: %s", paper_id, kind, e)
        return None


def _write_record(paper_id: str, kind: str, citation_key, numbers, computed, salvage_method: str):
    record = {
        "_paper_id": paper_id, "citation_key": citation_key, "kind": kind,
        "_salvage": True, "_salvage_method": salvage_method,
        "llm_extracted": numbers, "computed": computed,
    }
    write_json(EFFECTS_DIR / f"{paper_id}.{kind}.json", record)


def salvage_paper(paper_id: str, kinds_with_reasons: list[tuple[str, str]],
                  *, client, settings, cost, speed_mod, quality_mod) -> dict:
    """Salvage all (kind, reason) entries for one paper. Step 1 text-only first, then
    Step 2 PDF (with caching shared across kinds) only if step 1 was unconvertible."""
    pdf_exists = (PAPERS_DIR / f"{paper_id}.pdf").exists()
    results = {"step1_ok": 0, "step2_ok": 0, "still_failed": 0}

    needs_pdf: list[tuple[str, str, dict, str]] = []  # (kind, reason, numbers, notes)

    # ---- step 1: text-only ----
    for kind, reason in kinds_with_reasons:
        numbers, citation_key, _ = _call_text_pass(paper_id, kind, reason,
                                                    client=client, settings=settings, cost=cost)
        if numbers is None:
            needs_pdf.append((kind, reason, None, "step1 parse fail"))
            continue
        compute_fn = speed_mod.compute_speed if kind == "speed" else quality_mod.compute_quality
        computed = compute_fn(numbers)
        if _is_unconvertible(computed):
            needs_pdf.append((kind, reason, numbers, computed.get("notes", "")[:200]))
            # write intermediate result so it's audit-able
            _write_record(paper_id, kind, citation_key, numbers, computed, "step1_text_unconvertible")
        else:
            _write_record(paper_id, kind, citation_key, numbers, computed, "step1_text")
            results["step1_ok"] += 1
            log.info("step1 %s.%s OK method=%s", paper_id, kind, computed.get("computation_method"))

    if not needs_pdf or not pdf_exists:
        if needs_pdf and not pdf_exists:
            log.warning("paper %s has no PDF; skipping step2 for %d kinds", paper_id, len(needs_pdf))
        results["still_failed"] += len(needs_pdf)
        return results

    # ---- step 2: PDF, cache on first call so second call reads from cache ----
    citation_key, _, _ = _stage_1_artifacts(paper_id, needs_pdf[0][0])
    for i, (kind, reason, _, notes) in enumerate(needs_pdf):
        cache = (i == 0)  # first call writes cache; subsequent reads
        numbers = _call_pdf_pass(paper_id, kind, reason, notes,
                                 client=client, settings=settings, cost=cost, cache_pdf=cache)
        if numbers is None:
            results["still_failed"] += 1
            continue
        compute_fn = speed_mod.compute_speed if kind == "speed" else quality_mod.compute_quality
        computed = compute_fn(numbers)
        if _is_unconvertible(computed):
            _write_record(paper_id, kind, citation_key, numbers, computed, "step2_pdf_unconvertible")
            results["still_failed"] += 1
            log.info("step2 %s.%s STILL unconvertible", paper_id, kind)
        else:
            _write_record(paper_id, kind, citation_key, numbers, computed, "step2_pdf")
            results["step2_ok"] += 1
            log.info("step2 %s.%s OK method=%s", paper_id, kind, computed.get("computation_method"))

    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--paper", type=str, default=None)
    ap.add_argument("--kind", type=str, default=None, choices=[None, "speed", "quality"])
    ap.add_argument("--concurrency", type=int, default=4, help="Concurrent PAPERS (each may issue 2-4 sequential calls).")
    ap.add_argument("--skip-text", action="store_true", help="Go straight to PDF step (debug only).")
    args = ap.parse_args()

    settings = load_settings()
    client = get_client()
    cost = CostTracker(settings["pricing"])
    speed_mod, quality_mod = _load_compute_modules()

    targets = _failures_to_salvage()
    if args.paper:
        targets = [t for t in targets if t[0] == args.paper]
    if args.kind:
        targets = [t for t in targets if t[1] == args.kind]

    # Group by paper_id so speed+quality share the PDF cache
    by_paper: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for pid, kind, reason in targets:
        by_paper[pid].append((kind, reason))

    papers = list(by_paper.items())
    if args.limit:
        papers = papers[: args.limit]

    log.info("salvage papers: %d (total kind-targets: %d)", len(papers), sum(len(v) for _, v in papers))

    totals = {"step1_ok": 0, "step2_ok": 0, "still_failed": 0}

    def work(item):
        pid, kinds = item
        try:
            return salvage_paper(pid, kinds, client=client, settings=settings, cost=cost,
                                 speed_mod=speed_mod, quality_mod=quality_mod)
        except Exception as e:  # noqa: BLE001
            log.exception("salvage paper %s failed: %s", pid, e)
            return {"step1_ok": 0, "step2_ok": 0, "still_failed": len(kinds)}

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(work, item) for item in papers]
        for fut in as_completed(futs):
            r = fut.result()
            for k in totals:
                totals[k] += r.get(k, 0)

    log.info("salvage totals: step1_ok=%d  step2_ok=%d  still_failed=%d",
             totals["step1_ok"], totals["step2_ok"], totals["still_failed"])
    log.info("%s", cost.summary())


if __name__ == "__main__":
    main()
