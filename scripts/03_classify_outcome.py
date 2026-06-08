"""Stage 3: classify each outcome speed/quality + pick primaries (with exact-name validator)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CostTracker, LOG_DIR, OUTPUTS_DIR, call_llm, get_client, get_logger, load_prompt,
    load_settings, output_path, parse_json_response, read_json, write_json,
)

log = get_logger("03_classify_outcome")
STAGE = "03_outcome_classification"
PREV = "01_extraction"


def _fuzzy_match(candidate: str, choices: list[str]) -> str | None:
    """Return the choice with highest token-overlap with candidate, if confident (>=50%)."""
    def toks(s: str) -> set[str]:
        return {t for t in s.lower().replace("-", "_").split("_") if t}
    cand_t = toks(candidate)
    if not cand_t or not choices:
        return None
    best, best_score = None, 0.0
    for c in choices:
        c_t = toks(c)
        if not c_t:
            continue
        overlap = len(cand_t & c_t)
        score = overlap / max(len(cand_t), len(c_t))
        if score > best_score:
            best, best_score = c, score
    return best if best_score >= 0.5 else None


def run_one(paper_id: str, *, client, settings, cost: CostTracker, force: bool = False) -> dict | None:
    out = output_path(STAGE, paper_id)
    if not force and out.exists():
        return None
    prev = OUTPUTS_DIR / PREV / f"{paper_id}.json"
    if not prev.exists():
        log.warning("no extraction for %s", paper_id)
        return None
    ext = read_json(prev)
    payload = {
        "primary_task_description": ext.get("primary_task_description"),
        "outcomes": ext.get("outcomes", []),
    }
    prompt = load_prompt("classify_outcome.txt")
    user = prompt + "\n\nINPUT:\n" + json.dumps(payload, indent=2)
    model = settings.get("stage_models", {}).get("classify_outcome", settings["model"])
    res = call_llm(
        client=client, model=model,
        system="You output only valid JSON.",
        user=user,
        max_tokens=settings["max_tokens"]["classify_outcome"],
        temperature=settings["temperatures"]["classify_outcome"],
        retry_cfg=settings["retry"],
        log_path=LOG_DIR / "03_classify_outcome.jsonl",
        stage=STAGE, paper_id=paper_id,
    )
    cost.add(res.model, res.input_tokens, res.output_tokens)
    try:
        data = parse_json_response(res.text)
    except Exception as e:  # noqa: BLE001
        log.error("parse fail %s: %s", paper_id, e)
        write_json(out.with_suffix(".error.json"), {"paper_id": paper_id, "raw": res.text})
        return None

    # Validate primary_*_outcome against the input outcomes list; fuzzy-match or null
    valid_names = [o.get("outcome_name") for o in ext.get("outcomes") or [] if o.get("outcome_name")]
    valid_set = set(valid_names)
    notes = []
    for key in ("primary_speed_outcome", "primary_quality_outcome"):
        v = data.get(key)
        if v and v not in valid_set:
            best = _fuzzy_match(v, valid_names)
            if best:
                notes.append(f"{key} '{v}' → '{best}' (fuzzy-matched)")
                data[key] = best
            else:
                notes.append(f"{key} '{v}' hallucinated; no fuzzy match found; set to null")
                data[key] = None
    if notes:
        data.setdefault("_validation_notes", []).extend(notes)
        log.warning("name fix on %s: %s", paper_id, "; ".join(notes))

    data["citation_key"] = ext.get("citation_key")
    write_json(out, data)
    log.info("classified outcome %s -> speed=%s quality=%s",
             paper_id, data.get("primary_speed_outcome"), data.get("primary_quality_outcome"))
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", type=str, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    settings = load_settings()
    client = get_client()
    cost = CostTracker(settings["pricing"])
    ids = [args.paper] if args.paper else [p.stem for p in sorted((OUTPUTS_DIR / PREV).glob("*.json"))
                                            if not p.stem.endswith(".error")]
    for pid in ids:
        try:
            run_one(pid, client=client, settings=settings, cost=cost, force=args.force)
        except Exception as e:  # noqa: BLE001
            log.exception("fail %s: %s", pid, e)
    log.info("%s", cost.summary())


if __name__ == "__main__":
    main()
