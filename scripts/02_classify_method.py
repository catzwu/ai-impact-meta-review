"""Stage 2: classify the study design (rct / field / lab / DiD / observational / within / other)."""
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

log = get_logger("02_classify_method")
STAGE = "02_method_classification"
PREV = "01_extraction"


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
        "study_design_description": ext.get("study_design_description"),
        "population_description": ext.get("population_description"),
    }
    prompt = load_prompt("classify_method.txt")
    user = prompt + "\n\nINPUT:\n" + json.dumps(payload, indent=2)
    model = settings.get("stage_models", {}).get("classify_method", settings["model"])
    res = call_llm(
        client=client, model=model,
        system="You output only valid JSON.",
        user=user,
        max_tokens=settings["max_tokens"]["classify_method"],
        temperature=settings["temperatures"]["classify_method"],
        retry_cfg=settings["retry"],
        log_path=LOG_DIR / "02_classify_method.jsonl",
        stage=STAGE, paper_id=paper_id,
    )
    cost.add(res.model, res.input_tokens, res.output_tokens)
    try:
        data = parse_json_response(res.text)
    except Exception as e:  # noqa: BLE001
        log.error("parse fail %s: %s", paper_id, e)
        write_json(out.with_suffix(".error.json"), {"paper_id": paper_id, "raw": res.text})
        return None
    data["citation_key"] = ext.get("citation_key")
    write_json(out, data)
    log.info("classified method %s -> %s", paper_id, data.get("classification"))
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
