"""Stage 1b: structure verbatim quotes into the canonical extraction schema."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CostTracker, LOG_DIR, OUTPUTS_DIR, REPO_ROOT, call_llm, get_client, get_logger,
    load_prompt, load_settings, output_path, parse_json_response, read_json, write_json,
)

log = get_logger("01b_structure")
STAGE = "01_extraction"
PREV = "01a_quotes"


def run_one(paper_id: str, *, client, settings, cost: CostTracker, force: bool = False) -> dict | None:
    out = output_path(STAGE, paper_id)
    if not force and out.exists():
        return None
    prev = OUTPUTS_DIR / PREV / f"{paper_id}.json"
    if not prev.exists():
        log.warning("no quotes for %s", paper_id)
        return None
    quotes = read_json(prev)
    payload = {
        "citation_key": quotes.get("citation_key"),
        "authors": quotes.get("authors"),
        "title": quotes.get("title"),
        "year": quotes.get("year"),
        "venue": quotes.get("venue"),
        "quotes": quotes.get("quotes", {}),
        "notes_from_quote_stage": quotes.get("notes", ""),
    }
    prompt = load_prompt("extract.txt")
    user = prompt + "\n\n===QUOTES===\n" + json.dumps(payload, indent=2)
    model = settings.get("stage_models", {}).get("structure", settings["model"])
    res = call_llm(
        client=client, model=model,
        system="You output only valid JSON. Use only information present in the provided quotes.",
        user=user,
        max_tokens=settings["max_tokens"]["structure"],
        temperature=settings["temperatures"]["structure"],
        retry_cfg=settings["retry"],
        log_path=LOG_DIR / "01b_structure.jsonl",
        stage=STAGE, paper_id=paper_id,
    )
    cost.add(res.model, res.input_tokens, res.output_tokens)
    try:
        data = parse_json_response(res.text)
    except Exception as e:  # noqa: BLE001
        log.error("parse fail %s: %s", paper_id, e)
        write_json(out.with_suffix(".error.json"), {"paper_id": paper_id, "raw": res.text})
        return None
    data["_paper_id"] = paper_id
    write_json(out, data)
    log.info("structured %s (%d arms, %d outcomes, %d stats)",
             paper_id, len(data.get("arms") or []), len(data.get("outcomes") or []),
             len(data.get("reported_statistics") or []))
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
