"""Stage 1a: verbatim quote extraction (one LLM call per paper)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CostTracker, LOG_DIR, REPO_ROOT, call_llm, extract_pdf_text, get_client, get_logger,
    load_prompt, load_settings, output_path, paper_id_from_path, parse_json_response,
    write_json,
)

log = get_logger("01a_extract_quotes")
STAGE = "01a_quotes"


def run_one(pdf_path: Path, *, client, settings, cost: CostTracker, force: bool = False) -> dict | None:
    paper_id = paper_id_from_path(pdf_path)
    out = output_path(STAGE, paper_id)
    if not force and out.exists():
        log.info("skip %s (cached)", paper_id)
        return None
    text = extract_pdf_text(pdf_path)
    if not text.strip():
        log.error("empty PDF text for %s", paper_id)
        write_json(out.with_suffix(".error.json"), {"paper_id": paper_id, "error": "empty_pdf_text"})
        return None
    prompt = load_prompt("extract_quotes.txt")
    user = prompt + "\n\n===PAPER===\n" + text
    model = settings.get("stage_models", {}).get("extract_quotes", settings["model"])
    res = call_llm(
        client=client, model=model,
        system="You output only valid JSON. No markdown, no commentary.",
        user=user,
        max_tokens=settings["max_tokens"]["extract_quotes"],
        temperature=settings["temperatures"]["extract_quotes"],
        retry_cfg=settings["retry"],
        log_path=LOG_DIR / "01a_extract_quotes.jsonl",
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
    data["_source_pdf"] = str(pdf_path.relative_to(REPO_ROOT)) if pdf_path.is_relative_to(REPO_ROOT) else str(pdf_path)
    write_json(out, data)
    log.info("quoted %s (%d statistical_results)", paper_id,
             len((data.get("quotes") or {}).get("statistical_results", [])))
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers-dir", type=Path, default=REPO_ROOT / "papers")
    ap.add_argument("--paper", type=str, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    settings = load_settings()
    client = get_client()
    cost = CostTracker(settings["pricing"])
    pdfs = [args.papers_dir / f"{args.paper}.pdf"] if args.paper else sorted(args.papers_dir.glob("*.pdf"))
    for p in pdfs:
        if not p.exists():
            log.error("missing pdf: %s", p); continue
        try:
            run_one(p, client=client, settings=settings, cost=cost, force=args.force)
        except Exception as e:  # noqa: BLE001
            log.exception("fail %s: %s", p.name, e)
    log.info("%s", cost.summary())


if __name__ == "__main__":
    main()
