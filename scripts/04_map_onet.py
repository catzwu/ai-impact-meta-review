"""Stage 4: map paper to O*NET work activity or occupation. Uses prompt caching."""
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

log = get_logger("04_map_onet")
STAGE = "04_onet_mapping"
PREV = "01_extraction"
CONFIG = REPO_ROOT / "config"


def _load_onet() -> tuple[list, list]:
    act = json.loads((CONFIG / "onet_activities.json").read_text())
    occ = json.loads((CONFIG / "onet_occupations.json").read_text())
    return act, occ


def _validate_code(code: str, activities: list, occupations: list, mapping_type: str) -> bool:
    pool = activities if mapping_type == "work_activity" else occupations
    return any(item.get("code") == code for item in pool)


def _build_cached_system(activities: list, occupations: list) -> list[dict]:
    """Cached system block: prompt + O*NET list. Identical across all calls in the batch."""
    prompt = load_prompt("map_onet.txt")
    onet_payload = json.dumps({"onet_activities": activities, "onet_occupations": occupations}, indent=2)
    return [
        {"type": "text", "text": "You output only valid JSON. Use only codes provided in the input."},
        {
            "type": "text",
            "text": prompt + "\n\nONET_LISTS:\n" + onet_payload,
            "cache_control": {"type": "ephemeral"},
        },
    ]


def run_one(paper_id: str, *, client, settings, cost: CostTracker, cached_system: list[dict],
            activities: list, occupations: list, force: bool = False) -> dict | None:
    out = output_path(STAGE, paper_id)
    if not force and out.exists():
        return None
    prev = OUTPUTS_DIR / PREV / f"{paper_id}.json"
    if not prev.exists():
        log.warning("no extraction for %s", paper_id)
        return None
    ext = read_json(prev)
    paper_payload = {
        "primary_task_description": ext.get("primary_task_description"),
        "population_description": ext.get("population_description"),
    }
    user = "PAPER:\n" + json.dumps(paper_payload, indent=2)
    model = settings.get("stage_models", {}).get("map_onet", settings["model"])
    res = call_llm(
        client=client, model=model,
        system=cached_system,
        user=user,
        max_tokens=settings["max_tokens"]["map_onet"],
        temperature=settings["temperatures"]["map_onet"],
        retry_cfg=settings["retry"],
        log_path=LOG_DIR / "04_map_onet.jsonl",
        stage=STAGE, paper_id=paper_id,
    )
    cost.add(res.model, res.input_tokens, res.output_tokens,
             cache_write=res.cache_creation_input_tokens,
             cache_read=res.cache_read_input_tokens)
    try:
        data = parse_json_response(res.text)
    except Exception as e:  # noqa: BLE001
        log.error("parse fail %s: %s", paper_id, e)
        write_json(out.with_suffix(".error.json"), {"paper_id": paper_id, "raw": res.text})
        return None

    code = data.get("onet_code")
    mtype = data.get("mapping_type")
    if not _validate_code(code, activities, occupations, mtype):
        data["_code_validation_warning"] = f"Code {code} not found in {mtype} list"
        log.warning("invalid code %s for %s (%s)", code, paper_id, mtype)
        # Try label-match snap to a canonical code
        label = data.get("onet_label")
        pool = activities if mtype == "work_activity" else occupations
        for item in pool:
            if item.get("label") == label:
                data["_code_validation_warning"] += f"; snapped to '{item['code']}' by label match"
                data["onet_code"] = item["code"]
                break

    data["citation_key"] = ext.get("citation_key")
    write_json(out, data)
    log.info("mapped %s -> %s %s", paper_id, mtype, data.get("onet_code"))
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", type=str, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    settings = load_settings()
    client = get_client()
    cost = CostTracker(settings["pricing"])
    activities, occupations = _load_onet()
    cached_system = _build_cached_system(activities, occupations)
    ids = [args.paper] if args.paper else [p.stem for p in sorted((OUTPUTS_DIR / PREV).glob("*.json"))
                                            if not p.stem.endswith(".error")]
    for pid in ids:
        try:
            run_one(pid, client=client, settings=settings, cost=cost,
                    cached_system=cached_system, activities=activities, occupations=occupations,
                    force=args.force)
        except Exception as e:  # noqa: BLE001
            log.exception("fail %s: %s", pid, e)
    log.info("%s", cost.summary())


if __name__ == "__main__":
    main()
