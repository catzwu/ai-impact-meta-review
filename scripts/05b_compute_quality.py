"""Stage 5b: quality effect size (Hedges' g).

LLM extracts numbers; Python computes Cohen's d, applies Hedges correction, computes variance.

Fallback chain:
  A) means + SDs:   d = (M_ai - M_h) / s_pooled
  B) means + SEs:   SD = SE * sqrt(n)
  C) t + df + ns:   d = t * sqrt(1/n_h + 1/n_ai)
  D) F (df1=1) + ns: t = sqrt(F)
  E) reported d or g + ns

Sign flip: if smaller_is_better, multiply g by -1 so positive = AI helps.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CostTracker, LOG_DIR, OUTPUTS_DIR, REPO_ROOT, call_llm, get_client, get_logger,
    load_prompt, load_settings, output_path, parse_json_response, read_json, write_json,
)

log = get_logger("05b_compute_quality")
STAGE = "05_effect_sizes"
PREV_EXT = "01_extraction"
PREV_OUT = "03_outcome_classification"


def _hedges_correction(n_h: int, n_ai: int) -> float:
    return 1.0 - 3.0 / (4.0 * (n_h + n_ai) - 9.0)


def _var_d(d: float, n_h: int, n_ai: int) -> float:
    return (n_h + n_ai) / (n_h * n_ai) + (d ** 2) / (2.0 * (n_h + n_ai))


def _resolve_sd(sd, se, n):
    if sd is not None:
        return sd
    if se is not None and n is not None and n > 0:
        return se * math.sqrt(n)
    return None


def compute_quality(numbers: dict) -> dict:
    out: dict = {
        "cohens_d": None,
        "hedges_g": None,
        "variance": None,
        "computation_method": None,
        "confidence": None,
        "notes": numbers.get("notes", ""),
        "n_human": numbers.get("n_human"),
        "n_ai": numbers.get("n_ai"),
        "direction_convention": numbers.get("direction_convention"),
    }
    if not numbers.get("arm_pair_resolvable"):
        out["confidence"] = "low"
        out["notes"] += " | arm pair unresolvable"
        return out

    n_h, n_ai = numbers.get("n_human"), numbers.get("n_ai")
    direction = numbers.get("direction_convention")
    sign = -1.0 if direction == "smaller_is_better" else 1.0

    d = None
    method = None

    m_h, m_ai = numbers.get("m_human"), numbers.get("m_ai")
    sd_h = _resolve_sd(numbers.get("sd_human"), numbers.get("se_human"), n_h)
    sd_ai = _resolve_sd(numbers.get("sd_ai"), numbers.get("se_ai"), n_ai)

    # Level A/B: means + SDs/SEs
    if m_h is not None and m_ai is not None and sd_h is not None and sd_ai is not None and n_h and n_ai:
        s_pooled_sq = ((n_h - 1) * sd_h ** 2 + (n_ai - 1) * sd_ai ** 2) / (n_h + n_ai - 2)
        if s_pooled_sq > 0:
            s_pooled = math.sqrt(s_pooled_sq)
            d = (m_ai - m_h) / s_pooled
            method = "means_and_sds" if numbers.get("sd_human") is not None else "means_and_ses"

    # Level C: t + df + ns
    if d is None and numbers.get("reported_t") is not None and n_h and n_ai:
        t = numbers["reported_t"]
        d = t * math.sqrt(1.0 / n_h + 1.0 / n_ai)
        method = "t_statistic"

    # Level D: F (df1=1)
    if d is None and numbers.get("reported_F") is not None and numbers.get("reported_F_df1") == 1 and n_h and n_ai:
        t = math.sqrt(numbers["reported_F"])
        d = t * math.sqrt(1.0 / n_h + 1.0 / n_ai)
        method = "F_statistic_df1_1"

    # Within-subject: paired mean diff with SD of differences → Cohen's d_z
    if d is None:
        mdiff = numbers.get("mean_difference_within_subject")
        sd_diff = numbers.get("sd_difference_within_subject")
        if mdiff is not None and sd_diff is not None and sd_diff > 0:
            d = mdiff / sd_diff
            method = "within_subject_paired_d_z"

    # Level E: reported d / g
    if d is None and numbers.get("reported_cohens_d") is not None:
        d = numbers["reported_cohens_d"]
        method = "reported_d"
    if d is None and numbers.get("reported_hedges_g") is not None and n_h and n_ai:
        J = _hedges_correction(n_h, n_ai)
        d = numbers["reported_hedges_g"] / J
        method = "reported_g"

    if d is None or not n_h or not n_ai:
        out["confidence"] = "low"
        out["computation_method"] = "unconvertible"
        out["notes"] += " | insufficient numbers for d"
        return out

    d *= sign
    J = _hedges_correction(n_h, n_ai)
    g = J * d
    var_d = _var_d(d, n_h, n_ai)
    var_g = J ** 2 * var_d

    out["cohens_d"] = d
    out["hedges_g"] = g
    out["variance"] = var_g
    out["computation_method"] = method
    out["confidence"] = numbers.get("extraction_confidence", "medium")
    return out


def run_one(paper_id: str, *, client, settings, cost: CostTracker, force: bool = False) -> dict | None:
    out = OUTPUTS_DIR / STAGE / f"{paper_id}.quality.json"
    if not force and out.exists():
        return None
    ext_p = OUTPUTS_DIR / PREV_EXT / f"{paper_id}.json"
    oc_p = OUTPUTS_DIR / PREV_OUT / f"{paper_id}.json"
    if not (ext_p.exists() and oc_p.exists()):
        log.warning("missing prerequisites for %s", paper_id)
        return None
    ext = read_json(ext_p)
    oc = read_json(oc_p)
    primary_name = oc.get("primary_quality_outcome")
    if not primary_name:
        return None
    primary_obj = next((o for o in ext.get("outcomes", []) if o.get("outcome_name") == primary_name), None)
    stats_primary = [r for r in ext.get("reported_statistics", []) if r.get("outcome_name") == primary_name]
    payload = {
        "arms": ext.get("arms", []),
        "reported_statistics_primary": stats_primary,
        "primary_outcome": primary_obj,
    }
    prompt = load_prompt("compute_quality.txt")
    user = prompt + "\n\nINPUT:\n" + json.dumps(payload, indent=2)
    model = settings.get("stage_models", {}).get("compute_quality", settings["model"])
    res = call_llm(
        client=client, model=model,
        system="You output only valid JSON. Do not perform arithmetic; only extract numbers.",
        user=user,
        max_tokens=settings["max_tokens"]["compute_quality"],
        temperature=settings["temperatures"]["compute_quality"],
        retry_cfg=settings["retry"],
        log_path=LOG_DIR / "05b_compute_quality.jsonl",
        stage=STAGE, paper_id=paper_id,
    )
    cost.add(res.model, res.input_tokens, res.output_tokens)
    try:
        numbers = parse_json_response(res.text)
    except Exception as e:  # noqa: BLE001
        log.error("parse fail %s: %s", paper_id, e)
        write_json(out.with_suffix(".error.json"), {"paper_id": paper_id, "raw": res.text})
        return None
    computed = compute_quality(numbers)
    record = {
        "_paper_id": paper_id,
        "citation_key": ext.get("citation_key"),
        "kind": "quality",
        "llm_extracted": numbers,
        "computed": computed,
    }
    write_json(out, record)
    log.info("quality %s g=%s var=%s", paper_id, computed.get("hedges_g"), computed.get("variance"))
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", type=str, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    settings = load_settings()
    client = get_client()
    cost = CostTracker(settings["pricing"])
    ids = [args.paper] if args.paper else [p.stem for p in sorted((OUTPUTS_DIR / PREV_OUT).glob("*.json"))
                                            if not p.stem.endswith(".error")]
    for pid in ids:
        try:
            run_one(pid, client=client, settings=settings, cost=cost, force=args.force)
        except Exception as e:  # noqa: BLE001
            log.exception("fail %s: %s", pid, e)
    log.info("%s", cost.summary())


if __name__ == "__main__":
    main()
