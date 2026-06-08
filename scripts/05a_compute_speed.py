"""Stage 5a: speed effect size.

LLM extracts numbers; Python computes the log ratio and its variance deterministically.

Convention: positive log_ratio means AI helps.
  - smaller_is_better (time):       log(M_human / M_ai)
  - larger_is_better  (throughput): log(M_ai / M_human)

Fallbacks:
  - Means + SDs:           Var ≈ (SD_h / M_h)^2 / n_h + (SD_ai / M_ai)^2 / n_ai
  - Means + SE: SD = SE·sqrt(n)
  - Means + t-stat + ns:   derive SD_pooled from t-stat
  - reported_percent_change (paper's signed value) ± CI: log_ratio = log(1+pct), sign-flip for
    smaller_is_better
  - regression_coefficient (with log-transformed outcome): coef IS the log_ratio
  - regression_coefficient on levels with baseline mean: log(1 + coef/baseline)
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

log = get_logger("05a_compute_speed")
STAGE = "05_effect_sizes"
PREV_EXT = "01_extraction"
PREV_OUT = "03_outcome_classification"


def _resolve_sd(sd, se, n):
    if sd is not None:
        return sd
    if se is not None and n is not None and n > 0:
        return se * math.sqrt(n)
    return None


def compute_speed(numbers: dict) -> dict:
    out: dict = {
        "log_ratio": None,
        "log_ratio_variance": None,
        "percent_change_equivalent": None,
        "direction_convention": numbers.get("direction_convention"),
        "computation_method": None,
        "confidence": None,
        "notes": numbers.get("notes", ""),
        "n_human": numbers.get("n_human"),
        "n_ai": numbers.get("n_ai"),
    }
    if not numbers.get("arm_pair_resolvable"):
        out["notes"] += " | arm pair unresolvable"
        out["confidence"] = "low"
        return out

    m_h, m_ai = numbers.get("m_human"), numbers.get("m_ai")
    sd_h = _resolve_sd(numbers.get("sd_human"), numbers.get("se_human"), numbers.get("n_human"))
    sd_ai = _resolve_sd(numbers.get("sd_ai"), numbers.get("se_ai"), numbers.get("n_ai"))
    n_h, n_ai_ = numbers.get("n_human"), numbers.get("n_ai")
    direction = numbers.get("direction_convention")

    if m_h is not None and m_ai is not None and m_h > 0 and m_ai > 0:
        if direction == "smaller_is_better":
            log_ratio = math.log(m_h / m_ai)
        elif direction == "larger_is_better":
            log_ratio = math.log(m_ai / m_h)
        else:
            out["notes"] += " | unknown direction_convention; defaulting larger_is_better"
            log_ratio = math.log(m_ai / m_h)
        out["log_ratio"] = log_ratio
        out["percent_change_equivalent"] = math.exp(log_ratio) - 1
        if sd_h is not None and sd_ai is not None and n_h and n_ai_:
            var = (sd_h / m_h) ** 2 / n_h + (sd_ai / m_ai) ** 2 / n_ai_
            out["log_ratio_variance"] = var
            out["computation_method"] = "means_and_sds"
            out["confidence"] = numbers.get("extraction_confidence", "medium")
            return out
        # t-stat → variance path
        t = numbers.get("reported_t_statistic")
        if t is not None and n_h and n_ai_ and t != 0 and (m_ai - m_h) != 0:
            se_diff = abs(m_ai - m_h) / abs(t)
            sd_pooled_sq = se_diff ** 2 / (1.0 / n_h + 1.0 / n_ai_)
            var = sd_pooled_sq * (1.0 / (m_h ** 2 * n_h) + 1.0 / (m_ai ** 2 * n_ai_))
            out["log_ratio_variance"] = var
            out["computation_method"] = "means_and_t_stat"
            out["confidence"] = numbers.get("extraction_confidence", "medium")
            return out
        out["computation_method"] = "means_only_no_variance"
        out["confidence"] = "low"
        out["notes"] += " | SDs/SEs missing; variance not computable"
        return out

    # Within-subject paired mean difference: use baseline (m_human) if given, else fall back to pct path
    mdiff = numbers.get("mean_difference_within_subject")
    if mdiff is not None and m_h is not None and m_h > 0:
        new_m_ai = m_h + mdiff
        if new_m_ai > 0:
            if direction == "smaller_is_better":
                log_ratio = math.log(m_h / new_m_ai)
            else:
                log_ratio = math.log(new_m_ai / m_h)
            out["log_ratio"] = log_ratio
            out["percent_change_equivalent"] = math.exp(log_ratio) - 1
            out["computation_method"] = "within_subject_mean_diff_no_variance"
            out["confidence"] = "low"
            out["notes"] += " | within-subject mean diff; variance not computable without paired SD"
            return out

    pct = numbers.get("reported_percent_change")
    if pct is not None:
        # `pct` is paper's signed change. For smaller_is_better, AI helps when pct < 0 → flip sign.
        if direction == "smaller_is_better":
            log_ratio = -math.log(1 + pct) if (1 + pct) > 0 else None
        else:
            log_ratio = math.log(1 + pct) if (1 + pct) > 0 else None
        if log_ratio is None:
            out["computation_method"] = "unconvertible"
            out["confidence"] = "low"
            out["notes"] += " | invalid percent change (1 + pct <= 0)"
            return out
        out["log_ratio"] = log_ratio
        out["percent_change_equivalent"] = math.exp(log_ratio) - 1
        lo, hi = numbers.get("reported_percent_change_ci_low"), numbers.get("reported_percent_change_ci_high")
        if lo is not None and hi is not None and (1 + lo) > 0 and (1 + hi) > 0:
            log_lo, log_hi = math.log(1 + lo), math.log(1 + hi)
            se = abs(log_hi - log_lo) / (2 * 1.96)
            out["log_ratio_variance"] = se ** 2
            out["computation_method"] = "reported_pct_change_with_ci"
            out["confidence"] = numbers.get("extraction_confidence", "medium")
        else:
            out["computation_method"] = "reported_pct_change_no_ci"
            out["confidence"] = "low"
            out["notes"] += " | CI missing; variance not computable"
        return out

    # Regression coefficient (DiD, OLS, IV, RD, fixed-effects)
    coef = numbers.get("regression_coefficient_value")
    se_coef = numbers.get("regression_coefficient_se")
    if coef is not None:
        if numbers.get("regression_outcome_is_log"):
            out["log_ratio"] = coef
            if se_coef is not None:
                out["log_ratio_variance"] = se_coef ** 2
                out["computation_method"] = "regression_coef_log_outcome"
                out["confidence"] = numbers.get("extraction_confidence", "medium")
            else:
                out["computation_method"] = "regression_coef_log_outcome_no_se"
                out["confidence"] = "low"
            out["percent_change_equivalent"] = math.exp(coef) - 1
            return out
        baseline = numbers.get("regression_baseline_mean")
        if baseline is not None and baseline > 0 and (1 + coef / baseline) > 0:
            out["log_ratio"] = math.log(1 + coef / baseline)
            out["percent_change_equivalent"] = coef / baseline
            if se_coef is not None:
                out["log_ratio_variance"] = (se_coef / (baseline + coef)) ** 2
                out["computation_method"] = "regression_coef_level_with_baseline"
                out["confidence"] = "medium"
            else:
                out["computation_method"] = "regression_coef_level_with_baseline_no_se"
                out["confidence"] = "low"
            return out
        out["notes"] += " | regression coefficient on levels but baseline mean unknown"
        out["computation_method"] = "unconvertible"
        out["confidence"] = "low"
        return out

    out["notes"] += " | no usable numbers"
    out["confidence"] = "low"
    out["computation_method"] = "unconvertible"
    return out


def run_one(paper_id: str, *, client, settings, cost: CostTracker, force: bool = False) -> dict | None:
    out = OUTPUTS_DIR / STAGE / f"{paper_id}.speed.json"
    if not force and out.exists():
        return None
    ext_p = OUTPUTS_DIR / PREV_EXT / f"{paper_id}.json"
    oc_p = OUTPUTS_DIR / PREV_OUT / f"{paper_id}.json"
    if not (ext_p.exists() and oc_p.exists()):
        log.warning("missing prerequisites for %s", paper_id)
        return None
    ext = read_json(ext_p)
    oc = read_json(oc_p)
    primary_name = oc.get("primary_speed_outcome")
    if not primary_name:
        return None
    primary_obj = next((o for o in ext.get("outcomes", []) if o.get("outcome_name") == primary_name), None)
    stats_primary = [r for r in ext.get("reported_statistics", []) if r.get("outcome_name") == primary_name]
    payload = {
        "arms": ext.get("arms", []),
        "reported_statistics_primary": stats_primary,
        "primary_outcome": primary_obj,
    }
    prompt = load_prompt("compute_speed.txt")
    user = prompt + "\n\nINPUT:\n" + json.dumps(payload, indent=2)
    model = settings.get("stage_models", {}).get("compute_speed", settings["model"])
    res = call_llm(
        client=client, model=model,
        system="You output only valid JSON. Do not perform arithmetic; only extract numbers.",
        user=user,
        max_tokens=settings["max_tokens"]["compute_speed"],
        temperature=settings["temperatures"]["compute_speed"],
        retry_cfg=settings["retry"],
        log_path=LOG_DIR / "05a_compute_speed.jsonl",
        stage=STAGE, paper_id=paper_id,
    )
    cost.add(res.model, res.input_tokens, res.output_tokens)
    try:
        numbers = parse_json_response(res.text)
    except Exception as e:  # noqa: BLE001
        log.error("parse fail %s: %s", paper_id, e)
        write_json(out.with_suffix(".error.json"), {"paper_id": paper_id, "raw": res.text})
        return None
    computed = compute_speed(numbers)
    record = {
        "_paper_id": paper_id,
        "citation_key": ext.get("citation_key"),
        "kind": "speed",
        "llm_extracted": numbers,
        "computed": computed,
    }
    write_json(out, record)
    log.info("speed %s log_ratio=%s var=%s", paper_id, computed.get("log_ratio"), computed.get("log_ratio_variance"))
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
