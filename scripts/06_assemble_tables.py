"""Stage 6: assemble final tables (speed_table.csv, quality_table.csv, papers_excluded.csv). No LLM."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUTPUTS_DIR, get_logger, read_json  # noqa: E402

log = get_logger("06_assemble")

SPEED_COLS = [
    "citation_key", "authors", "year", "study_design",
    "mapping_type", "onet_code", "onet_label",
    "log_ratio", "log_ratio_variance", "percent_change_equivalent",
    "n_human", "n_ai", "computation_method", "confidence", "notes",
]
QUALITY_COLS = [
    "citation_key", "authors", "year", "study_design",
    "mapping_type", "onet_code", "onet_label",
    "hedges_g", "variance",
    "n_human", "n_ai", "computation_method", "confidence", "notes",
]
EXCLUDED_COLS = ["paper_id", "stage_failed", "reason"]


def _safe_read(p: Path):
    if not p.exists():
        return None
    try:
        return read_json(p)
    except Exception:  # noqa: BLE001
        return None


def assemble():
    final_dir = OUTPUTS_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    speed_rows, quality_rows, excluded = [], [], []

    ext_dir = OUTPUTS_DIR / "01_extraction"
    paper_ids = sorted(p.stem for p in ext_dir.glob("*.json") if not p.stem.endswith(".error"))

    for pid in paper_ids:
        ext = _safe_read(ext_dir / f"{pid}.json")
        method = _safe_read(OUTPUTS_DIR / "02_method_classification" / f"{pid}.json")
        outcome = _safe_read(OUTPUTS_DIR / "03_outcome_classification" / f"{pid}.json")
        onet = _safe_read(OUTPUTS_DIR / "04_onet_mapping" / f"{pid}.json")
        speed_eff = _safe_read(OUTPUTS_DIR / "05_effect_sizes" / f"{pid}.speed.json")
        quality_eff = _safe_read(OUTPUTS_DIR / "05_effect_sizes" / f"{pid}.quality.json")

        if ext is None:
            excluded.append({"paper_id": pid, "stage_failed": "01_extraction", "reason": "missing extraction"})
            continue
        if method is None:
            excluded.append({"paper_id": pid, "stage_failed": "02_method_classification", "reason": "missing"})
            continue
        # Out-of-scope: review/policy/simulation papers (check method BEFORE outcome)
        if method.get("classification") == "other":
            excluded.append({"paper_id": pid, "stage_failed": "02_method_classification",
                             "reason": f"study_design=other ({(method.get('rationale') or '')[:120]})"})
            continue
        if outcome is None:
            excluded.append({"paper_id": pid, "stage_failed": "03_outcome_classification", "reason": "missing"})
            continue
        if onet is None:
            excluded.append({"paper_id": pid, "stage_failed": "04_onet_mapping", "reason": "missing"})
            continue
        if speed_eff is None and quality_eff is None:
            excluded.append({"paper_id": pid, "stage_failed": "05_effect_sizes",
                             "reason": "no speed or quality primary outcome"})
            continue

        base = {
            "citation_key": ext.get("citation_key", pid),
            "authors": "; ".join(ext.get("authors") or []),
            "year": ext.get("year"),
            "study_design": method.get("classification"),
            "mapping_type": onet.get("mapping_type"),
            "onet_code": onet.get("onet_code"),
            "onet_label": onet.get("onet_label"),
        }

        def _row(eff):
            c = eff.get("computed", {})
            return {**base,
                    "n_human": c.get("n_human"),
                    "n_ai": c.get("n_ai"),
                    "computation_method": c.get("computation_method"),
                    "confidence": c.get("confidence"),
                    "notes": c.get("notes", "")}, c

        if speed_eff is not None:
            row, c = _row(speed_eff)
            if c.get("computation_method") in (None, "unconvertible"):
                excluded.append({"paper_id": pid, "stage_failed": "05_speed",
                                 "reason": c.get("notes", "unconvertible")})
            else:
                speed_rows.append({**row,
                                   "log_ratio": c.get("log_ratio"),
                                   "log_ratio_variance": c.get("log_ratio_variance"),
                                   "percent_change_equivalent": c.get("percent_change_equivalent")})
        if quality_eff is not None:
            row, c = _row(quality_eff)
            if c.get("computation_method") in (None, "unconvertible"):
                excluded.append({"paper_id": pid, "stage_failed": "05_quality",
                                 "reason": c.get("notes", "unconvertible")})
            else:
                quality_rows.append({**row,
                                     "hedges_g": c.get("hedges_g"),
                                     "variance": c.get("variance")})

    def _write(path: Path, cols: list[str], rows: list[dict]) -> None:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c) for c in cols})

    _write(final_dir / "speed_table.csv", SPEED_COLS, speed_rows)
    _write(final_dir / "quality_table.csv", QUALITY_COLS, quality_rows)
    _write(final_dir / "papers_excluded.csv", EXCLUDED_COLS, excluded)
    log.info("speed=%d quality=%d excluded=%d", len(speed_rows), len(quality_rows), len(excluded))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    assemble()


if __name__ == "__main__":
    main()
