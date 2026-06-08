"""Aggregate speed_table.csv + quality_table.csv into per-O*NET impact files for upload.

Outputs:
  outputs/final/onet_occupations_impact.csv
  outputs/final/onet_activities_impact.csv

If multiple studies map to the same O*NET code, the impact is the inverse-variance-weighted
mean across studies (if every study reports variance), or a simple arithmetic mean otherwise
(tagged in `*_aggregation_method`).

Convention: positive values = AI helps.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "outputs" / "final"


def _f(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def _weighted_mean(values: list[float], variances: list[float | None]) -> tuple[float | None, float | None, str]:
    if not values:
        return None, None, "n/a"
    if all(v is not None and v > 0 for v in variances):
        weights = [1.0 / v for v in variances]
        wsum = sum(weights)
        mean = sum(w * x for w, x in zip(weights, values)) / wsum
        se = math.sqrt(1.0 / wsum)
        return mean, se, "iv_weighted"
    mean = sum(values) / len(values)
    return mean, None, "simple_mean"


def collect_from_iters(speed_rows, quality_rows):
    """Aggregate raw per-study rows (already passed through any edit/dedup layer)
    into per-O*NET occ + act rows. `speed_rows` / `quality_rows` are iterables of
    dicts shaped like the speed_table.csv / quality_table.csv columns."""
    speed_by_key: dict[tuple, list[dict]] = defaultdict(list)
    quality_by_key: dict[tuple, list[dict]] = defaultdict(list)

    for r in speed_rows:
        key = (r.get("mapping_type", ""), r.get("onet_code", ""), r.get("onet_label", ""))
        if not key[1]:
            continue
        speed_by_key[key].append({
            "citation_key": r.get("citation_key", ""),
            "log_ratio": _f(r.get("log_ratio")),
            "log_ratio_variance": _f(r.get("log_ratio_variance")),
        })
    for r in quality_rows:
        key = (r.get("mapping_type", ""), r.get("onet_code", ""), r.get("onet_label", ""))
        if not key[1]:
            continue
        quality_by_key[key].append({
            "citation_key": r.get("citation_key", ""),
            "hedges_g": _f(r.get("hedges_g")),
            "variance": _f(r.get("variance")),
        })

    return _aggregate(speed_by_key, quality_by_key)


def collect():
    """File-based aggregation (legacy: reads from outputs/final/{speed,quality}_table.csv)."""
    with open(FINAL / "speed_table.csv") as f:
        speed_rows = list(csv.DictReader(f))
    with open(FINAL / "quality_table.csv") as f:
        quality_rows = list(csv.DictReader(f))
    return collect_from_iters(speed_rows, quality_rows)


def _aggregate(speed_by_key, quality_by_key):

    all_keys = set(speed_by_key) | set(quality_by_key)
    rows_occ, rows_act = [], []

    for key in sorted(all_keys, key=lambda k: (k[0], k[2])):
        mapping_type, code, label = key
        speed_studies = speed_by_key.get(key, [])
        quality_studies = quality_by_key.get(key, [])

        speed_vals = [s["log_ratio"] for s in speed_studies if s["log_ratio"] is not None]
        speed_vars = [s["log_ratio_variance"] for s in speed_studies if s["log_ratio"] is not None]
        speed_mean, speed_se, speed_method = _weighted_mean(speed_vals, speed_vars)

        qual_vals = [q["hedges_g"] for q in quality_studies if q["hedges_g"] is not None]
        qual_vars = [q["variance"] for q in quality_studies if q["hedges_g"] is not None]
        qual_mean, qual_se, qual_method = _weighted_mean(qual_vals, qual_vars)

        contributors = sorted(set(s["citation_key"] for s in speed_studies)
                              | set(q["citation_key"] for q in quality_studies))

        row = {
            "code": code,
            "label": label,
            "n_studies_speed": len(speed_vals),
            "speed_log_ratio_mean": speed_mean,
            "speed_log_ratio_se": speed_se,
            "speed_pct_change_equiv": (math.exp(speed_mean) - 1) if speed_mean is not None else None,
            "speed_aggregation_method": speed_method,
            "n_studies_quality": len(qual_vals),
            "quality_hedges_g_mean": qual_mean,
            "quality_hedges_g_se": qual_se,
            "quality_aggregation_method": qual_method,
            "contributing_studies": "; ".join(contributors),
        }

        if mapping_type == "occupation":
            rows_occ.append(row)
        else:
            rows_act.append(row)

    return rows_occ, rows_act


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def main() -> None:
    rows_occ, rows_act = collect()
    write_csv(FINAL / "onet_occupations_impact.csv", rows_occ)
    write_csv(FINAL / "onet_activities_impact.csv", rows_act)
    print(f"occupations rows: {len(rows_occ)}")
    print(f"activities rows:  {len(rows_act)}")
    print(f"wrote {FINAL / 'onet_occupations_impact.csv'}")
    print(f"wrote {FINAL / 'onet_activities_impact.csv'}")


if __name__ == "__main__":
    main()
