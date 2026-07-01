"""Occupational transitions data + augmentation lookup.

Loads the Jan-2021 BLS-style occupation transitions xlsx (soc1 -> soc2 with
transition_share conditional on soc1) and the latest analysis run's per-occupation
augmentation estimate so the /transitions page can render a heatmap and drill down.

Physical SOC majors (37, 45, 47, 49, 51, 53) are dropped from both source and
target — matching the convention in run_analysis.py.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from common import OUTPUTS_DIR
import run_analysis as RA

XLSX_PATH = Path(
    "/Users/catherinewu/Documents/nyu/yr2/msft/"
    "Occ Transitions Public Data Set (Jan 2021)/"
    "occupation_transitions_public_data_set.xlsx"
)
RUNS_DIR = OUTPUTS_DIR / "analysis_runs"

# Physical SOC majors to drop on both sides.
PHYSICAL_SOC_MAJORS = set(RA.DEFAULT_EXCLUDED_SOCS)  # {"37","45","47","49","51","53"}


def _load_raw() -> pd.DataFrame:
    df = pd.read_excel(XLSX_PATH)
    df = df.rename(columns=str.strip)
    # Coerce soc to 6-digit string form (e.g. "13-2011").
    df["soc1"] = df["soc1"].astype(str).str.strip()
    df["soc2"] = df["soc2"].astype(str).str.strip()
    df["soc1_major"] = df["soc1"].str[:2]
    df["soc2_major"] = df["soc2"].str[:2]
    # Drop physical majors on both sides.
    df = df[~df["soc1_major"].isin(PHYSICAL_SOC_MAJORS)]
    df = df[~df["soc2_major"].isin(PHYSICAL_SOC_MAJORS)]
    return df.reset_index(drop=True)


def _latest_run_dir() -> Path | None:
    if not RUNS_DIR.exists():
        return None
    runs = sorted(p for p in RUNS_DIR.iterdir() if p.is_dir())
    return runs[-1] if runs else None


def _estimates_by_soc6(run_dir: Path) -> dict[str, float]:
    """Map 6-digit SOC -> mean augmentation estimate across O*NET sub-codes.

    occupation_impacts.csv codes look like ``13-2011.00`` / ``29-1141.01``.
    Multiple sub-codes can share a 6-digit prefix; we average them.
    """
    p = run_dir / "occupation_impacts.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    df["soc6"] = df["code"].astype(str).str[:7]  # "13-2011"
    out = df.groupby("soc6")["estimate"].mean().to_dict()
    return {k: float(v) for k, v in out.items()}


def _minor_code(soc6: str) -> str:
    """13-2011 -> 13-2000 (the SOC minor group)."""
    return soc6[:4] + "000"


@lru_cache(maxsize=1)
def get_data() -> dict:
    """Return a JSON-serializable bundle for the /transitions page.

    Cached for the lifetime of the Flask process. To pick up new analysis runs
    without restart, call ``invalidate()``.

    The heatmap is at the SOC minor-group level (3-digit prefix, e.g. ``13-2000``).
    Drill-down is at the underlying 6-digit occupation level.

    Shape:
        {
          minors:   [{code: "13-2000", title, major: "13", n_occs}, ...],
          minor_matrix: [[share, ...], ...]      # row-stochastic over kept minors
          occs:     [{code, title, major, minor}, ...],
          rows:     [[[j, share], ...], ...],    # per-source-occ outgoing edges (6-digit)
          incoming: [[[i, share], ...], ...],    # per-target-occ incoming edges (6-digit)
          minor_to_occs: {minor_code: [occ_idx, ...]},
          totals:   [total_obs, ...],            # by source occ
          estimates: {soc6: float},
          soc_names: {major: name},
          run_id:   str | None,
        }
    """
    df = _load_raw()

    # ---- occupation-level (for drill-down) ----
    code_to_title: dict[str, str] = {}
    for code, name in zip(df["soc1"], df["soc1_name"]):
        code_to_title.setdefault(code, name)
    for code, name in zip(df["soc2"], df["soc2_name"]):
        code_to_title.setdefault(code, name)
    occ_codes = sorted(code_to_title.keys())
    occ_idx = {c: i for i, c in enumerate(occ_codes)}
    occs = [
        {"code": c, "title": code_to_title[c], "major": c[:2], "minor": _minor_code(c)}
        for c in occ_codes
    ]

    n_occ = len(occ_codes)
    rows: list[list[list[float]]] = [[] for _ in range(n_occ)]
    incoming: list[list[list[float]]] = [[] for _ in range(n_occ)]
    for s, t, share in zip(df["soc1"].values, df["soc2"].values, df["transition_share"].values):
        i, j = occ_idx[s], occ_idx[t]
        rows[i].append([j, float(share)])
        incoming[j].append([i, float(share)])
    for r in rows:
        r.sort(key=lambda x: -x[1])
    for r in incoming:
        r.sort(key=lambda x: -x[1])

    totals_map = df.groupby("soc1")["total_obs"].first().to_dict()
    totals = [float(totals_map.get(c, 0.0)) for c in occ_codes]

    # ---- minor-group aggregation for the heatmap ----
    # Weighted by source total_obs: share(g->h) = sum_{s in g, t in h} share(s,t) * obs(s)
    #                                            / sum_{s in g} obs(s)
    minor_codes = sorted({_minor_code(c) for c in occ_codes})
    minor_idx = {m: i for i, m in enumerate(minor_codes)}
    n_min = len(minor_codes)

    # minor_to_occs (for drill-down by minor group)
    minor_to_occs: dict[str, list[int]] = {m: [] for m in minor_codes}
    for i, c in enumerate(occ_codes):
        minor_to_occs[_minor_code(c)].append(i)

    # Source-minor total_obs for normalization
    src_minor_total = [0.0] * n_min
    for c in occ_codes:
        m = _minor_code(c)
        src_minor_total[minor_idx[m]] += totals_map.get(c, 0.0)

    minor_matrix = [[0.0] * n_min for _ in range(n_min)]
    for s, t, share in zip(df["soc1"].values, df["soc2"].values, df["transition_share"].values):
        g = minor_idx[_minor_code(s)]
        h = minor_idx[_minor_code(t)]
        minor_matrix[g][h] += float(share) * totals_map.get(s, 0.0)
    for g in range(n_min):
        denom = src_minor_total[g]
        if denom > 0:
            minor_matrix[g] = [v / denom for v in minor_matrix[g]]

    # Build minor label = "<code> · <largest member title>"
    # Use the occupation with the largest total_obs as a representative.
    rep_title: dict[str, str] = {}
    for m, occ_is in minor_to_occs.items():
        if not occ_is:
            rep_title[m] = m
            continue
        best_i = max(occ_is, key=lambda i: totals[i])
        rep_title[m] = code_to_title[occ_codes[best_i]]
    minors = [
        {
            "code": m,
            "title": rep_title[m],
            "major": m[:2],
            "n_occs": len(minor_to_occs[m]),
        }
        for m in minor_codes
    ]

    run_dir = _latest_run_dir()
    estimates = _estimates_by_soc6(run_dir) if run_dir else {}

    return {
        "minors": minors,
        "minor_matrix": minor_matrix,
        "occs": occs,
        "rows": rows,
        "incoming": incoming,
        "minor_to_occs": minor_to_occs,
        "totals": totals,
        "estimates": estimates,
        "soc_names": RA.SOC_NAMES,
        "run_id": run_dir.name if run_dir else None,
    }


def invalidate() -> None:
    get_data.cache_clear()
