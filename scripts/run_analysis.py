"""AI-impact imputation analysis — pipeline.py + run_final_speed_baseline.py
wrapped as a parameterizable function for the review app.

Reads:
  pipeline-run/outputs/final/onet_occupations_impact.csv
  pipeline-run/outputs/final/onet_activities_impact.csv
  ONET_XLSX_PATH (Work Activities.xlsx)
  FELTEN_AIOE_PATH (optional, speed-only baseline)

Writes per-run output dir under pipeline-run/outputs/analysis_runs/<run_id>/:
  occupation_impacts.csv
  activity_impacts.csv
  run.json   (params + metadata)
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import bmat, csr_matrix, diags

# Import pipeline.py from the user's analysis dir (parent of pipeline-run)
ANALYSIS_DIR = Path("/Users/catherinewu/Documents/nyu/yr2/msft")
sys.path.insert(0, str(ANALYSIS_DIR))
import importlib.util
spec = importlib.util.spec_from_file_location("pipeline", ANALYSIS_DIR / "pipeline.py")
PIPELINE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PIPELINE)

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from common import OUTPUTS_DIR  # noqa: E402

# --- Config paths ------------------------------------------------------------
ONET_XLSX_PATH = Path("/Users/catherinewu/Downloads/Work Activities.xlsx")
FELTEN_AIOE_PATH = ANALYSIS_DIR / "felten_aioe.csv"
RUNS_DIR = OUTPUTS_DIR / "analysis_runs"
OCC_OBS = OUTPUTS_DIR / "final" / "onet_occupations_impact.csv"
ACT_OBS = OUTPUTS_DIR / "final" / "onet_activities_impact.csv"

# Default SOC remap from pipeline.py + run_final_speed_baseline.py manual crosswalks
DEFAULT_SOC_REMAP = {"13-2051.00": "13-2099.01"}
SOFTWARE_DEV_AVG = ("15-1132", "15-1133")  # 15-1252.00 ← avg
RADIOLOGIST_REMAP = ("29-1224.00", "29-1069")  # 29-1224.00 ← 29-1069


SOC_NAMES = {
    "11":"Management","13":"Business & Financial","15":"Computer & Math",
    "17":"Architecture & Engineering","19":"Life, Physical, Social Science",
    "21":"Community & Social Service","23":"Legal","25":"Education",
    "27":"Arts, Design, Media","29":"Healthcare Practitioners","31":"Healthcare Support",
    "33":"Protective Service","35":"Food Prep & Serving","37":"Building & Grounds Cleaning",
    "39":"Personal Care & Service","41":"Sales","43":"Office & Admin Support",
    "45":"Farming, Fishing, Forestry","47":"Construction & Extraction",
    "49":"Installation, Maint, Repair","51":"Production","53":"Transportation & Material Moving",
}
DEFAULT_EXCLUDED_SOCS = ["37", "45", "47", "49", "51", "53"]
PHYS_ACTS = {
    "Handling and Moving Objects", "Performing General Physical Activities",
    "Controlling Machines and Processes", "Operating Vehicles, Mechanized Devices, or Equipment",
    "Repairing and Maintaining Mechanical Equipment", "Repairing and Maintaining Electronic Equipment",
    "Inspecting Equipment, Structures, or Materials",
}

# Cache the heavy matrix build (reads a 73k-row xlsx)
_HEAT_CACHE: dict[float, dict] = {}


@dataclass
class RunParams:
    metric: str = "speed"            # "speed" or "quality"
    beta: float = 2.5                # Stage B specificity
    # Analysis unit: "occupation" (default, 894 occs), "soc_minor" (~92 3-digit
    # minor groups, XX-X000), or "soc_major" (22 2-digit major groups).
    # Legacy ``aggregate_to_socmajor=True`` is still honored and maps to
    # aggregation_level="soc_major".
    aggregation_level: str = "occupation"
    aggregate_to_socmajor: bool = False  # deprecated; kept for backward compat
    # Pruning: manual SOC-major-group selection + per-activity weight threshold.
    # (The old network-based prune is still available via legacy params below but
    # manual_prune=True is the default path now.)
    manual_prune: bool = True
    excluded_soc_majors: list = field(default_factory=lambda: list(DEFAULT_EXCLUDED_SOCS))
    activity_weight_threshold: float = 10.0
    # Legacy network-prune params (used only when manual_prune=False)
    alpha: float = 0.7
    hops: int = 4
    c_occ: float = 1.0
    prune_activities: bool = False
    c_act: float = 1.5
    omega_ref: float = 100.0
    sigma_ref: float = 0.1
    eps: float = 1e-6
    use_baseline: bool = True
    omega_base: float = 0.5


def _build_W(beta: float):
    """Return (W, W_raw_df, occ_codes, occ_titles, activities, soc_groups, soc_agg, ordering).
    Cached by beta."""
    key = round(float(beta), 4)
    if key in _HEAT_CACHE:
        return _HEAT_CACHE[key]
    W_raw = PIPELINE.build_composite_matrix(str(ONET_XLSX_PATH))
    occ_codes = W_raw.index.get_level_values(0).to_numpy()
    occ_titles = W_raw.index.get_level_values(1).to_numpy()
    activities = W_raw.columns.to_numpy()
    W = PIPELINE.softmax_rows(W_raw.values, beta=key)
    major = np.array([c[:2] for c in occ_codes])
    groups = sorted(set(major))
    agg = np.vstack([W[major == g].mean(0) for g in groups])  # 22 x 41
    counts = [int((major == g).sum()) for g in groups]
    # Hierarchical clustering for stable display ordering (rows and cols)
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        col_order = leaves_list(linkage(agg.T, method="ward")).tolist()
        row_order = leaves_list(linkage(agg,   method="ward")).tolist()
    except Exception:
        col_order = list(range(agg.shape[1]))
        row_order = list(range(agg.shape[0]))
    cached = {
        "W": W, "W_raw": W_raw, "occ_codes": occ_codes, "occ_titles": occ_titles,
        "activities": activities, "soc_groups": groups, "soc_counts": counts,
        "major": major, "soc_agg": agg, "col_order": col_order, "row_order": row_order,
    }
    _HEAT_CACHE[key] = cached
    return cached


def heatmap_data(beta: float, metric: str,
                 occ_rows: Optional[list[dict]] = None,
                 act_rows: Optional[list[dict]] = None) -> dict:
    """Compute the heatmap JSON payload — clustered SOC-major × activity matrix
    plus the observed overlays for the given metric."""
    H = _build_W(beta)
    metric_col, _, _ = _metric_cols(metric)

    # Determine observed activities and observed-occupation SOC majors from the
    # in-memory aggregated rows (or fall back to the on-disk CSVs).
    if occ_rows is None or act_rows is None:
        occ_rows = pd.read_csv(OCC_OBS).to_dict(orient="records")
        act_rows = pd.read_csv(ACT_OBS).to_dict(orient="records")

    obs_acts = set()
    for r in act_rows:
        v = r.get(metric_col)
        if v not in (None, "") and not (isinstance(v, float) and pd.isna(v)):
            obs_acts.add(str(r.get("label", "")))
    obs_socs = set()
    for r in occ_rows:
        v = r.get(metric_col)
        if v not in (None, "") and not (isinstance(v, float) and pd.isna(v)):
            code = str(r.get("code", ""))
            for k, v2 in DEFAULT_SOC_REMAP.items():
                if code == k: code = v2
            if len(code) >= 2:
                obs_socs.add(code[:2])

    return {
        "soc_groups": [
            {"code": g, "name": SOC_NAMES.get(g, g), "n": H["soc_counts"][i],
             "observed": g in obs_socs}
            for i, g in enumerate(H["soc_groups"])
        ],
        "activities": [
            {"name": a, "observed": a in obs_acts, "is_physical": a in PHYS_ACTS}
            for a in H["activities"]
        ],
        "weights": H["soc_agg"].tolist(),                 # rows = soc_groups order, cols = activities order
        "col_order": H["col_order"],
        "row_order": H["row_order"],
        "default_excluded_socs": list(DEFAULT_EXCLUDED_SOCS),
        "physical_activities": sorted(list(PHYS_ACTS)),
    }


def _aggregate_aioe_baseline(occ_codes: np.ndarray, obs_mean: float, obs_sd: float) -> np.ndarray:
    """Build baseline_A per occupation via full-population moment match."""
    fel = pd.read_csv(FELTEN_AIOE_PATH)
    aioe = fel["language_modeling_aioe"].to_numpy(dtype=float)
    mu_a, sd_a = float(np.nanmean(aioe)), float(np.nanstd(aioe, ddof=0))
    if sd_a == 0:
        sd_a = 1.0
    fel["baseline_A"] = obs_mean + (aioe - mu_a) * (obs_sd / sd_a)
    fmap = dict(zip(fel["soc_code"].astype(str), fel["baseline_A"]))

    extra = {}
    sw1, sw2 = SOFTWARE_DEV_AVG
    if sw1 in fmap and sw2 in fmap:
        extra["15-1252.00"] = float(np.mean([fmap[sw1], fmap[sw2]]))
    rad_dst, rad_src = RADIOLOGIST_REMAP
    if rad_src in fmap:
        extra[rad_dst] = float(fmap[rad_src])

    out = []
    for c in occ_codes:
        c = str(c)
        if c in extra:
            out.append(extra[c])
        else:
            out.append(fmap.get(c[:7], np.nan))
    return np.asarray(out, dtype=float)


def _metric_cols(metric: str) -> tuple[str, str, str]:
    if metric == "speed":
        return "speed_log_ratio_mean", "speed_log_ratio_se", "n_studies_speed"
    elif metric == "quality":
        return "quality_hedges_g_mean", "quality_hedges_g_se", "n_studies_quality"
    raise ValueError(f"unknown metric: {metric}")


def _prepare_system(params: RunParams,
                    occ_rows: Optional[list[dict]] = None,
                    act_rows: Optional[list[dict]] = None) -> dict:
    """Build the linear system arrays shared by run() and run_loo().

    Everything up through Stage C (pruning) and the pre-solve Ω / L / y construction —
    but not the final `A = L + Ω + ...` assembly or solve. Returned as a namespace so
    the caller can perturb `om`/`y` (LOO) before assembling A."""
    metric_col, se_col, n_col = _metric_cols(params.metric)

    # Stage A
    W_raw = PIPELINE.build_composite_matrix(str(ONET_XLSX_PATH))
    occ_codes = W_raw.index.get_level_values(0).to_numpy()
    occ_titles = W_raw.index.get_level_values(1).to_numpy()
    activities = W_raw.columns.to_numpy()
    onet_shape = W_raw.shape

    # Stage B
    W = PIPELINE.softmax_rows(W_raw.values, beta=params.beta)
    m, n = W.shape

    # Load observations: prefer in-memory rows if provided
    if occ_rows is not None and act_rows is not None:
        occs = pd.DataFrame(occ_rows)
        acts = pd.DataFrame(act_rows)
        data_source = "in_memory"
    else:
        occs = pd.read_csv(OCC_OBS)
        acts = pd.read_csv(ACT_OBS)
        data_source = "csv_files"
    occs["code"] = occs["code"].astype(str).replace(DEFAULT_SOC_REMAP)
    oi = {c: i for i, c in enumerate(occ_codes)}
    ai = {a: j for j, a in enumerate(activities)}

    f = np.full(m, np.nan); g = np.full(n, np.nan)
    fn = np.full(m, np.nan); fse = np.full(m, np.nan)
    gn = np.full(n, np.nan); gse = np.full(n, np.nan)
    unmatched_occ, unmatched_act = [], []

    for _, r in occs.iterrows():
        if pd.isna(r.get(metric_col)):
            continue
        i = oi.get(str(r["code"]))
        if i is None:
            unmatched_occ.append(str(r["code"])); continue
        f[i] = r[metric_col]
        fn[i] = r.get(n_col, np.nan)
        fse[i] = r.get(se_col, np.nan)
    for _, r in acts.iterrows():
        if pd.isna(r.get(metric_col)):
            continue
        j = ai.get(str(r["label"]))
        if j is None:
            unmatched_act.append(str(r["label"])); continue
        g[j] = r[metric_col]
        gn[j] = r.get(n_col, np.nan)
        gse[j] = r.get(se_col, np.nan)
    obs_occ, obs_act = np.isfinite(f), np.isfinite(g)
    n_obs_occ, n_obs_act = int(obs_occ.sum()), int(obs_act.sum())

    # AIOE baseline (speed only) — computed BEFORE optional SOC-major collapse
    base = None
    baseline_active = False
    if params.use_baseline and params.metric == "speed" and FELTEN_AIOE_PATH.exists():
        obs_vals = f[obs_occ]
        obs_mean = float(np.mean(obs_vals)) if len(obs_vals) else 0.0
        obs_sd = float(np.std(obs_vals, ddof=0)) if len(obs_vals) > 1 else 0.0
        if obs_sd == 0:
            obs_sd = 0.1
        base = _aggregate_aioe_baseline(occ_codes, obs_mean, obs_sd)
        baseline_active = True

    # Optional: collapse the 894 individual occupations to SOC minor (3-digit,
    # XX-X000) or SOC major (2-digit) groups. Aggregate W (row-mean per group),
    # pool observations (inverse-variance weighted when SEs available; simple
    # mean otherwise), and average the AIOE baseline within each group.
    # Resolve aggregation_level (with legacy aggregate_to_socmajor fallback).
    agg_level = (params.aggregation_level or "occupation").lower()
    if params.aggregate_to_socmajor and agg_level == "occupation":
        agg_level = "soc_major"
    socmajor_aggregated = False
    socmajor_group_sizes: Optional[np.ndarray] = None
    if agg_level in ("soc_major", "soc_minor"):
        socmajor_aggregated = True  # name kept for downstream compat
        if agg_level == "soc_major":
            keyfn = lambda c: str(c)[:2]
            title_for = lambda g: SOC_NAMES.get(g, g)
        else:  # soc_minor: 3-digit prefix padded to XX-X000
            keyfn = lambda c: str(c)[:4] + "000"
            # Use most-numerous member occupation's title as a representative.
            title_for = None  # filled in below
        major = np.array([keyfn(c) for c in occ_codes])
        groups = sorted(set(major))
        m = len(groups)
        socmajor_group_sizes = np.array([int((major == g).sum()) for g in groups], dtype=float)
        W_agg = np.vstack([W[major == g].mean(0) for g in groups])
        f_agg = np.full(m, np.nan); fn_agg = np.full(m, np.nan); fse_agg = np.full(m, np.nan)
        base_agg = np.full(m, np.nan) if base is not None else None
        for gi, gcode in enumerate(groups):
            mask = (major == gcode) & np.isfinite(f)
            if mask.any():
                vals = f[mask]; ses = fse[mask]; ns = fn[mask]
                has_se = np.isfinite(ses) & (ses > 0)
                if has_se.all() and len(vals) > 0:
                    w = 1.0 / ses[has_se]**2
                    f_agg[gi]   = float(np.sum(w * vals[has_se]) / np.sum(w))
                    fse_agg[gi] = float(np.sqrt(1.0 / np.sum(w)))
                else:
                    f_agg[gi] = float(np.mean(vals))
                valid_n = ns[np.isfinite(ns)]
                if valid_n.size:
                    fn_agg[gi] = float(np.sum(valid_n))
            if base is not None:
                bmask = (major == gcode) & np.isfinite(base)
                if bmask.any():
                    base_agg[gi] = float(np.mean(base[bmask]))
        # Build titles.
        if agg_level == "soc_major":
            new_titles = np.array([title_for(g) for g in groups])
        else:
            # Representative title = title of the largest member by W row-norm
            # (or just the first member with a usable title).
            new_titles = []
            for gcode in groups:  # NB: don't shadow outer ``g`` (activity obs vector)
                idxs = np.where(major == gcode)[0]
                # Pick the member with the largest W row sum (most concentrated weight).
                if len(idxs):
                    best = idxs[np.argmax(W[idxs].sum(axis=1))]
                    new_titles.append(f"{occ_titles[best]} ({gcode[:4]})")
                else:
                    new_titles.append(gcode)
            new_titles = np.array(new_titles)
        # Swap into the per-occupation slots so the rest of the pipeline is unchanged.
        W = W_agg
        occ_codes = np.array(groups)
        occ_titles = new_titles
        f, fn, fse = f_agg, fn_agg, fse_agg
        base = base_agg
        obs_occ = np.isfinite(f)
        n_obs_occ = int(obs_occ.sum())

    # Stage C: prune
    excluded_set = set(params.excluded_soc_majors or [])
    if params.manual_prune:
        # Manual SOC-major-group selection + per-activity weight threshold.
        major = np.array([str(c)[:2] for c in occ_codes])
        ko = ~np.isin(major, list(excluded_set))
        # Observed occupations always kept (don't accidentally drop a given).
        ko = ko | obs_occ
        # Per-activity total weight across the kept occupations.
        # In SOC-major-aggregated mode, W stores per-group MEAN weights; multiply by
        # group size so the threshold semantic matches the bar chart and the
        # occupation-level mode (Σ over individual occupations).
        if socmajor_aggregated and socmajor_group_sizes is not None:
            col_weight = (W[ko] * socmajor_group_sizes[ko, None]).sum(axis=0)
        else:
            col_weight = W[ko].sum(axis=0)
        ka = col_weight >= float(params.activity_weight_threshold)
        # Observed activities always kept.
        ka = ka | obs_act
        if not ka.any():
            ka = np.ones(n, dtype=bool)
        if not ko.any():
            ko = np.ones(m, dtype=bool)
        Wf = W[np.ix_(ko, ka)]
        rs = Wf.sum(axis=1, keepdims=True)
        Wf = np.where(rs > 0, Wf / rs, 0)
        occ_reach = act_reach = None
    else:
        Wf, ko, ka, occ_reach, act_reach = PIPELINE.prune_graph(
            W, obs_occ, obs_act, c_occ=params.c_occ, c_act=params.c_act,
            alpha=params.alpha, K=params.hops,
        )
        if not params.prune_activities:
            ka = np.ones(n, dtype=bool)
            Wf = W[np.ix_(ko, ka)]
            rs = Wf.sum(axis=1, keepdims=True)
            Wf = np.where(rs > 0, Wf / rs, 0)

    f_k, g_k = f[ko], g[ka]
    fn_k, fse_k, gn_k, gse_k = fn[ko], fse[ko], gn[ka], gse[ka]
    act_names = activities[ka]
    occ_codes_k = occ_codes[ko]
    occ_titles_k = occ_titles[ko]
    base_k = base[ko] if base is not None else None
    mk, nk = Wf.shape

    # Stage D — assemble the pieces the solve step needs
    of = PIPELINE.build_omega(fn_k, fse_k, params.omega_ref, params.sigma_ref)
    og = PIPELINE.build_omega(gn_k, gse_k, params.omega_ref, params.sigma_ref)
    W_sp = csr_matrix(Wf)
    S = bmat([[None, W_sp], [W_sp.T, None]], format="csr")
    deg = np.asarray(S.sum(axis=1)).ravel()
    L = (diags(deg) - S).toarray()
    Nk = mk + nk

    x_obs = np.concatenate([f_k, g_k])
    om_obs = np.concatenate([np.where(np.isfinite(of), of, 0.0),
                             np.where(np.isfinite(og), og, 0.0)])
    observed = np.isfinite(x_obs) & (om_obs > 0)
    y = np.where(observed, x_obs, 0.0)
    om = np.where(observed, om_obs, 0.0)

    if baseline_active:
        base_full = np.concatenate([base_k, np.full(nk, np.nan)])
        use_base = np.isfinite(base_full) & (~observed)
        om_b = np.where(use_base, params.omega_base, 0.0)
        y_b = np.where(use_base, np.nan_to_num(base_full), 0.0)
    else:
        om_b = np.zeros(Nk)
        y_b = np.zeros(Nk)

    return {
        "L": L, "om": om, "y": y, "om_b": om_b, "y_b": y_b,
        "x_obs": x_obs, "observed": observed,
        "f_k": f_k, "g_k": g_k, "fn_k": fn_k, "gn_k": gn_k,
        "act_names": act_names, "occ_codes_k": occ_codes_k, "occ_titles_k": occ_titles_k,
        "base_k": base_k, "baseline_active": baseline_active,
        "mk": int(mk), "nk": int(nk),
        "n_obs_occ": n_obs_occ, "n_obs_act": n_obs_act,
        "unmatched_occ": unmatched_occ, "unmatched_act": unmatched_act,
        "metric_col": metric_col, "se_col": se_col, "n_col": n_col,
        "data_source": data_source, "onet_shape": list(onet_shape),
        "socmajor_aggregated": socmajor_aggregated, "agg_level": agg_level,
    }


def run(params: RunParams, run_id: Optional[str] = None,
        occ_rows: Optional[list[dict]] = None,
        act_rows: Optional[list[dict]] = None) -> dict:
    """If `occ_rows`/`act_rows` are provided, use them directly (in-memory) instead
    of reading from the on-disk `onet_*_impact.csv` files. The dicts must match the
    `build_upload_files.collect_from_iters` output shape."""
    started = dt.datetime.utcnow().isoformat() + "Z"
    if run_id is None:
        run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    out_dir = RUNS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    P = _prepare_system(params, occ_rows=occ_rows, act_rows=act_rows)
    mk, nk = P["mk"], P["nk"]
    Nk = mk + nk

    A = P["L"] + np.diag(P["om"]) + np.diag(P["om_b"]) + params.eps * np.eye(Nk)
    Ainv = np.linalg.inv(A)
    x = Ainv @ (P["om"] * P["y"] + P["om_b"] * P["y_b"])
    std = np.sqrt(np.diag(Ainv))
    fh, gh = x[:mk], x[mk:]
    fs, gs = std[:mk], std[mk:]

    occ_out = pd.DataFrame({
        "code": P["occ_codes_k"],
        "title": P["occ_titles_k"],
        "observed": P["f_k"],
        "n_studies": P["fn_k"],
        "estimate": fh.round(4),
        "posterior_std": fs.round(4),
    })
    if P["baseline_active"]:
        occ_out.insert(4, "aioe_baseline", np.round(P["base_k"], 4))
    act_out = pd.DataFrame({
        "activity": P["act_names"],
        "observed": P["g_k"],
        "n_studies": P["gn_k"],
        "estimate": gh.round(4),
        "posterior_std": gs.round(4),
    })
    # Replace commas in display names per spec §6.3
    occ_out["title"] = occ_out["title"].astype(str).str.replace(",", ";", regex=False)
    act_out["activity"] = act_out["activity"].astype(str).str.replace(",", ";", regex=False)

    occ_out.to_csv(out_dir / "occupation_impacts.csv", index=False)
    act_out.to_csv(out_dir / "activity_impacts.csv", index=False)

    meta = {
        "run_id": run_id,
        "type": "run",
        "started_utc": started,
        "finished_utc": dt.datetime.utcnow().isoformat() + "Z",
        "params": asdict(params),
        "onet_shape": P["onet_shape"],
        "n_observed_occ": P["n_obs_occ"],
        "n_observed_act": P["n_obs_act"],
        "n_kept_occ": int(mk),
        "n_kept_act": int(nk),
        "unmatched_occ_codes": P["unmatched_occ"],
        "unmatched_act_labels": P["unmatched_act"],
        "baseline_active": P["baseline_active"],
        "metric_col": P["metric_col"],
        "se_col": P["se_col"],
        "n_col": P["n_col"],
        "data_source": P["data_source"],
        "socmajor_aggregated": P["socmajor_aggregated"],
        "aggregation_level": P["agg_level"],
    }
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2))
    return meta


# ---------------------------------------------------------------------------
# Rank-recovery evaluation for LOO
#
# Claim being tested: point estimates are noisy/shrunk, but the ORDERING of nodes
# is trustworthy. For each fold we keep the held-out node's LOO prediction, then
# rank all nodes by their LOO predictions and compare with the ranking by actual
# values. Rank statistics are invariant to any monotone distortion of the
# estimates (e.g. uniform shrinkage toward the mean), so they isolate ordering.
#
# NB: an earlier "plug-in rank" (rank pred_i among the OTHER nodes' actuals) was
# rejected: removing node i's own actual from the comparison set leaks its truth
# into its rank (a constant predictor scored tau ~0.76). Ranking predictions
# against predictions has no such leakage and yields true permutations, so the
# footrule chance level and top-K set overlap are well defined.
# ---------------------------------------------------------------------------

LOO_RANK_METHOD = {
    "description": (
        "Each observed node is held out in turn and its value re-imputed by the graph. "
        "Nodes are then ranked by their held-out predictions (1 = largest effect) and "
        "compared with their ranking by actual values. Scored with Kendall's tau-b "
        "(primary), the concordance index C = (tau+1)/2, Spearman's rho, mean absolute "
        "rank displacement / Spearman's footrule (vs. its random-permutation expectation), "
        "and top-K precision. 95% CI on tau-b is a percentile bootstrap over folds."
    ),
    "citations": [
        "Kendall, M. G. (1938). A new measure of rank correlation. Biometrika, 30(1-2), 81-93.",
        "Kendall, M. G. (1945). The treatment of ties in ranking problems. Biometrika, 33(3), 239-251.",
        "Harrell, F. E., Califf, R. M., Pryor, D. B., Lee, K. L., & Rosati, R. A. (1982). Evaluating "
        "the yield of medical tests. JAMA, 247(18), 2543-2546.",
        "Diaconis, P., & Graham, R. L. (1977). Spearman's footrule as a measure of disarray. "
        "JRSS Series B, 39(2), 262-268.",
        "Spearman, C. (1904). The proof and measurement of association between two things. "
        "American Journal of Psychology, 15(1), 72-101.",
        "Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall.",
    ],
}


def _loo_ranks(actual: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (actual_rank, loo_rank): descending average ranks (1 = largest) of the
    actual values and of the held-out predictions, each within its own vector."""
    from scipy.stats import rankdata
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if len(actual) == 0:
        return np.array([]), np.array([])
    return rankdata(-actual, method="average"), rankdata(-pred, method="average")


def _add_loo_ranks(rows: list[dict]) -> list[dict]:
    """Add rank columns to LOO rows (returns new dicts; input untouched).

    actual_rank / loo_rank / rank_delta               — ranked across ALL observed nodes
    actual_rank_within / loo_rank_within / rank_delta_within — ranked within node_type"""
    out = [dict(r) for r in rows]
    if not out:
        return out

    def _fill(idx: list[int], suffix: str) -> None:
        if not idx:
            return
        a = np.array([float(out[k]["actual"]) for k in idx])
        p = np.array([float(out[k]["predicted"]) for k in idx])
        ar, lr = _loo_ranks(a, p)
        for k, x, y in zip(idx, ar, lr):
            out[k]["actual_rank" + suffix] = float(x)
            out[k]["loo_rank" + suffix] = float(y)
            out[k]["rank_delta" + suffix] = float(y - x)

    _fill(list(range(len(out))), "")
    for t in ("occupation", "activity"):
        _fill([k for k, r in enumerate(out) if r.get("node_type") == t], "_within")
    return out


def _rank_recovery(sub: pd.DataFrame, actual_rank_col: str = "actual_rank",
                   loo_rank_col: str = "loo_rank", n_boot: int = 2000,
                   seed: int = 0) -> dict:
    """Rank-recovery statistics on a set of LOO folds (see LOO_RANK_METHOD)."""
    n = int(len(sub))
    out: dict = {"n": n}
    if n < 3 or actual_rank_col not in sub.columns or loo_rank_col not in sub.columns:
        return out
    ar = sub[actual_rank_col].to_numpy(dtype=float)
    lr = sub[loo_rank_col].to_numpy(dtype=float)

    m = _rank_metrics(ar, lr)
    tau = m.get("kendall_tau")
    out.update({
        "kendall_tau": tau,
        "kendall_p": m.get("kendall_p"),
        "concordance_c": (tau + 1.0) / 2.0 if tau is not None and np.isfinite(tau) else None,
        "spearman_r": m.get("spearman_r"),
        "spearman_p": m.get("spearman_p"),
    })

    # Spearman footrule (Diaconis & Graham 1977)
    d = np.abs(lr - ar)
    footrule = float(d.sum())
    out["footrule"] = footrule
    out["footrule_norm"] = footrule / float((n * n) // 2)
    out["mean_abs_rank_delta"] = float(d.mean())
    # E|pi(i) - i| under a uniformly random permutation = (n^2 - 1) / (3n)
    out["mean_abs_rank_delta_chance"] = (n * n - 1) / (3.0 * n)

    # Top-K precision: |true top-K  ∩  predicted top-K| / K.
    topk = {}
    for K in (3, 5, 10):
        if K >= n:
            continue
        true_top = set(np.argsort(ar, kind="stable")[:K].tolist())
        pred_top = set(np.argsort(lr, kind="stable")[:K].tolist())
        hits = len(true_top & pred_top)
        topk[str(K)] = {"K": K, "hits": hits, "rate": hits / K, "chance": K / n}
    out["topk"] = topk

    # Percentile bootstrap CI on tau-b, resampling folds.
    try:
        from scipy.stats import kendalltau
        rng = np.random.default_rng(seed)
        taus = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            a, b = ar[idx], lr[idx]
            if np.std(a) == 0 or np.std(b) == 0:
                continue
            t = kendalltau(a, b).statistic
            if np.isfinite(t):
                taus.append(t)
        if len(taus) >= 100:
            out["tau_ci"] = [float(np.percentile(taus, 2.5)), float(np.percentile(taus, 97.5))]
            out["tau_ci_boot_n"] = len(taus)
    except Exception:
        pass
    # Non-finite -> None so the browser's JSON.parse accepts the payload.
    for k, v in list(out.items()):
        if isinstance(v, float) and not np.isfinite(v):
            out[k] = None
    return out


def _rank_summary(rows: list[dict]) -> dict:
    """Overall block uses cross-type ranks; per-type blocks use within-type ranks."""
    df = pd.DataFrame(rows)
    if df.empty:
        return {"overall": {"n": 0}, "occupation": {"n": 0}, "activity": {"n": 0},
                "method": LOO_RANK_METHOD}
    res = {"overall": _rank_recovery(df)}
    for t in ("occupation", "activity"):
        res[t] = _rank_recovery(df[df["node_type"] == t],
                                "actual_rank_within", "loo_rank_within")
    res["method"] = LOO_RANK_METHOD
    return res


def run_loo(params: RunParams, run_id: Optional[str] = None,
            occ_rows: Optional[list[dict]] = None,
            act_rows: Optional[list[dict]] = None) -> dict:
    """Leave-one-out cross validation: for each observed value, hold it out, re-solve
    the propagation, and record the imputed posterior at that node vs the actual.

    Writes:
      outputs/analysis_runs/<run_id>/loocv.csv    — one row per held-out observation
      outputs/analysis_runs/<run_id>/loocv.json   — {summary, rows}
      outputs/analysis_runs/<run_id>/run.json     — meta with type='loo' and loo_summary
    """
    started = dt.datetime.utcnow().isoformat() + "Z"
    if run_id is None:
        run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S_loo_") + uuid.uuid4().hex[:6]
    out_dir = RUNS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    P = _prepare_system(params, occ_rows=occ_rows, act_rows=act_rows)
    mk, nk = P["mk"], P["nk"]; Nk = mk + nk
    L, om, y, om_b, y_b = P["L"], P["om"], P["y"], P["om_b"], P["y_b"]
    observed, x_obs = P["observed"], P["x_obs"]

    obs_indices = np.where(observed)[0]
    loo_rows = []
    for i in obs_indices:
        om_i = om.copy(); y_i = y.copy()
        om_i[i] = 0.0; y_i[i] = 0.0
        # If a baseline exists for this node it can now activate (matches the
        # semantic of the observation truly being absent).
        om_b_i = om_b.copy(); y_b_i = y_b.copy()
        if P["baseline_active"] and i < mk and np.isfinite(P["base_k"][i]):
            om_b_i[i] = params.omega_base
            y_b_i[i] = float(P["base_k"][i])
        A = L + np.diag(om_i) + np.diag(om_b_i) + params.eps * np.eye(Nk)
        try:
            Ainv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            continue
        x = Ainv @ (om_i * y_i + om_b_i * y_b_i)
        std = np.sqrt(np.diag(Ainv))
        pred = float(x[i]); pstd = float(std[i]); actual = float(x_obs[i])
        if i < mk:
            node_type = "occupation"
            code = str(P["occ_codes_k"][i])
            label = str(P["occ_titles_k"][i]).replace(",", ";")
        else:
            node_type = "activity"
            code = ""
            label = str(P["act_names"][i - mk]).replace(",", ";")
        loo_rows.append({
            "node_type": node_type,
            "code": code, "label": label,
            "actual": round(actual, 4),
            "predicted": round(pred, 4),
            "residual": round(pred - actual, 4),
            "abs_error": round(abs(pred - actual), 4),
            "posterior_std": round(pstd, 4),
        })

    loo_rows = _add_loo_ranks(loo_rows)
    df = pd.DataFrame(loo_rows)
    df.to_csv(out_dir / "loocv.csv", index=False)

    def _summary(sub: pd.DataFrame) -> dict:
        n = len(sub)
        if n == 0:
            return {"n": 0}
        a = sub["actual"].to_numpy(dtype=float)
        p = sub["predicted"].to_numpy(dtype=float)
        r = p - a
        ss_tot = float(np.sum((a - a.mean())**2)) if n > 1 else 0.0
        out = {
            "n": n,
            "mae": float(np.mean(np.abs(r))),
            "rmse": float(np.sqrt(np.mean(r**2))),
            "bias": float(np.mean(r)),
            "r2": (1 - float(np.sum(r**2)) / ss_tot) if ss_tot > 0 else None,
        }
        try:
            from scipy.stats import pearsonr, spearmanr
            if n > 1 and float(np.std(a)) > 0 and float(np.std(p)) > 0:
                out["pearson_r"] = float(pearsonr(a, p).statistic)
                out["spearman_r"] = float(spearmanr(a, p).statistic)
            else:
                out["pearson_r"] = out["spearman_r"] = None
        except Exception:
            out["pearson_r"] = out["spearman_r"] = None
        return out

    summary = {
        "overall":    _summary(df),
        "occupation": _summary(df[df["node_type"] == "occupation"]),
        "activity":   _summary(df[df["node_type"] == "activity"]),
        "rank":       _rank_summary(loo_rows),
    }

    meta = {
        "run_id": run_id,
        "type": "loo",
        "started_utc": started,
        "finished_utc": dt.datetime.utcnow().isoformat() + "Z",
        "params": asdict(params),
        "onet_shape": P["onet_shape"],
        "n_observed_occ": P["n_obs_occ"],
        "n_observed_act": P["n_obs_act"],
        "n_kept_occ": int(mk),
        "n_kept_act": int(nk),
        "unmatched_occ_codes": P["unmatched_occ"],
        "unmatched_act_labels": P["unmatched_act"],
        "baseline_active": P["baseline_active"],
        "metric_col": P["metric_col"],
        "data_source": P["data_source"],
        "socmajor_aggregated": P["socmajor_aggregated"],
        "aggregation_level": P["agg_level"],
        "loo_summary": summary,
        "n_folds": int(len(df)),
    }
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2))
    (out_dir / "loocv.json").write_text(json.dumps(
        {"run_id": run_id, "summary": summary, "rows": loo_rows}, indent=2
    ))
    return meta


def load_loo(run_id: str) -> dict | None:
    d = RUNS_DIR / run_id
    if not (d / "loocv.json").exists():
        return None
    meta = json.loads((d / "run.json").read_text())
    payload = json.loads((d / "loocv.json").read_text())
    rows = payload.get("rows") or []
    # Back-fill rank-recovery stats for runs made before they existed (in memory only).
    if rows and "loo_rank" not in rows[0]:
        payload["rows"] = _add_loo_ranks(rows)
    summary = payload.setdefault("summary", {})
    if "rank" not in summary:
        summary["rank"] = _rank_summary(payload["rows"])
        summary["rank"]["backfilled"] = True
    meta["loo"] = payload
    return meta


def _rank_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    """Rank/agreement metrics on two aligned numeric vectors (NaNs already removed)."""
    n = len(a)
    if n < 2:
        return {
            "n": n, "pearson_r": None, "pearson_p": None,
            "spearman_r": None, "spearman_p": None,
            "kendall_tau": None, "kendall_p": None,
        }
    out = {"n": int(n)}
    try:
        from scipy.stats import pearsonr, spearmanr, kendalltau
        if np.std(a) > 0 and np.std(b) > 0:
            pear = pearsonr(a, b)
            out["pearson_r"] = float(pear.statistic)
            out["pearson_p"] = float(pear.pvalue)
        else:
            out["pearson_r"] = out["pearson_p"] = None
        spear = spearmanr(a, b)
        out["spearman_r"] = float(spear.statistic)
        out["spearman_p"] = float(spear.pvalue)
        kend = kendalltau(a, b)
        out["kendall_tau"] = float(kend.statistic)
        out["kendall_p"] = float(kend.pvalue)
    except Exception:
        out["pearson_r"] = out["pearson_p"] = None
        out["spearman_r"] = out["spearman_p"] = None
        out["kendall_tau"] = out["kendall_p"] = None
    return out


_LEVEL_RANK = {"occupation": 0, "soc_minor": 1, "soc_major": 2}


def _agg_level_of(meta: dict) -> str:
    p = meta.get("params") or {}
    lvl = (p.get("aggregation_level") or "occupation").lower()
    if p.get("aggregate_to_socmajor") and lvl == "occupation":
        lvl = "soc_major"
    return lvl if lvl in _LEVEL_RANK else "occupation"


def _aggregate_occ_up(df: pd.DataFrame, from_level: str, to_level: str) -> pd.DataFrame:
    """Roll occupation-level results up to a coarser SOC key. No-op when already
    at (or coarser than) `to_level`. Aggregation: mean of `estimate` and
    `observed`; title becomes the SOC-major name for soc_major, or an arbitrary
    member title for soc_minor."""
    if _LEVEL_RANK[from_level] >= _LEVEL_RANK[to_level]:
        return df
    if to_level == "soc_major":
        keyfn = lambda c: str(c)[:2]
    else:  # soc_minor
        keyfn = lambda c: str(c)[:4] + "000"
    df = df.copy()
    df["code"] = df["code"].astype(str)
    df["_k"] = df["code"].map(keyfn)
    for col in ("estimate", "observed"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    agg_spec: dict = {"estimate": "mean", "observed": "mean"}
    if "title" in df.columns:
        agg_spec["title"] = "first"
    out = df.groupby("_k", as_index=False).agg(agg_spec)
    out = out.rename(columns={"_k": "code"})
    if to_level == "soc_major" and "title" in out.columns:
        out["title"] = out["code"].map(lambda c: SOC_NAMES.get(c, c))
    return out


def compare_runs(a_id: str, b_id: str) -> dict | None:
    """Rank/agreement comparison of two runs' occupation and activity results.
    Joins on the display key (occupation `code`, activity `activity`), keeps rows
    where both runs produced an estimate, and returns per-view metrics + rows.

    If the two runs used different aggregation levels for occupations (occupation
    vs soc_minor vs soc_major), the finer-grained run is rolled up to the coarser
    run's level before joining, so the comparison is on a common key."""
    ra = RUNS_DIR / a_id
    rb = RUNS_DIR / b_id
    if not (ra / "run.json").exists() or not (rb / "run.json").exists():
        return None
    ma = json.loads((ra / "run.json").read_text())
    mb = json.loads((rb / "run.json").read_text())

    def _load(d: Path, name: str) -> pd.DataFrame | None:
        p = d / name
        if not p.exists():
            return None
        return pd.read_csv(p)

    lvl_a = _agg_level_of(ma)
    lvl_b = _agg_level_of(mb)
    # Coarsest common level = the one with the higher rank.
    common_level = lvl_a if _LEVEL_RANK[lvl_a] >= _LEVEL_RANK[lvl_b] else lvl_b

    result = {
        "a": ma, "b": mb,
        "level_a": lvl_a, "level_b": lvl_b,
        "common_level": common_level,
        "levels_differ": lvl_a != lvl_b,
        "views": {},
    }
    for view, name, keycol, labelcol in (
        ("occupation", "occupation_impacts.csv", "code",     "title"),
        ("activity",   "activity_impacts.csv",   "activity", "activity"),
    ):
        da = _load(ra, name); db = _load(rb, name)
        if da is None or db is None:
            continue
        # Roll occupations up to the coarser side's level before joining. Activities
        # aren't stratified by SOC, so their rows just pass through.
        if view == "occupation":
            da = _aggregate_occ_up(da, lvl_a, common_level)
            db = _aggregate_occ_up(db, lvl_b, common_level)
            # `code` at soc_major is "11"/"13"/... which pandas would otherwise
            # infer as int64 on one side and str on the other — force str.
            da["code"] = da["code"].astype(str)
            db["code"] = db["code"].astype(str)
        # Keep the label + estimate + observed columns from A; join estimate/observed from B
        cols_a = [keycol, "estimate", "observed"] + ([labelcol] if labelcol != keycol and labelcol in da.columns else [])
        cols_b = [keycol, "estimate", "observed"]
        left = da[cols_a].rename(columns={"estimate": "estimate_a", "observed": "observed_a"})
        right = db[cols_b].rename(columns={"estimate": "estimate_b", "observed": "observed_b"})
        merged = pd.merge(left, right, on=keycol, how="inner")
        # Drop rows where either estimate is NaN
        merged = merged[np.isfinite(pd.to_numeric(merged["estimate_a"], errors="coerce")) &
                        np.isfinite(pd.to_numeric(merged["estimate_b"], errors="coerce"))].copy()
        merged["estimate_a"] = pd.to_numeric(merged["estimate_a"], errors="coerce")
        merged["estimate_b"] = pd.to_numeric(merged["estimate_b"], errors="coerce")
        merged["rank_a"] = merged["estimate_a"].rank(ascending=False, method="average")
        merged["rank_b"] = merged["estimate_b"].rank(ascending=False, method="average")
        merged["rank_delta"] = (merged["rank_a"] - merged["rank_b"]).round(1)
        merged["delta"] = (merged["estimate_a"] - merged["estimate_b"]).round(4)
        a_vec = merged["estimate_a"].to_numpy()
        b_vec = merged["estimate_b"].to_numpy()
        metrics = _rank_metrics(a_vec, b_vec)
        # Top-K rank overlap (Jaccard on the top-K sets by each run's ordering)
        overlap = {}
        for K in (5, 10, 25, 50):
            if len(merged) < K:
                continue
            top_a = set(merged.nsmallest(K, "rank_a")[keycol].astype(str).tolist())
            top_b = set(merged.nsmallest(K, "rank_b")[keycol].astype(str).tolist())
            inter = top_a & top_b
            union = top_a | top_b
            overlap[str(K)] = {
                "intersection": len(inter),
                "jaccard": (len(inter) / len(union)) if union else None,
            }
        # NaN → None so jsonify emits valid JSON
        rows = merged.astype(object).where(lambda x: x.notna(), None).to_dict(orient="records")
        result["views"][view] = {
            "metrics": metrics,
            "topk_overlap": overlap,
            "n": int(len(merged)),
            "keycol": keycol, "labelcol": labelcol,
            "rows": rows,
        }
    return result


def list_runs() -> list[dict]:
    if not RUNS_DIR.exists():
        return []
    out = []
    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        meta_p = d / "run.json"
        if meta_p.exists():
            try:
                out.append(json.loads(meta_p.read_text()))
            except Exception:
                pass
    return out


def load_run(run_id: str) -> dict | None:
    d = RUNS_DIR / run_id
    meta_p = d / "run.json"
    if not meta_p.exists():
        return None
    meta = json.loads(meta_p.read_text())
    # NaN → None so jsonify emits valid JSON (Flask's jsonify writes raw NaN tokens otherwise,
    # which JSON.parse rejects in the browser).
    occ = pd.read_csv(d / "occupation_impacts.csv").astype(object).where(lambda x: x.notna(), None)
    act = pd.read_csv(d / "activity_impacts.csv").astype(object).where(lambda x: x.notna(), None)
    meta["occupation_impacts"] = occ.to_dict(orient="records")
    meta["activity_impacts"] = act.to_dict(orient="records")
    return meta


if __name__ == "__main__":
    meta = run(RunParams())
    print(json.dumps(meta, indent=2))
