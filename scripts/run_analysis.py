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
    # Analysis unit: False (default) = run on the 894 individual occupations;
    # True = collapse to the 22 SOC major groups.
    aggregate_to_socmajor: bool = False
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

    # Optional: collapse the 894 individual occupations to 22 SOC major groups.
    # Aggregate W (row-mean per major), pool observations (inverse-variance weighted
    # when SEs available; simple mean otherwise), and average the AIOE baseline
    # within each major.
    socmajor_aggregated = False
    socmajor_group_sizes: Optional[np.ndarray] = None
    if params.aggregate_to_socmajor:
        socmajor_aggregated = True
        major = np.array([str(c)[:2] for c in occ_codes])
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
        # Swap into the per-occupation slots so the rest of the pipeline is unchanged.
        W = W_agg
        occ_codes = np.array(groups)
        occ_titles = np.array([SOC_NAMES.get(g, g) for g in groups])
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

    # Stage D
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

    A = L + np.diag(om) + np.diag(om_b) + params.eps * np.eye(Nk)
    Ainv = np.linalg.inv(A)
    x = Ainv @ (om * y + om_b * y_b)
    std = np.sqrt(np.diag(Ainv))
    fh, gh = x[:mk], x[mk:]
    fs, gs = std[:mk], std[mk:]

    occ_out = pd.DataFrame({
        "code": occ_codes_k,
        "title": occ_titles_k,
        "observed": f_k,
        "n_studies": fn_k,
        "estimate": fh.round(4),
        "posterior_std": fs.round(4),
    })
    if baseline_active:
        occ_out.insert(4, "aioe_baseline", np.round(base_k, 4))
    act_out = pd.DataFrame({
        "activity": act_names,
        "observed": g_k,
        "n_studies": gn_k,
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
        "started_utc": started,
        "finished_utc": dt.datetime.utcnow().isoformat() + "Z",
        "params": asdict(params),
        "onet_shape": list(onet_shape),
        "n_observed_occ": n_obs_occ,
        "n_observed_act": n_obs_act,
        "n_kept_occ": int(mk),
        "n_kept_act": int(nk),
        "unmatched_occ_codes": unmatched_occ,
        "unmatched_act_labels": unmatched_act,
        "baseline_active": baseline_active,
        "metric_col": metric_col,
        "se_col": se_col,
        "n_col": n_col,
        "data_source": data_source,
        "socmajor_aggregated": socmajor_aggregated,
    }
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2))
    return meta


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
