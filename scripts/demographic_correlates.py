"""Correlate imputed occupation-level AI speed / quality effects with occupational
characteristics and demographic representation.

Mirrors the figure in Felten, Raj & Seamans (CACM 2024, `demos/FRS CACM 2024.pdf`)
and its Stata code (`demos/generative_ai_aioe_master.do`): 20-equal-count-bin
binscatters of the outcome against median salary, required education, creative-
ability weight, and % female / White / Black / Asian / Hispanic employment. Here
the y-series are our imputed speed and quality scores (z-scored so they share an
axis), with language-modeling AIOE drawn as a reference series.

Code systems in `demos/`:
- `occ_salary_data_2021.dta`             BLS OEWS 2021 -> SOC 2018, joined directly
- `generative_ai_aioe.dta`, `occ_required_education.dta`, `occ_creative_weight.dta`
                                          SOC 2010 -> mapped via the O*NET-SOC
                                          2010->2019 crosswalk (many->one averaged)
- `occ_representation.dta`               ACS 2021 codes (SOC 2018 based, some
                                          aggregated like 11-9030) -> exact match,
                                          else the nearest aggregated ACS code

Usage:
    .venv/bin/python scripts/demographic_correlates.py                 # fresh runs, no SOC exclusions
    .venv/bin/python scripts/demographic_correlates.py --speed-run <id> --quality-run <id>
    .venv/bin/python scripts/demographic_correlates.py --no-speed-baseline --quality-run <id>

Caveat: the default speed imputation uses the LM-AIOE baseline prior, so imputed
speed tracks AIOE (r~0.86) and its correlations largely restate Felten et al.
`--no-speed-baseline` shows what the study data alone imply.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
DEMOS_DIR = ROOT / "demos"
CROSSWALK = DEMOS_DIR / "crosswalks" / "onet_soc_2010_to_2019.csv"
OUT_DIR = ROOT / "outputs" / "demographics"
RUNS_DIR = ROOT / "outputs" / "analysis_runs"

N_BINS = 20

# (column, x-axis label, tick formatter) in the paper's panel order (a)-(h).
CHARACTERISTICS = [
    ("median_salary_2021", "Median Salary 2021 (Thousands of Dollars)", "num"),
    ("occ_req_education", "Occupation Required Education Level", "num"),
    ("avg_creative_weight", "Relative Weight of Creative Abilities", "pct"),
    ("mean_female", "Percent Female Employment", "pct"),
    ("mean_white", "Percent White Employment", "pct"),
    ("mean_black", "Percent Black Employment", "pct"),
    ("mean_asian", "Percent Asian Employment", "pct"),
    ("mean_hispanic", "Percent Hispanic Employment", "pct"),
]

# y-series: (column, legend label, color, marker). Categorical slots 1-2 of the
# dataviz reference palette; AIOE is a neutral-gray reference, not a series hue.
SERIES = [
    ("speed_z", "Speed (imputed)", "#2a78d6", "o"),
    ("quality_z", "Quality (imputed)", "#eb6834", "D"),
    ("lm_aioe_z", "Language-modeling AIOE (reference)", "#9a9a94", "x"),
]


# ---------------------------------------------------------------- characteristics

def _soc2010_to_2018(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Map a SOC-2010-keyed frame to 6-digit SOC 2018 via the O*NET crosswalk.
    A 2018 code fed by several 2010 codes gets their unweighted mean."""
    xw = pd.read_csv(CROSSWALK, dtype=str)
    pairs = pd.DataFrame({
        "occ_code": xw["O*NET-SOC 2010 Code"].str[:7],
        "soc": xw["O*NET-SOC 2019 Code"].str[:7],
    }).drop_duplicates()
    m = pairs.merge(df[["occ_code", *cols]], on="occ_code", how="inner")
    return m.groupby("soc", as_index=False)[cols].mean()


def _acs_lookup(codes: pd.Series, rep: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """ACS occupation codes are detailed SOC where possible, otherwise broad /
    minor groups with trailing zeros (11-9030, 25-1000). Try progressively
    coarser codes until one hits."""
    table = rep.set_index("occ_code")[cols]
    rows, matched = [], []
    for soc in codes:
        hit = None
        for k in range(4):
            cand = soc[: 7 - k] + "0" * k if k else soc
            if cand in table.index:
                hit = cand
                break
        rows.append(table.loc[hit].to_numpy() if hit else np.full(len(cols), np.nan))
        matched.append(hit)
    out = pd.DataFrame(rows, columns=cols)
    out.insert(0, "soc", codes.to_numpy())
    out["acs_code"] = matched
    return out


def load_characteristics(socs: pd.Series) -> pd.DataFrame:
    aioe = pd.read_stata(DEMOS_DIR / "generative_ai_aioe.dta")
    edu = pd.read_stata(DEMOS_DIR / "occ_required_education.dta")
    cre = pd.read_stata(DEMOS_DIR / "occ_creative_weight.dta")
    sal = pd.read_stata(DEMOS_DIR / "occ_salary_data_2021.dta")
    rep = pd.read_stata(DEMOS_DIR / "occ_representation.dta")

    s2010 = (aioe[["occ_code", "lm_aioe", "ig_aioe"]]
             .merge(edu, on="occ_code", how="outer")
             .merge(cre[["occ_code", "avg_creative_weight"]], on="occ_code", how="outer"))
    s2010 = _soc2010_to_2018(s2010, ["lm_aioe", "ig_aioe", "occ_req_education",
                                     "avg_creative_weight"])

    rep_cols = ["mean_female", "mean_white", "mean_black", "mean_asian",
                "mean_hispanic", "count_occ"]
    acs = _acs_lookup(socs, rep, rep_cols)

    out = (pd.DataFrame({"soc": socs.to_numpy()})
           .merge(sal.rename(columns={"occ_code": "soc"}), on="soc", how="left")
           .merge(s2010, on="soc", how="left")
           .merge(acs, on="soc", how="left"))
    return out


# ---------------------------------------------------------------- scores

def _run_imputation(metric: str, excluded: list[str], use_baseline: bool = True) -> str:
    sys.path.insert(0, str(SCRIPTS_DIR))
    import run_analysis as ra

    params = ra.RunParams(metric=metric, excluded_soc_majors=excluded,
                          use_baseline=use_baseline)
    res = ra.run(params)
    run_id = res.get("run_id") if isinstance(res, dict) else None
    if not run_id:  # fall back to newest run dir
        run_id = sorted(p.name for p in RUNS_DIR.iterdir() if p.is_dir())[-1]
    print(f"  {metric}: new run {run_id} ({json.dumps(asdict(params))[:80]}...)")
    return run_id


def _run_uses_baseline(run_id: str) -> bool:
    meta = json.loads((RUNS_DIR / run_id / "run.json").read_text())
    return bool(meta.get("params", {}).get("use_baseline", True))


def load_scores(speed_run: str, quality_run: str) -> pd.DataFrame:
    """Imputed estimates per O*NET-SOC code, averaged up to 6-digit SOC 2018."""
    frames = []
    for metric, run_id in (("speed", speed_run), ("quality", quality_run)):
        df = pd.read_csv(RUNS_DIR / run_id / "occupation_impacts.csv", dtype={"code": str})
        if not df["code"].str.match(r"^\d\d-\d{4}\.\d\d$").all():
            raise SystemExit(f"run {run_id} is not occupation-level")
        df["soc"] = df["code"].str[:7]
        g = df.groupby("soc").agg(
            **{metric: ("estimate", "mean"),
               f"{metric}_observed": ("observed", lambda s: s.notna().any()),
               "title": ("title", "first")})
        frames.append(g)
    scores = frames[0].drop(columns="title").join(frames[1], how="outer")
    return scores.reset_index()


# ---------------------------------------------------------------- analysis

def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for x, label, _ in CHARACTERISTICS:
        for y in ("speed", "quality", "lm_aioe"):
            sub = df[[x, y]].dropna()
            n = len(sub)
            r = sub[x].corr(sub[y])
            rho = sub[x].corr(sub[y], method="spearman")
            # No CIs: imputed scores are propagated from a handful of observed
            # occupations, so treating the n occupations as i.i.d. overstates precision.
            rows.append({"characteristic": x, "label": label, "score": y, "n": n,
                         "pearson_r": round(r, 3), "spearman_rho": round(rho, 3)})
    return pd.DataFrame(rows)


def binscatter_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Stata-binscatter-style: 20 equal-count bins on x, mean x and mean y per bin.
    Bins are formed on occupations with x and all y-series present, so every
    series in a panel shares the same bins."""
    ycols = [c for c, *_ in SERIES]
    out = []
    for x, *_ in CHARACTERISTICS:
        sub = df[[x, *ycols]].dropna()
        ranks = sub[x].rank(method="first")
        sub = sub.assign(bin=pd.qcut(ranks, N_BINS, labels=False))
        b = sub.groupby("bin").agg(x_mean=(x, "mean"), n=(x, "size"),
                                   **{f"{c}_mean": (c, "mean") for c in ycols})
        b.insert(0, "characteristic", x)
        out.append(b.reset_index())
    return pd.concat(out, ignore_index=True)


def plot(bins: pd.DataFrame, corr: pd.DataFrame, path_stem: Path, meta: str,
         series: list = SERIES, annot: str = "speed/quality") -> None:
    """`annot` picks the per-panel r line: the two imputed metrics, or (in
    --compare mode) the same metric with and without the AIOE baseline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    ink, muted, grid = "#1f1f1e", "#6b6a64", "#e6e5df"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(4, 2, figsize=(9, 12.5))
    for i, (ax, (x, label, fmt)) in enumerate(zip(axes.flat, CHARACTERISTICS)):
        b = bins[bins.characteristic == x]
        ax.axhline(0, color=muted, lw=0.8, zorder=1)
        ax.grid(axis="y", color=grid, lw=0.6, zorder=0)
        for col, name, color, marker in series:
            ref = col == "lm_aioe_z"
            ax.scatter(b["x_mean"], b[f"{col}_mean"], s=22 if ref else 30,
                       marker=marker, color=color, linewidths=1.0 if ref else 0.8,
                       edgecolors="white" if not ref else None,
                       alpha=0.9, zorder=2 if ref else 3, label=name)
        c = corr[corr.characteristic == x].set_index("score")
        if annot == "compare":
            line = (f"r(speed, with AIOE) = {c.loc['speed_with', 'pearson_r']:+.2f}    "
                    f"r(speed, study only) = {c.loc['speed_without', 'pearson_r']:+.2f}")
        else:
            line = (f"r(speed) = {c.loc['speed', 'pearson_r']:+.2f}    "
                    f"r(quality) = {c.loc['quality', 'pearson_r']:+.2f}")
        ax.text(0.0, 1.02, line, transform=ax.transAxes, va="bottom", ha="left",
                fontsize=8, color=ink)
        if fmt == "pct":
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_xlabel(label)
        ax.set_ylabel("Effect score (z)")
        ax.set_title(f"({'abcdefgh'[i]})", loc="center", y=-0.32, fontsize=9, color=muted)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.995))
    fig.text(0.5, 0.004, meta, ha="center", va="bottom", fontsize=7, color=muted, wrap=True)
    fig.tight_layout(rect=(0, 0.02, 1, 0.975), h_pad=2.2)
    for ext in ("png", "pdf"):
        fig.savefig(path_stem.with_suffix(f".{ext}"), dpi=200)
    plt.close(fig)


COMPARE_SERIES = [
    ("speed_with_z", "Speed (with AIOE baseline)", "#2a78d6", "o"),
    ("speed_without_z", "Speed (study data only)", "#eb6834", "D"),
    ("lm_aioe_z", "Language-modeling AIOE (reference)", "#9a9a94", "x"),
]


def compare(main_dir: Path) -> None:
    """Overlay the two speed imputations panel by panel. Both variants must have
    been generated already; quality and the characteristics are identical across
    them, so only the speed series differs."""
    alt_dir = main_dir / "no_aioe_baseline"
    for d in (main_dir, alt_dir):
        if not (d / "binscatter_bins.csv").exists():
            raise SystemExit(f"missing {d}/binscatter_bins.csv - generate it first")

    keys = ["characteristic", "bin"]
    mb = pd.read_csv(main_dir / "binscatter_bins.csv")
    ab = pd.read_csv(alt_dir / "binscatter_bins.csv")
    bins = (mb.rename(columns={"speed_z_mean": "speed_with_z_mean"})
              .merge(ab[[*keys, "speed_z_mean", "x_mean"]]
                     .rename(columns={"speed_z_mean": "speed_without_z_mean",
                                      "x_mean": "x_mean_alt"}), on=keys))
    drift = (bins["x_mean"] - bins["x_mean_alt"]).abs().max()
    if drift > 1e-9:  # bins must line up or the overlay compares different x
        raise SystemExit(f"bin x-means differ between variants (max {drift})")

    mc = pd.read_csv(main_dir / "correlations.csv")
    ac = pd.read_csv(alt_dir / "correlations.csv")
    corr = pd.concat([
        mc[mc.score == "speed"].assign(score="speed_with"),
        ac[ac.score == "speed"].assign(score="speed_without"),
        mc[mc.score.isin(["quality", "lm_aioe"])],
    ], ignore_index=True)

    wide = (corr.pivot(index=["characteristic", "label"], columns="score",
                       values="pearson_r").reset_index())
    wide["delta_speed"] = (wide["speed_with"] - wide["speed_without"]).round(3)
    order = {c: i for i, (c, *_) in enumerate(CHARACTERISTICS)}
    wide = wide.sort_values("characteristic", key=lambda s: s.map(order))
    wide.to_csv(main_dir / "baseline_comparison.csv", index=False)

    meta_main = json.loads((main_dir / "meta.json").read_text())
    meta_alt = json.loads((alt_dir / "meta.json").read_text())
    meta = (f"Speed imputation with the LM-AIOE baseline (run {meta_main['speed_run']}) vs "
            f"without it (run {meta_alt['speed_run']}); identical quality run and "
            f"{meta_main['n_occupations']} occupations. Without the baseline, speed is "
            f"propagated from {meta_main['n_observed_speed']} observed occupations alone.")
    plot(bins, corr, main_dir / "binscatter_baseline_comparison", meta,
         series=COMPARE_SERIES, annot="compare")

    cols = ["label", "speed_with", "speed_without", "delta_speed", "lm_aioe"]
    print(wide[cols].to_string(index=False))
    print(f"\nwrote {main_dir}/baseline_comparison.csv + "
          f"binscatter_baseline_comparison.png/.pdf")


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--speed-run", help="existing occupation-level speed run id")
    ap.add_argument("--quality-run", help="existing occupation-level quality run id")
    ap.add_argument("--excluded-socs", default="",
                    help="comma-separated SOC majors to prune when running fresh "
                         "imputations (default: none, i.e. whole economy)")
    ap.add_argument("--no-speed-baseline", action="store_true",
                    help="fresh speed run without the AIOE baseline prior, so speed "
                         "reflects only the study data (writes to <out-dir>/no_aioe_baseline)")
    ap.add_argument("--compare", action="store_true",
                    help="overlay the with- and without-AIOE-baseline speed variants "
                         "(both must already be generated) and exit")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    if args.compare:
        compare(args.out_dir)
        return
    if args.no_speed_baseline:
        if args.speed_run and _run_uses_baseline(args.speed_run):
            ap.error(f"--speed-run {args.speed_run} was imputed WITH the AIOE baseline")
        args.out_dir = args.out_dir / "no_aioe_baseline"

    excluded = [s for s in args.excluded_socs.split(",") if s]
    speed_run = args.speed_run or _run_imputation("speed", excluded,
                                                  use_baseline=not args.no_speed_baseline)
    quality_run = args.quality_run or _run_imputation("quality", excluded)

    scores = load_scores(speed_run, quality_run)
    chars = load_characteristics(scores["soc"])
    df = scores.merge(chars, on="soc", how="left")
    for col in ("speed", "quality", "lm_aioe"):
        df[f"{col}_z"] = _z(df[col])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    corr = correlations(df)
    bins = binscatter_bins(df)
    df.to_csv(args.out_dir / "occupation_characteristics.csv", index=False)
    corr.to_csv(args.out_dir / "correlations.csv", index=False)
    bins.to_csv(args.out_dir / "binscatter_bins.csv", index=False)

    coverage = {c: int(df[c].notna().sum()) for c, *_ in CHARACTERISTICS}
    meta = (f"{len(df)} SOC-2018 occupations; speed run {speed_run}, quality run "
            f"{quality_run}. {N_BINS} equal-count bins per panel; scores z-scored across "
            f"occupations. r = occupation-level Pearson correlation (unbinned)."
            + (" Speed imputed WITHOUT the AIOE baseline." if args.no_speed_baseline else
               " Speed imputation uses the LM-AIOE baseline prior."))
    plot(bins, corr, args.out_dir / "binscatter_characteristics", meta)
    (args.out_dir / "meta.json").write_text(json.dumps({
        "speed_run": speed_run, "quality_run": quality_run,
        "speed_aioe_baseline": not args.no_speed_baseline,
        "n_occupations": len(df),
        "n_observed_speed": int(df["speed_observed"].sum()),
        "n_observed_quality": int(df["quality_observed"].sum()),
        "coverage": coverage, "n_bins": N_BINS,
    }, indent=2))

    print(f"\n{len(df)} occupations; coverage: {coverage}")
    print(corr.pivot(index="label", columns="score", values="pearson_r")
          .reindex([l for _, l, _ in CHARACTERISTICS]).to_string())
    print(f"\nwrote {args.out_dir}")


if __name__ == "__main__":
    main()
