"""(Re)generate the curated set of analysis runs published on the static site.

Every run is computed from the LIVE review-table state (same path as /api/run and
/api/loo), so all published runs share one data snapshot. Writes the run dirs into
outputs/analysis_runs/ and the manifest config/site_runs.json, which
build_static_site.py uses to decide what to publish and how to group/label it.

Runs listed in the previous manifest are deleted before regenerating, so repeated
invocations don't pile up duplicates.

Run:
    python3 scripts/generate_site_runs.py
    python3 scripts/build_static_site.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import review_app as APP  # noqa: E402
import run_analysis as RA  # noqa: E402

MANIFEST = ROOT / "config" / "site_runs.json"

LEVEL_NAMES = {"occupation": "occupation", "soc_minor": "SOC minor group", "soc_major": "SOC major group"}

GROUPS = [
    {"id": "headline", "title": "Headline estimates",
     "blurb": "Default parameters (β = 2.5, manual pruning of the six physical-work SOC majors, activity "
              "threshold 10, AIOE baseline for speed) at each aggregation level."},
    {"id": "sensitivity", "title": "Sensitivity checks",
     "blurb": "One parameter changed from the occupation-level headline run. Agreement columns compare each "
              "run's estimates with that headline run (Spearman ρ over shared nodes; top-10 = overlap of the "
              "ten largest effects). High agreement means the ranking does not hinge on that choice."},
    {"id": "validation", "title": "Leave-one-out validation",
     "blurb": "Each observed node is held out in turn and re-imputed from the rest of the graph. Kendall τ-b "
              "measures whether held-out predictions recover the ORDER of observed effects; a 95% CI "
              "excluding 0 is a significant (positive or negative) result."},
]


def _specs() -> list[dict]:
    """(key, group, label, kind, params-overrides, reference key for agreement)."""
    out = []
    for metric in ("speed", "quality"):
        for lvl in ("occupation", "soc_minor", "soc_major"):
            out.append(dict(key=f"{metric}-{lvl}", group="headline", kind="run", metric=metric,
                            label=f"{metric.capitalize()} · {LEVEL_NAMES[lvl]}",
                            params=dict(aggregation_level=lvl),
                            reference=None if lvl == "occupation" else f"{metric}-occupation"))
    for metric in ("speed", "quality"):
        ref = f"{metric}-occupation"
        sens = [
            ("beta1", "β = 1 (flatter activity weights)", dict(beta=1.0)),
            ("beta10", "β = 10 (sharper activity weights)", dict(beta=10.0)),
            ("allsoc", "All 22 SOC majors kept (no exclusion)", dict(excluded_soc_majors=[])),
            ("excl11", "Also exclude Management (SOC 11)",
             dict(excluded_soc_majors=RA.DEFAULT_EXCLUDED_SOCS + ["11"])),
            ("thr5", "Activity weight threshold 5 (keep more activities)", dict(activity_weight_threshold=5.0)),
        ]
        if metric == "speed":  # the AIOE baseline only exists for speed
            sens.append(("nobase", "No AIOE baseline", dict(use_baseline=False)))
        for k, lab, p in sens:
            out.append(dict(key=f"{metric}-{k}", group="sensitivity", kind="run", metric=metric,
                            label=f"{metric.capitalize()} · {lab}", params=p, reference=ref))
    for metric in ("speed", "quality"):
        for lvl in ("occupation", "soc_minor", "soc_major"):
            out.append(dict(key=f"loo-{metric}-{lvl}", group="validation", kind="loo", metric=metric,
                            label=f"LOO · {metric.capitalize()} · {LEVEL_NAMES[lvl]}",
                            params=dict(aggregation_level=lvl), reference=None))
    out.append(dict(key="loo-speed-nobase", group="validation", kind="loo", metric="speed",
                    label="LOO · Speed · occupation · no AIOE baseline",
                    params=dict(use_baseline=False), reference=None))
    return out


def main():
    old = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"runs": []}
    for e in old.get("runs", []):
        d = RA.RUNS_DIR / e["run_id"]
        if d.exists():
            shutil.rmtree(d)

    occ_rows, act_rows, n_speed, n_qual = APP._observations_from_state()
    n_used = {"speed": n_speed, "quality": n_qual, "occ_codes": len(occ_rows), "act_codes": len(act_rows)}
    print(f"live state: {n_speed} speed + {n_qual} quality observations "
          f"-> {len(occ_rows)} occupation / {len(act_rows)} activity codes")

    entries = []
    key_to_id = {}
    for s in _specs():
        params = RA.RunParams(metric=s["metric"], **s["params"])
        fn = RA.run_loo if s["kind"] == "loo" else RA.run
        meta = fn(params, occ_rows=occ_rows, act_rows=act_rows)
        # Mirror /api/run: record the observation counts in run.json too.
        meta["n_observations_used"] = n_used
        (RA.RUNS_DIR / meta["run_id"] / "run.json").write_text(json.dumps(meta, indent=2))
        key_to_id[s["key"]] = meta["run_id"]
        entries.append({"key": s["key"], "run_id": meta["run_id"], "group": s["group"],
                        "kind": s["kind"], "label": s["label"], "reference": s["reference"]})
        print(f"  {s['key']:<22} -> {meta['run_id']}")

    for e in entries:
        ref = e.pop("reference")
        e["reference_run_id"] = key_to_id[ref] if ref else None
    MANIFEST.write_text(json.dumps({
        "canonical": key_to_id["speed-occupation"],
        "groups": GROUPS,
        "observations": n_used,
        "runs": entries,
    }, indent=2) + "\n")
    print(f"wrote {MANIFEST.relative_to(ROOT)} ({len(entries)} runs)")


if __name__ == "__main__":
    main()
