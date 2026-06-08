"""Pipeline orchestrator: runs stages sequentially, parallelizes within stage,
idempotent (skips done work unless --force), filters out 'other'-classified papers
from stage 3 onward."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from common import (  # noqa: E402
    CostTracker, OUTPUTS_DIR, REPO_ROOT, get_client, get_logger, load_settings,
    paper_id_from_path,
)

log = get_logger("orchestrator")


def _load(module_name: str, file_path: str | Path):
    p = Path(file_path)
    if not p.is_absolute():
        p = SCRIPTS_DIR / p
    spec = importlib.util.spec_from_file_location(module_name, p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


STAGE_MODULES = {
    "extract_quotes":    ("01a_extract_quotes",  "01a_extract_quotes.py",  "01a_quotes"),
    "structure":         ("01b_structure",       "01b_structure.py",       "01_extraction"),
    "classify_method":   ("02_classify_method",  "02_classify_method.py",  "02_method_classification"),
    "classify_outcome":  ("03_classify_outcome", "03_classify_outcome.py", "03_outcome_classification"),
    "map_onet":          ("04_map_onet",         "04_map_onet.py",         "04_onet_mapping"),
    "compute_speed":     ("05a_compute_speed",   "05a_compute_speed.py",   "05_effect_sizes"),
    "compute_quality":   ("05b_compute_quality", "05b_compute_quality.py", "05_effect_sizes"),
}

# Stages that should NOT be run on out-of-scope papers (study_design == "other").
_DOWNSTREAM_OF_METHOD = {
    "classify_outcome", "map_onet", "compute_speed", "compute_quality", "compute_effects",
}


def _list_paper_ids(papers_dir: Path) -> list[str]:
    return [paper_id_from_path(p) for p in sorted(papers_dir.glob("*.pdf"))]


def _filter_in_scope(stage_name: str, paper_ids: list[str]) -> tuple[list[str], list[str]]:
    if stage_name not in _DOWNSTREAM_OF_METHOD:
        return paper_ids, []
    kept, dropped = [], []
    for pid in paper_ids:
        method_path = OUTPUTS_DIR / "02_method_classification" / f"{pid}.json"
        if not method_path.exists():
            kept.append(pid)
            continue
        try:
            with open(method_path) as f:
                m = json.load(f)
            if m.get("classification") == "other":
                dropped.append(pid)
            else:
                kept.append(pid)
        except Exception:  # noqa: BLE001
            kept.append(pid)
    return kept, dropped


def run_stage(stage_name: str, paper_ids: list[str], *, settings, client, cost, papers_dir: Path,
              force: bool, concurrency: int) -> None:
    paper_ids, dropped = _filter_in_scope(stage_name, paper_ids)
    if dropped:
        log.info("stage %s: dropping %d out-of-scope (study_design=other) papers: %s",
                 stage_name, len(dropped), ", ".join(dropped[:5]) + ("..." if len(dropped) > 5 else ""))
    log.info("=== stage: %s (%d papers) ===", stage_name, len(paper_ids))

    if stage_name == "extract_quotes":
        mod = _load(*STAGE_MODULES["extract_quotes"][:2])
        def work(pid):
            return mod.run_one(papers_dir / f"{pid}.pdf", client=client, settings=settings, cost=cost, force=force)
    elif stage_name == "structure":
        mod = _load(*STAGE_MODULES["structure"][:2])
        def work(pid):
            return mod.run_one(pid, client=client, settings=settings, cost=cost, force=force)
    elif stage_name == "classify_method":
        mod = _load(*STAGE_MODULES["classify_method"][:2])
        def work(pid):
            return mod.run_one(pid, client=client, settings=settings, cost=cost, force=force)
    elif stage_name == "classify_outcome":
        mod = _load(*STAGE_MODULES["classify_outcome"][:2])
        def work(pid):
            return mod.run_one(pid, client=client, settings=settings, cost=cost, force=force)
    elif stage_name == "map_onet":
        mod = _load(*STAGE_MODULES["map_onet"][:2])
        activities, occupations = mod._load_onet()
        cached_system = mod._build_cached_system(activities, occupations)
        def work(pid):
            return mod.run_one(pid, client=client, settings=settings, cost=cost,
                               cached_system=cached_system,
                               activities=activities, occupations=occupations, force=force)
    elif stage_name == "compute_effects":
        speed_mod = _load(*STAGE_MODULES["compute_speed"][:2])
        quality_mod = _load(*STAGE_MODULES["compute_quality"][:2])
        def work(pid):
            r1 = speed_mod.run_one(pid, client=client, settings=settings, cost=cost, force=force)
            r2 = quality_mod.run_one(pid, client=client, settings=settings, cost=cost, force=force)
            return r1 or r2
    elif stage_name == "compute_speed":
        mod = _load(*STAGE_MODULES["compute_speed"][:2])
        def work(pid):
            return mod.run_one(pid, client=client, settings=settings, cost=cost, force=force)
    elif stage_name == "compute_quality":
        mod = _load(*STAGE_MODULES["compute_quality"][:2])
        def work(pid):
            return mod.run_one(pid, client=client, settings=settings, cost=cost, force=force)
    elif stage_name == "assemble":
        mod = _load("06_assemble_tables", SCRIPTS_DIR / "06_assemble_tables.py")
        mod.assemble()
        return
    else:
        raise ValueError(f"unknown stage: {stage_name}")

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(work, pid): pid for pid in paper_ids}
        for fut in as_completed(futs):
            pid = futs[fut]
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                log.exception("stage=%s paper=%s failed: %s", stage_name, pid, e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers-dir", type=Path, default=REPO_ROOT / "papers")
    ap.add_argument("--stage", type=str, default=None,
                    help="Run a single stage only. One of: extract_quotes, structure, classify_method, "
                         "classify_outcome, map_onet, compute_speed, compute_quality, compute_effects, assemble.")
    ap.add_argument("--paper", type=str, default=None, help="Run on a single paper id (filename stem).")
    ap.add_argument("--force", action="store_true", help="Re-run even if output exists.")
    args = ap.parse_args()

    settings = load_settings()
    client = get_client()
    cost = CostTracker(settings["pricing"])

    paper_ids = [args.paper] if args.paper else _list_paper_ids(args.papers_dir)
    concurrency = settings.get("concurrency", 4)

    stages = [args.stage] if args.stage else settings["stages"]
    for stage in stages:
        run_stage(stage, paper_ids, settings=settings, client=client, cost=cost,
                  papers_dir=args.papers_dir, force=args.force, concurrency=concurrency)

    log.info("%s", cost.summary())


if __name__ == "__main__":
    main()
