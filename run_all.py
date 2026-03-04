#!/usr/bin/env python3
"""
LLM4 - Single entry point: Phase 1 (baseline) then Phase 2 (stacking).
Saves all scores to output/ as JSON. Prints final comparison report.
Run on Kaggle P100 or locally. All paths relative.
When local_test is false, Phase 1 runs with seeds 42, 43, 44 (required for main claim).
"""

import os
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from config_utils import use_project_cache_only

use_project_cache_only()


def run(cmd, desc):
    print("")
    print("=" * 60)
    print(desc)
    print("=" * 60)
    r = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if r.returncode != 0:
        print(f"Failed: {desc}")
        sys.exit(r.returncode)


def main():
    scripts = PROJECT_ROOT / "scripts"
    data = PROJECT_ROOT / "data"
    rag = PROJECT_ROOT / "rag"

    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    local_test = config.get("local_test", False)

    # Phase 1
    run([sys.executable, str(data / "generate_arithmetic.py")], "Generate datasets")

    if local_test:
        run([sys.executable, str(scripts / "phase1_baseline.py")], "Phase 1: Clean baseline (100 samples, single seed)")
    else:
        for seed in [42, 43, 44]:
            run(
                [sys.executable, str(scripts / "phase1_baseline.py"), "--seed", str(seed)],
                f"Phase 1: Clean baseline (seed={seed})",
            )
        run([sys.executable, str(scripts / "aggregate_phase1_seeds.py")], "Phase 1: Aggregate seeds 42/43/44 (mean ± std)")

    # Phase 2: stacking (each stage loads previous)
    run([sys.executable, str(scripts / "stage0_train_base.py")], "Stage 0: Train base from scratch -> models/base")
    run([sys.executable, str(scripts / "stage1_cot_format.py")], "Stage 1: CoT (load base) -> models/cot")
    run([sys.executable, str(scripts / "stage2_probe_guided.py")], "Stage 2: Probe guided (load cot) -> models/probe_guided")
    run([sys.executable, str(scripts / "stage3_moe_train.py")], "Stage 3: MoE (load probe_guided) -> models/moe/")
    run([sys.executable, str(scripts / "stage4_prune.py")], "Stage 4: Pruning (load probe_guided) -> models/pruned")
    run([sys.executable, str(scripts / "stage5_qat_train.py")], "Stage 5: QAT (load pruned, max 3 epochs) -> models/final")

    run([sys.executable, str(rag / "build_index.py")], "Build RAG index from arithmetic_facts.jsonl")
    run([sys.executable, str(scripts / "stage6_rag_integrate.py")], "Stage 6: RAG evaluation (top 3 rules)")

    run([sys.executable, str(scripts / "compare_stages.py")], "Final comparison report")
    print("")
    print("Done. All scores in output/. Report: output/stacking_report.json")


if __name__ == "__main__":
    main()
