#!/usr/bin/env python3
"""
LLM4 - Full-pipeline smoke test (all stages, minimal data).
Runs Phase 1 + Stage 0 through Stage 6 with local_test config (e.g. 10 base / 20 large)
so the full pipeline completes quickly. Use to verify no bugs before 50k/100k.

Set in config.yaml: local_test: true, local_phase1_base_size: 10, local_phase1_large_size: 20
Then: python run_smoke.py
"""

import os
import subprocess
import sys
from pathlib import Path

import yaml

os.environ["FORCE_CPU"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


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
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not config.get("local_test", False):
        print("Smoke test expects local_test: true in config.yaml (e.g. base 10, large 20).")
        sys.exit(1)

    scripts = PROJECT_ROOT / "scripts"
    data = PROJECT_ROOT / "data"
    rag = PROJECT_ROOT / "rag"

    run([sys.executable, str(data / "generate_arithmetic.py")], "Generate datasets")
    run([sys.executable, str(scripts / "phase1_baseline.py")], "Phase 1: Clean baseline (smoke sizes)")
    run([sys.executable, str(scripts / "stage0_train_base.py")], "Stage 0: Train base -> models/base")
    run([sys.executable, str(scripts / "stage1_cot_format.py")], "Stage 1: CoT -> models/cot")
    run([sys.executable, str(scripts / "stage2_probe_guided.py")], "Stage 2: Probe guided -> models/probe_guided")
    run([sys.executable, str(scripts / "stage3_moe_train.py")], "Stage 3: MoE -> models/moe/")
    run([sys.executable, str(scripts / "stage4_prune.py")], "Stage 4: Pruning -> models/pruned")
    run([sys.executable, str(scripts / "stage5_qat_train.py")], "Stage 5: QAT -> models/final")
    run([sys.executable, str(rag / "build_index.py")], "Build RAG index")
    run([sys.executable, str(scripts / "stage6_rag_integrate.py")], "Stage 6: RAG evaluation")
    run([sys.executable, str(scripts / "compare_stages.py")], "Final comparison report")

    print("")
    print("Smoke test complete. Set local_test: false for full 50k/100k run (run_all.py).")


if __name__ == "__main__":
    main()
