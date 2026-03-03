#!/usr/bin/env python3
"""
LLM4 - Local test run: 100 samples, CPU only.
Runs data generation, Phase 1 (random + balanced random + guided), Stage 0 (base on 100),
evaluation with bootstrap CIs and perplexity, then comparison report.
Use this to verify the pipeline and statistics before running on Kaggle GPU.
Set config local_test: true (default in config.yaml for local dev).
"""

import os
import subprocess
import sys
from pathlib import Path

# Force CPU for local run (optional; device_utils already uses CPU when not on Kaggle)
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
    scripts = PROJECT_ROOT / "scripts"
    data = PROJECT_ROOT / "data"

    run([sys.executable, str(data / "generate_arithmetic.py")], "Generate datasets (train_100, balanced_train_100, test_200)")
    run([sys.executable, str(scripts / "phase1_baseline.py")], "Phase 1: random_100, balanced_random_100, guided_100 (from scratch, 100 samples)")
    run([sys.executable, str(scripts / "stage0_train_base.py")], "Stage 0: Train base from scratch on 100 examples -> models/base")
    run([sys.executable, str(scripts / "compare_stages.py")], "Comparison report (Phase 1 + Stage 0)")

    print("")
    print("Local test complete. All scores in output/ with bootstrap CIs and perplexity.")
    print("Switch config local_test: false and run run_all.py for full Kaggle pipeline.")


if __name__ == "__main__":
    main()
