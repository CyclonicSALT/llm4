#!/usr/bin/env python3
"""
LLM4 - Local test: Phase 1 + Stage 0 only (no stages 1-6).
Uses local_phase1_base_size / local_phase1_large_size from config (e.g. 100/500 or 10/20).
Set local_test: true, then: python run_local.py
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from config_utils import use_project_cache_only

use_project_cache_only()
os.environ["FORCE_CPU"] = "1"


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

    run([sys.executable, str(data / "generate_arithmetic.py")], "Generate datasets")
    run([sys.executable, str(scripts / "phase1_baseline.py")], "Phase 1: Clean baseline")
    run([sys.executable, str(scripts / "stage0_train_base.py")], "Stage 0: Train base -> models/base")
    run([sys.executable, str(scripts / "compare_stages.py")], "Comparison report")

    print("")
    print("Local test complete. Run run_smoke.py for full pipeline, or run_all.py for Kaggle.")


if __name__ == "__main__":
    main()
