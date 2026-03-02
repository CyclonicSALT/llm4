#!/usr/bin/env bash
# LLM4 - Full pipeline for Kaggle / Linux. Phase 1 then Phase 2. All paths relative.
set -e
cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "========================================"
echo "LLM4 - DATA EFFICIENCY (NO PRETRAINING)"
echo "========================================"
echo "Project root: $PROJECT_ROOT"
echo ""

python3 run_all.py

echo ""
echo "Pipeline complete. Report: output/stacking_report.json"
