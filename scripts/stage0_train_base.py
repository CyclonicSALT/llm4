"""
LLM4 Stage 0: Train base from scratch -> save models/base.
Uses train_phase1_base + phase1_base_size unless stage0_train_data (and stage0_samples) are set.
Each subsequent stage loads the previous stage's weights.
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_utils import load_config


def main():
    config = load_config()
    stage0_data = config.get("stage0_train_data")
    if stage0_data and str(stage0_data).strip():
        train_data = PROJECT_ROOT / str(stage0_data).replace("./", "")
        samples = str(config.get("stage0_samples") or config.get("local_phase1_base_size", 100))
    else:
        train_data = PROJECT_ROOT / config.get("train_phase1_base", "data/train_phase1_base.jsonl").replace("./", "")
        samples = str(config.get("local_phase1_base_size", 100))
    base_output = PROJECT_ROOT / config["base_output"].replace("./", "")
    output_dir = PROJECT_ROOT / config["output_dir"].replace("./", "")

    print(f"Stage 0: Train base from scratch ({samples} examples) -> models/base")
    if not train_data.exists():
        print("Run data/generate_arithmetic.py first.")
        sys.exit(1)

    subprocess.run([
        sys.executable,
        str(SCRIPT_DIR / "train_model.py"),
        "--data", str(train_data),
        "--output", str(base_output),
        "--samples", samples,
        "--from-scratch",
    ], cwd=PROJECT_ROOT, check=True)

    subprocess.run([
        sys.executable,
        str(SCRIPT_DIR / "evaluate_model_hf.py"),
        "--model", str(base_output),
        "--output", str(output_dir / "stage0_base_scores.json"),
        "--stage", "stage0_base",
    ], cwd=PROJECT_ROOT, check=True)
    print("Stage 0 complete.")


if __name__ == "__main__":
    main()
