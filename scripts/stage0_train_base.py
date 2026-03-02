"""
LLM4 Stage 0: Train base from scratch on 100 examples -> save models/base.
Each subsequent stage loads the previous stage's weights.
"""

import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config():
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    train_100 = PROJECT_ROOT / config["train_100"].replace("./", "")
    base_output = PROJECT_ROOT / config["base_output"].replace("./", "")
    output_dir = PROJECT_ROOT / config["output_dir"].replace("./", "")

    print("Stage 0: Train base from scratch (100 examples) -> models/base")
    if not train_100.exists():
        print("Run data/generate_arithmetic.py first.")
        sys.exit(1)

    subprocess.run([
        sys.executable,
        str(SCRIPT_DIR / "train_model.py"),
        "--data", str(train_100),
        "--output", str(base_output),
        "--samples", "100",
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
