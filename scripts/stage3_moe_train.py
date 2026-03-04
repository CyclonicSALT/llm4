"""
LLM4 Stage 3: MoE. Load models/probe_guided, train 4 LoRA experts -> save models/moe/
"""

import json
import pickle
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

TYPE_TO_EXPERT = {
    "single_digit_addition": "addition",
    "double_digit_addition": "addition",
    "addition_with_carrying": "addition",
    "three_number_addition": "addition",
    "single_digit_subtraction": "subtraction",
    "double_digit_subtraction": "subtraction",
    "single_digit_multiplication": "multiplication",
    "double_digit_multiplication": "multiplication",
    "simple_division": "division_and_mixed",
    "mixed_operations": "division_and_mixed",
}
EXPERT_TO_TYPES = {
    "addition": ["single_digit_addition", "double_digit_addition", "addition_with_carrying", "three_number_addition"],
    "subtraction": ["single_digit_subtraction", "double_digit_subtraction"],
    "multiplication": ["single_digit_multiplication", "double_digit_multiplication"],
    "division_and_mixed": ["simple_division", "mixed_operations"],
}


from config_utils import load_config


def extract_features(instruction: str):
    return [
        1 if "+" in instruction else 0,
        1 if "-" in instruction else 0,
        1 if "x" in instruction or "*" in instruction else 0,
        1 if "/" in instruction else 0,
        1 if "(" in instruction else 0,
    ]


def main():
    config = load_config()
    probe_guided_path = PROJECT_ROOT / config["probe_guided_output"].replace("./", "")
    moe_output = PROJECT_ROOT / config["moe_output"].replace("./", "")
    enhanced_train_path = PROJECT_ROOT / config["enhanced_train"].replace("./", "")
    train_large_path = PROJECT_ROOT / config.get("train_phase1_large", "data/train_phase1_large.jsonl").replace("./", "")
    samples_per_expert = config.get("stage3_moe_samples", 100)
    samples_per_expert = min(samples_per_expert, max(1, config.get("local_phase1_base_size", 100) // 4))

    print("Stage 3: MoE - load probe_guided, train 4 experts -> models/moe/")
    if not (probe_guided_path / "config.json").exists():
        print("models/probe_guided not found. Run Stage 2 first.")
        sys.exit(1)

    expert_names = ["addition", "subtraction", "multiplication", "division_and_mixed"]
    expert_label = {e: i for i, e in enumerate(expert_names)}

    # Router trained on phase1 large set
    X_router, y_router = [], []
    with open(train_large_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            X_router.append(extract_features(ex["instruction"]))
            y_router.append(expert_label[TYPE_TO_EXPERT[ex["type"]]])
    from sklearn.linear_model import LogisticRegression
    router = LogisticRegression(max_iter=500, random_state=42)
    router.fit(X_router, y_router)
    moe_output.mkdir(parents=True, exist_ok=True)
    with open(moe_output / "router.pkl", "wb") as f:
        pickle.dump(router, f)

    enhanced = []
    if Path(enhanced_train_path).exists():
        with open(enhanced_train_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    enhanced.append(json.loads(line))

    for expert in expert_names:
        types = EXPERT_TO_TYPES[expert]
        subset = [ex for ex in enhanced if ex.get("type") in types][:samples_per_expert]
        if not subset:
            print(f"No data for expert {expert}, skipping.")
            continue
        expert_dir = moe_output / f"expert_{expert}"
        expert_dir.mkdir(parents=True, exist_ok=True)
        data_path = expert_dir / "train_subset.jsonl"
        with open(data_path, "w", encoding="utf-8") as f:
            for ex in subset:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Training expert_{expert} on {len(subset)} examples (load probe_guided)...")
        subprocess.run([
            sys.executable,
            str(SCRIPT_DIR / "train_model.py"),
            "--data", str(data_path),
            "--output", str(expert_dir),
            "--samples", str(len(subset)),
            "--base", str(probe_guided_path),
        ], cwd=PROJECT_ROOT, check=True)

    output_dir = PROJECT_ROOT / config["output_dir"].replace("./", "")
    subprocess.run([
        sys.executable,
        str(SCRIPT_DIR / "evaluate_model_hf.py"),
        "--model", str(moe_output / "expert_addition"),
        "--output", str(output_dir / "stage3_moe_scores.json"),
        "--stage", "stage3_moe",
    ], cwd=PROJECT_ROOT, check=True)
    print("Stage 3 complete.")


if __name__ == "__main__":
    main()
