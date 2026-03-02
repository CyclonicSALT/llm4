"""
LLM4 Stage 2: Probe guided. Load models/cot, continue training on targeted data -> save models/probe_guided.
Uses cot model eval failures to build enhanced dataset.
"""

import json
import random
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.generate_arithmetic import GENERATORS


def load_config():
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    cot_output = PROJECT_ROOT / config["cot_output"].replace("./", "")
    probe_guided_output = PROJECT_ROOT / config["probe_guided_output"].replace("./", "")
    output_dir = PROJECT_ROOT / config["output_dir"].replace("./", "")
    cot_train_path = PROJECT_ROOT / config["cot_train"].replace("./", "")
    enhanced_train_path = PROJECT_ROOT / config["enhanced_train"].replace("./", "")
    failures_path = PROJECT_ROOT / config["failures_data"].replace("./", "")
    targeted_per_gap = config.get("targeted_examples_per_gap", 20)
    difficulty_levels = config.get("difficulty_levels", 3)

    print("Stage 2: Probe guided - load cot, train on targeted -> models/probe_guided")
    if not (cot_output / "config.json").exists():
        print("models/cot not found. Run Stage 1 first.")
        sys.exit(1)

    # Evaluate cot to get failures
    scores_path = output_dir / "stage2_cot_for_failures.json"
    if not scores_path.exists():
        subprocess.run([
            sys.executable,
            str(SCRIPT_DIR / "evaluate_model_hf.py"),
            "--model", str(cot_output),
            "--output", str(scores_path),
            "--stage", "cot",
            "--max-tokens", "128",
        ], cwd=PROJECT_ROOT, check=True)

    with open(scores_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    failures = [r for r in data.get("results", []) if not r.get("correct", True)]
    type_failures = {}
    for r in failures:
        t = r.get("type", "unknown")
        type_failures.setdefault(t, []).append(r)
    Path(failures_path).parent.mkdir(parents=True, exist_ok=True)
    with open(failures_path, "w", encoding="utf-8") as f:
        for r in failures:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(failures)} failures from CoT model.")

    rng = random.Random(43)
    targeted = []
    gap_types = sorted(type_failures.keys(), key=lambda t: -len(type_failures[t]))
    n_per_gap = min(targeted_per_gap * difficulty_levels, 60)
    per_gap = max(1, n_per_gap // max(1, len(gap_types)))
    for t in gap_types:
        if t not in GENERATORS:
            continue
        for _ in range(per_gap):
            instr, resp, operands, correct = GENERATORS[t](rng)
            targeted.append({
                "instruction": instr,
                "response": resp,
                "type": t,
                "operands": operands,
                "correct_answer": correct,
            })
            if len(targeted) >= 60:
                break
        if len(targeted) >= 60:
            break
    targeted = targeted[:60]

    cot_examples = []
    if cot_train_path.exists():
        with open(cot_train_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    cot_examples.append(json.loads(line))
    combined = cot_examples + targeted
    rng.shuffle(combined)
    combined = combined[: config.get("stage2_targeted_samples", 100)]
    enhanced_train_path = Path(enhanced_train_path)
    enhanced_train_path.parent.mkdir(parents=True, exist_ok=True)
    with open(enhanced_train_path, "w", encoding="utf-8") as f:
        for ex in combined:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Enhanced dataset: {len(combined)} examples")

    subprocess.run([
        sys.executable,
        str(SCRIPT_DIR / "train_model.py"),
        "--data", str(enhanced_train_path),
        "--output", str(probe_guided_output),
        "--samples", str(len(combined)),
        "--base", str(cot_output),
    ], cwd=PROJECT_ROOT, check=True)

    subprocess.run([
        sys.executable,
        str(SCRIPT_DIR / "evaluate_model_hf.py"),
        "--model", str(probe_guided_output),
        "--output", str(output_dir / "stage2_probe_guided_scores.json"),
        "--stage", "stage2_probe_guided",
        "--max-tokens", "128",
    ], cwd=PROJECT_ROOT, check=True)
    print("Stage 2 complete.")


if __name__ == "__main__":
    main()
