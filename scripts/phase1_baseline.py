"""
LLM4 Phase 1: Clean baseline - no pretraining.
Compare: random_5k, random_50k, guided_5k (probe-guided 5k).
All three trained from scratch (random weights). Scale controlled by config (phase1_base_size, phase1_large_size).
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

def build_guided_dataset(failures_path: Path, train_pool_path: Path, output_path: Path, targeted_cap: int, total: int):
    """Build guided set: up to targeted_cap from failure types, fill to total with train_pool."""
    with open(failures_path, "r", encoding="utf-8") as f:
        failures = [json.loads(line) for line in f if line.strip()]
    type_failures = {}
    for r in failures:
        t = r.get("type", "unknown")
        type_failures.setdefault(t, []).append(r)

    with open(train_pool_path, "r", encoding="utf-8") as f:
        train_pool = [json.loads(line) for line in f if line.strip()]

    rng = random.Random(43)
    from data.generate_arithmetic import GENERATORS

    targeted = []
    gap_types = sorted(type_failures.keys(), key=lambda t: -len(type_failures[t]))
    per_gap = max(1, targeted_cap // max(1, len(gap_types)))
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
            if len(targeted) >= targeted_cap:
                break
        if len(targeted) >= targeted_cap:
            break
    targeted = targeted[:targeted_cap]
    need = total - len(targeted)
    if need > 0:
        pool = [ex for ex in train_pool if ex not in targeted][:need]
        rng.shuffle(pool)
        guided = targeted + pool
    else:
        guided = targeted[:total]
    rng.shuffle(guided)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in guided:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Built guided dataset: {len(guided)} examples ({len(targeted)} targeted, {len(guided)-len(targeted)} from pool)")
    return len(guided)


def load_config():
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    data_dir = PROJECT_ROOT / "data"
    output_dir = PROJECT_ROOT / config["output_dir"].replace("./", "")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_size = int(config.get("phase1_base_size", 5000))
    large_size = int(config.get("phase1_large_size", 50000))
    train_base_path = PROJECT_ROOT / config["train_5000"].replace("./", "")
    train_large_path = PROJECT_ROOT / config["train_50000"].replace("./", "")
    guided_path = data_dir / "guided_5k.jsonl"
    failures_path = data_dir / "probe_failures.jsonl"
    targeted_cap = int(0.6 * base_size)

    print("=" * 60)
    print("LLM4 PHASE 1: CLEAN BASELINE (no pretraining)")
    print(f"Scale: base={base_size}, large={large_size}")
    print("=" * 60)

    # 1. Generate datasets
    print("\n[1] Generating datasets...")
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "data" / "generate_arithmetic.py")],
        cwd=PROJECT_ROOT,
        check=True,
    )

    # 2. Train random_5k from scratch, evaluate
    print(f"\n[2] Training random_5k (from scratch, {base_size} random examples)...")
    subprocess.run([
        sys.executable,
        str(SCRIPT_DIR / "train_model.py"),
        "--data", str(train_base_path),
        "--output", str(PROJECT_ROOT / config["random_5k_output"].replace("./", "")),
        "--samples", str(base_size),
        "--from-scratch",
    ], cwd=PROJECT_ROOT, check=True)

    print(f"Evaluating random_5k on 200 test problems...")
    subprocess.run([
        sys.executable,
        str(SCRIPT_DIR / "evaluate_model_hf.py"),
        "--model", config["random_5k_output"],
        "--output", str(output_dir / "random_5k_scores.json"),
        "--stage", "random_5k",
    ], cwd=PROJECT_ROOT, check=True)

    # 3. Extract failures and build probe-guided dataset (targeted_cap targeted, base_size total)
    with open(output_dir / "random_5k_scores.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    failures = [r for r in data.get("results", []) if not r.get("correct", True)]
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    with open(failures_path, "w", encoding="utf-8") as f:
        for r in failures:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(failures)} failures from random_5k.")

    build_guided_dataset(failures_path, train_base_path, guided_path, targeted_cap=targeted_cap, total=base_size)

    # 4. Train random_50k from scratch, evaluate
    print(f"\n[4] Training random_50k (from scratch, {large_size} random examples)...")
    subprocess.run([
        sys.executable,
        str(SCRIPT_DIR / "train_model.py"),
        "--data", str(train_large_path),
        "--output", str(PROJECT_ROOT / config["random_50k_output"].replace("./", "")),
        "--samples", str(large_size),
        "--from-scratch",
    ], cwd=PROJECT_ROOT, check=True)

    print("Evaluating random_50k...")
    subprocess.run([
        sys.executable,
        str(SCRIPT_DIR / "evaluate_model_hf.py"),
        "--model", config["random_50k_output"],
        "--output", str(output_dir / "random_50k_scores.json"),
        "--stage", "random_50k",
    ], cwd=PROJECT_ROOT, check=True)

    # 5. Train guided_5k from scratch, evaluate
    print(f"\n[5] Training guided_5k (from scratch, {base_size} probe-guided examples)...")
    subprocess.run([
        sys.executable,
        str(SCRIPT_DIR / "train_model.py"),
        "--data", str(guided_path),
        "--output", str(PROJECT_ROOT / config["guided_5k_output"].replace("./", "")),
        "--samples", str(base_size),
        "--from-scratch",
    ], cwd=PROJECT_ROOT, check=True)

    print("Evaluating guided_5k...")
    subprocess.run([
        sys.executable,
        str(SCRIPT_DIR / "evaluate_model_hf.py"),
        "--model", config["guided_5k_output"],
        "--output", str(output_dir / "guided_5k_scores.json"),
        "--stage", "guided_5k",
    ], cwd=PROJECT_ROOT, check=True)

    # 6. Comparison report
    def acc(path):
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("overall_accuracy", 0)

    r5k = acc(output_dir / "random_5k_scores.json")
    r50k = acc(output_dir / "random_50k_scores.json")
    g5k = acc(output_dir / "guided_5k_scores.json")

    report = {
        "random_5k_accuracy": r5k,
        "random_50k_accuracy": r50k,
        "guided_5k_accuracy": g5k,
        "verdict": None,
        "hypothesis_proven": None,
    }
    if r5k is not None and g5k is not None:
        report["verdict"] = "Targeting helps." if g5k > r5k else "Targeting did not beat random 5k."
    if g5k is not None and r50k is not None:
        report["hypothesis_proven"] = g5k >= r50k

    with open(output_dir / "phase1_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("")
    print("=" * 60)
    print("PHASE 1 COMPARISON REPORT")
    print("=" * 60)
    print(f"  random_5k:   {r5k:.1f}%" if r5k is not None else "  random_5k:   N/A")
    print(f"  random_50k:  {r50k:.1f}%  (10x more data)" if r50k is not None else "  random_50k:  N/A")
    print(f"  guided_5k:   {g5k:.1f}%  (same size as random_5k, targeted)" if g5k is not None else "  guided_5k:   N/A")
    print("=" * 60)
    if report.get("verdict"):
        print(f"  VERDICT: {report['verdict']}")
    if report.get("hypothesis_proven") is not None:
        print(f"  HYPOTHESIS (guided_5k >= random_50k): {'PROVEN' if report['hypothesis_proven'] else 'Not proven'}")
    print("")
    print(f"Report saved to {output_dir / 'phase1_report.json'}")


if __name__ == "__main__":
    main()
