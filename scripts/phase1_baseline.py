"""
LLM4 Phase 1: Clean baseline - no pretraining.
Compare: random_base, random_large, guided_base (probe-guided). All from scratch (random weights).
With --seed N, writes all outputs to output/seed_N/ so multi-seed runs don't overwrite; aggregate_phase1_seeds reads from there.
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_utils import load_config

def build_guided_dataset(failures_path: Path, train_pool_path: Path, output_path: Path, targeted_cap: int, total: int):
    """Build guided set: up to targeted_cap from failure types, fill to total with train pool."""
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


def main():
    parser = argparse.ArgumentParser(description="LLM4 Phase 1 baseline (no pretraining)")
    parser.add_argument("--seed", type=int, default=None, help="If set, write all outputs to output/seed_{seed}/ for multi-seed aggregation")
    args = parser.parse_args()

    config = load_config()
    data_dir = PROJECT_ROOT / "data"
    output_dir = PROJECT_ROOT / config["output_dir"].replace("./", "")
    if args.seed is not None:
        output_dir = output_dir / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    local_test = config.get("local_test", False)
    base_size = int(config.get("local_phase1_base_size", 100))
    large_size = int(config.get("local_phase1_large_size", 500))
    train_base_path = PROJECT_ROOT / config.get("train_phase1_base", "data/train_phase1_base.jsonl").replace("./", "")
    train_large_path = PROJECT_ROOT / config.get("train_phase1_large", "data/train_phase1_large.jsonl").replace("./", "")
    balanced_path = PROJECT_ROOT / config.get("balanced_train_phase1_base", "data/balanced_train_phase1_base.jsonl").replace("./", "")
    guided_path = data_dir / f"guided_{base_size}.jsonl"
    random_output = config.get(f"random_{base_size}_output", f"./models/random_{base_size}").replace("./", "")
    balanced_output = config.get(f"balanced_random_{base_size}_output", f"./models/balanced_random_{base_size}").replace("./", "")
    guided_output = config.get(f"guided_{base_size}_output", f"./models/guided_{base_size}").replace("./", "")
    scores_random = f"random_{base_size}_scores.json"
    scores_balanced = f"balanced_random_{base_size}_scores.json"
    scores_large = f"random_{large_size}_scores.json"
    scores_guided = f"guided_{base_size}_scores.json"

    failures_path = data_dir / "probe_failures.jsonl"
    targeted_cap = int(0.6 * base_size)

    print("=" * 60)
    print("LLM4 PHASE 1: CLEAN BASELINE (no pretraining)")
    print(f"Base={base_size}, large={large_size} (from config)")
    print("=" * 60)

    # 1. Generate datasets (skip if already run by launcher so we don't duplicate output)
    data_dir = PROJECT_ROOT / "data"
    need_generate = not train_base_path.exists() or not (data_dir / "test_200.jsonl").exists()
    if need_generate:
        print("\n[1] Generating datasets...")
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "data" / "generate_arithmetic.py")],
            cwd=PROJECT_ROOT,
            check=True,
        )
    else:
        print("\n[1] Datasets already present, skipping generation.")

    # 2. Train random (base_size) from scratch, evaluate
    print(f"\n[2] Training random (from scratch, {base_size} examples)...")
    subprocess.run([
        sys.executable,
        str(SCRIPT_DIR / "train_model.py"),
        "--data", str(train_base_path),
        "--output", str(PROJECT_ROOT / random_output),
        "--samples", str(base_size),
        "--from-scratch",
    ], cwd=PROJECT_ROOT, check=True)

    print("Evaluating random on test set...")
    subprocess.run([
        sys.executable,
        str(SCRIPT_DIR / "evaluate_model_hf.py"),
        "--model", random_output,
        "--output", str(output_dir / scores_random),
        "--stage", f"random_{base_size}",
    ], cwd=PROJECT_ROOT, check=True)

    # 3. Extract failures and build guided dataset
    with open(output_dir / scores_random, "r", encoding="utf-8") as f:
        data = json.load(f)
    failures = [r for r in data.get("results", []) if not r.get("correct", True)]
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    with open(failures_path, "w", encoding="utf-8") as f:
        for r in failures:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(failures)} failures from random run.")

    build_guided_dataset(failures_path, train_base_path, guided_path, targeted_cap=targeted_cap, total=base_size)

    # 4. Train balanced random (if available)
    if balanced_path and Path(balanced_path).exists():
        print(f"\n[4] Training balanced random (from scratch, {base_size} stratified examples)...")
        subprocess.run([
            sys.executable,
            str(SCRIPT_DIR / "train_model.py"),
            "--data", str(balanced_path),
            "--output", str(PROJECT_ROOT / balanced_output),
            "--samples", str(base_size),
            "--from-scratch",
        ], cwd=PROJECT_ROOT, check=True)
        print("Evaluating balanced random...")
        subprocess.run([
            sys.executable,
            str(SCRIPT_DIR / "evaluate_model_hf.py"),
            "--model", balanced_output,
            "--output", str(output_dir / scores_balanced),
            "--stage", f"balanced_random_{base_size}",
        ], cwd=PROJECT_ROOT, check=True)

    # 5. Train large random (when large_size > base_size)
    if large_size > base_size:
        large_output = config.get(f"random_{large_size}_output", f"./models/random_{large_size}").replace("./", "")
        print(f"\n[5] Training random_{large_size} (from scratch, {large_size} examples)...")
        subprocess.run([
            sys.executable,
            str(SCRIPT_DIR / "train_model.py"),
            "--data", str(train_large_path),
            "--output", str(PROJECT_ROOT / large_output),
            "--samples", str(large_size),
            "--from-scratch",
        ], cwd=PROJECT_ROOT, check=True)
        subprocess.run([
            sys.executable,
            str(SCRIPT_DIR / "evaluate_model_hf.py"),
            "--model", large_output,
            "--output", str(output_dir / scores_large),
            "--stage", f"random_{large_size}",
        ], cwd=PROJECT_ROOT, check=True)

    # 6. Train guided from scratch, evaluate
    print(f"\n[6] Training guided (from scratch, {base_size} probe-guided examples)...")
    subprocess.run([
        sys.executable,
        str(SCRIPT_DIR / "train_model.py"),
        "--data", str(guided_path),
        "--output", str(PROJECT_ROOT / guided_output),
        "--samples", str(base_size),
        "--from-scratch",
    ], cwd=PROJECT_ROOT, check=True)

    print("Evaluating guided...")
    subprocess.run([
        sys.executable,
        str(SCRIPT_DIR / "evaluate_model_hf.py"),
        "--model", guided_output,
        "--output", str(output_dir / scores_guided),
        "--stage", f"guided_{base_size}",
    ], cwd=PROJECT_ROOT, check=True)

    # 7. Comparison report
    def acc(path):
        if not path or not Path(path).exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("overall_accuracy", 0)

    r_base = acc(output_dir / scores_random)
    r_bal = acc(output_dir / scores_balanced) if scores_balanced else None
    r_large = acc(output_dir / scores_large)
    g_base = acc(output_dir / scores_guided)

    report = {
        "random_accuracy": r_base,
        "balanced_random_accuracy": r_bal,
        "random_large_accuracy": r_large,
        "guided_accuracy": g_base,
        "verdict": None,
        "hypothesis_proven": None,
    }
    if r_base is not None and g_base is not None:
        report["verdict"] = "Targeting helps." if g_base > r_base else "Targeting did not beat random."
    if g_base is not None and r_large is not None:
        report["hypothesis_proven"] = g_base >= r_large

    with open(output_dir / "phase1_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("")
    print("=" * 60)
    print("PHASE 1 COMPARISON REPORT")
    print("=" * 60)
    print(f"  random:           {r_base:.1f}%" if r_base is not None else "  random:           N/A")
    if r_bal is not None:
        print(f"  balanced random:  {r_bal:.1f}%  (stratified)")
    print(f"  random (large):   {r_large:.1f}%" if r_large is not None else "  random (large):   N/A")
    print(f"  guided:           {g_base:.1f}%" if g_base is not None else "  guided:           N/A")
    print("=" * 60)
    if report.get("verdict"):
        print(f"  VERDICT: {report['verdict']}")
    if report.get("hypothesis_proven") is not None:
        print(f"  HYPOTHESIS (guided >= random_large): {'PROVEN' if report['hypothesis_proven'] else 'Not proven'}")
    print("")
    print(f"Report saved to {output_dir / 'phase1_report.json'}")


if __name__ == "__main__":
    main()
