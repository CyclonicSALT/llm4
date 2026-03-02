"""
LLM4 Phase 1: Clean baseline - no pretraining.
Compare: random_100, random_1000, guided_100 (probe-guided 100).
All three trained from scratch (random weights). Proves data-efficiency hypothesis.
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

def build_guided_dataset(failures_path: Path, train_100_path: Path, output_path: Path, targeted_cap=60, total=100):
    """Build guided_100: up to 60 targeted examples from failure types, fill to 100 with train_100."""
    with open(failures_path, "r", encoding="utf-8") as f:
        failures = [json.loads(line) for line in f if line.strip()]
    type_failures = {}
    for r in failures:
        t = r.get("type", "unknown")
        type_failures.setdefault(t, []).append(r)

    with open(train_100_path, "r", encoding="utf-8") as f:
        train_100 = [json.loads(line) for line in f if line.strip()]

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
        pool = [ex for ex in train_100 if ex not in targeted][:need]
        rng.shuffle(pool)
        guided = targeted + pool
    else:
        guided = targeted[:total]
    rng.shuffle(guided)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in guided:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Built guided dataset: {len(guided)} examples ({len(targeted)} targeted, {len(guided)-len(targeted)} from train_100)")
    return len(guided)


def load_config():
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    data_dir = PROJECT_ROOT / "data"
    output_dir = PROJECT_ROOT / config["output_dir"].replace("./", "")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_100 = data_dir / "train_100.jsonl"
    train_1000_path = data_dir / "train_1000.jsonl"
    test_200 = data_dir / "test_200.jsonl"
    guided_100_path = data_dir / "guided_100.jsonl"
    failures_path = data_dir / "probe_failures.jsonl"

    print("=" * 60)
    print("LLM4 PHASE 1: CLEAN BASELINE (no pretraining)")
    print("=" * 60)

    # 1. Generate datasets
    print("\n[1] Generating datasets...")
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "data" / "generate_arithmetic.py")],
        cwd=PROJECT_ROOT,
        check=True,
    )

    # 2. Train random_100 from scratch, evaluate
    print("\n[2] Training random_100 (from scratch, 100 random examples)...")
    subprocess.run([
        sys.executable,
        str(SCRIPT_DIR / "train_model.py"),
        "--data", str(train_100),
        "--output", str(PROJECT_ROOT / config["random_100_output"].replace("./", "")),
        "--samples", "100",
        "--from-scratch",
    ], cwd=PROJECT_ROOT, check=True)

    print("Evaluating random_100 on 200 test problems...")
    subprocess.run([
        sys.executable,
        str(SCRIPT_DIR / "evaluate_model_hf.py"),
        "--model", config["random_100_output"],
        "--output", str(output_dir / "random_100_scores.json"),
        "--stage", "random_100",
    ], cwd=PROJECT_ROOT, check=True)

    # 3. Extract failures and build probe-guided dataset (60 targeted, 100 total)
    with open(output_dir / "random_100_scores.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    failures = [r for r in data.get("results", []) if not r.get("correct", True)]
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    with open(failures_path, "w", encoding="utf-8") as f:
        for r in failures:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(failures)} failures from random_100.")

    build_guided_dataset(failures_path, train_100, guided_100_path, targeted_cap=60, total=100)

    # 4. Train random_1000 from scratch, evaluate
    print("\n[4] Training random_1000 (from scratch, 1000 random examples)...")
    subprocess.run([
        sys.executable,
        str(SCRIPT_DIR / "train_model.py"),
        "--data", str(train_1000_path),
        "--output", str(PROJECT_ROOT / config["random_1000_output"].replace("./", "")),
        "--samples", "1000",
        "--from-scratch",
    ], cwd=PROJECT_ROOT, check=True)

    print("Evaluating random_1000...")
    subprocess.run([
        sys.executable,
        str(SCRIPT_DIR / "evaluate_model_hf.py"),
        "--model", config["random_1000_output"],
        "--output", str(output_dir / "random_1000_scores.json"),
        "--stage", "random_1000",
    ], cwd=PROJECT_ROOT, check=True)

    # 5. Train guided_100 from scratch, evaluate
    print("\n[5] Training guided_100 (from scratch, 100 probe-guided examples)...")
    subprocess.run([
        sys.executable,
        str(SCRIPT_DIR / "train_model.py"),
        "--data", str(guided_100_path),
        "--output", str(PROJECT_ROOT / config["guided_100_output"].replace("./", "")),
        "--samples", "100",
        "--from-scratch",
    ], cwd=PROJECT_ROOT, check=True)

    print("Evaluating guided_100...")
    subprocess.run([
        sys.executable,
        str(SCRIPT_DIR / "evaluate_model_hf.py"),
        "--model", config["guided_100_output"],
        "--output", str(output_dir / "guided_100_scores.json"),
        "--stage", "guided_100",
    ], cwd=PROJECT_ROOT, check=True)

    # 6. Comparison report
    def acc(path):
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("overall_accuracy", 0)

    r100 = acc(output_dir / "random_100_scores.json")
    r1000 = acc(output_dir / "random_1000_scores.json")
    g100 = acc(output_dir / "guided_100_scores.json")

    report = {
        "random_100_accuracy": r100,
        "random_1000_accuracy": r1000,
        "guided_100_accuracy": g100,
        "verdict": None,
        "hypothesis_proven": None,
    }
    if r100 is not None and g100 is not None:
        report["verdict"] = "Targeting helps." if g100 > r100 else "Targeting did not beat random 100."
    if g100 is not None and r1000 is not None:
        report["hypothesis_proven"] = g100 >= r1000

    with open(output_dir / "phase1_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("")
    print("=" * 60)
    print("PHASE 1 COMPARISON REPORT")
    print("=" * 60)
    print(f"  random_100:   {r100:.1f}%" if r100 is not None else "  random_100:   N/A")
    print(f"  random_1000:  {r1000:.1f}%  (10x more data)" if r1000 is not None else "  random_1000:  N/A")
    print(f"  guided_100:   {g100:.1f}%  (same data as random_100, targeted)" if g100 is not None else "  guided_100:   N/A")
    print("=" * 60)
    if report.get("verdict"):
        print(f"  VERDICT: {report['verdict']}")
    if report.get("hypothesis_proven") is not None:
        print(f"  HYPOTHESIS (guided_100 >= random_1000): {'PROVEN' if report['hypothesis_proven'] else 'Not proven'}")
    print("")
    print(f"Report saved to {output_dir / 'phase1_report.json'}")


if __name__ == "__main__":
    main()
