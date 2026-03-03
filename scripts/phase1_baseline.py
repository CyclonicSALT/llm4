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
    import argparse
    p = argparse.ArgumentParser(description="Phase 1 baseline (random, balanced random, guided)")
    p.add_argument("--seed", type=int, default=None, help="Seed for this run (for multi-seed: 42, 43, 44); writes to output/seed_X/ and models/*_seedX")
    args_main = p.parse_args()

    config = load_config()
    data_dir = PROJECT_ROOT / "data"
    output_dir = PROJECT_ROOT / config["output_dir"].replace("./", "")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_seed = args_main.seed
    if run_seed is not None:
        score_dir = output_dir / f"seed_{run_seed}"
        score_dir.mkdir(parents=True, exist_ok=True)
    else:
        score_dir = output_dir

    local_test = config.get("local_test", False)
    if local_test:
        base_size = 100
        large_size = 100
        train_base_path = PROJECT_ROOT / config["train_100"].replace("./", "")
        train_large_path = PROJECT_ROOT / config["train_1000"].replace("./", "")
        balanced_path = PROJECT_ROOT / config.get("balanced_train_100", "data/balanced_train_100.jsonl").replace("./", "")
        guided_path = data_dir / "guided_100.jsonl"
        random_output = config["random_100_output"].replace("./", "")
        balanced_output = config.get("balanced_random_100_output", "./models/balanced_random_100").replace("./", "")
        guided_output = config["guided_100_output"].replace("./", "")
        scores_random = "random_100_scores.json"
        scores_balanced = "balanced_random_100_scores.json"
        scores_large = "random_1000_scores.json"
        scores_guided = "guided_100_scores.json"
    else:
        base_size = int(config.get("phase1_base_size", 5000))
        large_size = int(config.get("phase1_large_size", 50000))
        train_base_path = PROJECT_ROOT / config["train_5000"].replace("./", "")
        train_large_path = PROJECT_ROOT / config["train_50000"].replace("./", "")
        balanced_path = data_dir / "balanced_train_5k.jsonl"  # generated below
        guided_path = data_dir / "guided_5k.jsonl"
        random_output = config["random_5k_output"].replace("./", "")
        balanced_output = config.get("balanced_random_5k_output", "./models/balanced_random_5k").replace("./", "")
        guided_output = config["guided_5k_output"].replace("./", "")
        scores_random = "random_5k_scores.json"
        scores_balanced = "balanced_random_5k_scores.json"
        scores_large = "random_50k_scores.json"
        scores_guided = "guided_5k_scores.json"

    if run_seed is not None:
        def _seed_path(s):
            p = Path(s.replace("./", ""))
            return str(p.parent / (p.name + f"_seed{run_seed}")).replace("\\", "/")
        random_output = _seed_path(random_output)
        balanced_output = _seed_path(balanced_output)
        guided_output = _seed_path(guided_output)

    failures_path = data_dir / "probe_failures.jsonl"
    targeted_cap = int(0.6 * base_size)

    train_seed = run_seed if run_seed is not None else config.get("seed", 42)

    print("=" * 60)
    print("LLM4 PHASE 1: CLEAN BASELINE (no pretraining)")
    if run_seed is not None:
        print(f"Multi-seed run: seed={run_seed}")
    if local_test:
        print("LOCAL TEST MODE: 100 samples, CPU")
    print(f"Scale: base={base_size}, large={large_size}")
    print("=" * 60)

    # 1. Generate datasets
    print("\n[1] Generating datasets...")
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "data" / "generate_arithmetic.py")],
        cwd=PROJECT_ROOT,
        check=True,
    )

    if not local_test:
        from data.generate_arithmetic import generate_balanced_problems, write_jsonl
        balanced_5k = generate_balanced_problems(5000, 44)
        write_jsonl(data_dir / "balanced_train_5k.jsonl", balanced_5k)
        balanced_path = data_dir / "balanced_train_5k.jsonl"

    # 2. Train random (unstratified) from scratch, evaluate
    print(f"\n[2] Training random (from scratch, {base_size} examples)...")
    subprocess.run([
        sys.executable,
        str(SCRIPT_DIR / "train_model.py"),
        "--data", str(train_base_path),
        "--output", str(PROJECT_ROOT / random_output),
        "--samples", str(base_size),
        "--from-scratch",
        "--seed", str(train_seed),
    ], cwd=PROJECT_ROOT, check=True)

    print(f"Evaluating random on test set...")
    subprocess.run([
        sys.executable,
        str(SCRIPT_DIR / "evaluate_model_hf.py"),
        "--model", random_output,
        "--output", str(score_dir / scores_random),
        "--stage", ("random_100" if local_test else "random_5k"),
    ], cwd=PROJECT_ROOT, check=True)

    # 3. Extract failures and build probe-guided dataset
    with open(score_dir / scores_random, "r", encoding="utf-8") as f:
        data = json.load(f)
    failures = [r for r in data.get("results", []) if not r.get("correct", True)]
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    with open(failures_path, "w", encoding="utf-8") as f:
        for r in failures:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(failures)} failures from random run.")

    build_guided_dataset(failures_path, train_base_path, guided_path, targeted_cap=targeted_cap, total=base_size)

    # 4. Train balanced random baseline (§11), evaluate
    if balanced_path.exists():
        print(f"\n[4] Training balanced random (from scratch, {base_size} stratified examples)...")
        subprocess.run([
            sys.executable,
            str(SCRIPT_DIR / "train_model.py"),
            "--data", str(balanced_path),
            "--output", str(PROJECT_ROOT / balanced_output),
            "--samples", str(base_size),
            "--from-scratch",
            "--seed", str(train_seed),
        ], cwd=PROJECT_ROOT, check=True)
        print("Evaluating balanced random...")
        subprocess.run([
            sys.executable,
            str(SCRIPT_DIR / "evaluate_model_hf.py"),
            "--model", balanced_output,
            "--output", str(score_dir / scores_balanced),
            "--stage", ("balanced_random_100" if local_test else "balanced_random_5k"),
        ], cwd=PROJECT_ROOT, check=True)

    # 5. Train large random (local_test: large_size==base_size so we skip; full run: train random_50k)
    if local_test and large_size > base_size:
        print(f"\n[5] Training random_1000 (from scratch, {large_size} examples)...")
        subprocess.run([
            sys.executable,
            str(SCRIPT_DIR / "train_model.py"),
            "--data", str(train_large_path),
            "--output", str(PROJECT_ROOT / config["random_1000_output"].replace("./", "")),
            "--samples", str(large_size),
            "--from-scratch",
        ], cwd=PROJECT_ROOT, check=True)
        subprocess.run([
            sys.executable,
            str(SCRIPT_DIR / "evaluate_model_hf.py"),
            "--model", config["random_1000_output"],
            "--output", str(score_dir / scores_large),
            "--stage", "random_1000",
        ], cwd=PROJECT_ROOT, check=True)
    elif not local_test:
        print(f"\n[5] Training random_50k (from scratch, {large_size} examples)...")
        rand_50k_out = config["random_50k_output"].replace("./", "")
        if run_seed is not None:
            rand_50k_out = str(Path(rand_50k_out).parent / (Path(rand_50k_out).name + f"_seed{run_seed}")).replace("\\", "/")
        subprocess.run([
            sys.executable,
            str(SCRIPT_DIR / "train_model.py"),
            "--data", str(train_large_path),
            "--output", str(PROJECT_ROOT / rand_50k_out),
            "--samples", str(large_size),
            "--from-scratch",
            "--seed", str(train_seed),
        ], cwd=PROJECT_ROOT, check=True)
        subprocess.run([
            sys.executable,
            str(SCRIPT_DIR / "evaluate_model_hf.py"),
            "--model", rand_50k_out,
            "--output", str(score_dir / scores_large),
            "--stage", "random_50k",
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
        "--seed", str(train_seed),
    ], cwd=PROJECT_ROOT, check=True)

    print("Evaluating guided...")
    subprocess.run([
        sys.executable,
        str(SCRIPT_DIR / "evaluate_model_hf.py"),
        "--model", guided_output,
        "--output", str(score_dir / scores_guided),
        "--stage", ("guided_100" if local_test else "guided_5k"),
    ], cwd=PROJECT_ROOT, check=True)

    # 7. Comparison report (for this seed)
    def acc(path):
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("overall_accuracy", 0)

    r_base = acc(score_dir / scores_random)
    r_bal = acc(score_dir / scores_balanced) if (score_dir / scores_balanced).exists() else None
    r_large = acc(score_dir / scores_large) if (score_dir / scores_large).exists() else None
    g_base = acc(score_dir / scores_guided)

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
    if g_base is not None and r_bal is not None:
        report["verdict_guided_vs_balanced"] = "Guided beats balanced random." if g_base > r_bal else "Guided did not beat balanced random."
    if g_base is not None and r_large is not None:
        report["hypothesis_proven"] = g_base >= r_large

    report_path = score_dir / "phase1_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("")
    print("=" * 60)
    print("PHASE 1 COMPARISON REPORT" + (f" (seed={run_seed})" if run_seed is not None else ""))
    print("=" * 60)
    print(f"  random:           {r_base:.1f}%" if r_base is not None else "  random:           N/A")
    if r_bal is not None:
        print(f"  balanced random:  {r_bal:.1f}%  (stratified)")
    if r_large is not None:
        print(f"  random (large):   {r_large:.1f}%")
    print(f"  guided:           {g_base:.1f}%" if g_base is not None else "  guided:           N/A")
    print("=" * 60)
    if report.get("verdict"):
        print(f"  VERDICT: {report['verdict']}")
    if report.get("verdict_guided_vs_balanced"):
        print(f"  GUIDED vs BALANCED: {report['verdict_guided_vs_balanced']}")
    if report.get("hypothesis_proven") is not None:
        print(f"  HYPOTHESIS (guided >= random_large): {'PROVEN' if report['hypothesis_proven'] else 'Not proven'}")
    print("")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
