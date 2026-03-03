"""
LLM4 - Compare all stages and Phase 1 baselines. Print final report, save JSON.
"""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config():
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        import yaml
        return yaml.safe_load(f)


def load_scores(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    config = load_config()
    output_dir = PROJECT_ROOT / config["output_dir"].replace("./", "")

    # Phase 1 (single-seed scores in output/, or multi-seed summary in phase1_multi_seed_report.json)
    multi_seed_path = output_dir / "phase1_multi_seed_report.json"
    if multi_seed_path.exists():
        with open(multi_seed_path, "r", encoding="utf-8") as f:
            multi = json.load(f)
        print("")
        print("Phase 1 (multi-seed mean ± std):")
        for key, v in multi.items():
            if not isinstance(v, dict) or "mean" not in v or "std" not in v:
                continue
            label = key.replace("_accuracy", "").replace("_", " ")
            print(f"  {label}: {v['mean']:.1f}% ± {v['std']:.1f}%")
        if multi.get("missing_seeds"):
            print("  (WARNING: incomplete seeds: {})".format(multi["missing_seeds"]))

    # Phase 1 (local test: 100/500; full run: 50k/100k in output/seed_* or phase1_multi_seed_report)
    phase1 = [
        ("random_100", "random_100_scores.json", 100),
        ("balanced_random_100", "balanced_random_100_scores.json", 100),
        ("random_500", "random_500_scores.json", 500),
        ("random_1000", "random_1000_scores.json", 1000),
        ("guided_100", "guided_100_scores.json", 100),
    ]
    # Phase 2 stacking
    stages = [
        ("stage0_base", "stage0_base_scores.json", 100),
        ("stage1_cot", "stage1_cot_scores.json", 100),
        ("stage2_probe_guided", "stage2_probe_guided_scores.json", 100),
        ("stage3_moe", "stage3_moe_scores.json", 100),
        ("stage4_pruned", "stage4_pruned_scores.json", 100),
        ("stage5_qat", "stage5_qat_scores.json", 100),
        ("stage6_rag", "stage6_rag_scores.json", 100),
    ]

    rows = []
    for label, filename, examples in phase1:
        path = output_dir / filename
        data = load_scores(path)
        acc = data.get("overall_accuracy", 0) if data else None
        boot = data.get("bootstrap_accuracy") if data else None
        if acc is not None and boot:
            acc_str = f"{acc:.1f}% [{boot.get('ci_low', 0):.1f}-{boot.get('ci_high', 0):.1f}%]"
        else:
            acc_str = f"{acc:.1f}%" if acc is not None else "N/A"
        ppl = data.get("perplexity") if data else None
        extra = f" ppl={ppl:.1f}" if ppl is not None else ""
        rows.append(("phase1", label, examples, acc_str + extra))

    for label, filename, examples in stages:
        path = output_dir / filename
        data = load_scores(path)
        acc = data.get("overall_accuracy", 0) if data else None
        boot = data.get("bootstrap_accuracy") if data else None
        if acc is not None and boot:
            acc_str = f"{acc:.1f}% [{boot.get('ci_low', 0):.1f}-{boot.get('ci_high', 0):.1f}%]"
        else:
            acc_str = f"{acc:.1f}%" if acc is not None else "N/A"
        ppl = data.get("perplexity") if data else None
        extra = f" ppl={ppl:.1f}" if ppl is not None else ""
        rows.append(("phase2", label, examples, acc_str + extra))

    print("")
    print("=" * 70)
    print("LLM4 - FINAL COMPARISON REPORT")
    print("=" * 70)
    print("")
    print("Phase 1 (no pretraining):")
    print("-" * 70)
    for _, label, ex, acc_str in rows:
        if _ == "phase1":
            print(f"  {label:<18} | {str(ex):<8} | {acc_str}")
    print("")
    print("Phase 2 (stacking: each stage loads previous):")
    print("-" * 70)
    for _, label, ex, acc_str in rows:
        if _ == "phase2":
            print(f"  {label:<22} | {str(ex):<8} | {acc_str}")
    print("=" * 70)

    # Verdicts
    g100 = load_scores(output_dir / "guided_100_scores.json")
    r100 = load_scores(output_dir / "random_100_scores.json")
    r1000 = load_scores(output_dir / "random_1000_scores.json")
    final_data = load_scores(output_dir / "stage5_qat_scores.json")
    if final_data is None:
        final_data = load_scores(output_dir / "stage6_rag_scores.json")

    if g100 and r100:
        print("")
        print("Phase 1 VERDICT: Targeting helps." if g100.get("overall_accuracy", 0) > r100.get("overall_accuracy", 0) else "Phase 1: guided_100 did not beat random_100.")
    if g100 and r1000:
        print("Hypothesis (guided_100 >= random_1000): " + ("PROVEN" if g100.get("overall_accuracy", 0) >= r1000.get("overall_accuracy", 0) else "Not proven"))
    if final_data and g100:
        print("Phase 2 VERDICT: Stacking beats guided_100 baseline." if final_data.get("overall_accuracy", 0) > g100.get("overall_accuracy", 0) else "Phase 2: stacked final vs guided_100 baseline - check scores above.")

    report = {
        "phase1": [{"label": r[1], "examples": r[2], "accuracy": r[3]} for r in rows if r[0] == "phase1"],
        "phase2": [{"label": r[1], "examples": r[2], "accuracy": r[3]} for r in rows if r[0] == "phase2"],
    }
    with open(output_dir / "stacking_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("")
    print(f"Report saved to {output_dir / 'stacking_report.json'}")


if __name__ == "__main__":
    main()
