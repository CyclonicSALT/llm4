"""
LLM4 - Compare all stages and Phase 1 baselines. Print final report, save JSON.
"""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_utils import load_config


def load_scores(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    config = load_config()
    output_dir = PROJECT_ROOT / config["output_dir"].replace("./", "")

    # Phase 1: base/large from config (no fixed sizes)
    local_test = config.get("local_test", False)
    base_size = int(config.get("local_phase1_base_size", 100))
    large_size = int(config.get("local_phase1_large_size", 500))
    phase1 = [
        (f"random_{base_size}", f"random_{base_size}_scores.json", base_size),
        (f"balanced_random_{base_size}", f"balanced_random_{base_size}_scores.json", base_size),
        (f"random_{large_size}", f"random_{large_size}_scores.json", large_size),
        (f"guided_{base_size}", f"guided_{base_size}_scores.json", base_size),
    ]
    # Phase 2 stacking (incl. RAG on/off ablation)
    stages = [
        ("stage0_base", "stage0_base_scores.json", base_size),
        ("stage1_cot", "stage1_cot_scores.json", base_size),
        ("stage2_probe_guided", "stage2_probe_guided_scores.json", base_size),
        ("stage3_moe", "stage3_moe_scores.json", base_size),
        ("stage4_pruned", "stage4_pruned_scores.json", base_size),
        ("stage5_qat", "stage5_qat_scores.json", base_size),
        ("stage6_no_rag", "stage6_no_rag_scores.json", base_size),
        ("stage6_rag", "stage6_rag_scores.json", base_size),
    ]

    rows = []
    for label, filename, examples in phase1:
        path = output_dir / filename
        data = load_scores(path)
        acc = data.get("overall_accuracy", 0) if data else None
        acc_str = f"{acc:.1f}%" if acc is not None else "N/A"
        rows.append(("phase1", label, examples, acc_str))

    for label, filename, examples in stages:
        path = output_dir / filename
        data = load_scores(path)
        acc = data.get("overall_accuracy", 0) if data else None
        acc_str = f"{acc:.1f}%" if acc is not None else "N/A"
        rows.append(("phase2", label, examples, acc_str))

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

    # Verdicts (dynamic base/large when local_test)
    guided_scores = output_dir / f"guided_{base_size}_scores.json"
    random_scores = output_dir / f"random_{base_size}_scores.json"
    random_large_scores = output_dir / f"random_{large_size}_scores.json"
    g = load_scores(guided_scores)
    r = load_scores(random_scores)
    r_large = load_scores(random_large_scores)
    final_data = load_scores(output_dir / "stage5_qat_scores.json")
    if final_data is None:
        final_data = load_scores(output_dir / "stage6_rag_scores.json")

    if g and r:
        print("")
        print("Phase 1 VERDICT: Targeting helps." if g.get("overall_accuracy", 0) > r.get("overall_accuracy", 0) else f"Phase 1: guided_{base_size} did not beat random_{base_size}.")
    if g and r_large:
        print(f"Hypothesis (guided_{base_size} >= random_{large_size}): " + ("PROVEN" if g.get("overall_accuracy", 0) >= r_large.get("overall_accuracy", 0) else "Not proven"))
    if final_data and g:
        print("Phase 2 VERDICT: Stacking beats guided baseline." if final_data.get("overall_accuracy", 0) > g.get("overall_accuracy", 0) else "Phase 2: stacked final vs guided baseline - check scores above.")

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
