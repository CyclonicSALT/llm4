"""
LLM4 Stage 6: RAG. Evaluate final model with and without RAG (ablation).
Runs two evals: no-RAG (plain prompt) -> stage6_no_rag_scores.json, then RAG (top 3 rules) -> stage6_rag_scores.json.
"""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from config_utils import load_config
from evaluate_model_hf import extract_answer_cot_safe


def build_plain_prompt(instruction: str) -> str:
    """No RAG: standard instruction-only prompt."""
    return f"### Instruction: {instruction}\n\n### Response:"


def run_eval(problems, tokenizer, model, prompt_fn, stage_label, use_rag: bool):
    """Run evaluation with given prompt builder; return (results, type_total, type_correct)."""
    import torch
    results = []
    n_probs = len(problems)
    for i, prob in enumerate(problems):
        instruction = prob["instruction"]
        correct_answer = prob["correct_answer"]
        ptype = prob["type"]
        full_prompt = prompt_fn(instruction)
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        model_response = reply.split("###")[0].split("\n\n\n")[0].strip()
        extracted = extract_answer_cot_safe(model_response)
        try:
            ext_num = int(float(extracted)) if extracted else None
        except (ValueError, TypeError):
            ext_num = None
        correct = ext_num == correct_answer if ext_num is not None else False
        results.append({
            "instruction": instruction,
            "correct_answer": correct_answer,
            "model_response": model_response,
            "extracted_number": extracted,
            "correct": correct,
            "type": ptype,
        })
        if (i + 1) % 20 == 0 or (i + 1) == n_probs:
            print(f"  [{stage_label}] Evaluated {i+1}/{n_probs}", flush=True)
    type_total = {}
    type_correct = {}
    for r in results:
        t = r["type"]
        type_total[t] = type_total.get(t, 0) + 1
        if r["correct"]:
            type_correct[t] = type_correct.get(t, 0) + 1
    return results, type_total, type_correct


def main():
    config = load_config()
    final_path = PROJECT_ROOT / config["final_output"].replace("./", "")
    test_path = PROJECT_ROOT / config["test_200"].replace("./", "")
    output_dir = PROJECT_ROOT / config["output_dir"].replace("./", "")

    if not (final_path / "config.json").exists():
        print("models/final not found. Run Stage 5 first.")
        sys.exit(1)

    sys.path.insert(0, str(PROJECT_ROOT))
    from rag.query_rag import build_augmented_prompt

    with open(test_path, "r", encoding="utf-8") as f:
        problems = [json.loads(line) for line in f if line.strip()]
    n_probs = len(problems)

    from device_utils import get_device_map, print_device_info
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print_device_info()
    tokenizer = AutoTokenizer.from_pretrained(str(final_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(final_path),
        trust_remote_code=True,
        dtype="float32",
        device_map=get_device_map(),
        low_cpu_mem_usage=True,
    )
    model.eval()

    # 1. No-RAG (ablation baseline)
    print("Stage 6a: Evaluation without RAG (plain prompt)")
    results_no_rag, type_total, type_correct = run_eval(
        problems, tokenizer, model, build_plain_prompt, "no_rag", use_rag=False
    )
    overall_c = sum(r["correct"] for r in results_no_rag)
    overall_t = len(results_no_rag)
    overall_pct = 100.0 * overall_c / overall_t
    out_data = {
        "stage": "stage6_no_rag",
        "overall_accuracy": overall_pct,
        "overall_correct": overall_c,
        "overall_total": overall_t,
        "per_type": {t: {"correct": type_correct.get(t, 0), "total": type_total.get(t, 0)} for t in type_total},
        "results": results_no_rag,
    }
    no_rag_path = output_dir / "stage6_no_rag_scores.json"
    with open(no_rag_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"  No-RAG OVERALL: {overall_c}/{overall_t} = {overall_pct:.1f}%")
    print(f"  Saved {no_rag_path}")

    # 2. With RAG (top 3 rules per question)
    print("")
    print("Stage 6b: Evaluation with RAG (top 3 rules per question)")
    def rag_prompt(instruction):
        return build_augmented_prompt(instruction, config)
    results_rag, type_total, type_correct = run_eval(
        problems, tokenizer, model, rag_prompt, "rag", use_rag=True
    )
    overall_c = sum(r["correct"] for r in results_rag)
    overall_pct = 100.0 * overall_c / overall_t
    out_data = {
        "stage": "stage6_rag",
        "overall_accuracy": overall_pct,
        "overall_correct": overall_c,
        "overall_total": overall_t,
        "per_type": {t: {"correct": type_correct.get(t, 0), "total": type_total.get(t, 0)} for t in type_total},
        "results": results_rag,
    }
    rag_path = output_dir / "stage6_rag_scores.json"
    with open(rag_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"  RAG OVERALL: {overall_c}/{overall_t} = {overall_pct:.1f}%")
    print(f"  Saved {rag_path}")
    print("Stage 6 complete.")


if __name__ == "__main__":
    main()
