"""
LLM4 Stage 6: RAG. Evaluate final model with RAG-augmented prompts (top 3 rules).
"""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_model_hf import extract_answer_cot_safe


def load_config():
    import yaml
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    final_path = PROJECT_ROOT / config["final_output"].replace("./", "")
    test_path = PROJECT_ROOT / config["test_200"].replace("./", "")
    output_dir = PROJECT_ROOT / config["output_dir"].replace("./", "")
    scores_path = output_dir / "stage6_rag_scores.json"

    print("Stage 6: RAG-augmented evaluation (top 3 rules per question)")
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
        torch_dtype="float32",
        device_map=get_device_map(),
        low_cpu_mem_usage=True,
    )
    model.eval()

    results = []
    for i, prob in enumerate(problems):
        instruction = prob["instruction"]
        correct_answer = prob["correct_answer"]
        ptype = prob["type"]
        full_prompt = build_augmented_prompt(instruction, config)
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        with __import__("torch").no_grad():
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
            print(f"  Evaluated {i+1}/{n_probs}", flush=True)

    type_total = {}
    type_correct = {}
    for r in results:
        t = r["type"]
        type_total[t] = type_total.get(t, 0) + 1
        if r["correct"]:
            type_correct[t] = type_correct.get(t, 0) + 1
    overall_c = sum(r["correct"] for r in results)
    overall_t = len(results)
    overall_pct = 100.0 * overall_c / overall_t

    print("")
    print("Stage 6: RAG evaluation")
    print(f"OVERALL: {overall_c}/{overall_t} = {overall_pct:.1f}%")

    out_data = {
        "stage": "stage6_rag",
        "overall_accuracy": overall_pct,
        "overall_correct": overall_c,
        "overall_total": overall_t,
        "per_type": {t: {"correct": type_correct.get(t, 0), "total": type_total.get(t, 0)} for t in type_total},
        "results": results,
    }
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {scores_path}")
    print("Stage 6 complete.")


if __name__ == "__main__":
    main()
