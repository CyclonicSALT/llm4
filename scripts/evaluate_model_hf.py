"""
LLM4 - Evaluate HuggingFace model on test set.
CoT-safe answer extraction: last number, or "answer is X", "= X", "result is X".
"""

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config():
    import yaml
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_answer_cot_safe(text: str):
    """
    Extract the model's numeric answer from possibly verbose/CoT response.
    - Prefer explicit patterns: "answer is X", "result is X", "= X" at end.
    - Else use the LAST number in the response (CoT often ends with the answer).
    """
    if not text or not text.strip():
        return None
    text = text.strip()
    # Explicit patterns (case-insensitive)
    for pat in [
        r"(?:answer|result)\s+is\s+(-?\d+\.?\d*)",
        r"=\s*(-?\d+\.?\d*)\s*$",
        r"(?:therefore|so)\s+(-?\d+\.?\d*)",
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1)
    # All numbers in text; take the last one (often the final answer after reasoning)
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return numbers[-1] if numbers else None


def main():
    parser = argparse.ArgumentParser(description="Evaluate HF model on test set (CoT-safe extraction)")
    parser.add_argument("--model", required=True, help="Path to HuggingFace model dir")
    parser.add_argument("--output", required=True, help="Path to save results JSON")
    parser.add_argument("--stage", type=str, default="eval", help="Label for this run")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max new tokens (CoT can be verbose)")
    args = parser.parse_args()

    config = load_config()
    test_path = PROJECT_ROOT / config["test_200"].replace("./", "")
    if not test_path.exists():
        print(f"Test file not found: {test_path}")
        sys.exit(1)

    problems = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    print(f"Loaded {len(problems)} test problems")

    from device_utils import get_device_map, print_device_info
    print_device_info()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        model_path = PROJECT_ROOT / args.model
    if not model_path.exists():
        print(f"Model path not found: {args.model}")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    # Ensure config.json has model_type (required by AutoModel; sometimes missing after from-scratch save)
    config_path = model_path / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if cfg.get("model_type") is None:
            cfg["model_type"] = "qwen2"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    # ignore_mismatched_sizes=True allows loading when vocab/embedding was resized (e.g. from-scratch)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        torch_dtype="float32",
        device_map=get_device_map(),
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True,
    )
    model.eval()

    prompt_template = "### Instruction: {instruction}\n\n### Response:"
    type_order = [
        "single_digit_addition", "single_digit_subtraction", "single_digit_multiplication",
        "double_digit_addition", "double_digit_subtraction", "double_digit_multiplication",
        "addition_with_carrying", "three_number_addition", "simple_division", "mixed_operations",
    ]
    results = []
    n_probs = len(problems)
    print(f"Evaluating 0/{n_probs}...", flush=True)

    for i, prob in enumerate(problems):
        instruction = prob["instruction"]
        correct_answer = prob["correct_answer"]
        ptype = prob["type"]
        full_prompt = prompt_template.format(instruction=instruction)
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        with __import__("torch").no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
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

    type_correct = {}
    type_total = {}
    for r in results:
        t = r["type"]
        type_total[t] = type_total.get(t, 0) + 1
        if r["correct"]:
            type_correct[t] = type_correct.get(t, 0) + 1

    overall_c = sum(r["correct"] for r in results)
    overall_t = len(results)
    overall_pct = 100.0 * overall_c / overall_t if overall_t else 0
    gaps = [t for t in type_order if type_total.get(t, 0) and (100.0 * type_correct.get(t, 0) / type_total[t]) < 50]

    print("")
    print("=" * 60)
    print(f"EVALUATION: {args.stage}")
    print("=" * 60)
    for t in type_order:
        total = type_total.get(t, 0)
        correct = type_correct.get(t, 0)
        pct = (100.0 * correct / total) if total else 0
        short = t.replace("_", " ")[:28].ljust(28)
        print(f"  {short} {correct}/{total} = {pct:.0f}%")
    print("-" * 60)
    print(f"  OVERALL{' ':22} {overall_c}/{overall_t} = {overall_pct:.1f}%")
    print("=" * 60)

    out_data = {
        "stage": args.stage,
        "overall_accuracy": overall_pct,
        "overall_correct": overall_c,
        "overall_total": overall_t,
        "per_type": {t: {"correct": type_correct.get(t, 0), "total": type_total.get(t, 0)} for t in type_order},
        "gaps": gaps,
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
