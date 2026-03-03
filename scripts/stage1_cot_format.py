"""
LLM4 Stage 1: CoT. Load models/base, continue training on CoT-formatted data -> save models/cot.
"""

import json
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config():
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cot_response(example: dict) -> str:
    ptype = example["type"]
    operands = example.get("operands", [])
    answer = example["correct_answer"]

    if ptype == "single_digit_addition":
        a, b = operands[0], operands[1]
        return f"Let me calculate. {a} + {b} = {answer}"
    if ptype == "single_digit_subtraction":
        a, b = operands[0], operands[1]
        return f"Let me calculate. {a} - {b} = {answer}"
    if ptype == "single_digit_multiplication":
        a, b = operands[0], operands[1]
        return f"Let me calculate. {a} x {b} = {answer}"
    if ptype == "double_digit_addition":
        a, b = operands[0], operands[1]
        a_ones, a_tens, b_ones, b_tens = a % 10, a // 10, b % 10, b // 10
        ones_sum = a_ones + b_ones
        carry = 1 if ones_sum >= 10 else 0
        ones_digit = ones_sum % 10
        tens_sum = a_tens + b_tens + carry
        carry_text = ". Carry 1." if carry else "."
        plus_carry = " + 1 (carried)" if carry else ""
        return (
            f"Let me work column by column.\nOnes: {a_ones} + {b_ones} = {ones_sum}. Write {ones_digit}{carry_text}\n"
            f"Tens: {a_tens} + {b_tens}{plus_carry} = {tens_sum}.\nResult: {answer}"
        )
    if ptype == "double_digit_subtraction":
        a, b = operands[0], operands[1]
        return f"Let me work through this. {a} - {b} = {answer}"
    if ptype == "addition_with_carrying":
        a, b = operands[0], operands[1]
        a_ones, a_tens, b_ones, b_tens = a % 10, a // 10, b % 10, b // 10
        ones_sum = a_ones + b_ones
        ones_digit = ones_sum % 10
        tens_sum = a_tens + b_tens + 1
        return (
            f"Ones: {a_ones} + {b_ones} = {ones_sum}. Write {ones_digit}, carry 1.\n"
            f"Tens: {a_tens} + {b_tens} + 1 = {tens_sum}.\nResult: {answer}"
        )
    if ptype == "double_digit_multiplication":
        a, b = operands[0], operands[1]
        b_ones, b_tens = b % 10, b // 10
        partial1 = a * b_ones
        partial2 = a * b_tens * 10
        return f"Let me break this down.\n{a} x {b_ones} = {partial1}\n{a} x {b_tens}0 = {partial2}\n{partial1} + {partial2} = {answer}"
    if ptype == "three_number_addition":
        a, b, c = operands[0], operands[1], operands[2]
        ab = a + b
        return f"Adding left to right.\n{a} + {b} = {ab}\n{ab} + {c} = {answer}"
    if ptype == "simple_division":
        a, b = operands[0], operands[1]
        return f"Let me think. {a} / {b} = ?\n{b} x {answer} = {a}.\nSo {a} / {b} = {answer}"
    if ptype == "mixed_operations":
        a, b, c = operands[0], operands[1], operands[2]
        inner = a + b
        return f"Brackets first.\n({a} + {b}) = {inner}\n{inner} x {c} = {answer}"
    return str(answer)


def main():
    config = load_config()
    local_test = config.get("local_test", False)
    train_base_path = PROJECT_ROOT / (config["train_100"] if local_test else config.get("train_5000", config["train_100"])).replace("./", "")
    cot_train_path = PROJECT_ROOT / config["cot_train"].replace("./", "")
    base_output = PROJECT_ROOT / config["base_output"].replace("./", "")
    cot_output = PROJECT_ROOT / config["cot_output"].replace("./", "")
    output_dir = PROJECT_ROOT / config["output_dir"].replace("./", "")

    print("Stage 1: CoT - load base, train on CoT data -> models/cot")
    if not (base_output / "config.json").exists():
        print("models/base not found. Run Stage 0 first.")
        sys.exit(1)

    examples = []
    with open(train_base_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    cot_examples = []
    for ex in examples:
        cot_examples.append({
            "instruction": ex["instruction"],
            "response": cot_response(ex),
            "type": ex["type"],
            "operands": ex.get("operands", []),
            "correct_answer": ex["correct_answer"],
        })
    cot_train_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cot_train_path, "w", encoding="utf-8") as f:
        for ex in cot_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Saved {len(cot_examples)} CoT examples to {cot_train_path}")

    n_cot = config.get("stage1_cot_samples", 100)
    if local_test:
        n_cot = min(n_cot, config.get("local_phase1_base_size", 100))
    subprocess.run([
        sys.executable,
        str(SCRIPT_DIR / "train_model.py"),
        "--data", str(cot_train_path),
        "--output", str(cot_output),
        "--samples", str(n_cot),
        "--base", str(base_output),
    ], cwd=PROJECT_ROOT, check=True)

    print("Evaluating CoT model (CoT-safe extraction)...")
    subprocess.run([
        sys.executable,
        str(SCRIPT_DIR / "evaluate_model_hf.py"),
        "--model", str(cot_output),
        "--output", str(output_dir / "stage1_cot_scores.json"),
        "--stage", "stage1_cot",
        "--max-tokens", "128",
    ], cwd=PROJECT_ROOT, check=True)
    print("Stage 1 complete.")


if __name__ == "__main__":
    main()
