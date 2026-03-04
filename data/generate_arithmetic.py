"""
LLM4 - Generate arithmetic datasets from config sizes.
Reads phase1_base_size and phase1_large_size from config.yaml; generates
train_phase1_base.jsonl, train_phase1_large.jsonl, balanced_train_phase1_base.jsonl,
test_200.jsonl, and RAG facts. No fixed training sizes — data matches your chosen sizes.
"""

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PROBLEM_TYPES = [
    "single_digit_addition",
    "single_digit_subtraction",
    "single_digit_multiplication",
    "double_digit_addition",
    "double_digit_subtraction",
    "double_digit_multiplication",
    "addition_with_carrying",
    "three_number_addition",
    "simple_division",
    "mixed_operations",
]


def single_digit_addition(rng):
    a, b = rng.randint(1, 9), rng.randint(1, 9)
    return f"What is {a} + {b}?", str(a + b), [a, b], a + b


def single_digit_subtraction(rng):
    a, b = rng.randint(1, 9), rng.randint(1, 9)
    if a < b:
        a, b = b, a
    return f"What is {a} - {b}?", str(a - b), [a, b], a - b


def single_digit_multiplication(rng):
    a, b = rng.randint(1, 9), rng.randint(1, 9)
    return f"What is {a} x {b}?", str(a * b), [a, b], a * b


def double_digit_addition(rng):
    a, b = rng.randint(10, 99), rng.randint(10, 99)
    return f"What is {a} + {b}?", str(a + b), [a, b], a + b


def double_digit_subtraction(rng):
    a, b = rng.randint(10, 99), rng.randint(10, 99)
    if a < b:
        a, b = b, a
    return f"What is {a} - {b}?", str(a - b), [a, b], a - b


def double_digit_multiplication(rng):
    a, b = rng.randint(10, 30), rng.randint(10, 30)
    return f"What is {a} x {b}?", str(a * b), [a, b], a * b


def addition_with_carrying(rng):
    a_tens, a_ones = rng.randint(1, 9), rng.randint(5, 9)
    b_tens, b_ones = rng.randint(1, 9), rng.randint(10 - a_ones, 9)
    a = a_tens * 10 + a_ones
    b = b_tens * 10 + b_ones
    return f"What is {a} + {b}?", str(a + b), [a, b], a + b


def three_number_addition(rng):
    a, b, c = rng.randint(10, 50), rng.randint(10, 50), rng.randint(10, 50)
    return f"What is {a} + {b} + {c}?", str(a + b + c), [a, b, c], a + b + c


def simple_division(rng):
    answer = rng.randint(2, 12)
    b = rng.randint(2, 12)
    a = answer * b
    return f"What is {a} / {b}?", str(answer), [a, b], answer


def mixed_operations(rng):
    a, b = rng.randint(1, 9), rng.randint(1, 9)
    c = rng.randint(2, 9)
    inner = a + b
    result = inner * c
    return f"What is ({a} + {b}) x {c}?", str(result), [a, b, c], result


GENERATORS = {
    "single_digit_addition": single_digit_addition,
    "single_digit_subtraction": single_digit_subtraction,
    "single_digit_multiplication": single_digit_multiplication,
    "double_digit_addition": double_digit_addition,
    "double_digit_subtraction": double_digit_subtraction,
    "double_digit_multiplication": double_digit_multiplication,
    "addition_with_carrying": addition_with_carrying,
    "three_number_addition": three_number_addition,
    "simple_division": simple_division,
    "mixed_operations": mixed_operations,
}


def generate_problems(per_type: int, seed: int):
    rng = random.Random(seed)
    out = []
    for _ in range(per_type):
        for ptype in PROBLEM_TYPES:
            instr, resp, operands, correct = GENERATORS[ptype](rng)
            out.append({
                "instruction": instr,
                "response": resp,
                "type": ptype,
                "operands": operands,
                "correct_answer": correct,
            })
    rng.shuffle(out)
    return out


def generate_balanced_problems(total: int, seed: int):
    """Stratified sampling: equal (or nearly equal) count per problem type. Returns exactly `total` items."""
    rng = random.Random(seed)
    n_types = len(PROBLEM_TYPES)
    if total <= 0:
        return []
    if total < n_types:
        # Fewer items than types: 1 item each for the first `total` types only
        per_type = 0
        remainder = total
    else:
        per_type = total // n_types
        remainder = total - per_type * n_types
    out = []
    for i, ptype in enumerate(PROBLEM_TYPES):
        n = per_type + (1 if i < remainder else 0)
        for _ in range(n):
            instr, resp, operands, correct = GENERATORS[ptype](rng)
            out.append({
                "instruction": instr,
                "response": resp,
                "type": ptype,
                "operands": operands,
                "correct_answer": correct,
            })
    rng.shuffle(out)
    return out


def write_jsonl(path, items):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def generate_arithmetic_facts(seed: int, count: int = 500):
    rng = random.Random(seed)
    facts = []
    for _ in range(50):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        if (a % 10) + (b % 10) >= 10:
            ones = (a % 10) + (b % 10)
            result = a + b
            facts.append({
                "fact": f"When adding where ones sum to 10+: {a} + {b}, ones: {a%10}+{b%10}={ones}, result: {result}",
                "category": "carrying",
                "related_types": ["addition_with_carrying", "three_number_addition"],
            })
    for _ in range(40):
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        facts.append({"fact": f"{a} + {b} = {a+b}", "category": "single_digit", "related_types": ["single_digit_addition"]})
    for _ in range(40):
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        if a < b:
            a, b = b, a
        facts.append({"fact": f"{a} - {b} = {a-b}", "category": "single_digit", "related_types": ["single_digit_subtraction"]})
    for _ in range(50):
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        facts.append({"fact": f"{a} x {b} = {a*b}", "category": "multiplication", "related_types": ["single_digit_multiplication", "double_digit_multiplication"]})
    for _ in range(40):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        facts.append({"fact": f"{a} + {b} = {a+b}. Add ones first, then tens.", "category": "double_digit", "related_types": ["double_digit_addition"]})
    for _ in range(40):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        if a < b:
            a, b = b, a
        facts.append({"fact": f"{a} - {b} = {a-b}. Subtract ones first, then tens.", "category": "double_digit", "related_types": ["double_digit_subtraction"]})
    for _ in range(50):
        b = rng.randint(2, 12)
        answer = rng.randint(2, 12)
        a = b * answer
        facts.append({"fact": f"{a} / {b} = {answer} because {b} x {answer} = {a}", "category": "division", "related_types": ["simple_division"]})
    for _ in range(40):
        x, y, z = rng.randint(1, 9), rng.randint(1, 9), rng.randint(2, 9)
        inner, result = x + y, (x + y) * z
        facts.append({"fact": f"({x} + {y}) x {z} = {inner} x {z} = {result}", "category": "mixed", "related_types": ["mixed_operations"]})
    for _ in range(50):
        a, b = rng.randint(10, 25), rng.randint(10, 25)
        b_ones, b_tens = b % 10, b // 10
        p1, p2 = a * b_ones, a * b_tens * 10
        facts.append({"fact": f"{a} x {b}: {a}x{b_ones}={p1}, {a}x{b_tens}0={p2}, total {a*b}", "category": "double_digit_multiplication", "related_types": ["double_digit_multiplication"]})
    for _ in range(50):
        a, b, c = rng.randint(10, 40), rng.randint(10, 40), rng.randint(10, 40)
        facts.append({"fact": f"{a}+{b}+{c}: {a}+{b}={a+b}, then +{c}={a+b+c}", "category": "three_number", "related_types": ["three_number_addition"]})
    rng.shuffle(facts)
    return facts[:count]


def main():
    from config_utils import load_config
    config = load_config()
    base_size = int(config.get("local_phase1_base_size", 100))
    large_size = int(config.get("local_phase1_large_size", 500))
    data_dir = Path(__file__).parent

    n_types = len(PROBLEM_TYPES)
    per_type_base = max(1, (base_size + n_types - 1) // n_types)
    per_type_large = max(1, (large_size + n_types - 1) // n_types)

    train_base = generate_problems(per_type_base, 42)[:base_size]
    assert len(train_base) == base_size, f"train_phase1_base: expected {base_size} examples, got {len(train_base)}"
    write_jsonl(data_dir / "train_phase1_base.jsonl", train_base)
    print(f"Generated train_phase1_base: {len(train_base)} problems")

    train_large = generate_problems(per_type_large, 42)[:large_size]
    assert len(train_large) == large_size, f"train_phase1_large: expected {large_size} examples, got {len(train_large)}"
    write_jsonl(data_dir / "train_phase1_large.jsonl", train_large)
    print(f"Generated train_phase1_large: {len(train_large)} problems")

    balanced_base = generate_balanced_problems(base_size, 43)
    assert len(balanced_base) == base_size, f"balanced_train_phase1_base: expected {base_size}, got {len(balanced_base)}"
    write_jsonl(data_dir / "balanced_train_phase1_base.jsonl", balanced_base)
    print(f"Generated balanced_train_phase1_base: {len(balanced_base)} problems (stratified)")

    # Stratified test set: 20 per problem type = 200 total (for per-type accuracy reporting)
    test_200 = generate_problems(20, 99)
    assert len(test_200) == 200, f"test_200: expected 200 examples, got {len(test_200)}"
    write_jsonl(data_dir / "test_200.jsonl", test_200)
    print(f"Generated test_200: {len(test_200)} problems (stratified: 20 per type)")

    rag_dir = data_dir / "rag_documents"
    rag_dir.mkdir(parents=True, exist_ok=True)
    facts = generate_arithmetic_facts(42, 500)
    write_jsonl(rag_dir / "arithmetic_facts.jsonl", facts)
    print(f"Generated arithmetic_facts: {len(facts)} rules")


if __name__ == "__main__":
    main()
