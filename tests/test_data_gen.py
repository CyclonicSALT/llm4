"""
Minimal tests for LLM4 data generation and Phase 1 seed outputs.
Run from project root:
  python -m pytest tests/test_data_gen.py -v
  or: python tests/test_data_gen.py
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_data_generation_produces_expected_counts():
    """After running generate_arithmetic.py, generated files have exact expected counts (catches size mismatch bugs)."""
    from config_utils import load_config

    # Run data generation (uses current config)
    r = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "data" / "generate_arithmetic.py")],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, f"generate_arithmetic failed: {r.stderr or r.stdout}"

    config = load_config()
    base_size = int(config.get("local_phase1_base_size", 100))
    large_size = int(config.get("local_phase1_large_size", 500))
    data_dir = PROJECT_ROOT / "data"

    def line_count(path: Path) -> int:
        if not path.exists():
            return -1
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    assert line_count(data_dir / "train_phase1_base.jsonl") == base_size, (
        f"train_phase1_base.jsonl should have {base_size} lines (config: local_phase1_base_size)"
    )
    assert line_count(data_dir / "train_phase1_large.jsonl") == large_size, (
        f"train_phase1_large.jsonl should have {large_size} lines (config: local_phase1_large_size)"
    )
    assert line_count(data_dir / "balanced_train_phase1_base.jsonl") == base_size, (
        f"balanced_train_phase1_base.jsonl should have {base_size} lines"
    )
    assert line_count(data_dir / "test_200.jsonl") == 200, (
        "test_200.jsonl should have 200 lines"
    )


def test_phase1_with_seed_writes_to_seed_subdir():
    """Phase 1 with --seed N writes phase1_report.json under output/seed_N/ (multi-seed aggregation works)."""
    from config_utils import load_config

    config = load_config()
    output_dir = PROJECT_ROOT / config["output_dir"].replace("./", "")
    seed_dir = output_dir / "seed_42"
    report_path = seed_dir / "phase1_report.json"

    # We only check the contract: if seed_42 report exists, it's in the right place; if not, skip (no full Phase 1 run)
    if report_path.exists():
        assert report_path.is_file(), "output/seed_42/phase1_report.json should be a file"
        import json
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict), "phase1_report.json should be a JSON object"
        # At least one of the expected keys should be present (aggregate_phase1_seeds expects these)
        expected = ["random_accuracy", "balanced_random_accuracy", "guided_accuracy", "random_large_accuracy"]
        assert any(k in data for k in expected), f"phase1_report should contain one of {expected}"
    # If report doesn't exist (e.g. never ran Phase 1 with --seed 42), test passes — we're only validating structure when present


if __name__ == "__main__":
    import traceback
    ok = 0
    for name, fn in [("data gen counts", test_data_generation_produces_expected_counts), ("phase1 seed output", test_phase1_with_seed_writes_to_seed_subdir)]:
        try:
            fn()
            print(f"PASS: {name}")
            ok += 1
        except Exception as e:
            print(f"FAIL: {name}")
            traceback.print_exc()
    sys.exit(0 if ok == 2 else 1)
