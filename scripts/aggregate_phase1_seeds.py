"""
LLM4 - Aggregate Phase 1 results across multiple seeds (e.g. 42, 43, 44).
Reads output/seed_42/phase1_report.json, output/seed_43/..., output/seed_44/...
and prints mean ± std for random, balanced_random, guided, random_large.
Robust to missing or failed runs: warns clearly and still aggregates available seeds.
"""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXPECTED_KEYS = ["random_accuracy", "balanced_random_accuracy", "guided_accuracy", "random_large_accuracy"]

from config_utils import load_config


def main():
    config = load_config()
    output_dir = PROJECT_ROOT / config["output_dir"].replace("./", "")
    seeds = [42, 43, 44]

    missing_seeds = []
    reports = []

    for s in seeds:
        path = output_dir / f"seed_{s}" / "phase1_report.json"
        if not path.exists():
            missing_seeds.append(s)
            print("", flush=True)
            print("WARNING: Missing Phase 1 report for seed {}: {}".format(s, path), flush=True)
            print("  -> Run may have failed (OOM, timeout, or crash). Re-run Phase 1 with --seed {} to fix.".format(s), flush=True)
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            missing_seeds.append(s)
            print("", flush=True)
            print("WARNING: Could not read {}: {}".format(path, e), flush=True)
            print("  -> File may be corrupted or incomplete (e.g. run killed mid-write).", flush=True)
            continue
        if not isinstance(data, dict):
            missing_seeds.append(s)
            print("WARNING: Invalid report for seed {} (not a dict); skipping.".format(s), flush=True)
            continue
        has_any = any(data.get(k) is not None for k in EXPECTED_KEYS)
        if not has_any:
            missing_seeds.append(s)
            print("WARNING: Report for seed {} has no accuracy keys; run may be incomplete. Skipping.".format(s), flush=True)
            continue
        reports.append((s, data))

    if missing_seeds:
        print("", flush=True)
        print("=" * 60, flush=True)
        print("INCOMPLETE SEEDS: {} missing or invalid (expected all of {}).".format(missing_seeds, seeds), flush=True)
        print("Aggregating only completed seeds. Means may be biased if failure was non-random (e.g. OOM on large run).", flush=True)
        print("=" * 60, flush=True)

    # Need at least 2 for std; 1 seed is meaningless for variance.
    if len(reports) < 2:
        print("ERROR: Need at least 2 valid seed reports to aggregate. Valid seeds: {}.".format([r[0] for r in reports]))
        sys.exit(1)

    def mean_std(vals):
        n = len(vals)
        m = sum(vals) / n
        var = sum((x - m) ** 2 for x in vals) / n
        return m, (var ** 0.5) if n > 1 else 0.0

    keys = EXPECTED_KEYS
    print("")
    print("=" * 60)
    print("PHASE 1 MULTI-SEED AGGREGATE (mean ± std)")
    if missing_seeds:
        print("  (based on {}/{} seeds; missing: {})".format(len(reports), len(seeds), missing_seeds))
    print("=" * 60)
    for key in keys:
        vals = [r[1].get(key) for _, r in reports if r[1].get(key) is not None]
        if not vals:
            continue
        m, s = mean_std(vals)
        label = key.replace("_accuracy", "").replace("_", " ").strip()
        print(f"  {label:<22} {m:.1f}% ± {s:.1f}%  (n={len(vals)} seeds)")
    print("=" * 60)

    out = {
        "n_seeds": len(reports),
        "expected_seeds": len(seeds),
        "missing_seeds": missing_seeds,
        "complete": len(missing_seeds) == 0,
    }
    for key in keys:
        vals = [r[1].get(key) for _, r in reports if r[1].get(key) is not None]
        if vals:
            m, s = mean_std(vals)
            out[key] = {"mean": m, "std": s, "n_seeds": len(vals)}
    out_path = output_dir / "phase1_multi_seed_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    if missing_seeds:
        print("Saved {} (INCOMPLETE: {} seed(s) missing or invalid).".format(out_path, len(missing_seeds)))
    else:
        print("Saved {}".format(out_path))


if __name__ == "__main__":
    main()
