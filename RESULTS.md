# Results summary

This file summarizes **runs** by category. Only **fully completed** runs are listed under **Results (comparable)** so we can track improvement across versions. Raw outputs live in `output/` (gitignored).

**Inclusion rule for Results:** Phase 1 must have finished (random, balanced random, guided, and large random when applicable) and written all score files; optional: full pipeline through Stage 6. Partial or aborted runs are not listed there.

---

## 1. Results (fully completed runs)

Use these runs to compare versions and see improvement. Only runs that fully completed are listed.

### Run: Local 100/500 (Phase 1 + Stage 0)

| Field | Value |
|-------|--------|
| **Entrypoint** | `run_local.py` |
| **Config** | `local_test: true`, base=100, large=500, CPU |
| **Completed** | Phase 1 (random_100, balanced_random_100, random_500, guided_100) + Stage 0 (base on 100) |
| **Stages 1–6** | No |

**Phase 1 (from scratch, no pretraining):**

| Run                 | Train n | Test accuracy | Bootstrap 95% CI   | Perplexity (answer tokens) |
|---------------------|--------|---------------|--------------------|----------------------------|
| random_100          | 100    | 0.0%          | [0.0%, 0.0%]       | 135,485                    |
| balanced_random_100 | 100    | 0.0%          | [0.0%, 0.0%]       | 105,170                    |
| random_500          | 500    | 0.0%          | [0.0%, 0.0%]       | 32,030                     |
| guided_100          | 100    | 0.0%          | [0.0%, 0.0%]       | 129,510                    |

**Stage 0 (base):** 0.0% (100 examples), perplexity 76,877.

**Takeaway:** Pipeline ran correctly. 0% is expected at 100/500 from random init on CPU. Perplexity ordering (random_500 best, then balanced_random_100, then others) is consistent with “more data helps” even when accuracy is still zero.

---

*Add new rows above when you have another **fully completed** run (e.g. full 50k/100k Phase 1, or Phase 1 + full stacking).*

---

## 2. Pipeline sanity (not for comparison)

Runs that only verify the pipeline; accuracy is not meaningful for the main hypothesis.

- **Smoke (10/20):** `run_smoke.py` with base=10, large=20. Confirms full pipeline (Phase 1 + Stage 0–6) runs without bugs. Not used for results.
- **Local 100/500:** Same config as the run in §1; when used only to “check it runs,” it’s sanity; when fully completed and reported above, it’s in §1.

---

## 3. Reference only (incomplete runs)

Not included in Results; kept for context.

### Prior Kaggle run (5k / 50k) — incomplete

Run did **not** complete (e.g. later stages missing or aborted). Used only to justify scale.

| Run        | Train n | Test accuracy | Note                |
|------------|--------|---------------|---------------------|
| random_5k  | 5,000  | 3.0% (6/200)  | Effectively no signal |
| guided_5k  | 5,000  | 2.0% (4/200)  | Same                |
| random_50k | 50,000 | 62.0% (124/200) | Clear learning    |

**Takeaway:** 5k was too small; 50k showed learning. That motivated Phase 1 config: **50k** base and **100k** large for the full pipeline (`local_test: false`).
