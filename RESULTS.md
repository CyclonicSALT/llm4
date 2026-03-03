# Results summary

Short summary of runs (no raw JSON). Full outputs live in `output/` (gitignored).

---

## Local small test trial (100 base, 500 brute-force, CPU)

Single full run of `run_local.py` with `local_test: true` to verify the pipeline.

**Phase 1 (from scratch, no pretraining):**

| Run                 | Train n | Test accuracy | Bootstrap 95% CI   | Perplexity (answer tokens) |
|---------------------|--------|---------------|--------------------|----------------------------|
| random_100          | 100    | 0.0%          | [0.0%, 0.0%]       | 135,485                    |
| balanced_random_100 | 100    | 0.0%          | [0.0%, 0.0%]       | 105,170                    |
| random_500          | 500    | 0.0%          | [0.0%, 0.0%]       | 32,030                     |
| guided_100          | 100    | 0.0%          | [0.0%, 0.0%]       | 129,510                    |

**Stage 0 (base):** 0.0% (100 examples), perplexity 76,877.

**Takeaway:** End-to-end pipeline ran correctly. 0% everywhere is expected at 100/500 examples from random init on CPU; the earlier Kaggle run showed learning only at 50k scale. Perplexity ordering (random_500 best, then balanced_random_100, then others) is consistent with “more data helps a bit” even when accuracy is still zero.

---

## Prior Kaggle run (5k / 50k)

From an unfinished Kaggle run (no pretraining):

| Run        | Train n | Test accuracy | Note                |
|------------|--------|---------------|---------------------|
| random_5k  | 5,000  | 3.0% (6/200)  | Effectively no signal |
| guided_5k  | 5,000  | 2.0% (4/200)  | Same                |
| random_50k | 50,000 | 62.0% (124/200) | Clear learning    |

**Takeaway:** 5k was too small to be informative; 50k is the scale where the model starts to learn. That motivated Phase 1 config: **50k** for base (guided vs random vs balanced random) and **100k** for the brute-force ceiling when running the full pipeline (`local_test: false`).
