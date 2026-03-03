# LLM4 – Data efficiency from scratch (no pretraining)

Test whether **probe-guided examples** can match **10× random data** when training a model with **no pretraining** (random weights). Same 0.5B architecture as Qwen2.5-0.5B-Instruct, zero pretrained knowledge. Default scale: **5k** base / **50k** large (config: `phase1_base_size`, `phase1_large_size`).

**Pull from GitHub to run on Kaggle:** clone this repo in a Kaggle notebook or script; all paths are relative. Use `local_test: false` in `config.yaml` for the full pipeline.

---

## Difference from LLM3

- **Random init**: `AutoModelForCausalLM.from_config(config)` — no pretrained weights.
- **Stacking**: Each stage loads the previous stage’s weights (no retraining from scratch).
- **CoT evaluation**: Answer extraction uses last number or patterns like “answer is X”, “= X”, “result is X”.
- **QAT**: Capped at 3 epochs to avoid overfitting.
- **RAG**: `rag/build_index.py` + `rag/query_rag.py`; top 3 rules injected into the prompt.

---

## Quick run

### Local test first (100 samples, CPU)

Set `local_test: true` in `config.yaml` (default for local dev), then:

```bash
python run_local.py
```

Runs: data generation → Phase 1 (random_100, balanced_random_100, guided_100) → Stage 0 (base on 100) → comparison report. All evals include **bootstrap confidence intervals** and **perplexity** on answer tokens.

### Full pipeline (e.g. Kaggle GPU)

Set `local_test: false` in `config.yaml`, then:

```bash
python run_all.py
```

- **Phase 1** runs with **seeds 42, 43, 44** (required for the main claim); results go to `output/seed_42/`, `output/seed_43/`, `output/seed_44/`, then **aggregated** to `output/phase1_multi_seed_report.json` (mean ± std).
- **Phase 2** runs once (stacking). Final comparison: `python scripts/compare_stages.py` (or run_all runs it at the end).

### Shell (e.g. Kaggle)

```bash
bash run_pipeline.sh
```

All paths are relative. Outputs go to `output/` as JSON.

---

## Phase 1: Clean baseline (5k / 50k or 100 in local test)

1. Generate datasets: `train_5000.jsonl`, `train_50000.jsonl`, `test_200.jsonl`, `balanced_train_5k.jsonl` (stratified), `arithmetic_facts.jsonl` (and `train_100` / `balanced_train_100` when `local_test: true`).
2. Train **random** from scratch (5k or 100 examples) → evaluate.
3. Use failures to build **guided** set (60% targeted) → train **guided** from scratch → evaluate.
4. Train **balanced random** from scratch (stratified by problem type) → evaluate.
5. Train **random_50k** from scratch (full run only) → evaluate.
6. **Multi-seed (full run):** Phase 1 is run with seeds 42, 43, 44; `scripts/aggregate_phase1_seeds.py` reports mean ± std. Missing or failed seeds (e.g. OOM) are warned; aggregation uses only completed runs.

**Verdicts:** `guided > random` → targeting helps; `guided > balanced random` → stronger claim; `guided >= random_50k` → data-efficiency hypothesis supported.

---

## Phase 2: Stacking (~45 min on P100)

Each stage **loads the previous stage**; no full retrain from scratch.

| Stage | Input | Action | Output |
|-------|--------|--------|--------|
| 0 | — | Train from scratch (5k or 100 examples) | `models/base` |
| 1 | base | CoT training (500 samples) | `models/cot` |
| 2 | cot | Probe-guided targeted (500) | `models/probe_guided` |
| 3 | probe_guided | 4 LoRA experts (500/expert) | `models/moe/` |
| 4 | probe_guided | Prune 20% FFN, recovery (500) | `models/pruned` |
| 5 | pruned | QAT (max 3 epochs, 500) | `models/final` |
| 6 | final + RAG | Eval with top-3 RAG rules | `output/stage6_rag_scores.json` |

RAG index: run `python rag/build_index.py` (uses `data/rag_documents/arithmetic_facts.jsonl`).

---

## Training & evaluation details

- **Validation:** Train/val split (default 90/10), validation loss every epoch, optional best-model loading (`config.yaml`: `val_split_ratio`, `load_best_model_at_end`).
- **From-scratch:** Cosine LR schedule, warmup, higher LR (`learning_rate_from_scratch`).
- **Evaluation:** Accuracy, **bootstrap confidence intervals** (default 500 iterations), **perplexity** on answer tokens; `--no-bootstrap` / `--no-perplexity` to disable.

---

## Success criteria

- **Phase 1**: `guided` accuracy > `random` → targeting helps; with multi-seed, report mean ± std.
- **Phase 2**: Final stacked model accuracy > `guided` baseline.
- **Main result**: If `guided` is close to `random_50k` on an untrained model, the data-efficiency hypothesis is supported.

---

## Layout

- `config.yaml` – Paths and hyperparameters; `local_test`, `val_split_ratio`, `seed`, `bootstrap_iterations`, etc.
- `run_local.py` – Local test entrypoint (100 samples, CPU).
- `run_all.py` – Full pipeline; multi-seed Phase 1 when `local_test: false`.
- `data/` – `generate_arithmetic.py` (and `generate_balanced_problems` for stratified data); generated `*.jsonl` created on first run.
- `scripts/` – `train_model.py`, `evaluate_model_hf.py`, `phase1_baseline.py` (supports `--seed`), `aggregate_phase1_seeds.py`, `compare_stages.py`, `stage0_train_base.py` … `stage6_rag_integrate.py`, `kaggle_export_bundle.py`.
- `rag/` – `build_index.py`, `query_rag.py`.
- `output/` – Scores (per-stage and, when multi-seed, `output/seed_*/`, `phase1_multi_seed_report.json`), `phase1_report.json`, `stacking_report.json`.

---

## Requirements

See `requirements.txt`. GPU training targets Kaggle P100; code is CPU-compatible (e.g. `FORCE_CPU=1` or run off-Kaggle with `local_test: true`).
