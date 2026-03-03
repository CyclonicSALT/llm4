# LLM4 – Data efficiency from scratch (no pretraining)

Test whether **probe-guided examples** can match **brute-force random** when training a model with **no pretraining** (random weights). Same 0.5B architecture as Qwen2.5-0.5B-Instruct, zero pretrained knowledge. Supports **local smoke test** (10/20 examples, full pipeline), **local dev** (100/500), and **full run** (50k/100k on Kaggle).

**Pull from GitHub** to run on Kaggle or locally; all paths are relative. Use `local_test: false` in `config.yaml` for the full 50k/100k pipeline.

---

## Difference from LLM3

- **Random init**: `AutoModelForCausalLM.from_config(config)` — no pretrained weights.
- **Stacking**: Each stage loads the previous stage’s weights (no retraining from scratch).
- **CoT evaluation**: Answer extraction uses last number or patterns like “answer is X”, “= X”, “result is X”.
- **QAT**: Capped at 3 epochs to avoid overfitting.
- **RAG**: `rag/build_index.py` + `rag/query_rag.py`; top 3 rules injected into the prompt.

---

## Quick run

### Smoke test (10/20, full pipeline, CPU)

To check that **the entire pipeline** (Phase 1 + Stage 0 through Stage 6: CoT, probe_guided, MoE, prune, QAT, RAG) runs without bugs:

Set in `config.yaml`: `local_test: true`, `local_phase1_base_size: 10`, `local_phase1_large_size: 20`. Then:

```bash
python run_smoke.py
```

Runs all stages with minimal data. When it completes, switch to 100/500 or 50k/100k for real runs.

### Local test (100/500, Phase 1 + Stage 0 only)

Set `local_test: true` and `local_phase1_base_size: 100`, `local_phase1_large_size: 500`:

```bash
python run_local.py
```

Runs: data generation → Phase 1 (random, balanced random, guided, large random) → Stage 0 (base) → comparison report.

### Full pipeline (e.g. Kaggle GPU)

Set `local_test: false` in `config.yaml` (and use 50k/100k scale if configured). Then:

```bash
python run_all.py
```

- **Phase 1**: random_50k, balanced_random_50k, guided_50k, random_100k (or single-seed 100/1000 when `local_test` was true).
- **Phase 2**: Stage 0 → CoT → probe_guided → MoE → prune → QAT → RAG → comparison report.

### Shell (e.g. Kaggle)

```bash
bash run_pipeline.sh
```

All paths are relative. Outputs go to `output/` as JSON.

---

## Config

- **`local_test`**: `true` = CPU, small data (smoke 10/20 or dev 100/500); `false` = full 50k/100k.
- **`local_phase1_base_size`**: Base size for Phase 1 when `local_test` (e.g. 10 for smoke, 100 for dev).
- **`local_phase1_large_size`**: Large (brute-force) size when `local_test` (e.g. 20 for smoke, 500 for dev).
- **Phase 1 (full run)**: `phase1_base_size` / `phase1_large_size` (e.g. 50k / 100k) when `local_test: false`.

---

## Phase 1: Clean baseline

1. Generate datasets: `train_100.jsonl`, `balanced_train_100.jsonl`, `train_1000.jsonl`, `test_200.jsonl`, `arithmetic_facts.jsonl` (and larger sets when needed).
2. Train **random** (base_size) from scratch → evaluate.
3. Use failures to build **probe-guided** dataset (targeted + pool to base_size).
4. Train **balanced random** (stratified) from scratch → evaluate.
5. If `local_test` and large_size > base_size: train **random_large** (large_size) from scratch → evaluate.
6. Train **guided** from scratch → evaluate.
7. Report: `phase1_report.json` and verdict (targeting vs random; hypothesis guided ≥ random_large).

---

## Phase 2: Stacking

Each stage **loads the previous stage**; no full retrain from scratch.

| Stage | Input | Action | Output |
|-------|--------|--------|--------|
| 0 | — | Train from scratch (base_size or 5k) | `models/base` |
| 1 | base | CoT training | `models/cot` |
| 2 | cot | Probe-guided targeted training | `models/probe_guided` |
| 3 | probe_guided | 4 LoRA experts | `models/moe/` |
| 4 | probe_guided | Prune 20% FFN, recovery finetune | `models/pruned` |
| 5 | pruned | QAT (max 3 epochs) | `models/final` |
| 6 | final + RAG | Eval with top-3 RAG rules | `output/stage6_rag_scores.json` |

RAG index: run `python rag/build_index.py` (uses `data/rag_documents/arithmetic_facts.jsonl`).

---

## Success criteria

- **Phase 1**: `guided` accuracy > `random` (base) → targeting helps.
- **Phase 2**: Final stacked model accuracy > `guided` baseline.
- **Main result**: If `guided` is close to `random_large` on a truly untrained model, the data-efficiency hypothesis is supported.

---

## Layout

- `config.yaml` – Paths and hyperparameters; `local_test`, `local_phase1_base_size`, `local_phase1_large_size` for smoke/local/full.
- `run_smoke.py` – Full pipeline with smoke sizes (10/20); requires `local_test: true`.
- `run_local.py` – Phase 1 + Stage 0 only (local_test sizes).
- `run_all.py` – Full pipeline (Phase 1 + all stages).
- `data/` – `generate_arithmetic.py`, `train_*.jsonl`, `balanced_train_100.jsonl`, `test_200.jsonl`, `rag_documents/arithmetic_facts.jsonl`.
- `scripts/` – `train_model.py` (`--from-scratch`, `--base`), `evaluate_model_hf.py`, `phase1_baseline.py`, `stage0_train_base.py` … `stage6_rag_integrate.py`, `compare_stages.py`.
- `rag/` – `build_index.py`, `query_rag.py`.
- `output/` – All scores, `phase1_report.json`, `stacking_report.json`.

---

## Requirements

See `requirements.txt`. GPU training targets Kaggle P100. Code is CPU-compatible; use `local_test: true` and `run_smoke.py` or `run_local.py` for local CPU runs.
