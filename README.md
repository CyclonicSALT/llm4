# LLM4 – Data efficiency from scratch (no pretraining)

Test whether **probe-guided examples** can match **brute-force random** when training a model with **no pretraining** (random weights). Same 0.5B architecture as Qwen2.5-0.5B-Instruct, zero pretrained knowledge. **No fixed training sizes** — choose any base and large size in config or via the UI; data is generated to match.

**Pull from GitHub** to run on Kaggle or locally; all paths are relative. Set `local_test: true` for CPU/small runs, `false` for GPU; base/large sizes are always from config (e.g. 10/20 smoke, 100/500, 5000/20000).

---

## Difference from LLM3

- **Random init**: `AutoModelForCausalLM.from_config(config)` — no pretrained weights.
- **Stacking**: Each stage loads the previous stage’s weights (no retraining from scratch).
- **CoT evaluation**: Answer extraction uses last number or patterns like “answer is X”, “= X”, “result is X”.
- **QAT**: Capped at 3 epochs to avoid overfitting.
- **RAG**: `rag/build_index.py` + `rag/query_rag.py`; top 3 rules injected into the prompt.

---

## Quick run

### Smoke test (full pipeline, CPU)

To check that **the entire pipeline** (Phase 1 + Stage 0 through Stage 6) runs without bugs, set e.g. `local_phase1_base_size: 10`, `local_phase1_large_size: 20` in `config.yaml`, then:

```bash
python run_smoke.py
```

Data is generated for your chosen sizes. Use any base/large for real runs (e.g. 100/500, 5000/20000).

### Local (Phase 1 + Stage 0 only)

Set base/large in config (or use the UI). Then:

```bash
python run_local.py
```

Runs: data generation → Phase 1 (random, balanced random, guided, large random) → Stage 0 (base) → comparison report.

### Full pipeline (e.g. Kaggle GPU)

Set `local_test: false` in `config.yaml`, choose base/large sizes, then:

```bash
python run_all.py
```

- **Phase 1**: random_base, balanced_random_base, guided_base, random_large (sizes from config).
- **Phase 2**: Stage 0 → CoT → probe_guided → MoE → prune → QAT → RAG → comparison report.

### Shell (e.g. Kaggle)

```bash
bash run_pipeline.sh
```

All paths are relative. Outputs go to `output/` as JSON.

---

## Config

- **`local_test`**: `true` = use CPU / small runs; `false` = GPU / longer runs.
- **`local_phase1_base_size`**: Phase 1 base training size (any positive integer). Data is generated to match.
- **`local_phase1_large_size`**: Phase 1 large (brute-force) size. No fixed tiers — choose any size (e.g. 10/20, 100/500, 5000/20000).
- **Phase 1 (full run)**: `phase1_base_size` / `phase1_large_size` (e.g. 50k / 100k) when `local_test: false`.

---

## Phase 1: Clean baseline

1. Generate datasets: `data/generate_arithmetic.py` reads config and creates `train_phase1_base.jsonl`, `train_phase1_large.jsonl`, `balanced_train_phase1_base.jsonl`, `test_200.jsonl`, `arithmetic_facts.jsonl` with counts matching your chosen base/large sizes.
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
- `data/` – `generate_arithmetic.py` (generates data to match config sizes), `train_phase1_base.jsonl`, `train_phase1_large.jsonl`, `balanced_train_phase1_base.jsonl`, `test_200.jsonl`, `rag_documents/arithmetic_facts.jsonl`.
- `scripts/` – `train_model.py` (`--from-scratch`, `--base`), `evaluate_model_hf.py`, `phase1_baseline.py`, `stage0_train_base.py` … `stage6_rag_integrate.py`, `compare_stages.py`.
- `rag/` – `build_index.py`, `query_rag.py`.
- `output/` – All scores, `phase1_report.json`, `stacking_report.json`.

---

## Requirements

See `requirements.txt`. GPU training targets Kaggle P100. Code is CPU-compatible; use `local_test: true` and `run_smoke.py` or `run_local.py` for local CPU runs.
