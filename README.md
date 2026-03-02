# LLM4 – Data efficiency from scratch (no pretraining)

Test whether **probe-guided examples** can match **10× random data** when training a model with **no pretraining** (random weights). Same 0.5B architecture as Qwen2.5-0.5B-Instruct, zero pretrained knowledge. Default scale: **5k** base / **50k** large (config: `phase1_base_size`, `phase1_large_size`).

## Difference from LLM3

- **Random init**: `AutoModelForCausalLM.from_config(config)` — no pretrained weights.
- **Stacking**: Each stage loads the previous stage’s weights (no retraining from scratch).
- **CoT evaluation**: Answer extraction uses last number or patterns like “answer is X”, “= X”, “result is X”.
- **QAT**: Capped at 3 epochs to avoid overfitting.
- **RAG**: `rag/build_index.py` + `rag/query_rag.py`; top 3 rules injected into the prompt.

## Quick run

**Single script (Phase 1 + Phase 2):**

```bash
python run_all.py
```

**Shell (e.g. Kaggle):**

```bash
bash run_pipeline.sh
```

All paths are relative; no Windows-specific paths. Outputs go to `output/` as JSON.

## Phase 1: Clean baseline (scaled: 5k / 50k)

1. Generate datasets: `train_5000.jsonl`, `train_50000.jsonl`, `test_200.jsonl`, `arithmetic_facts.jsonl` (and legacy train_100/train_1000).
2. Train **random_5k** from scratch on 5000 random examples → evaluate on 200 test.
3. Use random_5k failures to build **guided_5k** (60% targeted, 5000 total).
4. Train **random_50k** from scratch on 50000 random examples → evaluate.
5. Train **guided_5k** from scratch on 5000 probe-guided examples → evaluate.
6. Report: `random_5k`, `random_50k`, `guided_5k`.  
   - **Verdict**: If `guided_5k > random_5k`, targeting helps.  
   - **Hypothesis**: If `guided_5k >= random_50k`, data-efficiency hypothesis is supported.

## Phase 2: Stacking (~45 min on P100)

Each stage **loads the previous stage**; no full retrain from scratch.

| Stage | Input | Action | Output |
|-------|--------|--------|--------|
| 0 | — | Train from scratch on 5000 examples | `models/base` |
| 1 | base | CoT training (500 samples) | `models/cot` |
| 2 | cot | Probe-guided targeted (500) | `models/probe_guided` |
| 3 | probe_guided | 4 LoRA experts (500/expert) | `models/moe/` |
| 4 | probe_guided | Prune 20% FFN, recovery (500) | `models/pruned` |
| 5 | pruned | QAT (max 3 epochs, 500) | `models/final` |
| 6 | final + RAG | Eval with top-3 RAG rules | `output/stage6_rag_scores.json` |

RAG index: run `python rag/build_index.py` (uses `data/rag_documents/arithmetic_facts.jsonl`).

## Success criteria

- **Phase 1**: `guided_5k` accuracy > `random_5k` → targeting helps.  
- **Phase 2**: Final stacked model accuracy > `guided_5k` baseline.  
- **Main result**: If `guided_5k` is close to `random_50k` on a truly untrained model, the data-efficiency hypothesis is supported.

## Layout

- `config.yaml` – Paths and hyperparameters (relative only).
- `data/` – `generate_arithmetic.py`, `train_5000.jsonl`, `train_50000.jsonl`, `test_200.jsonl`, `guided_5k.jsonl` (built in Phase 1), `rag_documents/arithmetic_facts.jsonl`.
- `scripts/` – `train_model.py` (supports `--from-scratch` and `--base`), `evaluate_model_hf.py` (CoT-safe extraction), `phase1_baseline.py`, `stage0_train_base.py` … `stage6_rag_integrate.py`, `compare_stages.py`.
- `rag/` – `build_index.py`, `query_rag.py`.
- `output/` – All scores and `phase1_report.json`, `stacking_report.json`.

## Requirements

See `requirements.txt`. GPU training targets Kaggle P100; code is CPU-compatible with `FORCE_CPU=1` if needed.
