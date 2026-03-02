# LLM4 – Data efficiency from scratch (no pretraining)

Test whether **100 probe-guided examples** can match **1000 random examples** when training a model with **no pretraining** (random weights). Same 0.5B architecture as Qwen2.5-0.5B-Instruct, zero pretrained knowledge.

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

## Phase 1: Clean baseline (~20 min on P100)

1. Generate datasets: `train_100.jsonl`, `train_1000.jsonl`, `test_200.jsonl`, `arithmetic_facts.jsonl`.
2. Train **random_100** from scratch on 100 random examples → evaluate on 200 test.
3. Use random_100 failures to build **probe_guided** dataset (60 targeted + 40 to 100 total).
4. Train **random_1000** from scratch on 1000 random examples → evaluate.
5. Train **guided_100** from scratch on 100 probe-guided examples → evaluate.
6. Report: `random_100: X%`, `random_1000: Y%`, `guided_100: Z%`.  
   - **Verdict**: If `guided_100 > random_100`, targeting helps.  
   - **Hypothesis**: If `guided_100 >= random_1000`, data-efficiency hypothesis is supported.

## Phase 2: Stacking (~45 min on P100)

Each stage **loads the previous stage**; no full retrain from scratch.

| Stage | Input | Action | Output |
|-------|--------|--------|--------|
| 0 | — | Train from scratch on 100 examples | `models/base` |
| 1 | base | CoT training | `models/cot` |
| 2 | cot | Probe-guided targeted training | `models/probe_guided` |
| 3 | probe_guided | 4 LoRA experts | `models/moe/` |
| 4 | probe_guided | Prune 20% FFN, recovery finetune | `models/pruned` |
| 5 | pruned | QAT (max 3 epochs) | `models/final` |
| 6 | final + RAG | Eval with top-3 RAG rules | `output/stage6_rag_scores.json` |

RAG index: run `python rag/build_index.py` (uses `data/rag_documents/arithmetic_facts.jsonl`).

## Success criteria

- **Phase 1**: `guided_100` accuracy > `random_100` → targeting helps.  
- **Phase 2**: Final stacked model accuracy > `guided_100` baseline.  
- **Main result**: If `guided_100` is close to `random_1000` on a truly untrained model, the data-efficiency hypothesis is supported.

## Layout

- `config.yaml` – Paths and hyperparameters (relative only).
- `data/` – `generate_arithmetic.py`, `train_100.jsonl`, `train_1000.jsonl`, `test_200.jsonl`, `guided_100.jsonl` (built in Phase 1), `rag_documents/arithmetic_facts.jsonl`.
- `scripts/` – `train_model.py` (supports `--from-scratch` and `--base`), `evaluate_model_hf.py` (CoT-safe extraction), `phase1_baseline.py`, `stage0_train_base.py` … `stage6_rag_integrate.py`, `compare_stages.py`.
- `rag/` – `build_index.py`, `query_rag.py`.
- `output/` – All scores and `phase1_report.json`, `stacking_report.json`.

## Requirements

See `requirements.txt`. GPU training targets Kaggle P100; code is CPU-compatible with `FORCE_CPU=1` if needed.
