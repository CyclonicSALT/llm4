# Proposal: Implementing Suggested Improvements in LLM4

This document describes how each suggested improvement could be implemented in the codebase, what challenges they might pose, and when they should or should not be implemented. It also includes a **compute budget and priority order** for when only some changes are feasible.

---

## Compute budget and priority order

Several suggestions (multiple seeds, rank sweep, smarter RAG) increase compute. If time or budget is limited, use this order:

1. **Must-have for the main claim**
   - **Balanced random baseline** — one extra training run so “guided > random” is defensible.
   - **Multiple seeds (≥3) on Phase 1 and on that baseline** — required to support the core comparison; seed variance can otherwise make or break the result at this scale.

2. **High value, low cost**
   - **Validation split everywhere** — do it by default for every stage (see §1).
   - **Bootstrap confidence intervals** on the main metric(s) — high value, low cost; do it whenever you report or compare results (see §3).

3. **Next**
   - **Ablation table** (all stages on the same test set); **RAG on/off** if the code supports it (see §9).

4. **As time allows**
   - LoRA rank sweep, smarter RAG, failure-mode categorization, perplexity, stratified test set, LR schedule for from-scratch.

---

## 1. Validation split and validation loss tracking

**Recommendation: Do it everywhere.** With 500 examples per stage and (in Stage 0) random init, overfitting is a default risk. Validation is not optional for these settings.

**How it could be implemented**
- When loading the dataset (e.g. in `train_model.py` or in the stage scripts that call it), split once into train/val (e.g. 90/10 or 80/20) in a deterministic way (fixed seed).
- Pass `eval_dataset` into `SFTTrainer` and set `evaluation_strategy="epoch"` (or `"steps"` with `eval_steps`) in `SFTConfig`.
- The Trainer will compute validation loss (and any `compute_metrics` you add) and log it. No custom loss is required; it uses the same causal LM loss as training.
- Optionally set `load_best_model_at_end=True` and `metric_for_best_model="eval_loss"` for checkpoint selection or early stopping.

**Challenges**
- One canonical place for the split so every stage uses a consistent rule (or document that val is per-stage).
- With 500 examples per expert in Stage 3, 10% val = 50 examples; val loss will be noisy but still informative.
- If data is loaded from different files per stage, you need a consistent rule (e.g. same val fraction and seed) to avoid confusion.

**Reasons to do it**
- Catch overfitting early, especially in later stages with small data.
- Know when a stage has actually converged vs. just hit the epoch limit.
- Standard practice and expected when reporting.

**Reasons to skip**
- None recommended at this data scale; treat validation as unconditional.

---

## 2. Better LR schedule for from-scratch (Stage 0)

**How it could be implemented**
- In `train_model.py`, when building `SFTConfig`, set or override: `lr_scheduler_type="cosine"`, and `warmup_ratio=0.05` or `0.10` (5–10% of steps).
- For from-scratch only, use a higher `learning_rate` (e.g. `1e-4` or `2e-4`) when `args.from_scratch` is True — e.g. from config: `learning_rate=config.get("learning_rate_from_scratch", 2e-4)` so fine-tuning keeps a lower LR.
- No change to the optimizer (AdamW); only scheduler type, warmup, and LR value.

**Challenges**
- Avoid breaking fine-tuning: use a separate config key or branch on `--from-scratch` so only Stage 0 gets the more aggressive schedule.
- If from-scratch already converges well, the gain may be modest; risk is low.

**Reasons to do it**
- Aligns the schedule with “training from random init” rather than “fine-tuning pretrained.”
- Low implementation cost; may improve convergence speed or final loss.

**Reasons to skip**
- Current from-scratch already converges and you are not iterating on Stage 0.
- You prefer a single global LR for simplicity.

---

## 3. Bootstrapped confidence intervals on test metrics

**How it could be implemented**
- In the evaluation script (e.g. `evaluate_model_hf.py`): after computing accuracy (and optionally perplexity) on the full test set, run a bootstrap loop (e.g. 500 or 1000 iterations). Each iteration: sample N test examples with replacement, compute the metric on that sample, store it. Report mean and percentile interval (e.g. 2.5–97.5) or ±1.96×std.
- No change to how the model is loaded or how single-example correctness is computed; only aggregation and resampling.

**Challenges**
- Slightly longer eval (many resamples); with 200 examples, bootstrap is valid. Document N and number of bootstrap samples in the report.

**Reasons to do it**
- Makes “we get X%” statistically defensible (e.g. “85% ± 3%”).
- No extra data or training; easy to add.

**Reasons to skip**
- Purely internal use with no need to report uncertainty; or you already rely on multiple seeds and care more about seed variance than sampling variance.

---

## 4. Stratified test set

**How it could be implemented**
- When building the test set (in data generation or a dedicated split script), define strata: e.g. operation type, number of digits, “with carry” vs “no carry.”
- Either: (a) split so the test set has predefined counts per stratum (e.g. 25 per op type), or (b) stratify when sampling so proportions match the intended distribution.
- Use a fixed seed for reproducibility. Store stratum labels with each example (e.g. in JSONL) so evaluation can report per-stratum accuracy.

**Challenges**
- Need a clear definition of strata (op type exists in your data; digit length/carry may require parsing or metadata).
- Some strata may be rare; report per-stratum N so readers know.

**Reasons to do it**
- Prevents one subgroup from dominating the overall metric.
- Surfaces “good at addition, bad at division” instead of hiding it in one number.

**Reasons to skip**
- Test set is already diverse by chance and you only care about overall performance.
- You do not have clear strata in the data and do not want to add them.

---

## 5. Multiple seeds and averaging

**Recommendation: Required for the main result.** The core claim is “guided > random” (or “guided > balanced random”). At this scale, a single seed can flip that comparison. **At least 3 seeds (≥3) on Phase 1 and on the balanced baseline** should be treated as required for the main claim. **Later stages do not need multi-seed by default:** probe_guided, MoE, prune, QAT, and RAG can stay single-seed unless you are making stage-wise claims (e.g. "pruning does not hurt"). Ablation runs (comparing stages on the same test set) also typically use one seed per configuration. That keeps the extra compute to Phase 1 and the baseline, not a full tripling of every stage.

**How it could be implemented**
- Add a `--seed` (or `seed` in config) to the training entrypoint; pass it to `SFTConfig` (e.g. `seed=args.seed`) and to any data shuffling/splitting.
- Run the same stage multiple times (e.g. seeds 42, 43, 44), each writing to a different output dir or subdir (e.g. `output/seed_42/`).
- For evaluation: run the same eval script on each checkpoint; collect metrics per seed; report mean and std (or median and IQR) in a small table or in the existing compare/output scripts.

**Logging and aggregation:** With multiple seeds, stages, and an ablation table all producing JSON (or similar) outputs, use a **consistent naming convention** from the start (e.g. `stage0_seed42_scores.json`, or `scores/stage0/seed42.json`). Add a small **aggregation script** that reads these files and produces a single comparison table (e.g. mean ± std per stage/seed). Otherwise comparing results across seeds and stages becomes manual and error-prone.

**Challenges**
- 3–5× compute only for the stages where you run multiple seeds; focus on Phase 1 and the balanced baseline for the main claim.
- A small script or loop to launch N seeds and aggregate results avoids manual copy-paste.

**Reasons to do it**
- A single run can be misleading; averaging over seeds is the standard way to show “typical” performance.
- Essential for any serious comparison (e.g. guided vs. balanced random).

**Reasons to skip or limit**
- Only if compute is severely constrained; then run at least 2–3 seeds for the main baseline and the key comparison.

---

## 6. Failure mode categorization

**How it could be implemented**
- In the eval script (or a post-processing script): for each test example, store model output, ground truth, and correctness.
- Optionally run a simple classifier or rule set on (instruction, model_output, correct_answer) to assign a failure mode: e.g. “wrong_operation”, “carry_error”, “off_by_one”, “digit_alignment”, “other”. Rules can be regex/heuristic (e.g. right op symbol but wrong value → carry or off-by-one; wrong op symbol → wrong_operation).
- Alternatively, manually label a subset of errors (e.g. 50–100) and use that to interpret patterns.
- Output a small table: failure_mode, count, percentage of errors. Use this to decide what to add to the guided set.

**Challenges**
- Automatic categorization is imperfect; a few hand-written rules and an “other” bucket are likely needed.
- Manual labeling does not scale; use it for analysis and for designing the rules.

**Reasons to do it**
- Turns “78% accuracy” into “most errors are carry; add more carry examples.”
- Makes the next round of data construction targeted.

**Reasons to skip**
- You are not planning to iterate on data or the guided set.
- Aggregate accuracy is sufficient for your goals.

---

## 7. LoRA rank search and expert specialization

**How it could be implemented**
- **Rank:** In `config.yaml` (or in the stage that builds LoRA config), use something like `lora_r: 8` and have `train_model.py` read it. Run Stage 3 with `lora_r` in [4, 8, 16], same data and seed; compare val/test accuracy and optionally training time.
- **Expert specialization (performance):** In evaluation, run each of the four expert checkpoints on the full test set (or on a subset with type labels). Build a table: rows = problem type, columns = expert; cell = accuracy. If each expert wins on its own type, they specialize.
- **Expert specialization (weights):** Load the probe_guided base and each expert (after training). For each adapted layer, compute delta = expert_weight − base_weight (you would need to either save LoRA adapters before merge or reload base and each full expert and subtract). Compute pairwise similarity (e.g. cosine or Frobenius norm of difference) between experts’ deltas; low similarity suggests specialization. This requires a small script and careful handling of which layers were LoRA-adapted.

**Challenges**
- Rank sweep: three runs of Stage 3; manageable.
- Weight comparison: Stage 3 merges LoRA and saves full models, so “adapter only” is not saved. You would load base and each expert and diff full weights (or re-run Stage 3 with “save adapter only” for analysis).
- Router: you already have a fixed router (logistic regression on features); “routing” analysis is “which expert was chosen for which example” and “did that match the problem type?” — no code change, just eval and tables.

**Reasons to do it**
- Rank can be a free win or prevent underfitting.
- Specialization analysis shows whether the MoE is adding value or just four similar copies.

**Reasons to skip**
- You are satisfied with current MoE performance and do not need to justify the design.
- Weight-level analysis is overkill; a performance-by-type table is often enough.

---

## 8. Perplexity alongside accuracy

**How it could be implemented**
- In the evaluation script, when running the model on each test example, get logits for the target positions (the answer tokens). Compute the log probability of the correct answer sequence (sum of log probs per token, or mean). Exponentiate the mean negative log prob to get perplexity (or report mean log prob).
- This may require using the model’s forward with `labels` or extracting logits at answer positions and computing cross-entropy with the target tokens; many HF eval loops already have access to logits or loss.
- Log both accuracy and perplexity (and optionally per-example log prob) in the same output file or table.

**Challenges**
- Need a clear definition of the “answer” span (e.g. after “### Response:”) so you only score those tokens.
- Slightly more code in the eval script; ensure tokenization aligns with your prompt format.

**Reasons to do it**
- Distinguishes “barely right” from “confidently right” and helps judge calibration.
- Often a small code change once the eval loop exists.

**Reasons to skip**
- You only care about right/wrong for this project.
- The eval script is minimal and you prefer not to touch it.

---

## 9. Ablation table

**How it could be implemented**
- Define a single test set and a single metric (e.g. accuracy, and optionally perplexity).
- Run evaluation for: (1) base only, (2) probe_guided only, (3) one MoE expert (or router-driven MoE if you have it), (4) pruned model, (5) QAT model, (6) full RAG pipeline. Use the same eval script and same test file for all.
- Add a small script (e.g. `compare_stages.py` or extend existing) that reads the stage result files and prints a table: rows = stage, columns = metric(s). Optionally add a “no RAG” run for the same model to isolate RAG’s effect.

**Implementation note (RAG on/off):** The current architecture may not cleanly support “run the same model with RAG disabled” without some refactoring. Before committing to a RAG ablation, check whether the inference path can already run the same checkpoint with retrieval/injection turned off (e.g. via a flag or a separate code path). If not, a small code change (e.g. a `--no-rag` flag or a separate inference entrypoint) may be needed. Decide this before promising a RAG-on vs RAG-off row in the ablation table.

**Logging and aggregation:** Reuse the same convention and aggregation approach as for multi-seed (§5): stage outputs (e.g. `stage0_scores.json`, `stage3_moe_scores.json`) should follow a predictable naming and path pattern so one script can load all of them and build the ablation table. Doing this early avoids ad-hoc paths and makes it easy to add new stages or seeds later.

**Challenges**
- You must have (or train) checkpoints for each stage and run eval on each; some bookkeeping.
- RAG vs no-RAG: see implementation note above.

**Reasons to do it**
- Turns the pipeline into “evidence that each step helps (or doesn’t).”
- Identifies which stages are worth keeping and which are redundant or harmful.

**Reasons to skip**
- You are not trying to publish or convince others; you only care that the full pipeline works.

---

## 10. Smarter RAG (contrastive examples, dynamic k)

**How it could be implemented**
- **Contrastive examples:** In RAG retrieval or prompt construction, besides “top-k similar rules,” optionally retrieve “a similar problem where the model was wrong” plus the correct solution (you would need a small store of (instruction, wrong_output, correct_answer) or use a dev set and pre-run the model to collect failures). Inject one such example into the prompt. Requires a second retrieval path or index (e.g. by embedding of instruction) and a template.
- **Dynamic k:** When building the RAG prompt, classify or score “hardness” (e.g. by digit count, op count, or a small classifier). If hard, use k=5 or 6; if easy, use k=1 or 2. Implementation: a function that returns k given the query; then pass that k to the existing retrieval call.

**Challenges**
- Contrastive: need a curated set of (query, wrong, right) and a way to retrieve by similarity; more moving parts and risk of retrieval mistakes.
- Dynamic k: need a definition of “hard” and a threshold; may require some tuning.

**Reasons to do it**
- Could improve RAG quality and make the system more robust on hard problems.
- Good research/engineering content.

**Reasons to skip**
- Current RAG is good enough.
- You want to keep the pipeline simple; contrastive RAG is more involved.

---

## 11. Balanced random baseline

**How it could be implemented**
- When building the “random” baseline training set, do not sample uniformly from the pool. Instead, define strata (e.g. operation type, and optionally digit length or difficulty). Sample a target number per stratum (e.g. 5k total with 1.25k per op type, or proportional to a desired distribution) so the random set is balanced.
- Run Stage 0 (or the stage that uses “random” data) with this balanced random set; keep everything else (model, epochs, LR) the same. Compare to the probe-guided (or otherwise guided) run on the same test set.
- Implement in the data pipeline (e.g. in `generate_arithmetic.py` or a separate script that samples from existing data): group by stratum, then sample without replacement up to the per-stratum cap.

**Challenges**
- Need clear strata and enough data per stratum to fill 5k.
- One more training run and one more row in the comparison table.

**Reasons to do it**
- “Guided beats random” becomes “guided beats balanced random,” which is much harder to dismiss as luck.
- Stronger story for the value of probe/guidance.

**Reasons to skip**
- You are not claiming “guided is better than random”; you only care about absolute performance.
- You do not have stratified labels in the data and do not want to add them.

---

## Summary table

| Suggestion | Implementation effort | Main challenge | Recommendation |
|------------|------------------------|----------------|----------------|
| Validation split + val loss | Low | Consistent split across stages | **Do it everywhere** (unconditional). |
| LR schedule for from-scratch | Low | Don’t break fine-tuning | Do it if you iterate on Stage 0. |
| Bootstrap CIs | Low | Slightly longer eval | **High value, low cost;** do it when reporting or comparing results. |
| Stratified test set | Medium | Define strata, store labels | Do it if you care about subgroups. |
| Multiple seeds | Low (scripting) | More compute | **Required for main claim** (≥3 seeds on Phase 1 and balanced baseline). |
| Failure mode categorization | Medium | Heuristics or manual labels | Do it if you iterate on data. |
| LoRA rank + expert specialization | Medium | Weight diff if you want it | Rank sweep yes; weight analysis optional. |
| Perplexity | Low–medium | Define answer span | Do it if you care about calibration. |
| Ablation table | Medium | Eval all stages; RAG on/off may need refactor | Do it if you want to show each step. |
| Smarter RAG | Medium–high | Contrastive store, dynamic k | Experiment if RAG is central. |
| Balanced random baseline | Low–medium | Stratified sampling | **Do it if you claim “guided > random.”** |
