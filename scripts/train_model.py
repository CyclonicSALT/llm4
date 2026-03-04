"""
LLM4 - Training: from-scratch (random weights) or load previous stage and continue.
Supports --from-scratch (no pretrained weights) and --base <path> for stacking.
Saves full model (merge LoRA then save) so next stage can load it.
"""

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_utils import load_config, use_project_cache_only
from device_utils import get_device_map, use_cpu, print_device_info


def format_example(instruction: str, response: str) -> str:
    return f"### Instruction: {instruction}\n\n### Response: {response}"


def main():
    parser = argparse.ArgumentParser(description="Train model (from scratch or from previous stage)")
    parser.add_argument("--data", required=True, help="Path to training .jsonl")
    parser.add_argument("--output", required=True, help="Where to save model")
    parser.add_argument("--samples", type=int, default=None, help="Max samples (default: all)")
    parser.add_argument("--base", type=str, default=None, help="Load this model and continue (stacking)")
    parser.add_argument("--from-scratch", action="store_true", help="Random weights, no pretraining (Phase 1 / Stage 0)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if exists")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max epochs (e.g. QAT=3)")
    parser.add_argument("--no-lora", action="store_true", help="Full finetune (no LoRA); use for from-scratch small runs")
    args = parser.parse_args()

    if not args.from_scratch and not args.base:
        parser.error("Use either --from-scratch or --base <path>")
    if args.from_scratch and args.base:
        parser.error("Use only one of --from-scratch or --base")

    use_project_cache_only()
    config = load_config()
    base_model_name = config["base_model"]
    cache = config.get("model_cache", "./models/base")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    if args.samples is not None:
        examples = examples[: args.samples]
    print(f"Loaded {len(examples)} training examples")

    # Validation split (deterministic, fixed seed)
    val_fraction = config.get("val_fraction", 0.1)
    val_seed = config.get("val_seed", 42)
    n_val = max(0, min(len(examples) // 10, int(len(examples) * val_fraction)))
    if n_val > 0:
        import random
        rng = random.Random(val_seed)
        indices = list(range(len(examples)))
        rng.shuffle(indices)
        val_idx = set(indices[:n_val])
        train_ex = [ex for i, ex in enumerate(examples) if i not in val_idx]
        val_ex = [examples[i] for i in indices[:n_val]]
        train_texts = [format_example(ex["instruction"], ex["response"]) for ex in train_ex]
        val_texts = [format_example(ex["instruction"], ex["response"]) for ex in val_ex]
        print(f"Train/val split: {len(train_texts)} train, {len(val_texts)} val (val_fraction={val_fraction}, seed={val_seed})")
    else:
        train_ex = examples
        train_texts = [format_example(ex["instruction"], ex["response"]) for ex in examples]
        val_texts = None

    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    class ProgressCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.is_world_process_zero and logs and "loss" in logs:
                step, max_steps = state.global_step, state.max_steps
                print(f"  [Step {step}/{max_steps}] loss={logs.get('loss', 0):.4f}", flush=True)

    print_device_info()
    use_lora = not args.no_lora
    max_seq = config.get("max_seq_length", 256)
    num_epochs = config.get("epochs", 1)
    if args.max_epochs is not None:
        num_epochs = args.max_epochs

    if args.from_scratch:
        print("Loading config and tokenizer (no pretrained weights)...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            cache_dir=cache,
        )
        cfg = AutoConfig.from_pretrained(base_model_name, cache_dir=cache, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(cfg)
        model.resize_token_embeddings(len(tokenizer))
        import torch
        device = "cuda" if not use_cpu() and torch.cuda.is_available() else "cpu"
        model = model.to(device)
    else:
        base_path = Path(args.base).resolve()
        if not base_path.exists():
            base_path = PROJECT_ROOT / args.base
        if not base_path.exists():
            print(f"Base path not found: {args.base}")
            sys.exit(1)
        print(f"Loading model from {base_path}...")
        tokenizer = AutoTokenizer.from_pretrained(str(base_path), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(base_path),
            trust_remote_code=True,
            dtype="float32",
            device_map=get_device_map(),
            low_cpu_mem_usage=True,
        )

    if use_lora:
        lora_config = LoraConfig(
            r=config.get("lora_r", 8),
            lora_alpha=config.get("lora_alpha", 16),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    train_dataset = Dataset.from_dict({"text": train_texts})
    eval_dataset = Dataset.from_dict({"text": val_texts}) if val_texts else None

    lr = config.get("learning_rate", 2e-4)
    lr_scheduler = config.get("lr_scheduler_type", "linear")
    warmup_ratio = 0.0
    if args.from_scratch:
        lr = config.get("learning_rate_from_scratch", 2e-4)
        lr_scheduler = config.get("lr_scheduler_type_from_scratch", "cosine")
        warmup_ratio = config.get("warmup_ratio_from_scratch", 0.1)
        print(f"From-scratch: lr={lr}, scheduler={lr_scheduler}, warmup_ratio={warmup_ratio}")

    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.get("batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation", 16),
        num_train_epochs=num_epochs,
        learning_rate=lr,
        lr_scheduler_type=lr_scheduler,
        warmup_ratio=warmup_ratio,
        save_strategy=config.get("save_strategy", "steps"),
        save_steps=config.get("save_steps", 5),
        save_total_limit=config.get("save_total_limit", 2),
        logging_steps=1,
        use_cpu=use_cpu(),
        fp16=False,
        report_to=[],
        resume_from_checkpoint=bool(args.resume and any(output_dir.glob("checkpoint-*"))),
        dataset_text_field="text",
        max_length=max_seq,
        packing=False,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
        greater_is_better=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[ProgressCallback()],
    )

    print("Starting training...")
    start = time.perf_counter()
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    elapsed = time.perf_counter() - start
    print(f"Training finished in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    if use_lora:
        print("Merging LoRA and saving full model...")
        model = model.merge_and_unload()
    # Ensure config vocab_size and model_type so loading later works
    model.config.vocab_size = len(tokenizer)
    model.config.model_type = getattr(model.config, "model_type", None) or "qwen2"
    # Save merged model weights explicitly (trainer.save_model can write adapter-only; we need full model.safetensors)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))
    # Remove PEFT adapter artifacts if trainer wrote them, so loader sees only full model
    for f in ("adapter_config.json", "adapter_model.safetensors"):
        p = output_dir / f
        if p.exists():
            p.unlink()
            print(f"Removed {f} so directory loads as full model.")
    # Ensure config.json on disk has model_type (AutoModel requires it)
    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if cfg.get("model_type") is None:
            cfg["model_type"] = "qwen2"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
    if trainer.state.log_history and "loss" in trainer.state.log_history[-1]:
        print(f"Final loss: {trainer.state.log_history[-1]['loss']:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
