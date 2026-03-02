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

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from device_utils import get_device_map, use_cpu, print_device_info


def load_config():
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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

    texts = [format_example(ex["instruction"], ex["response"]) for ex in examples]

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
            torch_dtype="float32",
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

    dataset = Dataset.from_dict({"text": texts})

    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.get("batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation", 16),
        num_train_epochs=num_epochs,
        learning_rate=config.get("learning_rate", 2e-4),
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
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
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
    # Ensure config vocab_size matches tokenizer so loading later does not mismatch
    model.config.vocab_size = len(tokenizer)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    # Remove PEFT adapter artifacts so loader treats this as a full model, not base+adapter
    for f in ("adapter_config.json", "adapter_model.safetensors"):
        p = output_dir / f
        if p.exists():
            p.unlink()
            print(f"Removed {f} so directory loads as full model.")
    if trainer.state.log_history and "loss" in trainer.state.log_history[-1]:
        print(f"Final loss: {trainer.state.log_history[-1]['loss']:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
