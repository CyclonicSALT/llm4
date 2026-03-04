"""
LLM4 Stage 4: Pruning. Load models/probe_guided (not moe), prune 20% FFN, recovery finetune -> save models/pruned.
"""

import json
import subprocess
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_utils import load_config
from device_utils import get_device_map, use_cpu, print_device_info


def count_nonzero(model):
    return sum((p != 0).sum().item() for p in model.parameters())


def main():
    config = load_config()
    probe_guided_path = PROJECT_ROOT / config["probe_guided_output"].replace("./", "")
    pruned_output = PROJECT_ROOT / config["pruned_output"].replace("./", "")
    enhanced_train_path = PROJECT_ROOT / config["enhanced_train"].replace("./", "")
    output_dir = PROJECT_ROOT / config["output_dir"].replace("./", "")
    sparsity = config.get("pruning_target_sparsity", 0.2)
    preserve_attention = config.get("preserve_attention", True)

    print("Stage 4: Pruning - load probe_guided, prune 20% FFN, recovery -> models/pruned")
    if not (probe_guided_path / "config.json").exists():
        print("models/probe_guided not found. Run Stage 2 first.")
        sys.exit(1)

    print_device_info()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(probe_guided_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(probe_guided_path),
        trust_remote_code=True,
        dtype=torch.float32,
        device_map=get_device_map(),
        low_cpu_mem_usage=True,
    )
    before_params = sum(p.numel() for p in model.parameters())
    prunable = ["gate_proj", "up_proj", "down_proj"] if preserve_attention else []
    n_pruned = 0
    for name, param in model.named_parameters():
        if not any(p in name for p in prunable) or param.dim() < 2:
            continue
        with torch.no_grad():
            flat = param.data.abs().flatten()
            k = max(1, int(flat.numel() * sparsity))
            thresh = torch.kthvalue(flat, k).values.item()
            mask = param.data.abs() >= thresh
            param.data.mul_(mask)
            n_pruned += (~mask).sum().item()
    print(f"Pruned {n_pruned / 1e6:.2f}M weights ({100.0 * n_pruned / before_params:.1f}%)")

    pruned_output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(pruned_output))
    tokenizer.save_pretrained(str(pruned_output))

    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from transformers import TrainerCallback

    class ProgressCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.is_world_process_zero and logs and "loss" in logs:
                print(f"  [Step {state.global_step}/{state.max_steps}] loss={logs.get('loss', 0):.4f}", flush=True)

    examples = []
    if Path(enhanced_train_path).exists():
        with open(enhanced_train_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ex = json.loads(line)
                    examples.append(f"### Instruction: {ex['instruction']}\n\n### Response: {ex['response']}")
    prune_cap = config.get("stage4_prune_samples", 100)
    if config.get("local_test", False):
        prune_cap = min(prune_cap, config.get("local_phase1_base_size", 100))
    examples = examples[: prune_cap]
    dataset = Dataset.from_dict({"text": examples})

    training_args = SFTConfig(
        output_dir=str(pruned_output / "recovery"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_steps=100,
        learning_rate=1e-5,
        max_length=256,
        save_strategy="no",
        use_cpu=use_cpu(),
        report_to=[],
        dataset_text_field="text",
        packing=False,
        logging_steps=1,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[ProgressCallback()],
    )
    print("Recovery fine-tune...")
    trainer.train()
    trainer.save_model(str(pruned_output))
    tokenizer.save_pretrained(str(pruned_output))

    subprocess.run([
        sys.executable,
        str(SCRIPT_DIR / "evaluate_model_hf.py"),
        "--model", str(pruned_output),
        "--output", str(output_dir / "stage4_pruned_scores.json"),
        "--stage", "stage4_pruned",
    ], cwd=PROJECT_ROOT, check=True)
    print("Stage 4 complete.")


if __name__ == "__main__":
    main()
