"""
LLM4 Stage 5: QAT. Load models/pruned, SHORT training only (max 3 epochs) -> save models/final.
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


def main():
    config = load_config()
    pruned_path = PROJECT_ROOT / config["pruned_output"].replace("./", "")
    final_output = PROJECT_ROOT / config["final_output"].replace("./", "")
    enhanced_train_path = PROJECT_ROOT / config["enhanced_train"].replace("./", "")
    output_dir = PROJECT_ROOT / config["output_dir"].replace("./", "")
    qat_max_epochs = config.get("qat_max_epochs", 3)
    qat_samples = config.get("stage5_qat_samples", 100)
    if config.get("local_test", False):
        qat_samples = min(qat_samples, config.get("local_phase1_base_size", 100))

    print("Stage 5: QAT - load pruned, max 3 epochs -> models/final")
    if not (pruned_path / "config.json").exists():
        print("models/pruned not found. Run Stage 4 first.")
        sys.exit(1)

    print_device_info()
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    class ProgressCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.is_world_process_zero and logs and "loss" in logs:
                print(f"  [Step {state.global_step}/{state.max_steps}] loss={logs.get('loss', 0):.4f}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(pruned_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(pruned_path),
        trust_remote_code=True,
        dtype=torch.float32,
        device_map=get_device_map(),
        low_cpu_mem_usage=True,
    )

    examples = []
    if Path(enhanced_train_path).exists():
        with open(enhanced_train_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ex = json.loads(line)
                    examples.append(f"### Instruction: {ex['instruction']}\n\n### Response: {ex['response']}")
    train_qat = examples[:qat_samples]
    dataset = Dataset.from_dict({"text": train_qat})

    # Max 3 epochs: steps = min(epochs * len(dataset) / (batch * grad_accum), ...)
    num_epochs = min(3, qat_max_epochs)
    training_args = SFTConfig(
        output_dir=str(final_output),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=num_epochs,
        max_steps=-1,
        learning_rate=5e-6,
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
    print(f"QAT training (max {num_epochs} epochs, {len(train_qat)} examples)...")
    trainer.train()
    final_output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_output))
    tokenizer.save_pretrained(str(final_output))

    subprocess.run([
        sys.executable,
        str(SCRIPT_DIR / "evaluate_model_hf.py"),
        "--model", str(final_output),
        "--output", str(output_dir / "stage5_qat_scores.json"),
        "--stage", "stage5_qat",
    ], cwd=PROJECT_ROOT, check=True)
    print("Stage 5 complete.")


if __name__ == "__main__":
    main()
