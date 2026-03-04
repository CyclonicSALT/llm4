#!/usr/bin/env python3
"""
LLM4 Experiment Launcher UI.
Run pipelines (Local / Smoke / Full). Progress bars fill from actual script output
(parsed from stdout: steps, evaluated X/Y, %, loading, etc.).
"""

import os
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_utils import load_config, use_project_cache_only

CONFIG_PATH = PROJECT_ROOT / "config.yaml"
HF_TOKEN_FILE = PROJECT_ROOT / ".hf_token"


def parse_progress_line(line: str):
    """
    Extract 0-100 progress from pipeline stdout.
    Returns (pct, is_final) where is_final=True means this is a definitive X/Y or % value.
    """
    if not line or not line.strip():
        return None
    s = line.strip()
    # tqdm / percentage: "  50%|" or "100%"
    m = re.search(r"(\d+)%\s*\|?", s)
    if m:
        return (min(100, int(m.group(1))), True)
    # "Evaluated 120/200" or "  Evaluated 20/200" or "Evaluating 0/200..."
    m = re.search(r"Evaluat(?:ed|ing)\s+(\d+)/(\d+)", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if b:
            return (min(100, int(100 * a / b)), True)
    # "[Step 1/2] loss=..." or "[Step 3/10]"
    m = re.search(r"\[Step\s+(\d+)/(\d+)\]", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if b:
            return (min(100, int(100 * a / b)), True)
    # "Loading weights: 50%" or "Writing model shards: 50%"
    m = re.search(r"(?:Loading weights|Writing model shards):\s*(\d+)%", s)
    if m:
        return (min(100, int(m.group(1))), True)
    # "Adding EOS...", "Tokenizing...", "Batches: 50%", "Embedding: 100%"
    m = re.search(r"(?:Adding EOS|Tokenizing|Truncating|Batches|Embedding|Encoding).*?(\d+)%", s, re.IGNORECASE)
    if m:
        return (min(100, int(m.group(1))), True)
    # "Loaded N training examples" -> treat as ~30% (data loaded)
    if re.search(r"Loaded\s+\d+\s+(?:training\s+)?examples", s, re.IGNORECASE):
        return (30, False)
    # "Train/val split" -> ~40%
    if "Train/val split" in s or "train" in s.lower() and "val" in s.lower():
        return (40, False)
    # "Starting training" -> 5%
    if "Starting training" in s or "training" in s.lower() and "start" in s.lower():
        return (5, False)
    # "Generated X problems" / "Generated train_phase1_base"
    if re.search(r"Generated\s+\w+:\s*\d+", s):
        return (80, False)
    # "Merging LoRA" / "Saving..."
    if "Merging" in s or "Saving" in s or "save" in s.lower():
        return (90, False)
    # "[no_rag]" / "[rag]" Stage 6
    m = re.search(r"\[(?:no_rag|rag)\]\s+Evaluated\s+(\d+)/(\d+)", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if b:
            return (min(100, int(100 * a / b)), True)
    return None


def load_hf_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()
    if HF_TOKEN_FILE.exists():
        try:
            return HF_TOKEN_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return None


def save_config_updates(updates: dict):
    with open(CONFIG_PATH, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
    out = []
    for line in lines:
        new_line = line
        for key, value in updates.items():
            if line.strip().startswith(key + ":"):
                new_line = f"{key}: {str(value).lower()}\n" if isinstance(value, bool) else f"{key}: {value}\n"
                break
        out.append(new_line)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.writelines(out)


def get_steps(experiment: str, local_test: bool):
    data = PROJECT_ROOT / "data"
    scripts = PROJECT_ROOT / "scripts"
    rag = PROJECT_ROOT / "rag"
    steps = []
    if experiment == "local":
        steps = [
            ("Generate datasets", data / "generate_arithmetic.py", [], 8),
            ("Phase 1: Baseline", scripts / "phase1_baseline.py", [], 25),
            ("Stage 0: Train base", scripts / "stage0_train_base.py", [], 50),
            ("Compare stages", scripts / "compare_stages.py", [], 17),
        ]
    elif experiment == "smoke":
        steps = [
            ("Generate datasets", data / "generate_arithmetic.py", [], 2),
            ("Phase 1: Baseline", scripts / "phase1_baseline.py", [], 10),
            ("Stage 0: Train base", scripts / "stage0_train_base.py", [], 22),
            ("Stage 1: CoT", scripts / "stage1_cot_format.py", [], 10),
            ("Stage 2: Probe guided", scripts / "stage2_probe_guided.py", [], 10),
            ("Stage 3: MoE", scripts / "stage3_moe_train.py", [], 12),
            ("Stage 4: Prune", scripts / "stage4_prune.py", [], 8),
            ("Stage 5: QAT", scripts / "stage5_qat_train.py", [], 10),
            ("Build RAG index", rag / "build_index.py", [], 3),
            ("Stage 6: RAG", scripts / "stage6_rag_integrate.py", [], 5),
            ("Compare stages", scripts / "compare_stages.py", [], 8),
        ]
    else:
        if local_test:
            steps = [
                ("Generate datasets", data / "generate_arithmetic.py", [], 2),
                ("Phase 1: Baseline", scripts / "phase1_baseline.py", [], 10),
                ("Stage 0: Train base", scripts / "stage0_train_base.py", [], 20),
                ("Stage 1: CoT", scripts / "stage1_cot_format.py", [], 9),
                ("Stage 2: Probe guided", scripts / "stage2_probe_guided.py", [], 9),
                ("Stage 3: MoE", scripts / "stage3_moe_train.py", [], 11),
                ("Stage 4: Prune", scripts / "stage4_prune.py", [], 7),
                ("Stage 5: QAT", scripts / "stage5_qat_train.py", [], 9),
                ("Build RAG index", rag / "build_index.py", [], 3),
                ("Stage 6: RAG", scripts / "stage6_rag_integrate.py", [], 5),
                ("Compare stages", scripts / "compare_stages.py", [], 15),
            ]
        else:
            steps = [
                ("Generate datasets", data / "generate_arithmetic.py", [], 1),
                ("Phase 1 seed 42", scripts / "phase1_baseline.py", ["--seed", "42"], 6),
                ("Phase 1 seed 43", scripts / "phase1_baseline.py", ["--seed", "43"], 6),
                ("Phase 1 seed 44", scripts / "phase1_baseline.py", ["--seed", "44"], 6),
                ("Aggregate Phase 1", scripts / "aggregate_phase1_seeds.py", [], 2),
                ("Stage 0: Train base", scripts / "stage0_train_base.py", [], 18),
                ("Stage 1: CoT", scripts / "stage1_cot_format.py", [], 9),
                ("Stage 2: Probe guided", scripts / "stage2_probe_guided.py", [], 9),
                ("Stage 3: MoE", scripts / "stage3_moe_train.py", [], 11),
                ("Stage 4: Prune", scripts / "stage4_prune.py", [], 7),
                ("Stage 5: QAT", scripts / "stage5_qat_train.py", [], 9),
                ("Build RAG index", rag / "build_index.py", [], 2),
                ("Stage 6: RAG", scripts / "stage6_rag_integrate.py", [], 5),
                ("Compare stages", scripts / "compare_stages.py", [], 9),
            ]
    return steps


def run_steps(steps: list, env_extra: dict, log_queue: queue.Queue):
    env = os.environ.copy()
    env.update(env_extra)
    for idx, (label, script_path, args, _weight) in enumerate(steps):
        log_queue.put(("step_start", idx))
        try:
            cmd = [sys.executable, str(script_path)] + args
            proc = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            last_out = [None]
            for line in proc.stdout:
                s = line.rstrip()
                parsed = parse_progress_line(s)
                if parsed is not None:
                    pct, _ = parsed if isinstance(parsed, tuple) else (parsed, True)
                    log_queue.put(("step_progress", idx, min(100, pct)))
                if s != last_out[0]:
                    last_out[0] = s
                    log_queue.put(("out", s))
            proc.wait()
            log_queue.put(("step_done", idx, proc.returncode))
            if proc.returncode != 0:
                log_queue.put(("all_done", False))
                return
        except Exception as e:
            log_queue.put(("err", str(e)))
            log_queue.put(("step_done", idx, -1))
            log_queue.put(("all_done", False))
            return
    log_queue.put(("all_done", True))


def main():
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox

    config = load_config()
    base_size = config.get("local_phase1_base_size", 10)
    large_size = config.get("local_phase1_large_size", 20)
    local_test = config.get("local_test", True)

    root = tk.Tk()
    root.title("LLM4 Launcher")
    root.minsize(520, 520)
    root.geometry("640x620")

    var_experiment = tk.StringVar(value="local")
    var_base = tk.StringVar(value=str(base_size))
    var_large = tk.StringVar(value=str(large_size))
    var_local_test = tk.BooleanVar(value=local_test)
    var_running = tk.BooleanVar(value=False)
    var_overall_pct = tk.DoubleVar(value=0.0)
    var_current_step = tk.StringVar(value="—")
    var_current_pct = tk.DoubleVar(value=0.0)
    log_queue = queue.Queue()
    step_bars = []
    progress_container = None
    completed_weight_sum = [0.0]
    step_max_pct = {}
    current_step_index = [None]
    current_steps_list = []  # steps for current run (so poll_log can show names)

    main_f = ttk.Frame(root, padding=12)
    main_f.pack(fill=tk.BOTH, expand=True)

    ttk.Label(main_f, text="LLM4 Experiment Launcher", font=("Segoe UI", 12, "bold")).pack(anchor=tk.W)

    # Config row
    cfg_f = ttk.Frame(main_f)
    cfg_f.pack(fill=tk.X, pady=(8, 6))
    ttk.Radiobutton(cfg_f, text="Local", variable=var_experiment, value="local").pack(side=tk.LEFT, padx=(0, 12))
    ttk.Radiobutton(cfg_f, text="Smoke", variable=var_experiment, value="smoke").pack(side=tk.LEFT, padx=(0, 12))
    ttk.Radiobutton(cfg_f, text="Full", variable=var_experiment, value="full").pack(side=tk.LEFT, padx=(0, 16))
    ttk.Label(cfg_f, text="Base:").pack(side=tk.LEFT, padx=(8, 2))
    ttk.Spinbox(cfg_f, from_=1, to=100000, width=6, textvariable=var_base).pack(side=tk.LEFT, padx=2)
    ttk.Label(cfg_f, text="Large:").pack(side=tk.LEFT, padx=(8, 2))
    ttk.Spinbox(cfg_f, from_=1, to=100000, width=6, textvariable=var_large).pack(side=tk.LEFT, padx=2)
    ttk.Checkbutton(cfg_f, text="Local test (CPU)", variable=var_local_test).pack(side=tk.LEFT, padx=(12, 0))

    # Run button
    btn_run = ttk.Button(main_f, text="Run", command=None)
    btn_run.pack(pady=(0, 8))

    # ——— Progress section (main focus) ———
    prog_f = ttk.LabelFrame(main_f, text="Progress", padding=8)
    prog_f.pack(fill=tk.X, pady=4)

    ttk.Label(prog_f, text="Overall:").pack(anchor=tk.W)
    overall_bar = ttk.Progressbar(prog_f, length=400, maximum=100, variable=var_overall_pct, mode="determinate")
    overall_bar.pack(fill=tk.X, pady=(2, 8))
    overall_lbl = ttk.Label(prog_f, text="0%")
    overall_lbl.pack(anchor=tk.E)

    ttk.Label(prog_f, text="Current step:", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(6, 2))
    current_name_lbl = ttk.Label(prog_f, textvariable=var_current_step, foreground="gray")
    current_name_lbl.pack(anchor=tk.W)
    current_bar = ttk.Progressbar(prog_f, length=400, maximum=100, variable=var_current_pct, mode="determinate")
    current_bar.pack(fill=tk.X, pady=2)
    current_pct_lbl = ttk.Label(prog_f, text="0%")
    current_pct_lbl.pack(anchor=tk.E)

    steps_frame = ttk.Frame(prog_f)
    steps_frame.pack(fill=tk.X, pady=(8, 0))
    progress_container = steps_frame

    # Log
    ttk.Label(main_f, text="Log").pack(anchor=tk.W, pady=(8, 2))
    log_area = scrolledtext.ScrolledText(main_f, height=10, wrap=tk.WORD, font=("Consolas", 9))
    log_area.pack(fill=tk.BOTH, expand=True, pady=2)
    log_area.tag_configure("head", foreground="#1a56db")
    log_area.tag_configure("info", foreground="#6b7280")
    log_area.tag_configure("err", foreground="#b91c1c")

    def do_run():
        if var_running.get():
            return
        try:
            b = int(var_base.get())
            l = int(var_large.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Base and Large must be integers.")
            return
        if b < 1 or l < 1:
            messagebox.showerror("Invalid input", "Base and Large must be at least 1.")
            return
        save_config_updates({
            "local_phase1_base_size": b,
            "local_phase1_large_size": l,
            "local_test": var_local_test.get(),
        })
        exp = var_experiment.get()
        local_test_val = var_local_test.get()
        steps = get_steps(exp, local_test_val)
        current_steps_list[:] = steps

        for w in progress_container.winfo_children():
            w.destroy()
        step_bars.clear()
        step_max_pct.clear()
        completed_weight_sum[0] = 0.0
        current_step_index[0] = 0
        var_overall_pct.set(0.0)
        var_current_pct.set(0.0)
        var_current_step.set(steps[0][0] if steps else "—")
        overall_lbl.config(text="0%")
        current_pct_lbl.config(text="0%")

        for label, _path, _args, weight in steps:
            row = ttk.Frame(progress_container)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=label, width=26, anchor=tk.W).pack(side=tk.LEFT)
            pbar = ttk.Progressbar(row, length=180, maximum=100, value=0, mode="determinate")
            pbar.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
            step_bars.append((pbar, weight))

        var_running.set(True)
        btn_run.state(["disabled"])
        log_area.delete(1.0, tk.END)
        log_area.insert(tk.END, f"Running {exp} — base={b}, large={l}\n\n", "head")

        use_project_cache_only()
        env = dict(os.environ)
        if local_test_val:
            env["FORCE_CPU"] = "1"
        if load_hf_token():
            env["HF_TOKEN"] = load_hf_token()
        thread = threading.Thread(target=lambda: run_steps(steps, env, log_queue), daemon=True)
        thread.start()
        root.after(80, poll_log)

    def poll_log():
        try:
            while True:
                msg = log_queue.get_nowait()
                kind = msg[0]
                if kind == "out":
                    log_area.insert(tk.END, msg[1] + "\n")
                    log_area.see(tk.END)
                elif kind == "err":
                    log_area.insert(tk.END, f"Error: {msg[1]}\n", "err")
                elif kind == "step_start":
                    idx = msg[1]
                    current_step_index[0] = idx
                    if idx < len(step_bars):
                        var_current_step.set(current_steps_list[idx][0] if idx < len(current_steps_list) else "—")
                        var_current_pct.set(0.0)
                        current_pct_lbl.config(text="0%")
                        pbar, _ = step_bars[idx]
                        pbar.config(mode="determinate", value=0)
                elif kind == "step_progress":
                    idx, pct = msg[1], msg[2]
                    step_max_pct[idx] = max(step_max_pct.get(idx, 0), min(100, pct))
                    if idx < len(step_bars):
                        pbar, weight = step_bars[idx]
                        pbar.config(mode="determinate", value=step_max_pct[idx])
                        var_current_pct.set(step_max_pct[idx])
                        current_pct_lbl.config(text=f"{int(step_max_pct[idx])}%")
                    total_w = sum(w for _, w in step_bars)
                    if total_w:
                        overall = (completed_weight_sum[0] + step_bars[idx][1] * step_max_pct[idx] / 100.0) / total_w * 100.0
                        var_overall_pct.set(min(100.0, overall))
                        overall_lbl.config(text=f"{int(var_overall_pct.get())}%")
                elif kind == "step_done":
                    idx, code = msg[1], msg[2]
                    if idx < len(step_bars):
                        pbar, weight = step_bars[idx]
                        pbar.config(mode="determinate", value=100 if code == 0 else 0)
                        completed_weight_sum[0] += weight
                    total_w = sum(w for _, w in step_bars)
                    if total_w:
                        new_val = min(100.0, completed_weight_sum[0] / total_w * 100.0)
                        var_overall_pct.set(new_val)
                        overall_lbl.config(text=f"{int(new_val)}%")
                    var_current_pct.set(100.0 if code == 0 else 0.0)
                    current_pct_lbl.config(text="100%" if code == 0 else "0%")
                elif kind == "all_done":
                    success = msg[1]
                    var_running.set(False)
                    btn_run.state(["!disabled"])
                    var_overall_pct.set(100.0 if success else var_overall_pct.get())
                    overall_lbl.config(text="100%" if success else overall_lbl.cget("text"))
                    if success:
                        var_current_step.set("Done")
                        var_current_pct.set(100.0)
                        current_pct_lbl.config(text="100%")
                    else:
                        messagebox.showwarning("Run finished", "A step failed. Check the log.")
                    return
        except queue.Empty:
            pass
        root.after(80, poll_log)

    btn_run.config(command=do_run)
    root.mainloop()


if __name__ == "__main__":
    main()
