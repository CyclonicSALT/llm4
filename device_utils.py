"""
Device selection: use GPU on Kaggle (e.g. P100); CPU elsewhere or if FORCE_CPU.
"""
import os

FORCE_CPU_ENV = "FORCE_CPU"


def _force_cpu() -> bool:
    v = os.environ.get(FORCE_CPU_ENV, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def is_kaggle() -> bool:
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") != ""


def _use_gpu() -> bool:
    if _force_cpu():
        return False
    if not is_kaggle():
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def use_cpu() -> bool:
    return not _use_gpu()


def get_device_map() -> str:
    return "cpu" if use_cpu() else "auto"


def print_device_info():
    if use_cpu():
        msg = "Using CPU"
        if is_kaggle() and _force_cpu():
            msg += " (FORCE_CPU set)"
    else:
        import torch
        msg = f"Using GPU: {torch.cuda.get_device_name(0)} [Kaggle]"
    print(msg, flush=True)
