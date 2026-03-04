"""
LLM4 - Shared config loading. Single source of truth for config.yaml.
Use this everywhere to avoid drift and inconsistent encoding/paths.
"""

import os
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Subdir for all HuggingFace/transformers cache (keeps runs project-local; delete ./cache for a clean re-download).
CACHE_DIR = PROJECT_ROOT / "cache"


def use_project_cache_only():
    """Force HuggingFace/transformers to use project-local cache only (no global ~/.cache). Call early so every run is clean and self-contained."""
    os.environ.setdefault("HF_HOME", str(CACHE_DIR))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_DIR / "transformers"))
    os.environ.setdefault("HF_HUB_CACHE", str(CACHE_DIR / "hub"))


def load_config():
    """Load config.yaml from project root. Uses utf-8-sig for BOM tolerance."""
    with open(CONFIG_PATH, "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f)
