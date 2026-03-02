"""
LLM4 - RAG query: retrieve top-k arithmetic rules and build augmented prompt.
Injects top 3 most relevant rules into the prompt before each question.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_INDEX = None
_META = None
_MODEL = None
_CONFIG = None


def _load_index(config=None):
    global _INDEX, _META, _MODEL, _CONFIG
    if _INDEX is not None:
        return
    if config is None:
        import yaml
        with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    _CONFIG = config
    db_path = PROJECT_ROOT / config["rag_db_path"].replace("./", "")
    index_path = db_path / "embeddings.npy"
    meta_path = db_path / "facts.json"
    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"RAG index not found. Run rag/build_index.py first: {db_path}")
    import numpy as np
    _INDEX = np.load(str(index_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        _META = json.load(f)
    model_name = config.get("rag_embedding_model", "all-MiniLM-L6-v2")
    try:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(model_name)
    except Exception:
        _MODEL = None


def query_rules(instruction: str, top_k: int = 3):
    """Return list of top_k fact strings most relevant to instruction."""
    import numpy as np
    _load_index()
    if _MODEL is None:
        return []
    q_emb = _MODEL.encode([instruction], normalize_embeddings=True)
    if hasattr(q_emb, "cpu"):
        q_emb = q_emb.cpu().numpy()
    q_emb = np.asarray(q_emb, dtype=_INDEX.dtype)
    scores = (q_emb @ _INDEX.T).flatten()
    top_idx = scores.argsort()[-top_k:][::-1]
    return [_META[i]["fact"] for i in top_idx]


def build_augmented_prompt(instruction: str, config=None):
    """Build prompt with top 3 relevant arithmetic rules then the question."""
    if config is None:
        import yaml
        with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    top_k = config.get("rag_top_k", 3)
    try:
        _load_index(config)
        rules = query_rules(instruction, top_k=top_k)
    except Exception:
        rules = []
    if not rules:
        return f"### Instruction: {instruction}\n\n### Response:"
    context = "\n".join(f"- {r}" for r in rules)
    return f"Relevant rules:\n{context}\n\n### Instruction: {instruction}\n\n### Response:"
