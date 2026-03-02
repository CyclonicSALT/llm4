"""
LLM4 - Build RAG index from arithmetic_facts.jsonl.
Embeds facts with sentence-transformers and persists index for query_rag.
Uses relative paths; works on Kaggle.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config():
    import yaml
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    facts_path = PROJECT_ROOT / config["rag_documents"].replace("./", "")
    if not facts_path.exists():
        print(f"Facts file not found: {facts_path}. Run data/generate_arithmetic.py first.")
        sys.exit(1)

    db_path = PROJECT_ROOT / config["rag_db_path"].replace("./", "")
    db_path.mkdir(parents=True, exist_ok=True)
    model_name = config.get("rag_embedding_model", "all-MiniLM-L6-v2")

    facts = []
    with open(facts_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                facts.append(json.loads(line))

    print(f"Loading embedding model: {model_name}")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers not installed. pip install sentence-transformers")
        sys.exit(1)

    model = SentenceTransformer(model_name)
    texts = [d["fact"] for d in facts]
    print(f"Embedding {len(texts)} facts...")
    emb = model.encode(texts, show_progress_bar=True)
    emb = model.normalize_embeddings(emb)
    import numpy as np
    if hasattr(emb, "cpu"):
        emb = emb.cpu().numpy()
    embeddings = np.asarray(emb, dtype=np.float32)

    index_path = db_path / "embeddings.npy"
    meta_path = db_path / "facts.json"
    np.save(str(index_path), embeddings.astype(np.float32))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump([{"fact": d["fact"]} for d in facts], f, ensure_ascii=False)
    print(f"Index saved to {db_path} ({index_path.name}, {meta_path.name})")


if __name__ == "__main__":
    main()
