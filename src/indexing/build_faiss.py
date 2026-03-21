import argparse
from typing import Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.utils.io import load_yaml, read_jsonl, write_jsonl


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    corpus_path = cfg["data"]["corpus_out"]
    embed_model_name = cfg["retrieval"]["embed_model"]
    index_out = cfg["retrieval"]["index_out"]
    meta_out = cfg["retrieval"].get("meta_out", index_out + ".meta.jsonl")
    batch_size = int(cfg["retrieval"].get("batch_size", 64))

    rows = read_jsonl(corpus_path)
    texts = [r["text"] for r in rows]

    model = SentenceTransformer(embed_model_name)
    # SentenceTransformer will automatically use CUDA if available.
    embs: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
        batch = texts[i:i + batch_size]
        e = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embs.append(e)
    X = np.vstack(embs).astype("float32")
    X = l2_normalize(X)

    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized vectors
    index.add(X)

    faiss.write_index(index, index_out)

    # Save metadata in vector-id order (same order as rows)
    meta_rows = []
    for vid, r in enumerate(rows):
        meta_rows.append({
            "vector_id": vid,
            "chunk_id": r["chunk_id"],
            "doc_id": r["doc_id"],
            "title": r.get("title", ""),
            "text": r["text"],
        })
    write_jsonl(meta_out, meta_rows)

    print(f"Saved FAISS index to {index_out}")
    print(f"Saved metadata to {meta_out}")
    print(f"Vectors: {index.ntotal} | Dim: {dim}")


if __name__ == "__main__":
    main()
