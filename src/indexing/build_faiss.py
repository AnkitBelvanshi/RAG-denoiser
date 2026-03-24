import argparse
from typing import List

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
    ap.add_argument("--corpus", default=None, help="Override corpus JSONL path")
    ap.add_argument("--index_out", default=None, help="Override output FAISS path")
    ap.add_argument("--meta_out", default=None, help="Override output metadata JSONL path")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    # Defaults from config
    corpus_path_cfg = cfg["data"]["corpus_out"]
    index_out_cfg = cfg["retrieval"]["index_out"]
    meta_out_cfg = cfg["retrieval"].get("meta_out", index_out_cfg + ".meta.jsonl")

    # Overrides from CLI (if provided)
    corpus_path = args.corpus or corpus_path_cfg
    index_out = args.index_out or index_out_cfg
    meta_out = args.meta_out or meta_out_cfg

    embed_model_name = cfg["retrieval"]["embed_model"]
    batch_size = int(cfg["retrieval"].get("batch_size", 64))

    rows = read_jsonl(corpus_path)
    texts = [r["text"] for r in rows]

    model = SentenceTransformer(embed_model_name)

    embs: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
        batch = texts[i:i + batch_size]
        e = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embs.append(e)

    X = np.vstack(embs).astype("float32")
    X = l2_normalize(X)

    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized vectors
    index.add(X) # type: ignore

    faiss.write_index(index, index_out)

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

    print(f"Corpus: {corpus_path}")
    print(f"Saved FAISS index to {index_out}")
    print(f"Saved metadata to {meta_out}")
    print(f"Vectors: {index.ntotal} | Dim: {dim}")


if __name__ == "__main__":
    main()