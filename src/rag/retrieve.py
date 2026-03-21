from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.io import read_jsonl


def l2_normalize1(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x)
    return x / max(float(n), eps)


@dataclass
class RetrievedChunk:
    score: float
    chunk_id: str
    doc_id: str
    title: str
    text: str


class FaissRetriever:
    def __init__(self, index_path: str, meta_path: str, embed_model: str, top_k: int = 5):
        self.index = faiss.read_index(index_path)
        self.meta = read_jsonl(meta_path)
        # vector_id aligns with list index; keep direct list for speed
        self.embedder = SentenceTransformer(embed_model)
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int = None) -> List[RetrievedChunk]:
        k = int(top_k or self.top_k)
        q = self.embedder.encode([query], convert_to_numpy=True, show_progress_bar=False)[0].astype("float32")
        q = l2_normalize1(q).reshape(1, -1)
        scores, ids = self.index.search(q, k)
        out: List[RetrievedChunk] = []
        for score, vid in zip(scores[0].tolist(), ids[0].tolist()):
            if vid < 0:
                continue
            m = self.meta[vid]
            out.append(RetrievedChunk(
                score=float(score),
                chunk_id=m["chunk_id"],
                doc_id=m["doc_id"],
                title=m.get("title", ""),
                text=m["text"],
            ))
        return out
