"""Cross-encoder reranking helpers."""

import warnings
from dataclasses import dataclass
from typing import List, Optional

import torch
from sentence_transformers import CrossEncoder

from src.rag.retrieve import RetrievedChunk


@dataclass
class RerankConfig:
    model_name: str
    batch_size: int = 16
    device: Optional[str] = None  # e.g. "cuda", "cpu", or "auto"


def _resolve_device(requested: Optional[str]) -> str:
    normalized = (requested or "").strip().lower()
    if normalized in {"", "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn(
            f"Requested reranker device '{requested}' but CUDA is unavailable; falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cpu"
    return requested


class CrossEncoderReranker:
    def __init__(self, cfg: RerankConfig):
        self.cfg = cfg
        self.device = _resolve_device(cfg.device)
        self.model = CrossEncoder(cfg.model_name, device=self.device)

    def rerank(self, query: str, chunks: List[RetrievedChunk], top_n: int) -> List[RetrievedChunk]:
        if not chunks:
            return []
        pairs = [(query, c.text) for c in chunks]
        scores = self.model.predict(pairs, batch_size=int(self.cfg.batch_size), show_progress_bar=False)

        scored = []
        for c, s in zip(chunks, scores):
            scored.append(
                RetrievedChunk(
                    score=float(s),
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    title=c.title,
                    text=c.text,
                )
            )
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[: int(top_n)]