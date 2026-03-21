from dataclasses import dataclass
from typing import List, Tuple

from src.rag.retrieve import FaissRetriever, RetrievedChunk
from src.rag.generate import HFGenerator


def pack_context(chunks: List[RetrievedChunk], max_chars: int = 2000) -> str:
    # Concatenate top chunks with separators; simple and deterministic.
    parts = []
    total = 0
    for c in chunks:
        piece = c.text.strip()
        if not piece:
            continue
        if total + len(piece) + 2 > max_chars:
            break
        parts.append(piece)
        total += len(piece) + 2
    return "\n\n".join(parts)


@dataclass
class RAGResult:
    answer: str
    retrieved: List[RetrievedChunk]
    context: str


class RAGPipeline:
    def __init__(self, retriever: FaissRetriever, generator: HFGenerator, context_max_chars: int = 2000):
        self.retriever = retriever
        self.generator = generator
        self.context_max_chars = context_max_chars

    def answer(self, question: str, top_k: int = None) -> RAGResult:
        chunks = self.retriever.retrieve(question, top_k=top_k)
        context = pack_context(chunks, max_chars=self.context_max_chars)
        ans = self.generator.generate(question, context)
        return RAGResult(answer=ans, retrieved=chunks, context=context)
