import argparse
import hashlib
from typing import Dict, Iterable, List, Tuple

from datasets import load_dataset
from tqdm import tqdm

from src.utils.io import load_yaml, write_jsonl


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def chunk_text(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    step = max(1, max_chars - overlap_chars)
    while start < n:
        end = min(n, start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start += step
    return chunks


def build_squad_corpus(split: str, max_chars: int, overlap_chars: int) -> Iterable[Dict]:
    ds = load_dataset("squad", split=split)
    # SQuAD contains many QA pairs per context paragraph.
    # For a retrieval corpus, we deduplicate contexts and chunk them.
    seen = {}
    for ex in ds:
        ctx = ex["context"]
        if ctx not in seen:
            seen[ctx] = ex.get("title", "")
    contexts = list(seen.items())  # (context, title)
    for ctx, title in tqdm(contexts, desc="Chunking unique contexts"):
        doc_id = sha1_text(ctx)
        chunks = chunk_text(ctx, max_chars=max_chars, overlap_chars=overlap_chars)
        for i, ch in enumerate(chunks):
            yield {
                "chunk_id": f"{doc_id}_{i}",
                "doc_id": doc_id,
                "title": title,
                "text": ch,
            }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    split = cfg["data"]["dataset_split"]
    out_path = cfg["data"]["corpus_out"]
    max_chars = int(cfg["chunking"]["max_chars"])
    overlap_chars = int(cfg["chunking"]["overlap_chars"])

    rows = list(build_squad_corpus(split=split, max_chars=max_chars, overlap_chars=overlap_chars))
    write_jsonl(out_path, rows)
    print(f"Wrote {len(rows)} chunks to {out_path}")


if __name__ == "__main__":
    main()
