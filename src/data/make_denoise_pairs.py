import argparse
import json
import os
import random
from typing import Dict

from src.data.noise import corrupt_text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="Input clean corpus JSONL (chunks)")
    ap.add_argument("--out", required=True, help="Output JSONL of denoise pairs")
    ap.add_argument("--num_pairs", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--p_typo", type=float, default=0.03)
    ap.add_argument("--p_swap", type=float, default=0.01)
    ap.add_argument("--p_delete", type=float, default=0.01)
    ap.add_argument("--p_insert", type=float, default=0.01)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Load chunks
    chunks = []
    with open(args.inp, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    if not chunks:
        raise RuntimeError(f"No rows found in {args.inp}")

    n = min(args.num_pairs, len(chunks))
    sampled = rng.sample(chunks, n)

    with open(args.out, "w", encoding="utf-8") as fout:
        for row in sampled:
            clean = row.get("text", "")
            noisy = corrupt_text(
                clean,
                rng=rng,
                p_typo=args.p_typo,
                p_swap=args.p_swap,
                p_delete=args.p_delete,
                p_insert=args.p_insert,
            )
            fout.write(json.dumps({"input_text": noisy, "target_text": clean}, ensure_ascii=False) + "\n")

    print(f"Wrote {n} pairs -> {args.out}")

if __name__ == "__main__":
    main()