# src/data/check_noise_stats.py
import argparse
import json

def read_jsonl(path, limit=None):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            out.append(json.loads(line))
    return out

def char_diff_rate(a: str, b: str) -> float:
    # simple proxy: percent of positions that differ up to min length + length mismatch penalty
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    diffs = sum(1 for i in range(n) if a[i] != b[i])
    diffs += abs(len(a) - len(b))
    return diffs / max(len(a), len(b), 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--noisy", required=True)
    ap.add_argument("--limit", type=int, default=2000)
    args = ap.parse_args()

    clean = read_jsonl(args.clean, limit=args.limit)
    noisy = read_jsonl(args.noisy, limit=args.limit)

    # Align by chunk_id
    noisy_map = {x["chunk_id"]: x["text"] for x in noisy}
    rates = []
    missing = 0
    for c in clean:
        cid = c["chunk_id"]
        if cid not in noisy_map:
            missing += 1
            continue
        rates.append(char_diff_rate(c["text"], noisy_map[cid]))

    avg = sum(rates) / max(len(rates), 1)
    print(f"Compared chunks: {len(rates)} (missing noisy: {missing})")
    print(f"Average char-diff rate: {avg:.4f}")
    print("Sample diffs (first 5 non-trivial):")
    shown = 0
    for c in clean:
        cid = c["chunk_id"]
        if cid not in noisy_map:
            continue
        r = char_diff_rate(c["text"], noisy_map[cid])
        if r > 0.01:
            print("\n--- chunk_id:", cid, "diff_rate:", f"{r:.4f}")
            print("CLEAN:", c["text"][:220].replace("\n"," "))
            print("NOISY:", noisy_map[cid][:220].replace("\n"," "))
            shown += 1
            if shown >= 5:
                break

if __name__ == "__main__":
    main()