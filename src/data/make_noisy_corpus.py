import argparse
import random
from typing import Any, Dict, List

from tqdm import tqdm

from src.data.noise import NoiseConfig, perturb_text
from src.utils.io import load_yaml, read_jsonl, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config (e1_noisy_corpus.yaml).")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    noise_cfg = cfg["noise"]

    corpus_in = data_cfg["corpus_in"]
    corpus_out = data_cfg["corpus_out"]

    rows = read_jsonl(corpus_in)

    seed = int(noise_cfg.get("seed", 42))
    rng = random.Random(seed)
    chunk_fraction = float(noise_cfg.get("chunk_fraction", 1.0))
    edits_per_100 = float(noise_cfg.get("edits_per_100_chars", 2.5))
    preserve_ws = bool(noise_cfg.get("preserve_whitespace", True))
    op_weights = noise_cfg.get("op_weights", None)

    out_rows: List[Dict[str, Any]] = []
    for r in tqdm(rows, desc="Applying noise to chunks"):
        rr = dict(r)
        if rng.random() < chunk_fraction:
            cfg_i = NoiseConfig(
                seed=rng.randrange(0, 2**31 - 1),
                edits_per_100_chars=edits_per_100,
                op_weights=op_weights,
                preserve_whitespace=preserve_ws,
            )
            rr["text"] = perturb_text(rr["text"], cfg_i)
            rr["is_noisy"] = True
        else:
            rr["is_noisy"] = False
        out_rows.append(rr)

    write_jsonl(corpus_out, out_rows)
    num_noisy = sum(1 for r in out_rows if r.get("is_noisy"))
    print(f"Wrote noisy corpus to {corpus_out}")
    print(f"Chunks total: {len(out_rows)} | Noisy: {num_noisy} ({num_noisy / max(1, len(out_rows)):.3f})")


if __name__ == "__main__":
    main()
