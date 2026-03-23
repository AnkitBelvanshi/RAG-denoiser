# src/eval/summarize_runs.py
import argparse
import json
import os
from typing import Dict, List

def load_metrics(run_dir: str) -> Dict:
    path = os.path.join(run_dir, "metrics.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def pick(d: Dict, key: str):
    return d.get(key, None)

def fmt(x):
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="List of run dirs (each contains metrics.json)")
    ap.add_argument("--out_dir", default="outputs/summary", help="Where to write summary files")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows: List[Dict] = []
    for run_dir in args.runs:
        m = load_metrics(run_dir)
        rows.append({
            "experiment_id": pick(m, "experiment_id") or os.path.basename(run_dir),
            "num_questions": pick(m, "num_questions"),
            "EM": pick(m, "EM"),
            "F1": pick(m, "F1"),
            "retrieval_hit_rate": pick(m, "retrieval_hit_rate"),
            "latency_p50_sec": pick(m, "latency_p50_sec"),
            "latency_p95_sec": pick(m, "latency_p95_sec"),
            "run_dir": run_dir,
        })

    # Write CSV
    csv_path = os.path.join(args.out_dir, "results.csv")
    cols = ["experiment_id","num_questions","EM","F1","retrieval_hit_rate","latency_p50_sec","latency_p95_sec","run_dir"]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(fmt(r.get(c)) for c in cols) + "\n")

    # Write Markdown table
    md_path = os.path.join(args.out_dir, "results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Results Summary\n\n")
        f.write("| Experiment | #Q | EM | F1 | Hit@k | p50 latency (s) | p95 latency (s) |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['experiment_id']} | {r['num_questions']} | {fmt(r['EM'])} | {fmt(r['F1'])} | "
                f"{fmt(r['retrieval_hit_rate'])} | {fmt(r['latency_p50_sec'])} | {fmt(r['latency_p95_sec'])} |\n"
            )

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")

if __name__ == "__main__":
    main()