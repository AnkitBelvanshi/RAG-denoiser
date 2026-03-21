import argparse
import hashlib
import os
import time
from typing import Any, Dict, List

from datasets import load_dataset
from tqdm import tqdm

from src.eval.metrics import squad_em_f1
from src.rag.generate import GenerationConfig, HFGenerator
from src.rag.pipeline import RAGPipeline
from src.rag.retrieve import FaissRetriever
from src.utils.io import ensure_dir, load_yaml, write_json, write_jsonl
from src.utils.seed import set_seed


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument("--max_questions", type=int, default=None, help="Override number of questions to run.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("eval", {}).get("seed", 42))
    set_seed(seed)

    run_dir = cfg["eval"]["run_dir"]
    ensure_dir(run_dir)

    # Load dataset
    split = cfg["data"]["dataset_split"]
    ds = load_dataset(cfg["data"]["dataset_name"], split=split)
    max_q = args.max_questions or int(cfg["data"].get("max_questions_default", 300))
    ds = ds.select(range(min(len(ds), max_q)))

    # Build components
    retr_cfg = cfg["retrieval"]
    retriever = FaissRetriever(
        index_path=retr_cfg["index_out"],
        meta_path=retr_cfg.get("meta_out", retr_cfg["index_out"] + ".meta.jsonl"),
        embed_model=retr_cfg["embed_model"],
        top_k=int(retr_cfg.get("top_k", 5)),
    )

    gen_cfg = cfg["generation"]
    generator = HFGenerator(GenerationConfig(
        model=gen_cfg["model"],
        max_new_tokens=int(gen_cfg.get("max_new_tokens", 32)),
        temperature=float(gen_cfg.get("temperature", 0.0)),
        do_sample=bool(gen_cfg.get("do_sample", False)),
    ))

    pipe = RAGPipeline(retriever=retriever, generator=generator, context_max_chars=2000)

    preds_rows: List[Dict[str, Any]] = []
    ems: List[float] = []
    f1s: List[float] = []
    hit: List[int] = []
    latencies: List[float] = []

    compute_hit = bool(cfg["eval"].get("compute_retrieval_hit", True))

    for ex in tqdm(ds, desc="Running QA"):
        qid = ex["id"]
        question = ex["question"]
        answers = ex["answers"]["text"]
        gold_doc_id = sha1_text(ex["context"])

        t0 = time.time()
        result = pipe.answer(question)
        dt = time.time() - t0

        pred = result.answer
        em, f1 = squad_em_f1(pred, answers)

        ems.append(em)
        f1s.append(f1)
        latencies.append(dt)

        if compute_hit:
            is_hit = 0
            for ch in result.retrieved:
                if ch.doc_id == gold_doc_id:
                    is_hit = 1
                    break
            hit.append(is_hit)

        preds_rows.append({
            "id": qid,
            "question": question,
            "prediction": pred,
            "answers": answers,
            "em": em,
            "f1": f1,
            "latency_sec": dt,
            "retrieved": [
                {"score": c.score, "chunk_id": c.chunk_id, "doc_id": c.doc_id, "title": c.title}
                for c in result.retrieved
            ],
        })

    def percentile(xs: List[float], p: float) -> float:
        if not xs:
            return 0.0
        xs2 = sorted(xs)
        k = int(round((p / 100.0) * (len(xs2) - 1)))
        k = max(0, min(k, len(xs2) - 1))
        return float(xs2[k])

    metrics = {
        "experiment_id": cfg.get("experiment_id", ""),
        "timestamp": now_iso(),
        "num_questions": len(ds),
        "EM": float(sum(ems) / max(1, len(ems))),
        "F1": float(sum(f1s) / max(1, len(f1s))),
        "retrieval_hit_rate": float(sum(hit) / max(1, len(hit))) if hit else None,
        "latency_p50_sec": percentile(latencies, 50),
        "latency_p95_sec": percentile(latencies, 95),
        "config_path": args.config,
        "seed": seed,
    }

    write_json(os.path.join(run_dir, "metrics.json"), metrics)
    if bool(cfg["eval"].get("save_predictions", True)):
        write_jsonl(os.path.join(run_dir, "predictions.jsonl"), preds_rows)

    print("Done.")
    print(metrics)


if __name__ == "__main__":
    main()
