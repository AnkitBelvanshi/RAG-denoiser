import argparse
import hashlib
import os
import time
from typing import Any, Dict, List

from datasets import load_dataset
from tqdm import tqdm

from src.eval.metrics import squad_em_f1, answer_span_hit
from src.rag.denoise import ChunkDenoiser, DenoiseConfig
from src.rag.generate import GenerationConfig, HFGenerator
from src.rag.noise_gate import select_noisy_indices
from src.rag.pipeline import pack_context
from src.rag.rerank import CrossEncoderReranker, RerankConfig
from src.rag.retrieve import DualViewRetriever, FaissRetriever, RetrievedChunk
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

    # -------------------------
    # Build retriever
    # -------------------------
    retr_cfg = cfg["retrieval"]
    mode = retr_cfg.get("mode", "single")

    if mode == "dual_view":
        retriever = DualViewRetriever(
            raw_index_path=retr_cfg["raw_index_out"],
            raw_meta_path=retr_cfg["raw_meta_out"],
            norm_index_path=retr_cfg["norm_index_out"],
            norm_meta_path=retr_cfg["norm_meta_out"],
            embed_model=retr_cfg["embed_model"],
            top_k=int(retr_cfg.get("top_k", 5)),
            per_index_k=int(retr_cfg.get("per_index_k", retr_cfg.get("top_k", 5))),
        )
    else:
        retriever = FaissRetriever(
            index_path=retr_cfg["index_out"],
            meta_path=retr_cfg.get("meta_out", retr_cfg["index_out"] + ".meta.jsonl"),
            embed_model=retr_cfg["embed_model"],
            top_k=int(retr_cfg.get("top_k", 5)),
        )

    # -------------------------
    # Build generator
    # -------------------------
    gen_cfg = cfg["generation"]
    generator = HFGenerator(
        GenerationConfig(
            model=gen_cfg["model"],
            max_new_tokens=int(gen_cfg.get("max_new_tokens", 32)),
            temperature=float(gen_cfg.get("temperature", 0.0)),
            do_sample=bool(gen_cfg.get("do_sample", False)),
        )
    )

    # -------------------------
    # Optional reranker (E6)
    # -------------------------
    rerank_cfg = cfg.get("rerank", {})
    use_rerank = bool(rerank_cfg.get("enabled", False))

    reranker = None
    if use_rerank:
        reranker = CrossEncoderReranker(
            RerankConfig(
                model_name=rerank_cfg["model_name"],
                batch_size=int(rerank_cfg.get("batch_size", 16)),
                device=rerank_cfg.get("device", None),
            )
        )

    # candidate_k: how many chunks to retrieve BEFORE reranking
    candidate_k = int(rerank_cfg.get("candidate_k", retr_cfg.get("top_k", 5)))
    # top_n: how many chunks to keep AFTER reranking (and pass to LLM)
    top_n = int(rerank_cfg.get("top_n", retr_cfg.get("top_k", 5)))

    # -------------------------
    # Optional denoiser (E3/E4/E7)
    # -------------------------
    den_cfg = cfg.get("denoise", {})
    use_denoise = bool(den_cfg.get("enabled", False))

    denoiser = None
    gate_enabled = False
    gate_threshold = 0.0
    max_denoise_chunks = 0

    if use_denoise:
        if "adapter_path" not in den_cfg:
            raise ValueError("denoise.enabled is true but denoise.adapter_path is missing in config")

        denoiser = ChunkDenoiser(
            DenoiseConfig(
                base_model=den_cfg.get("base_model", "google/flan-t5-small"),
                adapter_path=den_cfg["adapter_path"],
                batch_size=int(den_cfg.get("batch_size", 8)),
                device=den_cfg.get("device", None),
                max_new_tokens=int(den_cfg.get("max_new_tokens", 64)),
            )
        )

        gate_enabled = bool(den_cfg.get("gated", True))
        gate_threshold = float(den_cfg.get("gate_threshold", 0.02))
        max_denoise_chunks = int(den_cfg.get("max_chunks", 2))

    preds_rows: List[Dict[str, Any]] = []
    ems: List[float] = []
    f1s: List[float] = []
    hit: List[int] = []
    latencies: List[float] = []
    span_hit_raw: List[int] = []
    span_hit_post: List[int] = []

    # denoise usage stats (thesis-friendly)
    denoise_queries = 0
    denoise_chunks_total = 0

    compute_hit = bool(cfg["eval"].get("compute_retrieval_hit", True))

    for ex in tqdm(ds, desc="Running QA"):
        qid = ex["id"]
        question = ex["question"]
        answers = ex["answers"]["text"]
        gold_doc_id = sha1_text(ex["context"])

        t0 = time.time()

        # 1) retrieve candidates
        candidates = retriever.retrieve(question, top_k=candidate_k)

        # 2) rerank (optional)
        if reranker is not None:
            top_chunks = reranker.rerank(question, candidates, top_n=top_n)
        else:
            top_chunks = candidates[:top_n]

        # Answer-span hit BEFORE denoise (diagnostic)
        raw_texts = [c.text for c in top_chunks]
        raw_span_h = answer_span_hit(raw_texts, answers)
        span_hit_raw.append(raw_span_h)

        denoised_indices: List[int] = []

        # 3) denoise (optional, gated)
        if denoiser is not None and top_chunks:
            if gate_enabled:
                denoised_indices, _scores = select_noisy_indices(
                    raw_texts, threshold=gate_threshold, max_chunks=max_denoise_chunks
                )
            else:
                denoised_indices = list(range(len(raw_texts)))

            if denoised_indices:
                denoise_queries += 1
                denoise_chunks_total += len(denoised_indices)

                to_fix = [raw_texts[i] for i in denoised_indices]
                fixed = denoiser.denoise_batch(to_fix)

                new_texts = list(raw_texts)
                for j, idx in enumerate(denoised_indices):
                    new_texts[idx] = fixed[j]

                top_chunks = [
                    RetrievedChunk(
                        score=c.score,
                        chunk_id=c.chunk_id,
                        doc_id=c.doc_id,
                        title=c.title,
                        text=new_texts[i],
                    )
                    for i, c in enumerate(top_chunks)
                ]

        # 4) generate from final context
        context = pack_context(top_chunks, max_chars=2000)
        pred = generator.generate(question, context)

        dt = time.time() - t0

        # QA metrics
        em, f1 = squad_em_f1(pred, answers)
        ems.append(em)
        f1s.append(f1)
        latencies.append(dt)

        # Answer-span hit AFTER denoise (what the LLM saw)
        post_texts = [c.text for c in top_chunks]
        span_h = answer_span_hit(post_texts, answers)
        span_hit_post.append(span_h)

        # Doc-level hit@k (coarse)
        if compute_hit:
            is_hit = 0
            for ch in top_chunks:
                if ch.doc_id == gold_doc_id:
                    is_hit = 1
                    break
            hit.append(is_hit)

        preds_rows.append(
            {
                "id": qid,
                "question": question,
                "prediction": pred,
                "answers": answers,
                "em": em,
                "f1": f1,
                "latency_sec": dt,
                "candidate_k": candidate_k,
                "top_n": top_n,
                "retrieved": [
                    {"score": c.score, "chunk_id": c.chunk_id, "doc_id": c.doc_id, "title": c.title}
                    for c in top_chunks
                ],
                "answer_span_hit_raw": raw_span_h,
                "answer_span_hit": span_h,
                "denoised_indices": denoised_indices,
                "rerank_enabled": use_rerank,
                "denoise_enabled": use_denoise,
                "denoise_gate_enabled": gate_enabled if use_denoise else False,
                "denoise_gate_threshold": gate_threshold if use_denoise else None,
                "denoise_max_chunks": max_denoise_chunks if use_denoise else None,
            }
        )

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
        "answer_span_hit_rate": float(sum(span_hit_post) / max(1, len(span_hit_post))) if span_hit_post else None,
        "answer_span_hit_rate_raw": float(sum(span_hit_raw) / max(1, len(span_hit_raw))) if span_hit_raw else None,
        "latency_p50_sec": percentile(latencies, 50),
        "latency_p95_sec": percentile(latencies, 95),
        "config_path": args.config,
        "seed": seed,
        "rerank_enabled": use_rerank,
        "candidate_k": candidate_k,
        "top_n": top_n,
        "denoise_enabled": use_denoise,
        "denoise_gate_enabled": gate_enabled if use_denoise else False,
        "denoise_gate_threshold": gate_threshold if use_denoise else None,
        "denoise_max_chunks": max_denoise_chunks if use_denoise else None,
        "denoise_query_fraction": (float(denoise_queries) / max(1, len(ds))) if use_denoise else None,
        "denoise_avg_chunks_per_query": (float(denoise_chunks_total) / max(1, len(ds))) if use_denoise else None,
    }

    write_json(os.path.join(run_dir, "metrics.json"), metrics)
    if bool(cfg["eval"].get("save_predictions", True)):
        write_jsonl(os.path.join(run_dir, "predictions.jsonl"), preds_rows)

    print("Done.")
    print(metrics)


if __name__ == "__main__":
    main()