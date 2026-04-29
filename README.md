# Lightweight Defenses Against Low-Level Perturbation Attacks in RAG

M.Tech Thesis — Lightweight denoising and retrieval hardening framework for robust Retrieval-Augmented Generation under typo/OCR noise.

## Problem

RAG systems are brittle to low-level textual perturbations (typos, character swaps, OCR artifacts) in their corpus. Even minor corruption can break retrieval and degrade answer quality. This project builds **modular, plug-and-play defenses** that any RAG pipeline can adopt.

## Key Results

| System | Low Noise (F1) | Med Noise (F1) | High Noise (F1) | p50 Latency |
|--------|---------------|----------------|-----------------|-------------|
| Baseline (no defense) | 0.5635 | 0.3244 | 0.1451 | 0.77s |
| Dual-View + Rerank + Gated Denoise | 0.5859 | 0.3761 | 0.2347 | 1.54s |

**Reranking recovers up to 61.7% F1** at high noise. The gated denoiser safely stays dormant for ASCII typo noise and activates only for OCR-style corruption, preventing the evidence destruction that always-on denoising causes (54% answer-span loss).

## Architecture

```
Query → Dual-View Retrieval → Cross-Encoder Reranking → Gated Denoising → Generator → Answer
         (raw + norm index)    (top 6 → top 3)          (selective)         (any LLM)
```

Each module is independent and optional:

| Module | What it does | Can be used alone? |
|--------|-------------|-------------------|
| **Dual-view retrieval** | Queries both raw and normalized FAISS indexes, merges results | Yes |
| **Cross-encoder reranker** | Reorders retrieved chunks by relevance | Yes |
| **Gated denoiser** | Scores chunks for noise, denoises only flagged ones | Yes |

## Project Structure

```
├── configs/                    # YAML configs for all experiments
│   ├── e0_clean.yaml          # Clean corpus baseline
│   ├── e1_*.yaml              # Noisy baselines (low/med/high)
│   ├── e4_*.yaml              # Gated denoise variants
│   ├── e6_*.yaml              # Dual-view + rerank
│   ├── e7_*.yaml              # Full system + ablations
│   ├── noisy_*_matched.yaml   # Matched noise corpus configs
│   └── build_*_v2_*.yaml      # FAISS index build configs
├── src/
│   ├── data/
│   │   ├── noise.py                    # Noise model (NoiseConfig, perturb_text)
│   │   ├── make_noisy_corpus.py        # Generate noisy corpora
│   │   ├── build_normalized_corpus.py  # Unicode normalization
│   │   ├── check_noise_stats.py        # Noise sanity checker
│   │   └── squad_build_corpus.py       # SQuAD → chunked JSONL
│   ├── indexing/
│   │   └── build_faiss.py     # FAISS index builder
│   ├── rag/
│   │   ├── pipeline.py        # RAG pipeline orchestration
│   │   ├── retrieve.py        # FAISS + dual-view retriever
│   │   ├── rerank.py          # Cross-encoder reranker
│   │   ├── noise_gate.py      # Heuristic noise scoring + gating
│   │   ├── denoise.py         # Seq2seq chunk denoiser (Flan-T5 + LoRA)
│   │   └── generate.py        # LLM generator wrapper
│   ├── eval/
│   │   ├── run_experiment.py  # Main experiment runner
│   │   ├── metrics.py         # EM, F1, Answer-Span Hit@k
│   │   └── summarize_runs.py  # Aggregate results to CSV/Markdown
│   └── utils/
│       ├── io.py              # YAML/JSONL helpers
│       └── seed.py            # Reproducibility
├── outputs/                   # (gitignored) corpora, indexes, runs
├── requirements.txt
└── README.md
```

## Setup

```bash
# Clone
git clone https://github.com/AnkitBelvanshi/RAG-denoiser.git
cd RAG-denoiser

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, CUDA-capable GPU recommended (tested on RTX 4060 8GB).

## Reproduce Results

### Step 1: Build clean corpus from SQuAD

```bash
python -m src.data.squad_build_corpus
```

### Step 2: Generate matched noisy corpora

```bash
python -m src.data.make_noisy_corpus --config configs/noisy_low_matched.yaml
python -m src.data.make_noisy_corpus --config configs/noisy_med_matched.yaml
python -m src.data.make_noisy_corpus --config configs/noisy_high_matched.yaml
```

### Step 3: Build normalized corpora

```bash
python -m src.data.build_normalized_corpus --inp outputs/corpora/squad_chunks_noisy_low_v2.jsonl --out outputs/corpora/squad_chunks_noisy_low_v2_norm.jsonl
python -m src.data.build_normalized_corpus --inp outputs/corpora/squad_chunks_noisy_med_v2.jsonl --out outputs/corpora/squad_chunks_noisy_med_v2_norm.jsonl
python -m src.data.build_normalized_corpus --inp outputs/corpora/squad_chunks_noisy_high_v2.jsonl --out outputs/corpora/squad_chunks_noisy_high_v2_norm.jsonl
```

### Step 4: Build FAISS indexes

```bash
# Raw indexes
python -m src.indexing.build_faiss --config configs/build_low_v2_raw.yaml
python -m src.indexing.build_faiss --config configs/build_med_v2_raw.yaml
python -m src.indexing.build_faiss --config configs/build_high_v2_raw.yaml

# Normalized indexes
python -m src.indexing.build_faiss --config configs/build_low_v2_norm.yaml
python -m src.indexing.build_faiss --config configs/build_med_v2_norm.yaml
python -m src.indexing.build_faiss --config configs/build_high_v2_norm.yaml
```

### Step 5: Train denoiser (LoRA adapter)

```bash
python -m src.denoise.train_denoiser
```

### Step 6: Run experiments

```bash
# Baselines
python -m src.eval.run_experiment --config configs/e1_low_v2.yaml
python -m src.eval.run_experiment --config configs/e1_med_v2.yaml
python -m src.eval.run_experiment --config configs/e1_high_v2.yaml

# Full system (dual-view + rerank + gated denoise)
python -m src.eval.run_experiment --config configs/e7_light_low_v2.yaml
python -m src.eval.run_experiment --config configs/e7_light_med_v2.yaml
python -m src.eval.run_experiment --config configs/e7_light_high_v2.yaml

# No-rerank ablation
python -m src.eval.run_experiment --config configs/e7_norerank_low_v2.yaml
python -m src.eval.run_experiment --config configs/e7_norerank_med_v2.yaml
python -m src.eval.run_experiment --config configs/e7_norerank_high_v2.yaml
```

### Step 7: Summarize

```bash
python -m src.eval.summarize_runs --runs outputs/runs/e1_noisy_low_v2 outputs/runs/e1_noisy_med_v2 outputs/runs/e1_noisy_high_v2 outputs/runs/e7_light_low_v2 outputs/runs/e7_light_med_v2 outputs/runs/e7_light_high_v2 outputs/runs/e7_norerank_low_v2 outputs/runs/e7_norerank_med_v2 outputs/runs/e7_norerank_high_v2 --out_dir outputs/summary_day12
```

## Plug-and-Play Usage

Add the gated denoiser to any existing RAG pipeline in ~10 lines:

```python
from src.rag.noise_gate import select_noisy_indices
from src.rag.denoise import ChunkDenoiser, DenoiseConfig

# Initialize once
denoiser = ChunkDenoiser(DenoiseConfig(
    base_model="google/flan-t5-small",
    adapter_path="outputs/models/denoiser_lora",
))

# After your retriever, before your LLM
texts = [chunk.text for chunk in retrieved_chunks]
noisy_ids, _ = select_noisy_indices(texts, threshold=0.05, max_chunks=1, percentile=90)

if noisy_ids:
    fixed = denoiser.denoise_batch([texts[i] for i in noisy_ids])
    for j, idx in enumerate(noisy_ids):
        retrieved_chunks[idx].text = fixed[j]

# Pass cleaned chunks to your LLM as usual
```

## Noise Model

Three matched severity levels, same seed (42), same operation weights:

| Level | edits_per_100_chars | Character Operations |
|-------|--------------------|--------------------|
| Low | 1.0 | swap (40%), delete (20%), insert (20%), keyboard substitute (20%) |
| Medium | 2.5 | same |
| High | 5.0 | same |

## Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| **EM** | Exact Match — binary, strict string match |
| **F1** | Token-level F1 — partial credit for overlapping tokens |
| **Hit@k** | Whether any retrieved chunk comes from the gold document |
| **SpanHit@k (raw)** | Whether retrieved chunks contain the answer string before denoising |
| **SpanHit@k (post)** | Whether chunks contain the answer string after denoising |
| **Latency p50/p95** | Median and 95th-percentile end-to-end time per query |

**Answer-Span Hit@k** is a contribution of this thesis. It revealed that always-on denoising destroys 54% of answer evidence (SpanHit drops from 0.76 to 0.35) — a critical failure invisible to EM/F1 alone.

## Key Findings

1. **Low-level noise severely degrades RAG:** F1 drops from 0.56 (low) → 0.32 (med) → 0.14 (high)
2. **Always-on denoising is harmful:** Destroys 54% of answer spans, reducing F1
3. **Gated denoiser is safe:** Correctly stays silent for ASCII typo noise, designed to activate for OCR artifacts
4. **Cross-encoder reranking is the primary defense:** +4% F1 (low), +16% (med), +62% (high noise)

## Models Used

| Component | Model | Size |
|-----------|-------|------|
| Embedder | `BAAI/bge-base-en-v1.5` | 110M |
| Reranker | `BAAI/bge-reranker-base` | 278M |
| Denoiser | `google/flan-t5-small` + LoRA | 77M + 0.8M |
| Generator | `google/flan-t5-base` | 248M |


## References

- Kim et al., EMNLP Findings 2024 — *Typos that Broke the RAG's Back* (GARAG)
- Lewis et al., 2020 — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
- Rajpurkar et al., 2016 — *SQuAD: 100,000+ Questions for Machine Comprehension of Text*
- Hu et al., 2022 — *LoRA: Low-Rank Adaptation of Large Language Models*

## License

MIT
