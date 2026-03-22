# RAG Denoising Thesis Starter (Day 1)

This repo provides a minimal, reproducible baseline RAG pipeline (E0) using SQuAD contexts as the knowledge base.

## Quickstart

### 1) Create environment and install dependencies
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

### 2) Build corpus from SQuAD contexts
```bash
python -m src.data.squad_build_corpus --config configs/e0_clean.yaml
```

### 3) Build FAISS index
```bash
python -m src.indexing.build_faiss --config configs/e0_clean.yaml
```

### 4) Run baseline experiment (E0)
```bash
python -m src.eval.run_experiment --config configs/e0_clean.yaml --max_questions 300
```

Outputs will be written to `outputs/`.

## Experiment IDs (Ablations)

- E0: baseline clean
- E1: baseline with noisy corpus (later)
- E2: query normalization only (later)
- E3: denoiser always-on (later)
- E4: denoiser gated (later)
- E5: dual-view retrieval (later)
- E6: dual-view + reranker (later)
- E7: full system (later)

Day 1 implements E0.

---

## Day 2 (E1): Noisy Knowledge Base

E1 simulates low-level perturbations (typos / character edits) in the knowledge base text.
This approximates GARAG-style corpus corruption and lets you measure how retrieval and QA degrade.

### Step 1 — Generate a noisy corpus from the clean chunks
```bash
python -m src.data.make_noisy_corpus --config configs/e1_noisy_corpus.yaml
```

### Step 2 — Build an index over the noisy corpus
```bash
python -m src.indexing.build_faiss --config configs/e1_noisy_corpus.yaml
```

### Step 3 — Run evaluation (E1)
```bash
python -m src.eval.run_experiment --config configs/e1_noisy_corpus.yaml --max_questions 300
```

Outputs:
- outputs/corpora/squad_chunks_noisy.jsonl
- outputs/indexes/e1_noisy.faiss
- outputs/indexes/e1_noisy_meta.jsonl
- outputs/runs/e1_noisy_corpus/metrics.json
- outputs/runs/e1_noisy_corpus/predictions.jsonl
