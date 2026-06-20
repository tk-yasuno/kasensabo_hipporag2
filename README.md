# RAG Benchmark — Japanese River & Sediment-Control Technical Standards

A reproducible benchmark comparing three RAG retrieval strategies on Japanese civil-engineering technical documents, evaluated by an AI-as-Judge scoring rubric.

## Overview

| Dimension | Details |
|---|---|
| **Document corpus** | 8 volumes of *Kasen·Dam·Sabo Technical Standards 2025* (河川砂防技術標準) |
| **Test set** | 200 QA pairs, sampled from 4,000 generated QA pairs (seed = 42) |
| **RAG strategies** | Naive RAG · Light RAG · HippoRAG2 (hierarchical coarse-to-fine) |
| **LLM backends** | Swallow-8B-LoRA-Q4 · ELYZA-JP-8B-LoRA-Q4 (served via Ollama) |
| **Judge model** | Qwen2.5-7B-Instruct (served via Ollama) |
| **Embedding model** | [`hotchpotch/static-embedding-japanese`](https://huggingface.co/hotchpotch/static-embedding-japanese) (1024-dim, IP similarity) |
| **GPU constraint** | 16 GB VRAM |

**6 conditions total** = 3 RAG types × 2 LLM models.

---

## Release Notes

### v0.1 (2026-06-21)

Initial release featuring:

- **Complete RAG benchmark pipeline** for Japanese technical documents
- **Three retrieval strategies**: Naive RAG, Light RAG, and HippoRAG2 (hierarchical coarse-to-fine)
- **Automated evaluation framework** with AI-as-Judge scoring (Qwen2.5-7B)
- **Experiment scripts**:
  - `04c_run_all.py`: Batch evaluation runner with configurable batch sizes
  - `04d_judge_only.py`: Re-run scoring without re-generating answers
  - Test scripts for Qwen GPU performance validation
- **Unsloth integration**: Compiled cache for various trainers (SFT, DPO, ORPO, etc.)
- **PowerShell automation**: `04b_run_all.ps1` for sequential condition execution
- **Configuration management**: `env_config.json` for Ollama model resolution
- **Comprehensive documentation**: Setup guides, lessons learned, and technical notes

**Key capabilities**:
- Supports 16 GB VRAM GPU constraint with model swapping
- Japanese text processing with custom tokenization
- Reproducible test set generation (seed-based sampling)
- Detailed per-question and aggregate metrics
- Visualization tools for score distribution and latency analysis

---

## RAG Strategies

### Naive RAG
All chunks are embedded into a single flat FAISS (`IndexFlatIP`) space. Query is encoded with the same model and the top-k chunks by inner-product similarity are returned.  
Role: **Baseline**.

### Light RAG
Hybrid retrieval: BM25 keyword score (char + bigram tokenizer) and dense embedding score are each normalized to [0, 1], then linearly fused with α = 0.5. Top-50 BM25 candidates are first selected, then re-scored and merged with the dense results.

### HippoRAG2 (Hierarchical Coarse-to-Fine)
Exploits the natural Volume → Chapter → Section hierarchy of the technical standards without building a knowledge graph:

1. **Level 1 (Volume selection)** — Query vector vs. volume representative vectors (mean-pooled chunk embeddings per volume). Top-2 volumes selected.  
2. **Level 2 (Chapter selection)** — Query vector vs. chapter representative vectors within the selected volumes. Top-3 chapters selected.  
3. **Level 3 (Chunk retrieval)** — Dense search restricted to the candidate chunk set from selected chapters. Returns top-k chunks.

Fallback: if candidate pool < top-k, full Naive search is used.

---

## Evaluation Rubric

Qwen2.5-7B-Instruct assigns a score 0–3 to each generated answer:

| Score | Criterion |
|---|---|
| **3** | Technically accurate and specific; cites standard name, section number, or key technical concept |
| **2** | Mostly correct but lacks specificity or citation |
| **1** | Partially correct; contains a significant error or omission |
| **0** | Incorrect, empty, or off-topic |

**Metrics reported**: average Judge score, perfect-score rate (score = 3), retrieval latency, generation latency, score distribution (0/1/2/3).

---

## Repository Structure

```
kasensabo_hipporag2/
├── data/
│   ├── kasen-dam-sabo_Train_set/   # 8 source Markdown volumes
│   ├── generated_QA/               # Generated QA pairs (JSONL)
│   └── rag/                        # Legacy indices (multilingual-e5-large)
│
├── experiments/
│   ├── requirements.txt            # Python dependencies
│   ├── 00_check_env.py             # Pre-flight check (Ollama / GPU / libs)
│   ├── 01_build_indices.py         # Build FAISS + BM25 + hierarchy metadata
│   ├── 01b_build_hipporag2_index.py  # Build volume / chapter representative vectors
│   ├── 02_prepare_testset.py       # Sample 200-question test set (seed=42)
│   ├── 03_rag_retrievers.py        # NaiveRetriever / LightRetriever / HippoRAG2Retriever
│   ├── 04_eval_rag.py              # Single-condition evaluation pipeline
│   ├── 04b_run_all.ps1             # Run all 6 conditions sequentially (PowerShell)
│   ├── 05_aggregate_results.py     # Aggregate per-condition summaries → CSV / JSON
│   ├── 05b_plot_results.py         # Generate bar / latency / distribution plots
│   │
│   ├── env_config.json             # [generated] resolved Ollama model names
│   ├── testset_200.jsonl           # [generated] 200-question test set
│   ├── indices/                    # [generated] FAISS, BM25, hierarchy, HippoRAG2 vectors
│   └── results/                    # [generated] per-condition JSONL + summary + figures
│
└── models/                         # LoRA adapter weights (not tracked by git)
```

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) running at `http://localhost:11434` with the following models pulled:
  - A Swallow-8B LoRA Q4 model (e.g., `swallow8b-lora-n4000-v09-q4`)
  - An ELYZA-JP-8B LoRA Q4 model (e.g., `elyza8b-lora-n4000-q4`)
  - `qwen2.5:7b` or `qwen2.5:7b-instruct-q4_k_m` (for AI-as-Judge)
- GPU with 16 GB VRAM (CPU fallback is supported but slow)

### 2. Create a virtual environment and install dependencies

```powershell
python -m venv .venv-hipp
.\.venv-hipp\Scripts\pip install -r experiments/requirements.txt
```

### 3. Run the full pipeline

```powershell
# Activate the environment
.\.venv-hipp\Scripts\Activate.ps1

# Step 0 — verify environment (outputs env_config.json)
python experiments/00_check_env.py

# Step 1 — build search indices
python experiments/01_build_indices.py
python experiments/01b_build_hipporag2_index.py

# Step 2 — prepare test set (200 questions, seed=42)
python experiments/02_prepare_testset.py

# Step 3 — (optional) smoke-test retrievers
python experiments/03_rag_retrievers.py --test

# Step 4 — evaluate all 6 conditions
pwsh experiments/04b_run_all.ps1

# Step 5 — aggregate and visualize
python experiments/05_aggregate_results.py
python experiments/05b_plot_results.py --no-show
```

Results are written to `experiments/results/`. Figures are saved under `experiments/results/figures/`.

### Dry-run (10 questions only)

```powershell
pwsh experiments/04b_run_all.ps1 -DryRun
# or single condition:
python experiments/04_eval_rag.py --model swallow --rag naive --dry-run
```

### Skip AI-as-Judge (generation only)

```powershell
python experiments/04_eval_rag.py --model swallow --rag naive --no-judge
```

---

## Output Files

| File | Description |
|---|---|
| `experiments/indices/embeddings.npy` | Chunk embeddings (N × 1024, float32) |
| `experiments/indices/faiss.index` | FAISS IndexFlatIP |
| `experiments/indices/bm25.pkl` | BM25Okapi index |
| `experiments/indices/hierarchy.json` | Volume → Chapter → chunk_id tree |
| `experiments/indices/hipporag2_volumes.json` | Volume representative vectors |
| `experiments/indices/hipporag2_chapters.json` | Chapter representative vectors |
| `experiments/testset_200.jsonl` | 200 test QA pairs |
| `experiments/results/{model}_{rag}_results.jsonl` | Per-question details (retrieved chunks, scores, latency) |
| `experiments/results/{model}_{rag}_summary.json` | Condition-level metrics |
| `experiments/results/summary.csv` | All-conditions summary table (UTF-8 BOM, Excel-friendly) |
| `experiments/results/figures/*.png` | Comparison plots |

---

## Dependencies

```
rank-bm25>=0.2.2
faiss-cpu>=1.8.0
sentence-transformers>=3.0.0
httpx>=0.27.0
numpy>=1.26.0
tqdm>=4.66.0
pandas>=2.2.0
matplotlib>=3.9.0
japanize-matplotlib>=1.1.3
```

See [experiments/requirements.txt](experiments/requirements.txt) for the pinned list.

---

## Notes

- **Embedding model**: `hotchpotch/static-embedding-japanese` is downloaded automatically by `sentence-transformers` on first use (~450 MB).
- **Existing indices** under `data/rag/` use `intfloat/multilingual-e5-large` and are **not** used in this experiment. `01_build_indices.py` re-encodes from the source text.
- **LoRA adapters** under `models/` are not tracked by git (add to `.gitignore`).
- **VRAM management**: each Ollama request includes `keep_alive: "5m"`. The last request of each evaluation phase sends `keep_alive: "0"` to unload the model immediately and free VRAM before the next model loads.

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).
