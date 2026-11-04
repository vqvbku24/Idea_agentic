> [!IMPORTANT]
> **EXPLORATORY - agent-proxy labels, human verification pending**

# English Terminology Dose-Response Pilot: Pre-Flight Inventory (Step E0)

This inventory confirms the presence, integrity, and operational readiness of all foundational components required for the English Terminology Dose-Response Pilot.

## 1. Corpus Data Artefacts

| Artefact | Path | Size (Bytes) | SHA-256 Checksum | Description / Content |
|---|---|---|---|---|
| **ACL Anthology Bulk Archive** | `data/cache/acl-anthology-master.zip` | 65,486,306 | `3597e23df041b603aa686ce9e703ddab09295610e254846927c4149d7ea23954` | Official bulk repository containing 1,772 XML volume metadata files (802 XML files for target years 2018–2024). Authoritative English titles, abstracts, authors, and venues. |

## 2. Model & Pipeline Codebase

| Component | Path | Size (Bytes) | SHA-256 Checksum | Capabilities & Configuration |
|---|---|---|---|---|
| **LLM Interface** | `src/pilot/llm.py` | 12,001 | `1bfaff3988af1fd3a985a09c5a14cbe9419ff939cd047b57b999637b876683aa` | Disk cache active (`cache/llm/`), SHA-256 caching key, deterministic `temperature=0.0`, call counter tracking (`call_count.json`), hard cap enforcement (`LLM_MAX_CALLS`), multi-key rotation and backoff. Configured models: Generator `qwen/qwen3.8-27b`, Judge `openai/gpt-oss-120b`. |
| **Retrievers** | `src/pilot/retrievers.py` | 11,781 | `f31246e3fc1cf5aed95f16f206fc3ef400cecc6ef1ee7efd65a5d086b15136a6` | Okapi BM25 (`BM25Ranker`), Dense BGE-M3 (`DenseBgeM3Ranker`), Hybrid Reciprocal Rank Fusion (`HybridRRFRanker`, $k=60$). |
| **Evaluator** | `src/pilot/evaluate.py` | 8,981 | `e45111724e18a2adc409722a6ca2291c5574c027fbfe4dc6161e5ef34476f872` | Exact Clopper-Pearson binomial intervals (`exact_binomial_interval`), paired bootstrap confidence intervals (`bootstrap_ci`), per-query ranks and Recalls. |
| **Colab Kit Script** | `scripts/run_retrieval_colab_full.py` | 13,801 | `097c274715eb94266125e684f6f7d4df7e861e78edb35c7b59717ac022221313` | Standalone runner for Google Colab GPU environment with embedding caching, SHA-256 manifest verification, and rank table export (`.csv` and `.parquet`). |
| **Colab Notebook** | `agentic_v2.ipynb` | 12,203 | `02acb2d4eb30f0813ddd14d704a4a90829862ccfa23b4c160a27b251c6e4e284` | Automated GPU runner for Google Drive mounting, Git checkout validation, retrieval execution, and artefact sync. |

## 3. Operational Readiness Checklist

- [x] **Bulk Metadata Cached**: `data/cache/acl-anthology-master.zip` verified locally (no external download needed).
- [x] **LLM Caching & Pacing**: Disk cache and Groq provider configured and smoke-tested.
- [x] **Local BM25**: Operational locally via `rank-bm25`.
- [x] **Colab Infrastructure**: Validated through successful completion and reproduction of the French follow-up study.
- [x] **Governance Rules**: Global rules G1–G9 active and strictly enforced.
