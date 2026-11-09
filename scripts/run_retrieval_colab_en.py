"""Colab GPU Retrieval Runner for English Terminology Dose-Response Pilot (Step E3).

Executes:
  - BM25
  - Dense BGE-M3 (BAAI/bge-m3) with Google Drive embedding cache
  - Hybrid Reciprocal Rank Fusion (hybrid_rrf, k=60)
  - Optional cross-encoder reranker (bge-reranker-v2-m3) on top 50 if memory allows
  - Exports full rank tables (CSV & Parquet) and SHA-256 manifest

EXPLORATORY - agent-proxy labels, human verification pending
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pilot.evaluate import exact_binomial_interval as clopper_pearson
from src.pilot.retrievers import BM25, DenseBgeM3, HybridRRF

BANNER = "EXPLORATORY - agent-proxy labels, human verification pending"


def get_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def check_git_cleanliness() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        dirty = bool(status)
        return {"commit": commit, "dirty": dirty, "status": status}
    except Exception as e:
        return {"commit": "unknown", "dirty": True, "error": str(e)}


def run_en_retrieval(drive_cache_dir: Path | None = None) -> None:
    print(f"[{BANNER}]", flush=True)
    print("=== English Terminology Dose-Response Colab GPU Runner ===", flush=True)

    git_info = check_git_cleanliness()
    print(f"Git commit: {git_info['commit']} (dirty: {git_info['dirty']})", flush=True)

    # 1. Load Corpus
    sources_path = ROOT / "data/en/sources_300.jsonl"
    distractors_path = ROOT / "data/en/distractors_5000.jsonl"
    queries_path = ROOT / "results/real/en/queries_test_en.jsonl"

    if not sources_path.exists():
        raise FileNotFoundError(f"Missing {sources_path}")
    if not distractors_path.exists():
        raise FileNotFoundError(f"Missing {distractors_path}")
    if not queries_path.exists():
        raise FileNotFoundError(f"Missing {queries_path}")

    sources = [json.loads(line) for line in open(sources_path, encoding="utf-8")]
    distractors = [json.loads(line) for line in open(distractors_path, encoding="utf-8")]
    corpus = sources + distractors
    corpus_dict = {p["paper_id"]: (p["title"] + " " + p["abstract"]) for p in corpus}
    corpus_size = len(corpus_dict)
    print(f"Loaded corpus: {corpus_size} documents ({len(sources)} sources, {len(distractors)} distractors)", flush=True)

    # 2. Load Queries
    queries = [json.loads(line) for line in open(queries_path, encoding="utf-8")]
    print(f"Loaded queries: {len(queries)} accepted queries", flush=True)

    # 3. Setup retrievers
    # A. BM25
    t0 = time.time()
    print("\n--- Indexing BM25 ---", flush=True)
    bm25 = BM25(corpus_dict)
    print(f"BM25 ready in {time.time()-t0:.2f}s", flush=True)

    # B. Dense BGE-M3
    dense_cache = drive_cache_dir if drive_cache_dir else (ROOT / "data/cache/embeddings")
    dense_cache.mkdir(parents=True, exist_ok=True)

    t1 = time.time()
    print(f"\n--- Initializing Dense BGE-M3 (cache dir: {dense_cache}) ---", flush=True)
    dense_index = DenseBgeM3(
        views={"native": corpus_dict},
        cache_dir=dense_cache,
        max_length=1024,
        batch_size=16,
    )
    dense = dense_index.for_view("native")
    print(f"Dense BGE-M3 ready in {time.time()-t1:.2f}s", flush=True)

    # C. Hybrid RRF
    hybrid = HybridRRF(bm25, dense, k=60)

    # 4. Optional Cross-Encoder Reranker
    reranker = None
    reranker_status = "NOT DONE (not attempted)"
    try:
        from FlagEmbedding import FlagReranker
        print("\n--- Attempting to load BAAI/bge-reranker-v2-m3 ---", flush=True)
        reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
        print("bge-reranker-v2-m3 loaded successfully!", flush=True)
        reranker_status = "LOADED"
    except Exception as e:
        reranker_status = f"NOT DONE (bge-reranker-v2-m3 could not be loaded: {e})"
        print(f"{reranker_status}; marking reranker as NOT DONE per rules.", flush=True)
        reranker = None

    retrievers = {
        "bm25": bm25,
        "dense_bge_m3": dense,
        "hybrid_rrf": hybrid,
    }

    if reranker is not None:
        class CrossEncoderReranker:
            def __init__(self, base_ranker: Any, reranker_model: Any, corpus: dict[str, str], top_k: int = 50):
                self.base_ranker = base_ranker
                self.reranker = reranker_model
                self.corpus = corpus
                self.top_k = top_k
                self.doc_ids = base_ranker.doc_ids

            def rank(self, query: str, candidates: set[str] | None = None) -> list[tuple[str, float]]:
                base_ranked = self.base_ranker.rank(query, candidates)
                to_rerank = base_ranked[:self.top_k]
                remainder = base_ranked[self.top_k:]
                if not to_rerank:
                    return base_ranked
                pairs = [[query, self.corpus.get(did, "")] for did, _ in to_rerank]
                scores = self.reranker.compute_score(pairs, normalize=True)
                if isinstance(scores, (float, int)):
                    scores = [scores]
                reranked = [(to_rerank[i][0], float(scores[i])) for i in range(len(to_rerank))]
                reranked.sort(key=lambda item: (-item[1], item[0]))
                return reranked + remainder

        retrievers["reranker_bge_v2"] = CrossEncoderReranker(hybrid, reranker, corpus_dict, top_k=50)

    # 5. Execute Retrieval & Export Rankings
    print("\n--- Running retrieval across all queries and models ---", flush=True)
    export_rows = []
    levels = ["R100", "R50", "R25", "R0"]
    metrics_by_model_level: dict[str, dict[str, list[int]]] = {
        m: {lvl: [] for lvl in levels} for m in retrievers
    }

    start_ret = time.time()
    for q_idx, q in enumerate(queries, 1):
        sid = q["source_id"]
        lvl = q["level"]
        q_text = q["query_text"]
        qid = f"{sid}_{lvl}_{q_idx}"

        for m_name, model in retrievers.items():
            ranked = model.rank(q_text)
            doc_ids = [did for did, _ in ranked]
            source_rank = doc_ids.index(sid) + 1 if sid in doc_ids else 9999
            metrics_by_model_level[m_name][lvl].append(source_rank)

            top200_ids = doc_ids[:200]
            top200_scores = [round(float(s), 4) for _, s in ranked[:200]]

            export_rows.append({
                "banner": BANNER,
                "query_id": qid,
                "source_id": sid,
                "level": lvl,
                "retriever": m_name,
                "corpus_size": corpus_size,
                "source_rank": source_rank,
                "top200_doc_ids": json.dumps(top200_ids),
                "top200_scores": json.dumps(top200_scores),
            })

        if q_idx % 25 == 0 or q_idx == len(queries):
            print(f"  Processed {q_idx}/{len(queries)} queries ({time.time()-start_ret:.1f}s)", flush=True)

    # 6. Save Full Rank Exports (CSV & Parquet)
    out_dir = ROOT / "results/real/en"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_export = pd.DataFrame(export_rows)
    csv_path = out_dir / "rank_export_en.csv"
    parquet_path = out_dir / "rank_export_en.parquet"

    df_export.to_csv(csv_path, index=False, encoding="utf-8")
    df_export.to_parquet(parquet_path, index=False)
    print(f"\nWrote CSV: {csv_path} ({csv_path.stat().st_size} bytes)", flush=True)
    print(f"Wrote Parquet: {parquet_path} ({parquet_path.stat().st_size} bytes)", flush=True)

    # 7. Compute Summary Metrics Table
    summary_rows = []
    print("\n=== Retrieval Summary Metrics Table (5,270 docs) ===")
    print(f"{'Retriever':<15} | {'Level':<6} | {'N':<4} | {'Recall@10 [95% CI]':<24} | {'Recall@50':<10} | {'Recall@100':<10} | {'MRR':<8}")
    print("-" * 88)

    for m_name in retrievers:
        for lvl in levels:
            ranks = metrics_by_model_level[m_name][lvl]
            n = len(ranks)
            if n == 0:
                continue
            h10 = sum(1 for r in ranks if r <= 10)
            h50 = sum(1 for r in ranks if r <= 50)
            h100 = sum(1 for r in ranks if r <= 100)
            mrr = sum(1.0 / r for r in ranks if r > 0) / n

            r10 = h10 / n
            r50 = h50 / n
            r100 = h100 / n
            ci10 = clopper_pearson(h10, n, 0.95)

            ci_str = f"[{ci10[0]:.3f}, {ci10[1]:.3f}]"
            print(f"{m_name:<15} | {lvl:<6} | {n:<4} | {r10:.3f} {ci_str:<16} | {r50:.3f}      | {r100:.3f}       | {mrr:.4f}")

            summary_rows.append({
                "banner": BANNER,
                "retriever": m_name,
                "level": lvl,
                "n_queries": n,
                "recall_10": round(r10, 4),
                "recall_10_ci95_low": round(ci10[0], 4),
                "recall_10_ci95_high": round(ci10[1], 4),
                "recall_50": round(r50, 4),
                "recall_100": round(r100, 4),
                "mrr": round(mrr, 4),
            })

    metrics_csv = out_dir / "retrieval_en_metrics.csv"
    pd.DataFrame(summary_rows).to_csv(metrics_csv, index=False, encoding="utf-8")
    print(f"\nWrote metrics CSV: {metrics_csv}", flush=True)

    # 8. SHA-256 Manifest
    manifest_en = {
        "banner": BANNER,
        "study": "English Terminology Dose-Response Pilot",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_info["commit"],
        "git_dirty": git_info["dirty"],
        "corpus_size": corpus_size,
        "n_queries": len(queries),
        "reranker_status": reranker_status,
        "files": {
            "rank_export_en_csv": {
                "sha256": get_sha256(csv_path),
                "size_bytes": csv_path.stat().st_size,
            },
            "rank_export_en_parquet": {
                "sha256": get_sha256(parquet_path),
                "size_bytes": parquet_path.stat().st_size,
            },
            "retrieval_en_metrics_csv": {
                "sha256": get_sha256(metrics_csv),
                "size_bytes": metrics_csv.stat().st_size,
            },
        },
    }
    manifest_path = out_dir / "rank_export_en_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_en, f, indent=2, ensure_ascii=False)
    print(f"Wrote Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-cache", type=Path, default=None, help="Google Drive cache path for embeddings")
    args = parser.parse_args()
    run_en_retrieval(drive_cache_dir=args.drive_cache)

