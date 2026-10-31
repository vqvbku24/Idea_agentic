from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pilot.corpus import Document, build_index_views
from pilot.retrievers import BM25
from pilot.evaluate import bootstrap_ci, exact_binomial_interval, auc

BANNER = "EXPLORATORY - agent-proxy labels, human verification pending"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_corpus(root: Path, distractor_count: int = 500) -> tuple[list[Document], dict[str, str]]:
    """Load sources, dev papers, and distractors. Returns documents and dict of doc_id -> title."""
    papers_raw = {row["doc_id"]: row for row in load_jsonl(root / "results/real/papers_x2.jsonl")}
    
    # 20 eligible sources
    with (root / "results/real/extraction_details.csv").open(newline="", encoding="utf-8") as f:
        eligible_sources = [
            row["paper_id"] for row in csv.DictReader(f)
            if row["eligible"].lower() == "true"
        ]

    sources = [
        Document(
            doc_id=pid,
            kind="source",
            lang="fr",
            title=papers_raw[pid]["title"],
            abstract_fr=papers_raw[pid].get("abstract_fr", ""),
            abstract_en=papers_raw[pid].get("abstract_en", ""),
            body_fr=papers_raw[pid].get("body_fr", ""),
            year=int(papers_raw[pid].get("year", 0)),
            keywords_en=papers_raw[pid].get("keywords_en", ""),
        )
        for pid in sorted(eligible_sources)
    ]

    # 100 French distractors
    fr_dist_raw = load_jsonl(root / "results/real/french_distractors_native_x2.jsonl")
    french_distractors = [
        Document(
            doc_id=row["doc_id"],
            kind="fr_distractor",
            lang="fr",
            title=row["title"],
            abstract_fr=row.get("abstract_fr", ""),
            abstract_en=row.get("abstract_en", ""),
            body_fr=row.get("body_fr", ""),
            year=int(row.get("year", 0)),
        )
        for row in fr_dist_raw
    ]

    # English Distractors (500 or 100)
    en_dist_raw = load_jsonl(root / "data/processed/english_distractors_metadata.jsonl")[:distractor_count]
    english_distractors = [
        Document(
            doc_id=row["paper_id"],
            kind="en_distractor",
            lang="en",
            title=row.get("title", ""),
            abstract_en=row.get("abstract_meta", ""),
            year=int(row.get("year", 0)),
        )
        for row in en_dist_raw
    ]

    docs = sources + french_distractors + english_distractors
    titles = {d.doc_id: d.title for d in docs}
    return docs, titles


def evaluate_bm25_suite(
    queries: list[dict],
    views: dict[str, dict[str, str]],
    corpus_size: int,
    titles: dict[str, str],
) -> tuple[list[dict], list[dict]]:
    """Evaluate BM25 across views and variants. Returns (metric_rows, failure_records)."""
    # Group queries by variant (only accepted queries)
    variants = ["V0", "V1", "V2", "V3", "H-agent"]
    target_views = ["native", "native_abstract", "en_abstract"]

    # Build BM25 indexes
    indexes = {view: BM25(views[view]) for view in target_views}

    metrics: list[dict] = []
    failures: list[dict] = []

    for view_name in target_views:
        index = indexes[view_name]
        for var in variants:
            # Filter accepted queries for this variant
            var_queries = [q for q in queries if q["variant"] == var and q["accepted"]]
            if not var_queries:
                continue

            per_paper_10: list[float] = []
            per_paper_5: list[float] = []
            per_paper_1: list[float] = []
            reciprocal_ranks: list[float] = []
            score_pairs: list[tuple[float, int]] = []

            for q in var_queries:
                sid = q["source_id"]
                q_text = q["query_text"]

                # Rank with source included
                ranked = index.rank(q_text)
                ranked_ids = [doc_id for doc_id, _ in ranked]
                score_dict = dict(ranked)

                if sid in ranked_ids:
                    rank = ranked_ids.index(sid) + 1
                    target_score = score_dict[sid]
                else:
                    rank = math.inf
                    target_score = 0.0

                hit1 = float(rank <= 1)
                hit5 = float(rank <= 5)
                hit10 = float(rank <= 10)

                per_paper_1.append(hit1)
                per_paper_5.append(hit5)
                per_paper_10.append(hit10)
                reciprocal_ranks.append(1.0 / rank if math.isfinite(rank) else 0.0)

                # Rank without source for absent-control AUC
                absent_candidates = set(index.doc_ids) - {sid}
                absent_ranked = index.rank(q_text, absent_candidates)
                absent_top = absent_ranked[0][1] if absent_ranked else 0.0
                present_top = ranked[0][1] if ranked else 0.0
                score_pairs.extend([(present_top, 1), (absent_top, 0)])

                # Record top-10 miss failure
                if rank > 10 and var in {"V1", "V2", "V3", "H-agent"}:
                    top3 = []
                    for did, sc in ranked[:3]:
                        top3.append({
                            "doc_id": did,
                            "title": titles.get(did, "Unknown"),
                            "score": round(sc, 4),
                        })
                    failures.append({
                        "corpus_size": corpus_size,
                        "view": view_name,
                        "variant": var,
                        "source_id": sid,
                        "source_title": titles.get(sid, "Unknown"),
                        "query_text": q_text,
                        "rank": rank if math.isfinite(rank) else ">corpus",
                        "top3": top3,
                    })

            # Calculate bootstrap and exact CIs
            n = len(var_queries)
            r1 = sum(per_paper_1) / n
            r5 = sum(per_paper_5) / n
            r10, r10_lo, r10_hi = bootstrap_ci(per_paper_10, resamples=1000, seed=13)
            exact_lo, exact_hi = exact_binomial_interval(int(sum(per_paper_10)), n)
            mrr, mrr_lo, mrr_hi = bootstrap_ci(reciprocal_ranks, resamples=1000, seed=13)
            abs_auc = auc(score_pairs)

            metrics.append({
                "banner": BANNER,
                "corpus_size": corpus_size,
                "view": view_name,
                "variant": var,
                "n_queries": n,
                "recall@1": round(r1, 4),
                "recall@5": round(r5, 4),
                "recall@10": round(r10, 4),
                "recall@10_ci_95": f"[{r10_lo:.3f}, {r10_hi:.3f}]",
                "recall@10_exact_cp_95": f"[{exact_lo:.3f}, {exact_hi:.3f}]",
                "mrr": round(mrr, 4),
                "mrr_ci_95": f"[{mrr_lo:.3f}, {mrr_hi:.3f}]",
                "false_gap@10": round(1.0 - r10, 4),
                "absent_control_auc": round(abs_auc, 4),
            })

    return metrics, failures


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print("=== RUNNING BM25 RETRIEVAL PREVIEW (CPU) ===")
    print(f"Banner: {BANNER}\n")

    queries_path = ROOT / "results/real/queries.jsonl"
    if not queries_path.exists():
        print(f"Error: {queries_path} not found. Ensure Step 4 completes first.")
        sys.exit(1)

    queries = load_jsonl(queries_path)
    print(f"Loaded {len(queries)} total queries from {queries_path}")

    all_metrics: list[dict] = []
    all_failures: list[dict] = []

    # Run for both distractor counts: 500 (620-doc) and 100 (220-doc)
    for dist_count in [500, 100]:
        docs, titles = load_corpus(ROOT, distractor_count=dist_count)
        corpus_size = len(docs)
        print(f"\nBuilding views and evaluating on corpus size: {corpus_size} docs ({dist_count} distractors)...")
        views = build_index_views(docs)
        metrics, failures = evaluate_bm25_suite(queries, views, corpus_size, titles)
        all_metrics.extend(metrics)
        all_failures.extend(failures)

    # Save metrics CSV
    out_csv = ROOT / "results/real/retrieval_preview_bm25_metrics.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\nSaved metrics to {out_csv}")

    # Format Markdown tables with BANNER
    md_lines = [
        f"> [!IMPORTANT]\n> **{BANNER}**\n\n",
        "# BM25 Retrieval Preview Report (CPU)\n\n",
        f"**Date**: 2026-10-31\n",
        f"**Corpus Configurations**: 620 documents (500 distractors) and 220 documents (100 distractors)\n",
        f"**Views**: `native`, `native_abstract`, `en_abstract`\n\n",
    ]

    for corpus_size in [620, 220]:
        md_lines.append(f"## Corpus Size: {corpus_size} Documents\n\n")
        md_lines.append(f"> [!NOTE]\n> {BANNER}\n\n")
        md_lines.append("| View | Variant | N | Recall@1 | Recall@5 | Recall@10 | Recall@10 95% Bootstrap CI | Recall@10 95% Exact CP | MRR | MRR 95% CI | False Gap@10 | Absent AUC |\n")
        md_lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|\n")

        for m in all_metrics:
            if m["corpus_size"] == corpus_size:
                md_lines.append(
                    f"| `{m['view']}` | `{m['variant']}` | {m['n_queries']} | {m['recall@1']:.2f} | {m['recall@5']:.2f} | "
                    f"**{m['recall@10']:.2f}** | {m['recall@10_ci_95']} | {m['recall@10_exact_cp_95']} | "
                    f"{m['mrr']:.3f} | {m['mrr_ci_95']} | {m['false_gap@10']:.2f} | {m['absent_control_auc']:.2f} |\n"
                )
        md_lines.append("\n")

    out_md = ROOT / "results/real/retrieval_preview_bm25_report.md"
    out_md.write_text("".join(md_lines), encoding="utf-8")
    print(f"Saved Markdown report to {out_md}")

    # Step 6: Failure cases file
    # results/real/failure_cases_bm25_preview.md (one entry per top-10 miss for V1-V3 and H-agent: query, source title, rank, top-3 retrieved)
    # Leave the "why it failed" line empty.
    fail_lines = [
        f"> [!IMPORTANT]\n> **{BANNER}**\n\n",
        "# Failure Cases: BM25 Retrieval Preview (Top-10 Misses on 620-Document Corpus)\n\n",
        f"This document catalogs all top-10 retrieval misses for queries V1, V2, V3, and H-agent under BM25 on the 620-document corpus.\n\n",
    ]

    # Filter failures for 620 corpus, primary view native and en_abstract
    failures_620 = [f for f in all_failures if f["corpus_size"] == 620 and f["view"] in ("native", "en_abstract")]
    
    # Deduplicate or list by view and variant
    current_key = None
    for f in failures_620:
        key = (f["view"], f["variant"])
        if key != current_key:
            current_key = key
            fail_lines.append(f"## View: `{f['view']}` | Variant: `{f['variant']}`\n\n")

        fail_lines.append(f"### Paper: `{f['source_id']}`\n")
        fail_lines.append(f"- **Source Title**: {f['source_title']}\n")
        fail_lines.append(f"- **Query**: {f['query_text']}\n")
        fail_lines.append(f"- **Target Rank**: {f['rank']}\n")
        fail_lines.append(f"- **Top 3 Retrieved**:\n")
        for idx, t in enumerate(f["top3"], 1):
            fail_lines.append(f"  {idx}. `{t['doc_id']}` — {t['title']} (score: {t['score']})\n")
        fail_lines.append(f"- **Why it failed**: \n\n")

    out_fail = ROOT / "results/real/failure_cases_bm25_preview.md"
    out_fail.write_text("".join(fail_lines), encoding="utf-8")
    print(f"Saved Failure Cases to {out_fail}")

    print("\nRetrieval preview and failure cases completed successfully!")


if __name__ == "__main__":
    main()

