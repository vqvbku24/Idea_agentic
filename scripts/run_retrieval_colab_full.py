from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pilot.corpus import Document, build_index_views
from pilot.retrievers import BM25, DenseBgeM3, HybridRRF, DocTranslateBM25
from pilot.queries import Query
from pilot.evaluate import evaluate_rankers, rank_without_source, write_metrics

BANNER = "EXPLORATORY - agent-proxy labels, human verification pending"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_corpus(root: Path, distractor_count: int = 500) -> list[Document]:
    papers_raw = {row["doc_id"]: row for row in load_jsonl(root / "results/real/papers_x2.jsonl")}
    
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

    en_dist_raw = load_jsonl(root / "data/processed/english_distractors_metadata.jsonl")[:distractor_count]
    english_distractors = [
        Document(
            doc_id=row["paper_id"],
            kind="en_distractor",
            lang="en",
            title=row["title"],
            abstract_en=row.get("abstract_en", ""),
            year=int(row.get("year", 0)),
        )
        for row in en_dist_raw
    ]

    return sources + french_distractors + english_distractors


def load_all_queries(root: Path) -> list[Query]:
    queries_path = root / "results/real/queries.jsonl"
    if not queries_path.exists():
        raise FileNotFoundError(f"Missing {queries_path}")

    raw_queries = load_jsonl(queries_path)
    queries = [
        Query(
            source_id=q["source_id"],
            variant=q["variant"],
            text=q["query_text"],
            kept_facets=tuple(q.get("kept_facets", [])),
            lexical_leak=float(q.get("lexical_leak", 0.0)),
            accepted=bool(q.get("accepted", False)),
        )
        for q in raw_queries
    ]

    # Optional H-human queries from data/human/human_queries.csv or data/human/human_queries_followup.csv
    human_files = [
        root / "data/human/human_queries_followup.csv",
        root / "data/human/human_queries.csv",
    ]
    for hf in human_files:
        if hf.exists():
            with hf.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = row.get("query_text", "").strip()
                    sid = row.get("source_id", "").strip()
                    if text and sid:
                        queries.append(
                            Query(
                                source_id=sid,
                                variant="H-human",
                                text=text,
                                kept_facets=(),
                                lexical_leak=0.0,
                                accepted=True,
                            )
                        )
            break

    return queries


def build_rankers(docs: list[Document], dense_cache: Path, trans_cache: Path) -> dict[tuple[str, str], Any]:
    views = build_index_views(docs)
    dense = DenseBgeM3(views, cache_dir=dense_cache, max_length=1024)

    french_by_view = {
        view: {doc.doc_id: texts[doc.doc_id] for doc in docs if doc.lang == "fr" and doc.doc_id in texts}
        for view, texts in views.items()
    }
    english_by_view = {
        view: {doc.doc_id: texts[doc.doc_id] for doc in docs if doc.lang == "en" and doc.doc_id in texts}
        for view, texts in views.items()
    }

    rankers = {}
    target_views = ["native", "native_abstract", "en_abstract"]

    for view in target_views:
        texts = views[view]
        bm25 = BM25(texts)
        dense_view = dense.for_view(view)
        rankers[("bm25", view)] = bm25
        rankers[("dense_bge_m3", view)] = dense_view
        rankers[("hybrid_rrf", view)] = HybridRRF(bm25, dense_view, k=60)

        # Doc translate fix (a + b)
        if view == "en_abstract":
            # Fix (b): French documents already have author English abstract in en_abstract view.
            # Do NOT pass already-English text through French-to-English MT!
            trans_index = BM25({**french_by_view[view], **english_by_view[view]})
            mock_dt = object.__new__(DocTranslateBM25)
            mock_dt.index = trans_index
            mock_dt.doc_ids = trans_index.doc_ids
            mock_dt.translated_texts = trans_index.documents
            rankers[("doc_translate_bm25", view)] = mock_dt
        else:
            translated = DocTranslateBM25(
                french_by_view[view],
                cache_dir=trans_cache,
                max_tokens=400,
            )
            # Combine translated French with English distractors
            combined_docs = {**translated.translated_texts, **english_by_view[view]}
            translated.index = BM25(combined_docs)
            translated.doc_ids = translated.index.doc_ids
            translated.translated_texts = combined_docs
            rankers[("doc_translate_bm25", view)] = translated

    return rankers


def export_ranks_for_corpus(
    queries: list[Query],
    rankers: dict[tuple[str, str], Any],
    distractor_count: int,
    corpus_size: int,
) -> list[dict]:
    rows = []
    total = len(queries) * len(rankers)
    done = 0

    print(f"Exporting ranks for distractor_count={distractor_count} (corpus_size={corpus_size})...")
    for q_idx, query in enumerate(queries):
        query_id = f"{query.source_id}_{query.variant}_{q_idx}"
        for (retriever, view), ranker in rankers.items():
            # Full ranking (source present)
            ranked = ranker.rank(query.text)
            doc_ids = [doc_id for doc_id, _ in ranked]
            scores = [round(float(score), 6) for _, score in ranked]

            if query.source_id in doc_ids:
                source_rank = str(doc_ids.index(query.source_id) + 1)
            else:
                source_rank = f">{len(doc_ids)}"

            present_top_score = scores[0] if scores else 0.0

            # Absent ranking (source excluded)
            absent_ranked = rank_without_source(ranker, query)
            absent_top_score = round(float(absent_ranked[0][1]), 6) if absent_ranked else 0.0

            top200_ids = doc_ids[:200]
            top200_sc = scores[:200]

            rows.append({
                "query_id": query_id,
                "source_id": query.source_id,
                "variant": query.variant,
                "accepted": query.accepted,
                "retriever": retriever,
                "view": view,
                "distractor_count": distractor_count,
                "corpus_size": corpus_size,
                "source_rank": source_rank,
                "top200_doc_ids": json.dumps(top200_ids),
                "top200_scores": json.dumps(top200_sc),
                "present_top_score": present_top_score,
                "absent_top_score": absent_top_score,
                "banner": BANNER,
            })
            done += 1

    return rows


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"=== COLAB RUN: FULL RETRIEVAL & RANK EXPORT SUITE ===")
    print(f"Banner: {BANNER}\n")

    # Dirty-tree verification
    try:
        git_status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        print("Git status check:")
        if git_status:
            print("  Tree has uncommitted/untracked files:\n" + git_status[:300])
        else:
            print("  Git tree clean.")
    except Exception as e:
        print("  Git status check skipped:", e)

    # Load all queries (accepted AND rejected + human if present)
    queries = load_all_queries(ROOT)
    accepted_count = sum(1 for q in queries if q.accepted)
    print(f"Loaded {len(queries)} queries ({accepted_count} accepted, {len(queries) - accepted_count} rejected)")

    dense_cache = ROOT / "data/cache/embeddings-v2"
    trans_cache = ROOT / "data/cache/translations-v2"

    all_rank_rows = []

    # Run for both distractor counts: 500 and 100
    for dist_count in [500, 100]:
        print(f"\n--- Loading corpus with distractor_count={dist_count} ---")
        docs = load_corpus(ROOT, distractor_count=dist_count)
        corpus_size = len(docs)
        print(f"Corpus size: {corpus_size} documents")

        print("Building rankers...")
        rankers = build_rankers(docs, dense_cache, trans_cache)

        # If distractor_count == 500, also compute the summary metrics table
        if dist_count == 500:
            print(f"Evaluating {len(rankers)} rankers across accepted queries for summary report...")
            results = evaluate_rankers(queries, rankers, top_k=10, resamples=1000)
            for r in results:
                r["banner"] = BANNER

            out_csv = ROOT / "results/real/retrieval_full_metrics.csv"
            write_metrics(results, out_csv)
            print(f"Saved full metrics to {out_csv}")

            # Build report markdown
            md = [
                f"> [!IMPORTANT]\n> **{BANNER}**\n\n",
                "# Full Retrieval Evaluation Report (Colab Cached Run)\n\n",
                "| Retriever | View | Variant | N | Recall@1 | Recall@5 | Recall@10 | Recall@10 95% Bootstrap CI | Recall@10 Exact CP | MRR | False Gap@10 | Absent AUC |\n",
                "|---|---|---|---|---|---|---|---|---|---|---|---|\n",
            ]
            for r in results:
                md.append(
                    f"| `{r['retriever']}` | `{r['view']}` | `{r['variant']}` | {r['n']} | "
                    f"{r['recall@1']:.2f} | {r['recall@5']:.2f} | **{r['recall@10']:.2f}** | "
                    f"[{r['recall@10_ci_low']:.3f}, {r['recall@10_ci_high']:.3f}] | "
                    f"[{r['recall@10_exact_low']:.3f}, {r['recall@10_exact_high']:.3f}] | "
                    f"{r['mrr']:.3f} | {r['false_gap@10']:.2f} | {r['top1_score_auc_present_absent']:.2f} |\n"
                )
            out_md = ROOT / "results/real/retrieval_full_report.md"
            out_md.write_text("".join(md), encoding="utf-8")
            print(f"Saved report to {out_md}")

        # Export detailed per-query ranks
        rows = export_ranks_for_corpus(queries, rankers, dist_count, corpus_size)
        all_rank_rows.extend(rows)

    # Save rank export
    rank_export_csv = ROOT / "results/real/rank_export.csv"
    rank_export_csv.parent.mkdir(parents=True, exist_ok=True)
    with rank_export_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rank_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rank_rows)
    print(f"\nSaved {len(all_rank_rows)} rank records to {rank_export_csv}")

    # Parquet export if pandas / pyarrow installed
    rank_export_parquet = ROOT / "results/real/rank_export.parquet"
    parquet_saved = False
    try:
        import pandas as pd
        df_ranks = pd.DataFrame(all_rank_rows)
        df_ranks.to_parquet(rank_export_parquet, index=False)
        print(f"Saved parquet export to {rank_export_parquet}")
        parquet_saved = True
    except Exception as e:
        print(f"Parquet export skipped ({e}); CSV is available.")

    # Manifest with SHA-256
    manifest = {
        "banner": BANNER,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_records": len(all_rank_rows),
        "distractor_counts": [500, 100],
        "queries_count": len(queries),
        "files": {
            "rank_export.csv": {
                "sha256": sha256_file(rank_export_csv),
                "size_bytes": rank_export_csv.stat().st_size,
            },
        },
    }
    if parquet_saved and rank_export_parquet.exists():
        manifest["files"]["rank_export.parquet"] = {
            "sha256": sha256_file(rank_export_parquet),
            "size_bytes": rank_export_parquet.stat().st_size,
        }

    manifest_path = ROOT / "results/real/rank_export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest to {manifest_path}")

    print("\nRank export suite completed successfully!")


if __name__ == "__main__":
    main()
