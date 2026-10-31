from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pilot.corpus import Document, build_index_views
from pilot.retrievers import BM25, DenseBgeM3, HybridRRF, DocTranslateBM25
from pilot.queries import Query
from pilot.evaluate import evaluate_rankers, write_metrics

BANNER = "EXPLORATORY - agent-proxy labels, human verification pending"


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
            title=row.get("title", ""),
            abstract_en=row.get("abstract_meta", ""),
            year=int(row.get("year", 0)),
        )
        for row in en_dist_raw
    ]

    return sources + french_distractors + english_distractors


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"=== COLAB RUN 2: FULL RETRIEVAL SUITE ===")
    print(f"Banner: {BANNER}\n")

    queries_path = ROOT / "results/real/queries.jsonl"
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
    accepted_count = sum(1 for q in queries if q.accepted)
    print(f"Loaded {len(queries)} queries ({accepted_count} accepted)")

    # 620 documents (500 English distractors)
    print("Building 620-document corpus...")
    docs = load_corpus(ROOT, distractor_count=500)
    views = build_index_views(docs)

    dense_cache = ROOT / "data/cache/embeddings-v2"
    trans_cache = ROOT / "data/cache/translations-v2"

    print("Initializing DenseBgeM3 (cached embeddings)...")
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

        # Doc translate
        translated = DocTranslateBM25(
            french_by_view[view],
            cache_dir=trans_cache,
            max_tokens=400,
        )
        translated.index = BM25({**translated.translated_texts, **english_by_view[view]})
        translated.doc_ids = translated.index.doc_ids
        translated.translated_texts = translated.index.documents
        rankers[("doc_translate_bm25", view)] = translated

    print(f"Evaluating {len(rankers)} rankers across accepted queries...")
    results = evaluate_rankers(queries, rankers, top_k=10, resamples=1000)

    # Prepend banner to every row
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
    print("\nColab full retrieval suite completed successfully!")


if __name__ == "__main__":
    main()

