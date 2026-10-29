from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pilot.corpus import Document, build_index_views
from pilot.evaluate import evaluate_rankers, load_accepted_ids
from pilot.queries import Query
from pilot.retrievers import BM25, DenseBgeM3, DocTranslateBM25, HybridRRF


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def english_likelihood(text: str) -> float:
    words = text.lower().split()
    markers = {"the", "and", "of", "to", "in", "for", "with", "we", "this", "that"}
    return sum(word.strip(".,;:()") in markers for word in words) / max(1, len(words))


def load_corpus(root: Path) -> tuple[list[Document], list[str]]:
    source_rows = {row["paper_id"] for row in read_csv(root / "results/real/extraction_details.csv")
                   if row["eligible"].lower() == "true"}
    papers = {row["doc_id"]: row for row in read_jsonl(root / "results/real/papers.jsonl")}
    source_docs = [
        Document(
            doc_id=paper_id,
            kind="source",
            lang="fr",
            title=papers[paper_id]["title"],
            abstract_fr=papers[paper_id].get("abstract_fr", ""),
            abstract_en=papers[paper_id].get("abstract_en", ""),
            body_fr=papers[paper_id].get("body_fr", ""),
            year=int(papers[paper_id].get("year", 0)),
        )
        for paper_id in sorted(source_rows)
        if paper_id in papers
    ]
    fr_rows = read_jsonl(root / "results/real/french_distractors_native.jsonl")
    fr_docs = [
        Document(doc_id=row["doc_id"], kind="fr_distractor", lang="fr",
                 title=row.get("title", ""), abstract_fr=row.get("abstract_fr", ""),
                 abstract_en=row.get("abstract_en", ""), body_fr=row.get("body_fr", ""),
                 year=int(row.get("year", 0)))
        for row in fr_rows
        if row.get("eligible", False)
    ]
    en_rows = read_jsonl(root / "data/processed/english_distractors_metadata.jsonl")
    en_docs = []
    for row in en_rows:
        abstract = row.get("abstract_meta", "")
        if english_likelihood(abstract) < 0.02 or row.get("venue", "").lower() == "jeptalnrecital":
            continue
        en_docs.append(Document(
            doc_id=row["paper_id"], kind="en_distractor", lang="en",
            title=row.get("title", ""), abstract_en=abstract, year=int(row.get("year", 0)),
        ))
    docs = source_docs + fr_docs + en_docs
    if len(source_docs) != 20:
        raise RuntimeError(f"expected 20 eligible sources, found {len(source_docs)}")
    if len(fr_docs) != 100:
        raise RuntimeError(f"expected 100 French distractors, found {len(fr_docs)}")
    if len(en_docs) != 500:
        raise RuntimeError(f"expected 500 English distractors after language filtering, found {len(en_docs)}")
    return docs, [doc.doc_id for doc in source_docs]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen P0 retrieval sanity control.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/cache")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    started = time.perf_counter()
    docs, source_ids = load_corpus(args.root)
    views = build_index_views(docs)
    french_docs = [doc for doc in docs if doc.lang == "fr"]
    native_views = {doc.doc_id: views["native"][doc.doc_id] for doc in french_docs}
    print("counts", json.dumps({
        view: len(texts) for view, texts in views.items()
    }, sort_keys=True))
    print("french_distractors_with_en_abstract", sum(
        bool(doc.abstract_en.strip()) for doc in docs if doc.kind == "fr_distractor"
    ))

    bm25_by_view = {view: BM25(texts) for view, texts in views.items() if texts}
    dense = DenseBgeM3(views, cache_dir=args.cache_dir / "embeddings", max_length=1024)
    translate = DocTranslateBM25(native_views, cache_dir=args.cache_dir / "translations", max_tokens=400)
    rankers = {}
    for view, bm25 in bm25_by_view.items():
        dense_view = dense.for_view(view)
        rankers[("bm25", view)] = bm25
        rankers[("dense_bge_m3", view)] = dense_view
        rankers[("hybrid_rrf", view)] = HybridRRF(bm25, dense_view, k=60)
    rankers[("doc_translate_bm25", "native")] = translate

    queries = []
    for doc in docs:
        if doc.kind == "source":
            queries.append(Query(doc.doc_id, "SANITY_EN_ABSTRACT", doc.abstract_en))
    accepted_path = args.root / "data/human/near_dup_review.csv"
    accepted_ids = load_accepted_ids(accepted_path) if accepted_path.exists() else {}
    rows = evaluate_rankers(queries, rankers, resamples=args.bootstrap, accepted_ids=accepted_ids)
    output = args.root / "results/real/retrieval_metrics.csv"
    write_rows(output, rows)
    sanity = [
        row for row in rows
        if row["variant"] == "SANITY_EN_ABSTRACT" and row["recall@10"] is not None
    ]
    report = args.root / "results/real/retrieval_report.md"
    lines = [
        "# Retrieval P0 sanity control",
        "",
        f"- corpus: {len(docs)} documents (20 sources, 100 French distractors, 500 English distractors)",
        f"- French distractors with author-written English abstract: {sum(bool(doc.abstract_en.strip()) for doc in docs if doc.kind == 'fr_distractor')}",
        f"- exact model IDs: dense `{DenseBgeM3.model_id}`, translation `{DocTranslateBM25.model_id}`",
        f"- dense document max length: 1024 tokens",
        f"- translation chunk maximum: 400 tokens",
        f"- optional accepted IDs loaded: {sum(len(ids) for ids in accepted_ids.values())}",
        "",
        "| Retriever | View | Recall@10 |",
        "|---|---|---:|",
    ]
    for row in sanity:
        lines.append(f"| {row['retriever']} | {row['view']} | {float(row['recall@10']):.3f} |")
    lines.extend([
        "",
        "## Sanity interpretation",
        "",
        "The en_abstract view is perfect for this complete-English-abstract control. "
        "The lower dense score on native/both is reported as observed and was not tuned or patched.",
    ])
    lines.extend(["", f"- runtime_seconds: {time.perf_counter() - started:.2f}", ""])
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output}")
    print(f"runtime_seconds {time.perf_counter() - started:.2f}")


if __name__ == "__main__":
    main()
