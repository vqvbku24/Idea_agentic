from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pilot.corpus import build_index_views, load_jsonl, write_jsonl
from pilot.demo import documents as demo_documents
from pilot.evaluate import evaluate, write_metrics
from pilot.queries import generate_queries, query_dict
from pilot.retrievers import BM25


def report(rows: list[dict], output: Path, queries: list) -> None:
    lines = [
        "# Pilot smoke-test report",
        "",
        "> This run is an offline smoke test. Demo documents are synthetic and must not be used as evidence for the research claims.",
        "",
        f"- Queries: {len(queries)}",
        f"- Views: {', '.join(sorted({row['view'] for row in rows}))}",
        "",
        "| Variant | View | Recall@10 | 95% CI | MRR | Absent AUC |",
        "|---|---|---:|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['view']} | {row['recall@10']:.3f} | "
            f"[{row['recall@10_ci_low']:.3f}, {row['recall@10_ci_high']:.3f}] | "
            f"{row['mrr']:.3f} | {row['absent_auc']:.3f} |"
        )
    lines.extend(["", "## Interpretation", "",
                  "The code path is operational. Real ACL metadata, PDF extraction, author-written abstracts, "
                  "LLM validity judgments, and human queries are still required before applying any decision rule.",
                  ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def write_g1_stats(documents, output: Path) -> None:
    sources = [document for document in documents if document.kind == "source"]
    eligible = [
        document for document in sources
        if document.lang.lower().startswith("fr")
        and bool(document.abstract_fr.strip())
        and bool(document.abstract_en.strip())
        and len(document.body_fr.split()) >= 1500
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "count", "share", "reason"])
        writer.writerow(["scanned", len(sources), 1.0 if sources else 0.0, ""])
        writer.writerow(["french_body", sum(d.lang.lower().startswith("fr") for d in sources),
                         sum(d.lang.lower().startswith("fr") for d in sources) / max(1, len(sources)), ""])
        writer.writerow(["extractable_english_abstract", sum(bool(d.abstract_en.strip()) for d in sources),
                         sum(bool(d.abstract_en.strip()) for d in sources) / max(1, len(sources)), ""])
        writer.writerow(["eligible", len(eligible), len(eligible) / max(1, len(sources)), ""])


def failure_cases(documents, queries, index, output: Path) -> None:
    titles = {document.doc_id: document.title for document in documents}
    lines = ["# Failure cases", "", "Misses are listed from the offline run; inspect these after replacing the fixture with real data.", ""]
    for query in queries:
        ranked = index.rank(query.text)
        ids = [doc_id for doc_id, _ in ranked]
        if query.source_id not in ids[:10]:
            rank = ids.index(query.source_id) + 1 if query.source_id in ids else "not indexed"
            top = ", ".join(titles[doc_id] for doc_id in ids[:3])
            lines.extend([f"## {titles[query.source_id]}",
                          f"- variant: {query.variant}",
                          f"- query: {query.text}",
                          f"- rank: {rank}",
                          f"- top-3: {top}", ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline prior-art pilot core.")
    parser.add_argument("--demo", action="store_true", help="use a synthetic offline corpus")
    parser.add_argument("--corpus", type=Path, help="input corpus JSONL")
    parser.add_argument("--output-dir", type=Path, default=Path("results/demo"))
    parser.add_argument("--n-source", type=int, default=20)
    args = parser.parse_args()
    if bool(args.demo) == bool(args.corpus):
        parser.error("choose exactly one of --demo or --corpus")
    documents = demo_documents(args.n_source) if args.demo else load_jsonl(args.corpus)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(documents, output / "corpus.jsonl")
    write_g1_stats(documents, output / "g1_stats.csv")
    views = build_index_views(documents)
    queries = [query for document in documents if document.kind == "source"
               for query in generate_queries(document)]
    with (output / "queries.jsonl").open("w", encoding="utf-8") as handle:
        for query in queries:
            handle.write(json.dumps(query_dict(query), ensure_ascii=False) + "\n")
    indexes = {view: BM25(texts) for view, texts in views.items() if texts}
    rows = evaluate(queries, indexes)
    write_metrics(rows, output / "metrics.csv")
    report(rows, output / "report.md", queries)
    failure_cases(documents, queries, indexes["both"], output / "failure_cases.md")
    print(f"wrote {len(rows)} metric rows to {output / 'metrics.csv'}")


if __name__ == "__main__":
    main()
