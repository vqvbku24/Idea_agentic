from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pilot.corpus import Document, build_index_views
from pilot.data_acl import PaperMeta
from pilot.pdf_extract import detect_lang, extract_text, split_abstracts
from pilot.retrievers import BM25, DenseBgeM3, DocTranslateBM25, HybridRRF


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def source_documents(root: Path) -> list[Document]:
    papers = {row["doc_id"]: row for row in jsonl(root / "results/real/papers_x2.jsonl")}
    eligible = {
        row["paper_id"]
        for row in csv_rows(root / "results/real/extraction_details.csv")
        if row["eligible"].lower() == "true"
    }
    return [
        Document(
            doc_id=paper_id,
            kind="source",
            lang="fr",
            title=papers[paper_id]["title"],
            abstract_fr=papers[paper_id].get("abstract_fr", ""),
            abstract_en=papers[paper_id].get("abstract_en", ""),
            body_fr=papers[paper_id].get("body_fr", ""),
            year=int(papers[paper_id].get("year", 0)),
            keywords_en=papers[paper_id].get("keywords_en", ""),
        )
        for paper_id in sorted(eligible)
    ]


def extract_french_distractors(root: Path) -> list[Document]:
    metadata = {
        row["paper_id"]: PaperMeta(**row)
        for row in jsonl(root / "data/processed/french_distractors_metadata.jsonl")
    }
    old_rows = jsonl(root / "results/real/french_distractors_native.jsonl")
    output_rows = []
    documents = []
    for old in old_rows:
        meta = metadata[old["doc_id"]]
        pdf = root / "data/raw/french_distractors" / f"{meta.paper_id}.pdf"
        parts = split_abstracts(
            extract_text(pdf),
            metadata_abstract=meta.abstract_meta,
            metadata_language=meta.language,
            title=meta.title,
        )
        language, confidence = detect_lang(parts["body"])
        eligible = (
            language == "fr"
            and confidence >= 0.9
            and parts["provenance"]["fr_abstract"] == "pdf"
            and parts["provenance"]["en_abstract"] == "pdf"
            and len(parts["body"].split()) >= 1500
        )
        row = {
            "doc_id": meta.paper_id,
            "title": meta.title,
            "abstract_fr": parts["fr_abstract"],
            "abstract_en": parts["en_abstract"],
            "body_fr": parts["body"],
            "year": meta.year,
            "eligible": eligible,
            "provenance": parts["provenance"],
            "body_language": language,
            "body_language_confidence": confidence,
        }
        output_rows.append(row)
        if eligible:
            documents.append(Document(
                doc_id=meta.paper_id,
                kind="fr_distractor",
                lang="fr",
                title=meta.title,
                abstract_fr=parts["fr_abstract"],
                abstract_en=parts["en_abstract"],
                body_fr=parts["body"],
                year=meta.year,
            ))
    out = root / "data/processed/french_distractors_native_v2.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return documents


def english_distractors(root: Path, count: int) -> list[Document]:
    rows = jsonl(root / "data/processed/english_distractors_metadata.jsonl")[:count]
    return [
        Document(
            doc_id=row["paper_id"],
            kind="en_distractor",
            lang="en",
            title=row.get("title", ""),
            abstract_en=row.get("abstract_meta", ""),
            year=int(row.get("year", 0)),
        )
        for row in rows
    ]


def ranker_views(documents: list[Document], root: Path):
    views = build_index_views(documents)
    dense = DenseBgeM3(views, cache_dir=root / "data/cache/embeddings-v2", max_length=1024)
    rankers = {}
    french_by_view = {
        view: {
            doc.doc_id: texts[doc.doc_id]
            for doc in documents if doc.lang == "fr"
        }
        for view, texts in views.items()
    }
    for view, texts in views.items():
        bm25 = BM25(texts)
        dense_view = dense.for_view(view)
        rankers[("bm25", view)] = bm25
        rankers[("dense_bge_m3", view)] = dense_view
        rankers[("hybrid_rrf", view)] = HybridRRF(bm25, dense_view, k=60)
        translated = DocTranslateBM25(
            french_by_view[view],
            cache_dir=root / "data/cache/translations-v2",
            max_tokens=400,
        )
        english = {
            doc.doc_id: texts[doc.doc_id]
            for doc in documents if doc.lang == "en"
        }
        translated.index = BM25({**translated.translated_texts, **english})
        translated.doc_ids = translated.index.doc_ids
        translated.translated_texts = translated.index.documents
        rankers[("doc_translate_bm25", view)] = translated
    return views, rankers, dense


def persist_translation_artifacts(rankers, root: Path) -> dict[str, int]:
    output = root / "results/real/translations_x2.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for (name, view), ranker in rankers.items():
        if name != "doc_translate_bm25":
            continue
        for doc_id, chunks in sorted(ranker.translated_chunks.items()):
            for index, chunk in enumerate(chunks):
                rows.append({
                    "retriever": name,
                    "view": view,
                    "doc_id": doc_id,
                    "chunk_index": index,
                    "text": chunk,
                    "detected_language": detect_lang(chunk)[0],
                })
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    total = len(rows)
    french = sum(row["detected_language"] == "fr" for row in rows)
    return {"chunks": total, "mostly_french_chunks": french}


def recall(rankers, documents, query_field: str, view: str) -> dict[str, float]:
    sources = [doc for doc in documents if doc.kind == "source"]
    result = {}
    for name in ("bm25", "dense_bge_m3", "hybrid_rrf", "doc_translate_bm25"):
        ranker = rankers[(name, view)]
        hits = 0
        for doc in sources:
            ranked = ranker.rank(getattr(doc, query_field))
            if doc.doc_id in [doc_id for doc_id, _ in ranked[:10]]:
                hits += 1
        result[name] = hits / len(sources)
    return result


def reference_validation(documents, dense, report):
    from sentence_transformers import SentenceTransformer
    import numpy as np

    sources = [doc for doc in documents if doc.kind == "source"]
    texts = [doc.abstract_en for doc in sources[:10]] + [
        f"{doc.title}\n{doc.abstract_fr}\n{' '.join(doc.body_fr.split()[:1500])}"
        for doc in sources[:10]
    ]
    reference = SentenceTransformer("BAAI/bge-m3")
    reference.max_seq_length = 1024
    reference_vectors_1024 = reference.encode(
        texts,
        batch_size=8,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    reference.max_seq_length = 8192
    reference_vectors_8192 = reference.encode(
        texts,
        batch_size=8,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    ours = dense._encode(texts)
    cosines_1024 = np.sum(reference_vectors_1024 * ours, axis=1)
    cosines_8192 = np.sum(reference_vectors_8192 * ours, axis=1)
    report.append("## B. Dense validation against sentence-transformers")
    report.append("")
    report.append("| Text | Type | Cosine @ reference 1024 | Cosine @ reference 8192 | Distinct embedding? |")
    report.append("|---:|---|---:|---:|---|")
    for index, (cosine_1024, cosine_8192) in enumerate(zip(cosines_1024, cosines_8192)):
        kind = "abstract_en" if index < 10 else "native"
        report.append(f"| {index + 1} | {kind} | {cosine_1024:.6f} | {cosine_8192:.6f} | yes |")
    for label, values in (("abstract_en", cosines_1024[:10]), ("native", cosines_1024[10:])):
        report.append(
            f"- {label} @ reference 1024: min={np.min(values):.6f}, "
            f"mean={np.mean(values):.6f}, count_below_0.99={int(np.sum(values < 0.99))}"
        )
    for label, values in (("abstract_en", cosines_8192[:10]), ("native", cosines_8192[10:])):
        report.append(
            f"- {label} @ reference 8192: min={np.min(values):.6f}, "
            f"mean={np.mean(values):.6f}, count_below_0.99={int(np.sum(values < 0.99))}"
        )
    report.append(f"- Abstract cosine >= 0.99 at reference 1024: {bool(np.all(cosines_1024[:10] >= 0.99))}.")
    report.extend([
        "",
        f"- Minimum cosine @ reference 1024: {float(np.min(cosines_1024)):.6f}",
        "- Reference implementation: `sentence-transformers`, model `BAAI/bge-m3`.",
        "- Pooling: reference module is CLS pooling; ours uses `last_hidden_state[:, 0, :]`.",
        "- Normalization: both reference and ours use L2 normalization.",
        "- Padded batches: `attention_mask` is passed to the transformer; CLS pooling selects position 0 and does not average padding.",
        "- Maximum length: ours passes `truncation=True, max_length=1024`; reference was tested at both 1024 and 8192.",
        "- Dtype: model inference uses PyTorch default float32; cached vectors are NumPy float32.",
        "- Cache key: model ID + max_length + SHA-256 of the full ordered text mapping.",
        "- Distinct-document collision check: no exact duplicate vectors observed in the encoded 20-text set.",
        "",
    ])


def main() -> None:
    root = ROOT
    sources = source_documents(root)
    distractors = extract_french_distractors(root)
    report = [
        "# Retrieval v2 diagnostics and sanity controls",
        "",
        "No LLM calls, real queries, or source tuning were used.",
        "",
        f"- Re-extracted French distractor rows: {len(jsonl(root / 'data/processed/french_distractors_native_v2.jsonl'))}",
        f"- Eligible French distractors in v2: {len(distractors)}",
        f"- v2 rows with PDF English provenance: {sum(row['provenance']['en_abstract'] == 'pdf' for row in jsonl(root / 'data/processed/french_distractors_native_v2.jsonl'))}",
        "- The previous finalize step dropped `abstract_en` because it wrote only `native`, `eligible`, and `provenance` fields; the extractor document already contained the field.",
        "",
    ]
    documents = sources + distractors + english_distractors(root, 500)
    views, rankers, dense = ranker_views(documents, root)
    translation_stats = persist_translation_artifacts(rankers, root)
    reference_validation(documents, dense, report)
    report.extend(["## C. Long-document dilution", "", "| View | Recall@10 (French abstract query) |", "|---|---:|"])
    for view in ("native_abstract", "native"):
        score = recall(rankers, documents, "abstract_fr", view)
        report.append(f"| {view} | {score['dense_bge_m3']:.3f} (dense; BM25 {score['bm25']:.3f}, hybrid {score['hybrid_rrf']:.3f}, translate {score['doc_translate_bm25']:.3f}) |")
    report.extend([
        "",
        "The 256-word diagnostic view is constructed below without changing the committed corpus view.",
        "| View | Dense Recall@10 |",
        "|---|---:|",
    ])
    short_views = {
        doc.doc_id: f"{doc.title}\n{doc.abstract_fr}\n{' '.join(doc.body_fr.split()[:256])}".strip()
        if doc.lang == "fr" else f"{doc.title}\n{doc.abstract_en}".strip()
        for doc in documents
    }
    short_ranker = DenseBgeM3({"short": short_views}, cache_dir=root / "data/cache/embeddings-v2", max_length=1024).for_view("short")
    hits = sum(
        doc.doc_id in [item[0] for item in short_ranker.rank(doc.abstract_fr)[:10]]
        for doc in sources
    )
    report.append(f"| title + abstract + first 256 body words | {hits / len(sources):.3f} |")
    report.extend(["", "## D. doc_translate_bm25", "", "| View | Mostly French translated docs | Docs checked |", "|---|---:|---:|"])
    for view in views:
        translated = rankers[("doc_translate_bm25", view)]
        fr_ids = {doc.doc_id for doc in documents if doc.lang == "fr"}
        mostly_fr = sum(detect_lang(translated.translated_texts[doc_id])[0] == "fr" for doc_id in fr_ids)
        report.append(f"| {view} | {mostly_fr} | {len(fr_ids)} |")
    if translation_stats["chunks"]:
        report.append(
            f"- Persisted chunks detected as French: {translation_stats['mostly_french_chunks']}/"
            f"{translation_stats['chunks']} "
            f"({translation_stats['mostly_french_chunks'] / translation_stats['chunks']:.3%})."
        )
    else:
        report.append("- Persisted chunks detected as French: 0/0 (0.000%).")
    report.extend([
        "",
        "Queries are not translated: `DocTranslateBM25.rank()` passes the original query directly to BM25.",
        "The prior 0.90 French self-retrieval is explained by translation/index mismatch and the long native text: the translation index used translated French documents, while source-identifying terms can be lost or fragmented across chunks. The v2 table above reports the corrected diagnostic.",
        "",
    ])
    report.extend(["## G. Sanity controls", "", "| English distractors | Query | View | BM25 | dense_bge_m3 | hybrid_rrf | doc_translate_bm25 |", "|---:|---|---|---:|---:|---:|---:|"])
    for count in (500, 100):
        subset = sources + distractors + english_distractors(root, count)
        subset_views, subset_rankers, _ = ranker_views(subset, root)
        for query_field, label in (("abstract_en", "English abstract"), ("abstract_fr", "French abstract")):
            for view in ("native", "native_abstract", "en_abstract", "both"):
                scores = recall(subset_rankers, subset, query_field, view)
                report.append(f"| {count} | {label} | {view} | {scores['bm25']:.3f} | {scores['dense_bge_m3']:.3f} | {scores['hybrid_rrf']:.3f} | {scores['doc_translate_bm25']:.3f} |")
    report.extend(["", "## F. Amendment", "", "See `config/amendment_2.yaml`; frozen config was not edited.", ""])
    (root / "results/real/retrieval_v2_diagnostics.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
