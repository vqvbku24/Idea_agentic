from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pilot.data_acl import (
    PaperMeta,
    download_pdf,
    list_french_distractor_candidates,
    write_metadata,
)
from pilot.pdf_extract import detect_lang, extract_text, split_abstracts


REAL = Path("results/real")
EVENTS = Path("events.jsonl")
SOURCE_METADATA = Path("data/processed/extended_metadata.jsonl")
DISTRACTOR_METADATA = Path("data/processed/french_distractors_metadata.jsonl")


def read_metadata(path: Path) -> list[PaperMeta]:
    return [PaperMeta(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def process(meta: PaperMeta, pdf: Path) -> tuple[dict, dict | None]:
    text = extract_text(pdf)
    parts = split_abstracts(
        text,
        metadata_abstract=meta.abstract_meta,
        metadata_language=meta.language,
        title=meta.title,
    )
    language, confidence = detect_lang(parts["body"])
    reasons = []
    if language != "fr" or confidence < 0.9:
        reasons.append("body_not_french_or_low_confidence")
    if parts["provenance"]["fr_abstract"] != "pdf":
        reasons.append("french_abstract_not_from_pdf")
    if parts["provenance"]["en_abstract"] != "pdf":
        reasons.append("english_abstract_not_from_pdf")
    if len(parts["body"].split()) < 1500:
        reasons.append("body_under_1500_words")
    row = {
        "paper_id": meta.paper_id,
        "title": meta.title,
        "eligible": not reasons,
        "failure_reasons": reasons,
        "body_words": len(parts["body"].split()),
        "provenance": parts["provenance"],
    }
    return row, {
        "doc_id": meta.paper_id,
        "title": meta.title,
        "abstract_fr": parts["fr_abstract"],
        "abstract_en": parts["en_abstract"],
        "body_fr": parts["body"],
        "year": meta.year,
        "provenance": parts["provenance"],
        "eligible": not reasons,
    }


def write_scan_log(rows: list[dict]) -> None:
    with (REAL / "scan_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scan_index", "paper_id", "eligible", "failure_reasons"])
        for index, row in enumerate(rows, 1):
            writer.writerow([index, row["paper_id"], str(row["eligible"]).lower(), json.dumps(row["failure_reasons"])])


def write_g1(rows: list[dict]) -> None:
    n = len(rows)
    eligible = sum(row["eligible"] for row in rows)
    failures = Counter(reason for row in rows for reason in row["failure_reasons"])
    french = sum(not ("body_not_french_or_low_confidence" in row["failure_reasons"]) for row in rows)
    english = sum(row["provenance"].get("en_abstract") == "pdf" for row in rows)
    unbiased = (eligible - 1) / (n - 1) if n > 1 else float("nan")
    with (REAL / "g1_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "count", "share", "reason"])
        writer.writerow(["scanned", n, 1.0 if n else 0.0, ""])
        writer.writerow(["french_body", french, french / max(1, n), ""])
        writer.writerow(["extractable_english_abstract", english, english / max(1, n), ""])
        writer.writerow(["eligible", eligible, eligible / max(1, n), ""])
        writer.writerow(["unbiased_estimate_(r-1)/(n-1)", "", unbiased, ""])
        for reason, count in sorted(failures.items()):
            writer.writerow(["failure", count, count / max(1, n), reason])


def event(event_type: str, **payload: object) -> None:
    with EVENTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": event_type, **payload}, ensure_ascii=False) + "\n")


def regenerate_source_events(rows: list[dict]) -> None:
    existing = []
    if EVENTS.exists():
        existing = [
            line for line in EVENTS.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line).get("type") != "SourceSelected"
        ]
    with EVENTS.open("w", encoding="utf-8") as handle:
        for line in existing:
            handle.write(line + "\n")
        for index, row in enumerate(rows, 1):
            handle.write(json.dumps({
                "type": "SourceSelected",
                "paper_id": row["paper_id"],
                "scan_index": index,
            }) + "\n")


def write_acquisition_counts(source_rows: list[dict], distractor_scanned: int, distractor_eligible: int) -> None:
    english_count = len(read_metadata(Path("data/processed/english_distractors_metadata.jsonl")))
    french_count = distractor_scanned
    with (REAL / "acquisition_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["set", "requested", "discovered", "downloaded", "failure_reason"])
        writer.writerow(["new_sources", 20, len(source_rows), len(source_rows), ""])
        writer.writerow(["english_distractors", 500, english_count, 0, "metadata-only"])
        writer.writerow([
            "french_distractors", 100, french_count, distractor_scanned,
            "" if distractor_eligible >= 80 else "fewer than 80 eligible after fixed rules",
        ])


def token_counts(text: str) -> Counter[str]:
    return Counter(re.findall(r"[a-z]{2,}", text.lower()))


def cosine(left: Counter[str], right: Counter[str], idf: dict[str, float]) -> float:
    common = set(left) | set(right)
    numerator = sum(left[t] * right[t] * idf.get(t, 1.0) ** 2 for t in common)
    left_norm = math.sqrt(sum((left[t] * idf.get(t, 1.0)) ** 2 for t in left))
    right_norm = math.sqrt(sum((right[t] * idf.get(t, 1.0)) ** 2 for t in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def write_near_duplicates(sources: list[dict], english: list[PaperMeta], french: list[dict]) -> None:
    candidates = [
        (paper.paper_id, paper.title, paper.abstract_meta or "")
        for paper in english
    ] + [
        (paper["doc_id"], paper["title"], paper["abstract_en"])
        for paper in french if paper["abstract_en"].strip()
    ]
    documents = [token_counts(text) for _, _, text in candidates]
    document_frequency = Counter(token for document in documents for token in document)
    idf = {token: math.log((1 + len(documents)) / (1 + frequency)) + 1
           for token, frequency in document_frequency.items()}
    lines = ["# Near-duplicate review", "", "Cosine similarity over lowercase word TF-IDF; no candidates were removed.", ""]
    for source in sources:
        query = token_counts(source["abstract_en"])
        scored = sorted(
            ((cosine(query, document, idf), paper_id, title) for (paper_id, title, _), document in zip(candidates, documents)),
            reverse=True,
        )[:3]
        lines.append(f"## {source['doc_id']} — {source['title']}")
        for score, paper_id, title in scored:
            lines.append(f"- `{paper_id}` — {title} — similarity `{score:.6f}`")
        lines.append("")
    (REAL / "near_duplicates.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REAL.mkdir(parents=True, exist_ok=True)
    source_rows = []
    source_docs = []
    for meta in read_metadata(SOURCE_METADATA):
        pdf = Path("data/raw/new") / f"{meta.paper_id}.pdf"
        row, document = process(meta, pdf)
        source_rows.append(row)
        if document:
            source_docs.append(document)
    write_scan_log(source_rows)
    write_g1(source_rows)
    regenerate_source_events(source_rows)
    eligible_sources = [document for document in source_docs if document["eligible"]]
    json_sources = [document["doc_id"] for document in eligible_sources]
    (Path("data/processed") / "sources_final.json").write_text(
        json.dumps(json_sources, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    source_ids = set(json_sources)
    candidates = list_french_distractor_candidates(13, source_ids)
    selected = candidates[:100]
    write_metadata(selected, DISTRACTOR_METADATA)
    state: dict[str, float] = {}
    distractor_rows = []
    distractor_docs = []
    scanned = 0
    for meta in candidates[:250]:
        scanned += 1
        pdf = Path("data/raw/french_distractors") / f"{meta.paper_id}.pdf"
        if not download_pdf(meta, pdf.parent, state=state, events_path=EVENTS):
            distractor_rows.append({"paper_id": meta.paper_id, "eligible": False, "failure_reasons": ["download_failed"]})
            continue
        row, document = process(meta, pdf)
        distractor_rows.append(row)
        event("DistractorProcessed", paper_id=meta.paper_id, eligible=row["eligible"])
        if document:
            distractor_docs.append(document)
        if scanned >= 100 and sum(item["eligible"] for item in distractor_rows) >= 100:
            break
    with (REAL / "french_distractors_native.jsonl").open("w", encoding="utf-8") as handle:
        for document in distractor_docs:
            native = " ".join(
                [document["title"], document["abstract_fr"], *document["body_fr"].split()[:1500]]
            ).strip()
            handle.write(json.dumps({
                "doc_id": document["doc_id"],
                "title": document["title"],
                "native": native,
                "eligible": document["eligible"],
                "provenance": document["provenance"],
            }, ensure_ascii=False) + "\n")
    write_metadata(candidates[:scanned], DISTRACTOR_METADATA)
    with (REAL / "distractor_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["set", "n_scanned", "n_eligible", "eligibility_rate"])
        writer.writerow(["french_distractors", scanned, sum(row["eligible"] for row in distractor_rows),
                         sum(row["eligible"] for row in distractor_rows) / max(1, scanned)])
    write_acquisition_counts(source_rows, scanned, sum(row["eligible"] for row in distractor_rows))
    write_near_duplicates(eligible_sources, read_metadata(Path("data/processed/english_distractors_metadata.jsonl")), distractor_docs)
    print(f"sources_scanned={len(source_rows)} sources_eligible={len(eligible_sources)}")
    print(f"french_distractors_scanned={scanned} french_distractors_eligible={sum(row['eligible'] for row in distractor_rows)}")


if __name__ == "__main__":
    main()
