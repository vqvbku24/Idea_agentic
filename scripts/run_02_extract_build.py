from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pilot.data_acl import PaperMeta
from pilot.pdf_extract import detect_lang, extract_text, split_abstracts


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract abstracts and apply fixed source eligibility rules.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/selected_metadata.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/real"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    papers = []
    with args.metadata.open(encoding="utf-8") as handle:
        for line in handle:
            papers.append(PaperMeta(**json.loads(line)))
    rows = []
    failures: dict[str, int] = {}
    extracted = []
    review_rows = []
    for meta in papers:
        pdf = args.raw_dir / f"{meta.paper_id}.pdf"
        base = {"paper_id": meta.paper_id, "title": meta.title, "year": meta.year}
        if not pdf.exists():
            failures["download_failed"] = failures.get("download_failed", 0) + 1
            rows.append({**base, "eligible": False, "failure_reason": "download_failed"})
            review_rows.append({
                **base, "fr_abstract": "", "en_abstract": "", "body": "",
                "provenance": "{}", "suspicious": "download_failed",
            })
            continue
        try:
            text = extract_text(pdf)
            parts = split_abstracts(
                text,
                metadata_abstract=meta.abstract_meta,
                metadata_language=meta.language,
                title=meta.title,
            )
            lang, confidence = detect_lang(parts["body"])
            body_words = len(parts["body"].split())
            en_lang, en_confidence = detect_lang(parts["en_abstract"])
            title_tokens = " ".join(meta.title.lower().split())
            first_1500 = " ".join(parts["body"].split()[:1500]).lower()
            title_occurrences = first_1500.count(title_tokens) if title_tokens else 0
            suspicious = []
            if parts["en_abstract"] and en_lang == "fr" and en_confidence >= 0.9:
                suspicious.append("en_abstract_detected_as_french")
            if len(parts["fr_abstract"].split()) < 40:
                suspicious.append("fr_abstract_shorter_than_40_words")
            if len(parts["en_abstract"].split()) < 40:
                suspicious.append("en_abstract_shorter_than_40_words")
            if title_occurrences > 1:
                suspicious.append("source_title_repeated_in_first_1500_body_words")
            reasons = []
            if lang != "fr" or confidence < 0.9:
                reasons.append("body_not_french_or_low_confidence")
            if parts["provenance"]["fr_abstract"] != "pdf":
                reasons.append("french_abstract_not_from_pdf")
            if parts["provenance"]["en_abstract"] != "pdf":
                reasons.append("english_abstract_not_from_pdf")
            if body_words < 1500:
                reasons.append("body_under_1500_words")
            for reason in reasons:
                failures[reason] = failures.get(reason, 0) + 1
            eligible = not reasons
            rows.append({**base, "eligible": eligible, "body_language": lang,
                         "body_language_confidence": confidence,
                         "english_provenance": parts["provenance"]["en_abstract"],
                         "body_words": body_words,
                         "title_occurrences_first_1500": title_occurrences,
                         "suspicious": ";".join(suspicious),
                         "failure_reason": ";".join(reasons)})
            review_rows.append({
                **base,
                "fr_abstract": parts["fr_abstract"][:300],
                "en_abstract": parts["en_abstract"][:300],
                "body": parts["body"][:300],
                "provenance": json.dumps(parts["provenance"], ensure_ascii=False),
                "suspicious": ";".join(suspicious),
            })
            extracted.append({
                "doc_id": meta.paper_id, "kind": "source", "lang": "fr", "title": meta.title,
                "abstract_fr": parts["fr_abstract"], "abstract_en": parts["en_abstract"],
                "body_fr": parts["body"], "year": meta.year, "provenance": parts["provenance"],
                "body_language": lang, "body_language_confidence": confidence, "eligible": eligible,
            })
        except Exception as exc:
            failures[type(exc).__name__] = failures.get(type(exc).__name__, 0) + 1
            rows.append({**base, "eligible": False, "failure_reason": type(exc).__name__})
            review_rows.append({
                **base, "fr_abstract": "", "en_abstract": "", "body": "",
                "provenance": "{}", "suspicious": f"extraction_failed:{type(exc).__name__}",
            })
    with (args.output_dir / "papers.jsonl").open("w", encoding="utf-8") as handle:
        for paper in extracted:
            handle.write(json.dumps(paper, ensure_ascii=False) + "\n")
    with (args.output_dir / "g1_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "count", "share", "reason"])
        total = len(papers)
        writer.writerow(["scanned", total, 1.0 if total else 0.0, ""])
        french_body = sum(
            r.get("body_language") == "fr" and r.get("body_language_confidence", 0) >= 0.9
            for r in rows
        )
        english_abstract = sum(r.get("english_provenance") == "pdf" for r in rows)
        writer.writerow(["french_body", french_body, french_body / max(1, total), ""])
        writer.writerow(["extractable_english_abstract", english_abstract,
                         english_abstract / max(1, total), ""])
        writer.writerow(["eligible", sum(r["eligible"] for r in rows), sum(r["eligible"] for r in rows) / max(1, total), ""])
        for reason, count in sorted(failures.items()):
            writer.writerow(["failure", count, count / max(1, total), reason])
    with (args.output_dir / "extraction_details.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "paper_id", "title", "year", "eligible", "body_language",
            "body_language_confidence", "english_provenance", "body_words", "failure_reason",
            "title_occurrences_first_1500", "suspicious",
        ])
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "extraction_review.md").open("w", encoding="utf-8") as handle:
        handle.write("# Extraction review\n\n")
        handle.write("This review is descriptive. Suspicious cases are flagged, not auto-fixed.\n\n")
        for row in review_rows:
            handle.write(f"## {row['paper_id']} — {row['title']} ({row['year']})\n\n")
            handle.write(f"- provenance: `{row['provenance']}`\n")
            handle.write(f"- suspicious: `{row['suspicious'] or 'none'}`\n\n")
            handle.write(f"**French abstract (first 300 chars):** {row['fr_abstract']}\n\n")
            handle.write(f"**English abstract (first 300 chars):** {row['en_abstract']}\n\n")
            handle.write(f"**Body (first 300 chars):** {row['body']}\n\n")
    print(f"scanned={len(papers)} eligible={sum(r['eligible'] for r in rows)}")


if __name__ == "__main__":
    main()
