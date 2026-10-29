from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pilot.data_acl import PaperMeta
from pilot.pdf_extract_v2 import extract_pdf_v2

ASSIGNMENT_IDS = [
    "2018.jeptalnrecital-long.14", "2019.jeptalnrecital-court.18",
    "2015.jeptalnrecital-court.13", "2017.jeptalnrecital-court.5",
    "2023.jeptalnrecital-long.11", "2015.jeptalnrecital-recital.3",
    "2021.jeptalnrecital-taln.4", "2020.jeptalnrecital-taln.1",
    "2021.jeptalnrecital-deft.1", "2019.jeptalnrecital-court.7",
]
AUDIT_IDS = ASSIGNMENT_IDS


def load_meta(path: Path) -> list[PaperMeta]:
    return [PaperMeta(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def eligible(parts: dict) -> tuple[bool, list[str]]:
    reasons = []
    if parts["body_language"] != "fr" or parts["body_language_confidence"] < 0.9:
        reasons.append("body_not_french_or_low_confidence")
    if parts["provenance"]["fr_abstract"] != "pdf":
        reasons.append("french_abstract_not_from_pdf")
    if parts["provenance"]["en_abstract"] != "pdf":
        reasons.append("english_abstract_not_from_pdf")
    if len(parts["body"].split()) < 1500:
        reasons.append("body_under_1500_words")
    return not reasons, reasons


def process(meta: PaperMeta, pdf: Path) -> tuple[dict, dict]:
    parts = extract_pdf_v2(
        pdf,
        metadata_abstract=meta.abstract_meta,
        metadata_language=meta.language,
        title=meta.title,
    )
    ok, reasons = eligible(parts)
    row = {
        "paper_id": meta.paper_id, "title": meta.title, "year": meta.year,
        "eligible": ok, "failure_reasons": reasons,
        "body_words": len(parts["body"].split()),
        "provenance": parts["provenance"], "flags": parts["flags"],
    }
    doc = {
        "doc_id": meta.paper_id, "kind": "source", "lang": "fr",
        "title": meta.title, "abstract_fr": parts["fr_abstract"],
        "abstract_en": parts["en_abstract"], "body_fr": parts["body"],
        "keywords_fr": parts["keywords_fr"], "keywords_en": parts["keywords_en"],
        "year": meta.year, "provenance": parts["provenance"],
        "flags": parts["flags"], "extractor_version": 2, "eligible": ok,
    }
    return row, doc


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_stats(path: Path, rows: list[dict]) -> None:
    failures: dict[str, int] = {}
    for row in rows:
        for reason in row["failure_reasons"]:
            failures[reason] = failures.get(reason, 0) + 1
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "count", "share", "reason"])
        total = len(rows)
        writer.writerow(["scanned", total, 1.0, ""])
        writer.writerow(["french_body", sum("body_not_french_or_low_confidence" not in r["failure_reasons"] for r in rows), "", ""])
        writer.writerow(["extractable_english_abstract", sum(r["provenance"].get("en_abstract") == "pdf" for r in rows), "", ""])
        writer.writerow(["eligible", sum(r["eligible"] for r in rows), sum(r["eligible"] for r in rows) / max(1, total), ""])
        for reason, count in sorted(failures.items()):
            writer.writerow(["failure", count, count / max(1, total), reason])


def defect_counts(rows: list[dict]) -> dict[str, int]:
    counts = {"lowercase_starts": 0, "title_prefixed_en_abstract": 0, "front_matter_in_body": 0, "english_keywords_in_body": 0}
    for row in rows:
        en = row.get("abstract_en", "")
        body = row.get("body_fr", "")
        title = row.get("title", "")
        if en and en[0].islower():
            counts["lowercase_starts"] += 1
        if title and en.casefold().startswith(title.casefold()):
            counts["title_prefixed_en_abstract"] += 1
        if re.search(r"@|(?:université|university|cnrs|laboratoire|laboratory)\b", body, re.I):
            counts["front_matter_in_body"] += 1
        if re.search(r"\b(?:keywords?|mots[- ]cl[eé]s)\b", body, re.I):
            counts["english_keywords_in_body"] += 1
    return counts


def write_g1_comparison(out: Path, v2_rows: list[dict]) -> None:
    v1 = {row["stage"]: int(row["count"]) for row in csv.DictReader((out / "g1_stats.csv").open(encoding="utf-8")) if row["count"]}
    v2 = {
        "scanned": len(v2_rows),
        "french_body": sum("body_not_french_or_low_confidence" not in r["failure_reasons"] for r in v2_rows),
        "extractable_english_abstract": sum(r["provenance"].get("en_abstract") == "pdf" for r in v2_rows),
        "eligible": sum(r["eligible"] for r in v2_rows),
    }
    with (out / "g1_stats_x2.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "v1_count", "v2_count", "v1_share", "v2_share"])
        for stage in ("scanned", "french_body", "extractable_english_abstract", "eligible"):
            writer.writerow([
                stage, v1.get(stage, ""), v2[stage],
                v1.get(stage, "") / max(1, v1.get("scanned", 1)) if stage in v1 else "",
                v2[stage] / max(1, v2["scanned"]),
            ])


def write_blind(all_docs: dict[str, dict], authors: dict[str, list[str]], root: Path) -> None:
    def clean(text: str, title: str) -> str:
        value = re.sub(re.escape(title), "", text or "", flags=re.I)
        value = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "", value)
        value = re.sub(r"\b(?:keywords?|mots[- ]cl[eé]s)\b[^.]{0,180}[.:]", "", value, flags=re.I)
        return value
    def rows(ids, fields):
        output = []
        for paper_id in ids:
            doc = all_docs[paper_id]
            row = {"paper_id": paper_id}
            for field in fields:
                value = doc.get(field, "")
                if field == "body_fr":
                    value = " ".join(value.split()[:1500])
                    for author in authors.get(paper_id, []):
                        if author:
                            value = re.sub(re.escape(author), "", value, flags=re.I)
                row[field] = clean(value, doc["title"])
            output.append(row)
        return output
    write_jsonl(root / "data/human/t3_input.jsonl", rows(ASSIGNMENT_IDS, ["abstract_en", "abstract_fr"]))
    write_jsonl(root / "data/human/t4_passA_input.jsonl", rows(AUDIT_IDS, ["abstract_en"]))
    write_jsonl(root / "data/human/t4_passB_input.jsonl", rows(AUDIT_IDS, ["body_fr"]))

    forbidden = re.compile(r"@|(?:keywords?|mots[- ]cl[eé]s)", re.I)
    for path in ("t3_input.jsonl", "t4_passA_input.jsonl", "t4_passB_input.jsonl"):
        raw = (root / "data/human" / path).read_text(encoding="utf-8")
        if forbidden.search(raw):
            raise RuntimeError(f"blind leak check failed: {path}")


def main() -> None:
    root = ROOT
    out = root / "results/real"
    out.mkdir(parents=True, exist_ok=True)
    source_ids = json.loads((root / "data/processed/sources_final.json").read_text(encoding="utf-8"))
    metas = {m.paper_id: m for m in load_meta(root / "data/processed/extended_metadata.jsonl")}
    source_rows, source_docs = [], []
    for paper_id in source_ids:
        row, doc = process(metas[paper_id], root / "data/raw/extended" / f"{paper_id}.pdf")
        source_rows.append(row); source_docs.append(doc)
    replacements = []
    if not all(row["eligible"] for row in source_rows):
        for meta in load_meta(root / "data/processed/extended_metadata.jsonl"):
            if meta.paper_id in source_ids:
                continue
            row, doc = process(meta, root / "data/raw/extended" / f"{meta.paper_id}.pdf")
            if row["eligible"]:
                bad = next(index for index, value in enumerate(source_rows) if not value["eligible"])
                replacements.append({"old_id": source_rows[bad]["paper_id"], "new_id": meta.paper_id, "reasons": source_rows[bad]["failure_reasons"]})
                source_rows[bad], source_docs[bad] = row, doc
                if all(value["eligible"] for value in source_rows):
                    break
    with (out / "source_changes_x2.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["old_id", "new_id", "reasons"])
        writer.writeheader(); writer.writerows(replacements)
    write_jsonl(out / "papers_x2.jsonl", source_docs)
    write_stats(out / "g1_stats_x2.csv", source_rows)
    write_g1_comparison(out, source_rows)

    dev_ids = [json.loads(line)["paper_id"] for line in (root / "data/processed/development_exclusion.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    dev_docs = []
    dev_meta = {m.paper_id: m for m in load_meta(root / "data/processed/selected_metadata_new.jsonl")}
    for paper_id in dev_ids:
        dev_docs.append(process(dev_meta[paper_id], root / "data/raw" / f"{paper_id}.pdf")[1])
    write_jsonl(out / "development_papers_x2.jsonl", dev_docs)

    distractor_docs, distractor_rows = [], []
    for meta in load_meta(root / "data/processed/french_distractors_metadata.jsonl")[:250]:
        pdf = root / "data/raw/french_distractors" / f"{meta.paper_id}.pdf"
        row, doc = process(meta, pdf)
        row["scan_index"] = len(distractor_rows) + 1
        distractor_rows.append(row)
        if row["eligible"]:
            doc["kind"] = "fr_distractor"; distractor_docs.append(doc)
        if len(distractor_docs) >= 100:
            break
    write_jsonl(out / "french_distractors_native_x2.jsonl", distractor_docs)
    write_jsonl(root / "data/processed/french_distractors_native_v2.jsonl", distractor_docs)

    all_docs = {doc["doc_id"]: doc for doc in source_docs + dev_docs + distractor_docs}
    author_map = {meta.paper_id: meta.authors for meta in metas.values()}
    author_map.update({meta.paper_id: meta.authors for meta in dev_meta.values()})
    write_blind(all_docs, author_map, root)
    review = ["# Extraction review x2", "", f"- sources: {len(source_docs)}", f"- distractors eligible: {len(distractor_docs)}", ""]
    for doc in source_docs:
        review.extend([f"## {doc['doc_id']}", f"- flags: {', '.join(doc['flags']) or 'none'}", f"- provenance: `{json.dumps(doc['provenance'])}`", f"- French abstract: {doc['abstract_fr'][:300]}", f"- English abstract: {doc['abstract_en'][:300]}", f"- Body: {doc['body_fr'][:300]}", ""])
    (out / "extraction_review_x2.md").write_text("\n".join(review), encoding="utf-8")
    old_docs = json.loads("[]")
    old_docs = jsonl(out / "papers.jsonl") if (out / "papers.jsonl").exists() else []
    defects = {"v1": defect_counts(old_docs), "v2": defect_counts(source_docs)}
    with (out / "defect_counts_x2.json").open("w", encoding="utf-8") as handle:
        json.dump(defects, handle, indent=2)
    defect_report = ["# Extractor-v2 residual defect review", "", "| Defect | v1 | v2 |", "|---|---:|---:|"]
    for key in defects["v1"]:
        defect_report.append(f"| {key} | {defects['v1'][key]} | {defects['v2'][key]} |")
    defect_report.extend(["", "Residual cases are listed for human review; no automatic post-extraction fixes were applied.", ""])
    (out / "extraction_defects_x2.md").write_text("\n".join(defect_report), encoding="utf-8")
    print(f"sources={len(source_docs)} eligible={sum(r['eligible'] for r in source_rows)} replacements={len(replacements)}")
    print(f"dev={len(dev_docs)} distractors_scanned={len(distractor_rows)} distractors_eligible={len(distractor_docs)}")


if __name__ == "__main__":
    main()
