from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pilot.pdf_extract import detect_lang


FINAL = Path("data/processed/sources_final.json")
PAPERS = Path("results/real/papers.jsonl")
DEV = Path("data/processed/development_exclusion.jsonl")
REAL = Path("results/real")


def records() -> dict[str, dict]:
    return {
        row["doc_id"]: row
        for row in (json.loads(line) for line in PAPERS.read_text(encoding="utf-8").splitlines())
        if row.get("doc_id")
    }


def track(paper_id: str) -> str:
    match = re.search(r"jeptalnrecital-([^.]+)\.", paper_id)
    return match.group(1) if match else "unknown"


def write_review(final_ids: list[str], papers: dict[str, dict]) -> None:
    counts = Counter(track(paper_id) for paper_id in final_ids)
    lines = ["# Final source extraction review", "",
             "| Track | Count |", "|---|---:|"]
    lines.extend(f"| {name} | {counts[name]} |" for name in sorted(counts))
    lines.append("")
    for paper_id in final_ids:
        paper = papers[paper_id]
        suspicious = []
        en_lang, en_confidence = detect_lang(paper["abstract_en"])
        if en_lang == "fr" and en_confidence >= 0.9:
            suspicious.append("en_abstract_detected_as_french")
        if len(paper["abstract_en"].split()) < 40:
            suspicious.append("en_abstract_shorter_than_40_words")
        if len(paper["abstract_fr"].split()) < 40:
            suspicious.append("fr_abstract_shorter_than_40_words")
        if not paper["abstract_en"].strip():
            suspicious.append("en_abstract_missing")
        first_body = " ".join(paper["body_fr"].split()[:1500]).lower()
        title_count = first_body.count(" ".join(paper["title"].lower().split()))
        if title_count > 1:
            suspicious.append("source_title_repeated_in_first_1500_body_words")
        lines.extend([
            f"## {paper_id}",
            f"- track: `{track(paper_id)}`",
            f"- provenance: `{json.dumps(paper['provenance'], ensure_ascii=False)}`",
            f"- suspicious: `{';'.join(suspicious) or 'none'}`",
            "",
            f"**French abstract (first 300 chars):** {paper['abstract_fr'][:300]}",
            "",
            f"**English abstract (first 300 chars):** {paper['abstract_en'][:300]}",
            "",
            f"**Body (first 300 chars):** {paper['body_fr'][:300]}",
            "",
        ])
    (REAL / "extraction_review.md").write_text("\n".join(lines), encoding="utf-8")


def write_human(final_ids: list[str]) -> list[str]:
    selected = random.Random(13).sample(final_ids, 10)
    human_dir = Path("data/human")
    human_dir.mkdir(parents=True, exist_ok=True)
    with (human_dir / "human_assignments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["paper_id"])
        writer.writerows([[paper_id] for paper_id in selected])
    with (human_dir / "human_queries.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(["source_id", "query_text", "written_by", "date"])
    return selected


def write_facet_audit(final_ids: list[str], papers: dict[str, dict]) -> list[str]:
    selected = random.Random(13).sample(final_ids, 10)
    with (REAL / "facet_audit_sheet.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["paper_id", "facet", "in_en_abstract", "in_fr_body"])
        for paper_id in selected:
            for facet in ("problem", "mechanism", "setting", "outcome"):
                writer.writerow([paper_id, facet, "", ""])
    with (REAL / "facet_audit_materials.jsonl").open("w", encoding="utf-8") as handle:
        for paper_id in selected:
            paper = papers[paper_id]
            handle.write(json.dumps({
                "paper_id": paper_id,
                "english_abstract": paper["abstract_en"],
                "first_1500_body_words": " ".join(paper["body_fr"].split()[:1500]),
            }, ensure_ascii=False) + "\n")
    return selected


def main() -> None:
    final_ids = json.loads(FINAL.read_text(encoding="utf-8"))
    papers = records()
    missing = [paper_id for paper_id in final_ids if paper_id not in papers]
    if missing:
        raise RuntimeError(f"final source records missing from papers.jsonl: {missing}")
    dev_ids = [
        json.loads(line)["paper_id"]
        for line in DEV.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    Path("config").mkdir(exist_ok=True)
    Path("config/dev_prompt_subset.json").write_text(
        json.dumps(random.Random(13).sample(dev_ids, 3), indent=2) + "\n",
        encoding="utf-8",
    )
    write_review(final_ids, papers)
    print("human_assignments", write_human(final_ids))
    print("facet_audit", write_facet_audit(final_ids, papers))
    print("dev_prompt_subset", json.loads(Path("config/dev_prompt_subset.json").read_text()))


if __name__ == "__main__":
    main()
