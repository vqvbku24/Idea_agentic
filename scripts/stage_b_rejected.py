from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLIND_IDS = [
    "2018.jeptalnrecital-long.14", "2019.jeptalnrecital-court.18",
    "2015.jeptalnrecital-court.13", "2017.jeptalnrecital-court.5",
    "2023.jeptalnrecital-long.11", "2015.jeptalnrecital-recital.3",
    "2021.jeptalnrecital-taln.4", "2020.jeptalnrecital-taln.1",
    "2021.jeptalnrecital-deft.1", "2019.jeptalnrecital-court.7",
]


def artifact_root() -> Path:
    for candidate in (ROOT / "results/real", ROOT / "results/real/real"):
        if (candidate / "papers_x2.jsonl").exists():
            return candidate
    raise FileNotFoundError("x2 artefacts not found")


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def clean(text: str, title: str, body: bool) -> str:
    value = re.sub(re.escape(title), "", text or "", flags=re.I)
    value = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "", value)
    value = re.sub(r"\b(?:keywords?|mots[- ]cl[eé]s)\b[^.]{0,180}[.:]", "", value, flags=re.I)
    if body:
        value = re.sub(r"(?im)^[^\n]*(?:université|university|cnrs|laboratoire|laboratory)[^\n]*$", "", value)
    return " ".join(value.split())


def strip_title_like_first_line(text: str) -> tuple[str, str]:
    value = " ".join(text.split())
    match = re.search(r"\s+(?=(?:In this|We |This |Our |To |For |A |An |The |As ))", value)
    if not match:
        return value, ""
    prefix = value[:match.start()].strip(" .:-")
    if not 3 <= len(prefix.split()) <= 18:
        return value, ""
    if sum(word[:1].isupper() for word in prefix.split() if word) < 2:
        return value, ""
    return value[match.end() - 1:].lstrip(), prefix


def main() -> None:
    data = artifact_root()
    docs = {row["doc_id"]: row for name in ("papers_x2.jsonl", "development_papers_x2.jsonl") for row in jsonl(data / name)}
    metas = {}
    for name in ("extended_metadata.jsonl", "selected_metadata_new.jsonl"):
        path = ROOT / "data/processed" / name
        if path.exists():
            metas.update({row["paper_id"]: row for row in jsonl(path)})
    checks = []
    for filename, fields in (
        ("t3_input.jsonl", ("abstract_en", "abstract_fr")),
        ("t4_passA_input.jsonl", ("abstract_en",)),
        ("t4_passB_input.jsonl", ("body_fr",)),
    ):
        output = []
        for paper_id in BLIND_IDS:
            doc = docs[paper_id]
            row = {"paper_id": paper_id}
            for field in fields:
                row[field] = clean(doc[field], doc["title"], field == "body_fr")
            output.append(row)
            text = " ".join(str(value) for key, value in row.items() if key != "paper_id")
            meta = metas.get(paper_id, {})
            authors = meta.get("authors", [])
            checks.extend([
                {"file": filename, "paper_id": paper_id, "check": "french_title",
                 "status": "FAIL" if doc["title"].casefold() in text.casefold() else "PASS"},
                {"file": filename, "paper_id": paper_id, "check": "english_title_like_start",
                 "status": "FAIL" if re.match(r"^(?:[A-Z][a-z]+ ){2,}", text) else "PASS"},
                {"file": filename, "paper_id": paper_id, "check": "english_keywords",
                 "status": "FAIL" if re.search(r"\b(?:keywords?|key words)\b", text, re.I) else "PASS"},
                {"file": filename, "paper_id": paper_id, "check": "at_sign",
                 "status": "FAIL" if "@" in text else "PASS"},
                {"file": filename, "paper_id": paper_id, "check": "author_names",
                 "status": "FAIL" if any(author.casefold() in text.casefold() for author in authors if author) else "PASS"},
                {"file": filename, "paper_id": paper_id, "check": "affiliations",
                 "status": "FAIL" if re.search(r"\b(?:université|university|cnrs|laboratoire|laboratory)\b", text, re.I) else "PASS"},
            ])
            if filename == "t4_passB_input.jsonl":
                checks.extend([
                    {"file": filename, "paper_id": paper_id, "check": "english_abstract_in_passB",
                     "status": "FAIL" if doc["abstract_en"].casefold() in text.casefold() else "PASS"},
                    {"file": filename, "paper_id": paper_id, "check": "first_page_front_matter",
                     "status": "FAIL" if not ("@" not in text and not re.search(r"\b(?:université|university|cnrs|laboratoire|laboratory)\b", text, re.I)) else "PASS"},
                ])
        (ROOT / "data/human" / filename).write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in output) + "\n", encoding="utf-8"
        )
    out = ROOT / "results/real"
    with (out / "blind_leak_checks_x2.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "paper_id", "check", "status"])
        writer.writeheader()
        writer.writerows(checks)

    source_ids = [row["doc_id"] for row in jsonl(data / "papers_x2.jsonl")]
    sensitivity = []
    for paper_id in source_ids:
        doc = docs[paper_id]
        text, removed = strip_title_like_first_line(doc.get("abstract_en", ""))
        if removed:
            sensitivity.append({
                "doc_id": paper_id,
                "title_like_prefix_removed": removed,
                "abstract_en_without_title_like_first_line": text,
            })
    (out / "sensitivity_native_abstract_x2.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in sensitivity) + ("\n" if sensitivity else ""),
        encoding="utf-8",
    )
    report = [
        "# Stage B decision: v2.1 rejected",
        "",
        "- Signed v2.1 review items: 26.",
        "- False positives (including unsure): 20.",
        "- Threshold: at most 2, and none among the 20 sources.",
        "- Outcome: **rejected**.",
        "- `src/pilot/pdf_extract.py` and `src/pilot/pdf_extract_v2.py` were not changed.",
        f"- Rebuilt blind inputs and leak checks: {len(checks)} checks, {sum(row['status'] == 'FAIL' for row in checks)} FAIL.",
        f"- Sensitivity output contains {len(sensitivity)} source abstracts with a removable title-like prefix; it is diagnostic only.",
    ]
    (out / "stage_b_rejected_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
