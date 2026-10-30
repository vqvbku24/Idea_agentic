from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pilot.data_acl import PaperMeta
from pilot.retrievers import DenseBgeM3

BLIND_IDS = [
    "2018.jeptalnrecital-long.14", "2019.jeptalnrecital-court.18",
    "2015.jeptalnrecital-court.13", "2017.jeptalnrecital-court.5",
    "2023.jeptalnrecital-long.11", "2015.jeptalnrecital-recital.3",
    "2021.jeptalnrecital-taln.4", "2020.jeptalnrecital-taln.1",
    "2021.jeptalnrecital-deft.1", "2019.jeptalnrecital-court.7",
]


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def first_span(text: str, pattern: str) -> str:
    match = re.search(pattern, text, re.I)
    if not match:
        return ""
    start = max(0, match.start() - 60)
    return " ".join(text[start:match.end() + 140].split())


def write_blind_and_checks(docs: dict[str, dict], metas: dict[str, PaperMeta]) -> list[dict]:
    def clean(text: str, title: str, authors: list[str], body: bool = False) -> str:
        value = re.sub(re.escape(title), "", text or "", flags=re.I)
        value = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "", value)
        value = re.sub(r"\b(?:keywords?|mots[- ]cl[eé]s)\b[^.]{0,180}[.:]", "", value, flags=re.I)
        if body:
            for author in authors:
                value = re.sub(re.escape(author), "", value, flags=re.I)
            value = re.sub(r"(?im)^[^\n]*(?:université|university|cnrs|laboratoire|laboratory)[^\n]*$", "", value)
        return " ".join(value.split())

    records = []
    for filename, fields in (
        ("t3_input.jsonl", ("abstract_en", "abstract_fr")),
        ("t4_passA_input.jsonl", ("abstract_en",)),
        ("t4_passB_input.jsonl", ("body_fr",)),
    ):
        output = []
        for paper_id in BLIND_IDS:
            doc, meta = docs[paper_id], metas[paper_id]
            row = {"paper_id": paper_id}
            for field in fields:
                row[field] = clean(
                    " ".join(doc.get("body_fr", "").split()[:1500]) if field == "body_fr" else doc.get(field, ""),
                    doc["title"],
                    meta.authors,
                    body=field == "body_fr",
                )
            output.append(row)
        path = ROOT / "data/human" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in output) + "\n",
            encoding="utf-8",
        )
        for row in output:
            doc, meta = docs[row["paper_id"]], metas[row["paper_id"]]
            text = " ".join(str(value) for key, value in row.items() if key != "paper_id")
            checks = {
                "french_title": doc["title"].casefold() not in text.casefold(),
                "english_title_like_start": not bool(re.match(r"^(?:[A-Z][a-z]+ ){2,}", text)),
                "english_keywords": not bool(re.search(r"\b(?:keywords?|key words)\b", text, re.I)),
                "at_sign": "@" not in text,
                "author_names": not any(author.casefold() in text.casefold() for author in meta.authors if author),
                "affiliations": not bool(re.search(r"\b(?:université|university|cnrs|laboratoire|laboratory)\b", text, re.I)),
            }
            if filename == "t4_passB_input.jsonl":
                checks["english_abstract_in_passB"] = doc["abstract_en"].casefold() not in text.casefold()
                checks["first_page_front_matter"] = checks["affiliations"] and checks["at_sign"]
            for check, passed in checks.items():
                records.append({"file": filename, "paper_id": row["paper_id"], "check": check, "status": "PASS" if passed else "FAIL"})
    return records


def main() -> None:
    out = ROOT / "results/real"
    docs = {row["doc_id"]: row for row in jsonl(out / "papers_x2.jsonl")}
    docs.update({row["doc_id"]: row for row in jsonl(out / "development_papers_x2.jsonl")})
    metas = {}
    for path in ("data/processed/extended_metadata.jsonl", "data/processed/selected_metadata_new.jsonl"):
        for row in jsonl(ROOT / path):
            metas[row["paper_id"]] = PaperMeta(**row)
    checks = write_blind_and_checks(docs, metas)
    with (out / "blind_leak_checks_x2.csv").open("w", encoding="utf-8") as handle:
        handle.write("file,paper_id,check,status\n")
        for row in checks:
            handle.write(f"{row['file']},{row['paper_id']},{row['check']},{row['status']}\n")

    residual = ["# Residual extractor-v2 defects", "", "| Set | Paper ID | Defect | Offending span |", "|---|---|---|---|"]
    for set_name, path in (("source", out / "papers_x2.jsonl"), ("dev", out / "development_papers_x2.jsonl"), ("distractor", out / "french_distractors_native_x2.jsonl")):
        for row in jsonl(path):
            body = row.get("body_fr", "")
            for defect, pattern in (
                ("front_matter", r"@|(?:université|university|cnrs|laboratoire|laboratory)\b"),
                ("english_keywords", r"\b(?:keywords?|key words)\b"),
            ):
                span = first_span(body, pattern)
                if span:
                    safe_span = span.replace("|", "\\|")
                    residual.append(f"| {set_name} | {row['doc_id']} | {defect} | {safe_span} |")
    (out / "residual_defects_x2.md").write_text("\n".join(residual) + "\n", encoding="utf-8")

    sources = [docs[row["doc_id"]] for row in jsonl(out / "papers_x2.jsonl")]
    texts = [doc["abstract_en"] for doc in sources[:10]] + [
        f"{doc['title']}\n{doc['abstract_fr']}\n{' '.join(doc['body_fr'].split()[:1500])}"
        for doc in sources[:10]
    ]
    dense = DenseBgeM3({"audit": {str(i): text for i, text in enumerate(texts)}}, ROOT / "data/cache/reference-audit", 1024)
    import numpy as np
    from sentence_transformers import SentenceTransformer
    reference = SentenceTransformer("BAAI/bge-m3")
    rows = []
    ours = dense.embeddings["audit"]
    for max_length in (1024, 8192):
        reference.max_seq_length = max_length
        vectors = reference.encode(texts, batch_size=8, normalize_embeddings=True, show_progress_bar=False)
        cosine = np.sum(vectors * ours, axis=1)
        for kind, values in (("abstract_en", cosine[:10]), ("native", cosine[10:])):
            rows.append((max_length, kind, float(values.min()), float(values.mean()), int(np.sum(values < 0.99))))
    with (out / "dense_reference_x2.csv").open("w", encoding="utf-8") as handle:
        handle.write("reference_max_length,text_type,min_cosine,mean_cosine,count_below_0.99\n")
        for row in rows:
            handle.write(",".join(map(str, row)) + "\n")
    diagnostics_path = out / "retrieval_v2_diagnostics.md"
    diagnostics = diagnostics_path.read_text(encoding="utf-8")
    diagnostics = diagnostics.replace("See `config/amendment_1.yaml`", "See `config/amendment_2.yaml`")
    diagnostics += "\n## B2. Dense reference recheck with matched max length\n\n"
    diagnostics += "| Reference max length | Text type | Min cosine | Mean cosine | Count < 0.99 |\n|---:|---|---:|---:|---:|\n"
    for row in rows:
        diagnostics += f"| {row[0]} | {row[1]} | {row[2]:.6f} | {row[3]:.6f} | {row[4]} |\n"
    abstract_1024 = next(row for row in rows if row[0] == 1024 and row[1] == "abstract_en")
    diagnostics += f"\nAbstract cosines >= 0.99 at reference max length 1024: {'yes' if abstract_1024[4] == 0 else 'no'}.\n"
    diagnostics_path.write_text(diagnostics, encoding="utf-8")
    print(f"leak_checks={len(checks)} failures={sum(row['status'] == 'FAIL' for row in checks)}")
    print(f"dense_reference_rows={len(rows)} abstract_1024_below_0.99={abstract_1024[4]}")


if __name__ == "__main__":
    main()
