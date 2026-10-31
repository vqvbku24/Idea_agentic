import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MARKERS = r"(?:This paper|In this paper|We |This article|In this article|This work|In this work)"


def detect_and_strip_title(text: str) -> tuple[str, str]:
    if not text:
        return "", ""
    match = re.search(r"\s+(?=" + MARKERS + r"\b)", text)
    if not match:
        return "", text
    prefix = text[: match.start()].strip()
    words = prefix.split()
    if not prefix or not prefix[0].isupper() or "." in prefix or len(words) < 3 or len(words) > 25:
        return "", text
    cleaned = text[match.start() :].strip()
    return prefix, cleaned


def clean_sources() -> list[dict]:
    sources = json.load(open(ROOT / "data/processed/sources_final.json", encoding="utf-8"))
    papers = {}
    with open(ROOT / "results/real/papers_x2.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            papers[d["doc_id"]] = d

    records = []
    for pid in sources:
        doc = papers[pid]
        raw_en = doc.get("abstract_en", "")
        prefix, cleaned = detect_and_strip_title(raw_en)
        records.append({
            "paper_id": pid,
            "field": "abstract_en",
            "stripped_prefix": prefix,
            "cleaned_text": cleaned,
        })
    return records


def main() -> None:
    records = clean_sources()
    out_path = ROOT / "results/real/llm_input_cleaning.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["paper_id", "field", "stripped_prefix", "cleaned_text"])
        writer.writeheader()
        writer.writerows(records)

    stripped = [r for r in records if r["stripped_prefix"]]
    print(f"Stripped {len(stripped)} prefixes out of {len(records)} sources.")
    for r in stripped:
        print(f"  [{r['paper_id']}]: {r['stripped_prefix']!r}")


if __name__ == "__main__":
    main()
