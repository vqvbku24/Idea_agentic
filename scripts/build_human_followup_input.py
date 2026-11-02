from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.llm_input_cleaning import detect_and_strip_title

BANNER = "EXPLORATORY - agent-proxy labels, human verification pending"


def build_human_followup_input(seed: int = 13) -> None:
    sources_path = ROOT / "data/processed/sources_final.json"
    papers_path = ROOT / "results/real/papers_x2.jsonl"

    with sources_path.open(encoding="utf-8") as f:
        sources = sorted(json.load(f))

    with papers_path.open(encoding="utf-8") as f:
        papers = {d["doc_id"]: d for d in (json.loads(line) for line in f if line.strip())}

    rng = random.Random(seed)
    chosen_sources = rng.sample(sources, 10)

    # Build blind input records: paper_id, abstract_fr, abstract_en only (no title, authors, venue)
    records = []
    flagged_prefixes = []

    for pid in chosen_sources:
        p = papers[pid]
        a_fr = p.get("abstract_fr", "")
        a_en = p.get("abstract_en", "")

        # Strict leak check: check if starts with title-like prefix
        prefix, _ = detect_and_strip_title(a_en)
        if prefix:
            flagged_prefixes.append({
                "paper_id": pid,
                "prefix": prefix,
            })

        # Do NOT auto-redact as instructed
        records.append({
            "paper_id": pid,
            "abstract_fr": a_fr,
            "abstract_en": a_en,
        })

    out_jsonl = ROOT / "data/human/h_followup_input.jsonl"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Create empty human_queries_followup.csv
    out_csv = ROOT / "data/human/human_queries_followup.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_id", "query_text", "written_by", "date"])

    print(f"[{BANNER}]")
    print(f"Created {out_jsonl} with 10 blind sources.")
    print(f"Created empty {out_csv}.")
    print(f"Flagged title-like prefixes to confirm by human ({len(flagged_prefixes)}):")
    for item in flagged_prefixes:
        print(f"  - [{item['paper_id']}]: {item['prefix']!r}")


if __name__ == "__main__":
    build_human_followup_input()
