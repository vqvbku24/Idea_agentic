from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.leak_checker import apply_redaction, check_blind_entry, load_confirmed_redactions
from scripts.llm_input_cleaning import detect_and_strip_title

BANNER = "EXPLORATORY - agent-proxy labels, human verification pending"


def build_human_followup_input(seed: int = 13) -> None:
    sources_path = ROOT / "data/processed/sources_final.json"
    papers_path = ROOT / "results/real/papers_x2.jsonl"

    with sources_path.open(encoding="utf-8") as f:
        sources = sorted(json.load(f))

    with papers_path.open(encoding="utf-8") as f:
        papers = {d["doc_id"]: d for d in (json.loads(line) for line in f if line.strip())}

    confirmed_redactions = load_confirmed_redactions()

    rng = random.Random(seed)
    chosen_sources = rng.sample(sources, 10)

    # Build blind input records: paper_id, abstract_fr, abstract_en only (no title, authors, venue)
    records = []
    unconfirmed_flagged = []
    applied_redactions_log = []

    for pid in chosen_sources:
        p = papers[pid]
        a_fr = p.get("abstract_fr", "")
        a_en = p.get("abstract_en", "")

        # Apply confirmed redactions if present
        if pid in confirmed_redactions:
            redaction_info = confirmed_redactions[pid]
            prefix_to_strip = redaction_info.get("prefix_text", "")
            redacted_en = apply_redaction(a_en, prefix_to_strip)
            applied_redactions_log.append({
                "paper_id": pid,
                "prefix_stripped": prefix_to_strip.strip(),
            })
            a_en = redacted_en
        else:
            # Check if an unconfirmed title-like prefix exists
            prefix, _ = detect_and_strip_title(a_en)
            if prefix:
                unconfirmed_flagged.append({
                    "paper_id": pid,
                    "prefix": prefix,
                })

        records.append({
            "paper_id": pid,
            "abstract_fr": a_fr,
            "abstract_en": a_en,
        })

    # Save rebuilt blind input
    out_jsonl = ROOT / "data/human/h_followup_input.jsonl"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[{BANNER}]\n")
    print(f"Rebuilt {out_jsonl} with {len(records)} blind sources.")
    print(f"Applied human-confirmed redactions ({len(applied_redactions_log)}):")
    for item in applied_redactions_log:
        print(f"  - [{item['paper_id']}]: stripped {item['prefix_stripped']!r}")

    print(f"\nOther sources in the 10 with unconfirmed title-like runs ({len(unconfirmed_flagged)}):")
    if unconfirmed_flagged:
        for item in unconfirmed_flagged:
            print(f"  - [{item['paper_id']}]: {item['prefix']!r}")
    else:
        print("  None. All remaining 8 sources start with standard abstract text.")

    # Run strict leak check on all rebuilt entries
    print("\n=== STRICT LEAK CHECK (PASS/FAIL TABLE) ===")
    leak_results = []
    for r in records:
        pid = r["paper_id"]
        doc = papers[pid]
        checks = check_blind_entry(
            filename="h_followup_input.jsonl",
            row=r,
            doc_title=doc.get("title", ""),
            doc_abstract_en=doc.get("abstract_en", ""),
            authors=doc.get("authors", []),
        )
        checks_by_name = {c["check"]: c["status"] for c in checks}
        overall = "PASS" if all(s == "PASS" for s in checks_by_name.values()) else "FAIL"
        leak_results.append({
            "paper_id": pid,
            "overall": overall,
            "french_title": checks_by_name.get("french_title", "N/A"),
            "english_keywords": checks_by_name.get("english_keywords", "N/A"),
            "at_sign": checks_by_name.get("at_sign", "N/A"),
            "author_names": checks_by_name.get("author_names", "N/A"),
            "affiliations": checks_by_name.get("affiliations", "N/A"),
        })

    # Print formatted table
    header = f"| {'Paper ID':<28} | {'Overall':<7} | {'FR Title':<8} | {'Keywords':<8} | {'@ Sign':<6} | {'Authors':<7} | {'Affil':<6} |"
    separator = f"|{'-'*30}|{'-'*9}|{'-'*10}|{'-'*10}|{'-'*8}|{'-'*9}|{'-'*8}|"
    print(header)
    print(separator)
    for res in leak_results:
        print(
            f"| {res['paper_id']:<28} | {res['overall']:<7} | {res['french_title']:<8} | "
            f"{res['english_keywords']:<8} | {res['at_sign']:<6} | {res['author_names']:<7} | {res['affiliations']:<6} |"
        )

    all_pass = all(res["overall"] == "PASS" for res in leak_results)
    print(f"\nStrict leak check summary: {'ALL PASS (10/10)' if all_pass else 'FAILURES DETECTED'}")


if __name__ == "__main__":
    build_human_followup_input()
