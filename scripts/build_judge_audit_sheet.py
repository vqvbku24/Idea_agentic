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


def build_judge_audit_sheet(seed: int = 13) -> None:
    queries_path = ROOT / "results/real/queries.jsonl"
    papers_path = ROOT / "results/real/papers_x2.jsonl"

    with queries_path.open(encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    with papers_path.open(encoding="utf-8") as f:
        papers = {d["doc_id"]: d for d in (json.loads(line) for line in f if line.strip())}

    accepted = [q for q in queries if q.get("accepted", False)]
    rejected = [q for q in queries if not q.get("accepted", False)]

    rng = random.Random(seed)

    # 1. Sample 10 accepted items: 2 from each variant
    accepted_by_v: dict[str, list[dict]] = {}
    for q in accepted:
        accepted_by_v.setdefault(q["variant"], []).append(q)

    sample_accepted = []
    for v in ["V0", "V1", "V2", "V3", "H-agent"]:
        pool = sorted(accepted_by_v[v], key=lambda x: (x["source_id"], x["query_text"]))
        sample_accepted.extend(rng.sample(pool, 2))

    # 2. Sample 10 rejected items:
    # At least 5 V2 rejections; rest spread over V1, V3, V0, H-agent.
    # Available: V1 (1), V3 (1), V0 (0), V2 (7), H-agent (4).
    v1_rej = [q for q in rejected if q["variant"] == "V1"]
    v3_rej = [q for q in rejected if q["variant"] == "V3"]
    v2_rej = sorted([q for q in rejected if q["variant"] == "V2"], key=lambda x: (x["source_id"], x["query_text"]))
    h_rej = sorted([q for q in rejected if q["variant"] == "H-agent"], key=lambda x: (x["source_id"], x["query_text"]))

    # Sample 6 V2 (>= 5) and 2 H-agent to get exactly 10 rejected items
    v2_sample = rng.sample(v2_rej, 6)
    h_sample = rng.sample(h_rej, 2)
    sample_rejected = v1_rej + v3_rej + v2_sample + h_sample

    all_20 = sample_accepted + sample_rejected
    # Shuffle the combined 20 items with seed 13 to fully blind the human reviewer
    rng.shuffle(all_20)

    audit_rows = []
    key_rows = []

    for idx, q in enumerate(all_20, start=1):
        item_id = f"item_{idx:02d}"
        source_id = q["source_id"]
        paper = papers[source_id]
        raw_en = paper.get("abstract_en", "")
        prefix, cleaned = detect_and_strip_title(raw_en)
        source_abstract = cleaned if prefix else raw_en

        kept_facets_list = q.get("kept_facets", [])
        kept_facets_str = ", ".join(kept_facets_list)
        query_text = q.get("query_text", "")

        # Audit sheet row (human columns left strictly EMPTY)
        audit_rows.append({
            "item_id": item_id,
            "source_english_abstract": source_abstract,
            "kept_facets": kept_facets_str,
            "query_text": query_text,
            "human_faithful": "",
            "human_leaks": "",
            "human_fluent": "",
        })

        # Judge details for sealed key
        judge_json = q.get("judge_json", {})
        preserved = set(judge_json.get("preserved_facets", []))
        kept_set = set(kept_facets_list)
        judge_faithful = kept_set.issubset(preserved)
        judge_leaks = bool(judge_json.get("leaks_identifiers", False))
        judge_fluent = bool(judge_json.get("fluent", False))
        judge_accepted = bool(q.get("accepted", False))
        verdict = "accepted" if judge_accepted else "rejected"

        key_rows.append({
            "item_id": item_id,
            "source_id": source_id,
            "variant": q.get("variant"),
            "accepted": judge_accepted,
            "judge_verdict": verdict,
            "judge_faithful": judge_faithful,
            "judge_leaks": judge_leaks,
            "judge_fluent": judge_fluent,
            "judge_json": json.dumps(judge_json),
            "lexical_leak": q.get("lexical_leak", 0.0),
        })

    # Save audit sheet
    audit_csv = ROOT / "results/real/followup/judge_audit_sheet.csv"
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    with audit_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "item_id", "source_english_abstract", "kept_facets", "query_text",
            "human_faithful", "human_leaks", "human_fluent"
        ])
        writer.writeheader()
        writer.writerows(audit_rows)

    # Save sealed key
    key_csv = ROOT / "results/real/followup/sealed/judge_key.csv"
    key_csv.parent.mkdir(parents=True, exist_ok=True)
    with key_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "item_id", "source_id", "variant", "accepted", "judge_verdict",
            "judge_faithful", "judge_leaks", "judge_fluent", "judge_json", "lexical_leak"
        ])
        writer.writeheader()
        writer.writerows(key_rows)

    print(f"[{BANNER}]")
    print(f"Generated {audit_csv} with 20 items (human columns empty).")
    print(f"Generated {key_csv} with sealed judge verdicts.")


if __name__ == "__main__":
    build_judge_audit_sheet()
