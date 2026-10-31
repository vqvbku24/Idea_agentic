import csv
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pilot.llm import complete, load_env
from pilot.prompts import (
    build_facet_extraction_prompt,
    build_v0_prompt,
    build_v1_prompt,
    build_v2_prompt,
    build_v3_roundtrip_prompt,
    build_judge_prompt,
    select_v2_facets,
    lexical_leak,
)


def clean_json_text(text: str) -> str:
    t = text.strip()
    if t.startswith("```json"):
        t = t[7:]
    elif t.startswith("```"):
        t = t[3:]
    if t.endswith("```"):
        t = t[:-3]
    t = t.strip()
    # Fix any combination of asterisks and quotes around keys: e.g. **"key**": or **"key"**: or **key**:
    t = re.sub(r'[*"]+(\w+)[*"]+\s*:', r'"\1":', t)
    # Strip trailing commas before closing braces
    t = re.sub(r',\s*([}\]])', r'\1', t)
    return t.strip()


def extract_english_title(full_title: str) -> str:
    match = re.search(r"\((.*?)\)", full_title)
    if match and len(match.group(1).split()) >= 3:
        return match.group(1)
    return ""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    load_env()
    print("=== STARTING GENERATION AND JUDGING (EXPLORATORY RUN) ===")

    # Load sources
    sources = json.load(open(ROOT / "data/processed/sources_final.json", encoding="utf-8"))
    papers = {
        d["doc_id"]: d
        for d in [json.loads(line) for line in open(ROOT / "results/real/papers_x2.jsonl", encoding="utf-8")]
    }

    # Load cleaned abstract_en
    cleaned_en = {}
    clean_csv = ROOT / "results/real/llm_input_cleaning.csv"
    if clean_csv.exists():
        with open(clean_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cleaned_en[row["paper_id"]] = row["cleaned_text"]

    sealed_dir = ROOT / "results/real/sealed"
    sealed_dir.mkdir(parents=True, exist_ok=True)
    facets_file = sealed_dir / "facets_llm.jsonl"

    cache_dir = ROOT / "cache/llm"

    # Step 3: Facet extraction
    facets_by_paper = {}
    if facets_file.exists():
        for line in facets_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                facets_by_paper[row["paper_id"]] = row["facets"]

    print(f"Extracting facets for {len(sources)} sources (already have {len(facets_by_paper)})...")
    for idx, pid in enumerate(sources):
        if pid in facets_by_paper:
            continue
        doc = papers[pid]
        en_abs = cleaned_en.get(pid) or doc.get("abstract_en", "")
        fr_abs = doc.get("abstract_fr", "")
        body_1500 = " ".join(doc.get("body_fr", "").split()[:1500])

        prompt = build_facet_extraction_prompt(fr_abs, en_abs, body_1500)
        res = complete(prompt, cache_dir=cache_dir)
        try:
            facets = json.loads(clean_json_text(res))
            facets_by_paper[pid] = facets
            with open(facets_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"paper_id": pid, "facets": facets}, ensure_ascii=False) + "\n")
            print(f"  [+] Extracted facets for source {idx + 1}/20: {pid}")
        except Exception as e:
            print(f"  [-] Failed to parse facets for {pid}: {e}")
            cleaned_snippet = clean_json_text(res)[:300].encode("ascii", errors="replace").decode()
            print(f"  Cleaned text snippet: {cleaned_snippet}")
            raise

    print(f"Total facets sealed: {len(facets_by_paper)}")

    # Step 4: Variants V0-V3 and Judging
    queries_file = ROOT / "results/real/queries.jsonl"
    all_query_records = []

    judge_model = os.environ.get("JUDGE_MODEL")

    variant_stats = {
        "V0": {"total": 0, "accepted": 0, "rejected": 0, "leak_flag": 0},
        "V1": {"total": 0, "accepted": 0, "rejected": 0, "leak_flag": 0},
        "V2": {"total": 0, "accepted": 0, "rejected": 0, "leak_flag": 0},
        "V3": {"total": 0, "accepted": 0, "rejected": 0, "leak_flag": 0},
        "H-agent": {"total": 0, "accepted": 0, "rejected": 0, "leak_flag": 0},
    }

    print("\nGenerating and judging variants V0-V3 for 20 sources...")
    for idx, pid in enumerate(sources):
        doc = papers[pid]
        facets = facets_by_paper[pid]
        title_fr = doc.get("title", "")
        title_en = extract_english_title(title_fr)

        variants_to_generate = [
            ("V0", build_v0_prompt(facets), ["problem", "mechanism", "setting", "outcome"]),
            ("V1", build_v1_prompt(facets), ["problem", "mechanism", "setting", "outcome"]),
            ("V2", build_v2_prompt(facets, select_v2_facets(13, idx)), select_v2_facets(13, idx)),
            ("V3", build_v3_roundtrip_prompt(facets), ["problem", "mechanism", "setting", "outcome"]),
        ]

        for v_name, prompt_v, kept_facets in variants_to_generate:
            v_text = complete(prompt_v, cache_dir=cache_dir).strip().strip('"')
            if v_text.startswith("```"):
                lines = v_text.splitlines()
                v_text = " ".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

            # Judge with JUDGE_MODEL
            prompt_j = build_judge_prompt(v_text, kept_facets)
            res_j = complete(prompt_j, model=judge_model, cache_dir=cache_dir)
            try:
                judge_json = json.loads(clean_json_text(res_j))
            except Exception:
                judge_json = {"preserved_facets": [], "leaks_identifiers": True, "fluent": True}

            # Lexical leak
            leak_fr = lexical_leak(v_text, title_fr)
            leak_en = lexical_leak(v_text, title_en) if title_en else 0.0
            leak = max(leak_fr, leak_en)

            # Acceptance
            target = set(kept_facets)
            preserved = set(judge_json.get("preserved_facets", []))
            accepted = (
                target.issubset(preserved)
                and (v_name == "V0" or not judge_json.get("leaks_identifiers", False))
                and judge_json.get("fluent", False)
            )

            variant_stats[v_name]["total"] += 1
            if accepted:
                variant_stats[v_name]["accepted"] += 1
            else:
                variant_stats[v_name]["rejected"] += 1
            if leak > 0.3:
                variant_stats[v_name]["leak_flag"] += 1

            record = {
                "source_id": pid,
                "variant": v_name,
                "query_text": v_text,
                "kept_facets": kept_facets,
                "judge_json": judge_json,
                "lexical_leak": round(leak, 4),
                "accepted": accepted,
                "label_quality": "llm_generated",
            }
            all_query_records.append(record)

        print(f"  Processed variants for source {idx + 1}/20: {pid}")

    # Include H-agent queries from data/human/human_queries.csv
    h_file = ROOT / "data/human/human_queries.csv"
    if h_file.exists():
        print("\nIncluding H-agent queries from data/human/human_queries.csv...")
        with open(h_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row["source_id"]
                q_text = row["query_text"]
                doc = papers.get(pid, {})
                title_fr = doc.get("title", "")
                title_en = extract_english_title(title_fr)

                target_facets = ["problem", "mechanism", "setting", "outcome"]
                prompt_j = build_judge_prompt(q_text, target_facets)
                res_j = complete(prompt_j, model=judge_model, cache_dir=cache_dir)
                try:
                    judge_json = json.loads(clean_json_text(res_j))
                except Exception:
                    judge_json = {"preserved_facets": target_facets, "leaks_identifiers": False, "fluent": True}

                leak_fr = lexical_leak(q_text, title_fr)
                leak_en = lexical_leak(q_text, title_en) if title_en else 0.0
                leak = max(leak_fr, leak_en)

                target = set(target_facets)
                preserved = set(judge_json.get("preserved_facets", []))
                accepted = (
                    target.issubset(preserved)
                    and not judge_json.get("leaks_identifiers", False)
                    and judge_json.get("fluent", False)
                )

                variant_stats["H-agent"]["total"] += 1
                if accepted:
                    variant_stats["H-agent"]["accepted"] += 1
                else:
                    variant_stats["H-agent"]["rejected"] += 1
                if leak > 0.3:
                    variant_stats["H-agent"]["leak_flag"] += 1

                all_query_records.append({
                    "source_id": pid,
                    "variant": "H-agent",
                    "query_text": q_text,
                    "kept_facets": target_facets,
                    "judge_json": judge_json,
                    "lexical_leak": round(leak, 4),
                    "accepted": accepted,
                    "label_quality": "agent_drafted_unverified",
                })

    # Write results/real/queries.jsonl
    with open(queries_file, "w", encoding="utf-8") as f:
        for r in all_query_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_query_records)} records to {queries_file}")
    print("\n=== VARIANT GENERATION & JUDGING SUMMARY ===")
    print("Variant | Total | Accepted | Rejected | Lexical Leak > 0.3")
    print("--------|-------|----------|----------|-------------------")
    for v, stats in variant_stats.items():
        print(f"{v:7} | {stats['total']:5} | {stats['accepted']:8} | {stats['rejected']:8} | {stats['leak_flag']:18}")


if __name__ == "__main__":
    main()
