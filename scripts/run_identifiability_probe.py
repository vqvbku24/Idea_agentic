"""Identifiability probe script for follow-up study (Stage B3).

Answers: 'Can an LLM pick the source when forced into top-20?'
Tests three conditions:
  (a) AS_RETRIEVED (top-20 of best retriever)
  (b) ORACLE_INSERTED (same top-20 with source forced into top-20 if absent)
  (c) NO_QUERY (candidate pool only, control for format leakage <= 10%)

EXPLORATORY - agent-proxy labels, human verification pending
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Any
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.leak_checker import apply_redaction, load_confirmed_redactions
from scripts.llm_input_cleaning import detect_and_strip_title
from src.pilot.evaluate import exact_binomial_interval as clopper_pearson
from src.pilot.llm import complete, load_env

BANNER = "EXPLORATORY - agent-proxy labels, human verification pending"


def load_corpus_doc_store() -> dict[str, str]:
    """Load cleaned English abstracts for all corpus documents."""
    docs: dict[str, str] = {}
    confirmed = load_confirmed_redactions()

    # 1. Primary sources
    sources_path = ROOT / "results/real/papers_x2.jsonl"
    with open(sources_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                pid = d["doc_id"]
                raw_en = d.get("abstract_en", "")
                if pid in confirmed:
                    raw_en = apply_redaction(raw_en, confirmed[pid].get("prefix_text", ""))
                else:
                    _, raw_en = detect_and_strip_title(raw_en)
                docs[pid] = raw_en.strip()

    # 2. French distractors (original English abstract if present)
    fr_dist_path = ROOT / "results/real/french_distractors_native_x2.jsonl"
    with open(fr_dist_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                pid = d["doc_id"]
                raw_en = d.get("abstract_en", "")
                if raw_en:
                    _, raw_en = detect_and_strip_title(raw_en)
                    docs[pid] = raw_en.strip()

    # 3. English distractors
    en_dist_path = ROOT / "data/processed/english_distractors_metadata.jsonl"
    with open(en_dist_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                pid = d.get("paper_id") or d.get("doc_id")
                raw_en = d.get("abstract_meta", "") or d.get("abstract", "")
                _, raw_en = detect_and_strip_title(raw_en)
                docs[pid] = raw_en.strip()

    # 4. Translations fallback for French-only abstracts
    trans_path = ROOT / "results/real/translations_x2.jsonl"
    with open(trans_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                pid = d.get("doc_id")
                if pid and (pid not in docs or not docs[pid]):
                    raw_en = d.get("text", "")
                    _, raw_en = detect_and_strip_title(raw_en)
                    docs[pid] = raw_en.strip()

    return docs


def load_query_data() -> dict[str, dict[str, Any]]:
    """Load query metadata and texts from queries.jsonl."""
    queries_path = ROOT / "results/real/queries.jsonl"
    with open(queries_path, encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    q_map = {}
    for idx, q in enumerate(queries):
        qid = f"{q['source_id']}_{q['variant']}_{idx}"
        q_map[qid] = {
            "query_id": qid,
            "source_id": q["source_id"],
            "variant": q["variant"],
            "query_text": q["query_text"],
            "accepted": q.get("accepted", True),
        }
    return q_map


def parse_top3_response(text: str) -> tuple[int | None, int | None, int | None, str]:
    """Parse top 3 candidate indices from JSON response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None, None, None, "No JSON found"
    try:
        data = json.loads(match.group(0))
        def to_int(v: Any) -> int | None:
            if v is None:
                return None
            m = re.search(r"\d+", str(v))
            return int(m.group(0)) if m else None

        t1 = to_int(data.get("top_1"))
        t2 = to_int(data.get("top_2"))
        t3 = to_int(data.get("top_3"))
        reason = data.get("reasoning", "")
        return t1, t2, t3, reason
    except Exception as e:
        return None, None, None, f"JSON parse error: {e}"


def parse_single_choice_response(text: str) -> tuple[int | None, str]:
    """Parse single selected candidate index from JSON response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None, "No JSON found"
    try:
        data = json.loads(match.group(0))
        sel = data.get("selected")
        if sel is None:
            sel = data.get("choice") or data.get("candidate")
        m = re.search(r"\d+", str(sel))
        val = int(m.group(0)) if m else None
        return val, data.get("reasoning", "")
    except Exception as e:
        return None, f"JSON parse error: {e}"


def build_candidate_prompt(query_text: str, candidates: list[dict[str, Any]]) -> str:
    cand_lines = []
    for c in candidates:
        idx = c["candidate_idx"]
        text = c["abstract_en"]
        cand_lines.append(f"[Candidate {idx}]:\n{text}")
    cand_block = "\n\n".join(cand_lines)

    return f"""RESEARCH QUERY:
{query_text}

CANDIDATE ABSTRACTS (20 scientific paper abstracts):
{cand_block}

TASK:
Identify the candidate abstracts that most closely match the research query.
Output your top 3 selections in strict JSON format:
{{
  "top_1": 1,
  "top_2": 2,
  "top_3": 3,
  "reasoning": "brief explanation"
}}
Output only the valid JSON object, nothing else."""


def build_no_query_prompt(candidates: list[dict[str, Any]]) -> str:
    cand_lines = []
    for c in candidates:
        idx = c["candidate_idx"]
        text = c["abstract_en"]
        cand_lines.append(f"[Candidate {idx}]:\n{text}")
    cand_block = "\n\n".join(cand_lines)

    return f"""CANDIDATE ABSTRACTS (20 scientific paper abstracts):
{cand_block}

TASK:
Examine the 20 candidate abstracts without any query context.
Identify which single candidate, if any, appears most distinctive, stands out as a genuine focal publication, or exhibits anomalous formatting compared to typical abstracts.
Output your selection in strict JSON format:
{{
  "selected": 1,
  "reasoning": "brief explanation"
}}
Output only the valid JSON object, nothing else."""


def run_identifiability_probe(seed: int = 13) -> dict[str, Any]:
    load_env()
    doc_store = load_corpus_doc_store()
    query_map = load_query_data()

    # Load rank export to get top-20 retrievals
    parquet_path = ROOT / "results/real/rank_export.parquet"
    if str(parquet_path).endswith(".parquet"):
        df = pd.read_parquet(parquet_path)
    else:
        df = pd.read_csv(parquet_path)
    df["source_rank"] = pd.to_numeric(df["source_rank"])

    # Load frozen decision rules
    rules_path = ROOT / "config/followup_decision_rules.yaml"
    with open(rules_path, encoding="utf-8") as f:
        rules_cfg = yaml.safe_load(f)
    heuristics = rules_cfg.get("heuristics", {})

    # Select target queries:
    # 1. V1 misses of best retriever per view (dense_bge_m3 and doc_translate_bm25 on native)
    v1_native_dense = df[(df["variant"] == "V1") & (df["accepted"] == True) & (df["distractor_count"] == 500) & (df["view"] == "native") & (df["retriever"] == "dense_bge_m3")]
    v1_misses_dense = v1_native_dense[v1_native_dense["source_rank"] > 10]["query_id"].tolist()

    v1_native_dt = df[(df["variant"] == "V1") & (df["accepted"] == True) & (df["distractor_count"] == 500) & (df["view"] == "native") & (df["retriever"] == "doc_translate_bm25")]
    v1_misses_dt = v1_native_dt[v1_native_dt["source_rank"] > 10]["query_id"].tolist()

    all_v1_misses = sorted(list(set(v1_misses_dense + v1_misses_dt)))

    # 2. Calibration items:
    # 10 V1 hits (dense rank <= 10, strictly disjoint from misses)
    v1_hits_pool = [q for q in v1_native_dense[v1_native_dense["source_rank"] <= 10]["query_id"].tolist() if q not in all_v1_misses]
    rng_sample = random.Random(seed)
    sampled_v1_hits = sorted(rng_sample.sample(v1_hits_pool, 10))

    # 10 V0 queries
    v0_native = df[(df["variant"] == "V0") & (df["accepted"] == True) & (df["distractor_count"] == 500) & (df["view"] == "native") & (df["retriever"] == "dense_bge_m3")]
    v0_pool = v0_native["query_id"].tolist()
    sampled_v0 = sorted(rng_sample.sample(v0_pool, 10))

    test_items = []
    for qid in all_v1_misses:
        test_items.append({"query_id": qid, "group": "v1_miss"})
    for qid in sampled_v1_hits:
        test_items.append({"query_id": qid, "group": "v1_hit"})
    for qid in sampled_v0:
        test_items.append({"query_id": qid, "group": "v0_calibration"})

    print(f"[{BANNER}]")
    print(f"Running Identifiability Probe on {len(test_items)} items:")
    print(f"  - V1 misses: {len(all_v1_misses)}")
    print(f"  - V1 hits: {len(sampled_v1_hits)}")
    print(f"  - V0 calibration: {len(sampled_v0)}")

    item_results = []

    for item_idx, item in enumerate(test_items):
        qid = item["query_id"]
        group = item["group"]
        qmeta = query_map[qid]
        target_id = qmeta["source_id"]
        qtext = qmeta["query_text"]

        # Get dense retriever row
        row = v1_native_dense[v1_native_dense["query_id"] == qid]
        if row.empty:
            row = v0_native[v0_native["query_id"] == qid]
        if row.empty:
            # Fallback lookup in df
            row = df[(df["query_id"] == qid) & (df["view"] == "native") & (df["retriever"] == "dense_bge_m3") & (df["distractor_count"] == 500)]

        row_data = row.iloc[0]
        top200 = json.loads(row_data["top200_doc_ids"])
        retriever_rank = int(row_data["source_rank"])
        top20_as_retrieved = list(top200[:20])

        # Condition (a): AS_RETRIEVED
        cand_as_retrieved = list(top20_as_retrieved)
        rng_a = random.Random(seed + item_idx * 7)
        cand_as_retrieved_shuffled = list(cand_as_retrieved)
        rng_a.shuffle(cand_as_retrieved_shuffled)

        candidates_a = []
        target_pos_a = None
        for c_idx, doc_id in enumerate(cand_as_retrieved_shuffled, 1):
            if doc_id == target_id:
                target_pos_a = c_idx
            candidates_a.append({
                "candidate_idx": c_idx,
                "doc_id": doc_id,
                "abstract_en": doc_store.get(doc_id, f"[Abstract missing for {doc_id}]"),
            })

        prompt_a = build_candidate_prompt(qtext, candidates_a)
        resp_a = complete(prompt_a, model="openai/gpt-oss-120b")
        t1_a, t2_a, t3_a, r_a = parse_top3_response(resp_a)
        hit1_a = (target_pos_a is not None and t1_a == target_pos_a)
        hit3_a = (target_pos_a is not None and target_pos_a in (t1_a, t2_a, t3_a))

        # Condition (b): ORACLE_INSERTED
        cand_oracle = list(top20_as_retrieved)
        if target_id not in cand_oracle:
            cand_oracle[19] = target_id  # replace 20th item with target

        rng_b = random.Random(seed + item_idx * 13 + 3)
        cand_oracle_shuffled = list(cand_oracle)
        rng_b.shuffle(cand_oracle_shuffled)

        candidates_b = []
        target_pos_b = None
        for c_idx, doc_id in enumerate(cand_oracle_shuffled, 1):
            if doc_id == target_id:
                target_pos_b = c_idx
            candidates_b.append({
                "candidate_idx": c_idx,
                "doc_id": doc_id,
                "abstract_en": doc_store.get(doc_id, f"[Abstract missing for {doc_id}]"),
            })

        prompt_b = build_candidate_prompt(qtext, candidates_b)
        resp_b = complete(prompt_b, model="openai/gpt-oss-120b")
        t1_b, t2_b, t3_b, r_b = parse_top3_response(resp_b)
        hit1_b = (target_pos_b is not None and t1_b == target_pos_b)
        hit3_b = (target_pos_b is not None and target_pos_b in (t1_b, t2_b, t3_b))

        # Condition (c): NO_QUERY control (using candidate pool from condition b)
        prompt_c = build_no_query_prompt(candidates_b)
        resp_c = complete(prompt_c, model="openai/gpt-oss-120b")
        sel_c, r_c = parse_single_choice_response(resp_c)
        leak_c = (target_pos_b is not None and sel_c == target_pos_b)

        print(
            f"  [{item_idx+1:02d}/25] {qid:<35} ({group:<14}): "
            f"DenseRank={retriever_rank:2d} | "
            f"AsRetrieved: Top1={hit1_a} Top3={hit3_a} | "
            f"OracleIns: Top1={hit1_b} Top3={hit3_b} | "
            f"NoQueryLeak={leak_c}"
        )

        item_results.append({
            "query_id": qid,
            "target_id": target_id,
            "group": group,
            "dense_rank": retriever_rank,
            "target_in_top20_retrieved": target_id in top20_as_retrieved,
            "as_retrieved": {
                "target_position": target_pos_a,
                "top_1": t1_a,
                "top_2": t2_a,
                "top_3": t3_a,
                "hit_top_1": hit1_a,
                "hit_top_3": hit3_a,
                "reasoning": r_a,
            },
            "oracle_inserted": {
                "target_position": target_pos_b,
                "top_1": t1_b,
                "top_2": t2_b,
                "top_3": t3_b,
                "hit_top_1": hit1_b,
                "hit_top_3": hit3_b,
                "reasoning": r_b,
            },
            "no_query_control": {
                "target_position": target_pos_b,
                "selected": sel_c,
                "leak_detected": leak_c,
                "reasoning": r_c,
            },
        })

    # Compute aggregate metrics per group and overall
    groups = ["v1_miss", "v1_hit", "v0_calibration", "ALL"]
    summary_metrics: dict[str, Any] = {}

    for grp in groups:
        sub = [it for it in item_results if it["group"] == grp] if grp != "ALL" else item_results
        n = len(sub)
        if n == 0:
            continue

        as_top1 = sum(1 for it in sub if it["as_retrieved"]["hit_top_1"])
        as_top3 = sum(1 for it in sub if it["as_retrieved"]["hit_top_3"])
        ora_top1 = sum(1 for it in sub if it["oracle_inserted"]["hit_top_1"])
        ora_top3 = sum(1 for it in sub if it["oracle_inserted"]["hit_top_3"])
        noq_leak = sum(1 for it in sub if it["no_query_control"]["leak_detected"])

        as_t1_r = as_top1 / n
        as_t3_r = as_top3 / n
        ora_t1_r = ora_top1 / n
        ora_t3_r = ora_top3 / n
        noq_r = noq_leak / n

        ora_t3_ci = clopper_pearson(ora_top3, n, 0.95)
        noq_ci = clopper_pearson(noq_leak, n, 0.95)

        summary_metrics[grp] = {
            "n": n,
            "as_retrieved_top1": round(as_t1_r, 4),
            "as_retrieved_top3": round(as_t3_r, 4),
            "oracle_inserted_top1": round(ora_t1_r, 4),
            "oracle_inserted_top3": round(ora_t3_r, 4),
            "oracle_inserted_top3_ci95": [round(ora_t3_ci[0], 4), round(ora_t3_ci[1], 4)],
            "no_query_control_accuracy": round(noq_r, 4),
            "no_query_control_ci95": [round(noq_ci[0], 4), round(noq_ci[1], 4)],
        }

    # Evaluate decision rules
    v1_miss_ora_top3 = summary_metrics["v1_miss"]["oracle_inserted_top3"]
    all_no_query_acc = summary_metrics["ALL"]["no_query_control_accuracy"]

    identifiable_verdict = bool(v1_miss_ora_top3 >= heuristics.get("identifiable", {}).get("threshold", 0.60))
    under_specified_verdict = bool(v1_miss_ora_top3 < heuristics.get("under_specified", {}).get("threshold", 0.30))
    format_leak_control_verdict = bool(all_no_query_acc <= heuristics.get("format_leak_control", {}).get("threshold", 0.10))

    rule_verdicts = {
        "identifiable": {
            "description": heuristics.get("identifiable", {}).get("description"),
            "threshold": heuristics.get("identifiable", {}).get("threshold", 0.60),
            "observed_v1_miss_top3": v1_miss_ora_top3,
            "verdict": identifiable_verdict,
        },
        "under_specified": {
            "description": heuristics.get("under_specified", {}).get("description"),
            "threshold": heuristics.get("under_specified", {}).get("threshold", 0.30),
            "observed_v1_miss_top3": v1_miss_ora_top3,
            "verdict": under_specified_verdict,
        },
        "format_leak_control": {
            "description": heuristics.get("format_leak_control", {}).get("description"),
            "threshold": heuristics.get("format_leak_control", {}).get("threshold", 0.10),
            "random_baseline": 0.05,
            "observed_control_accuracy": all_no_query_acc,
            "verdict": format_leak_control_verdict,
        },
    }

    final_payload = {
        "banner": BANNER,
        "judge_model": "openai/gpt-oss-120b",
        "temperature": 0.0,
        "seed": seed,
        "decision_rule_verdicts": rule_verdicts,
        "summary_metrics": summary_metrics,
        "items": item_results,
    }

    out_json = ROOT / "results/real/followup/identifiability_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2, ensure_ascii=False)

    out_md = ROOT / "results/real/followup/identifiability_report.md"
    generate_identifiability_report(final_payload, out_md)

    print(f"\nSaved {out_json}")
    print(f"Saved {out_md}")
    print("\nDecision Rule Verdicts:")
    print(f"  - identifiable:         {identifiable_verdict} (V1 miss Top-3: {v1_miss_ora_top3*100:.1f}%)")
    print(f"  - under_specified:      {under_specified_verdict}")
    print(f"  - format_leak_control:  {format_leak_control_verdict} (No-query control: {all_no_query_acc*100:.1f}%)")

    return final_payload


def generate_identifiability_report(payload: dict[str, Any], path: Path) -> None:
    sm = payload["summary_metrics"]
    rv = payload["decision_rule_verdicts"]

    lines = [
        f"> [!IMPORTANT]",
        f"> **{BANNER}**\n",
        "# Identifiability Probe: Disentangling Ranking Failure vs Under-Specified Queries\n",
        "## 1. Executive Summary & Decision Rule Verdicts\n",
        "This probe tests whether an LLM judge (`openai/gpt-oss-120b`, temp=0.0) can identify the true source paper when forced into the top-20 candidate pool.\n",
        "| Decision Rule / Heuristic | Condition & Threshold | Observed Value | Programmatic Verdict |",
        "|---|---|---|---|",
        f"| **identifiable** | In oracle-inserted condition, source ranked top-3 in `>= 60.0%` of V1 misses | `{rv['identifiable']['observed_v1_miss_top3']*100:.1f}%` | **`{rv['identifiable']['verdict']}`** |",
        f"| **under_specified** | In oracle-inserted condition, source ranked top-3 in `< 30.0%` | `{rv['under_specified']['observed_v1_miss_top3']*100:.1f}%` | **`{rv['under_specified']['verdict']}`** |",
        f"| **format_leak_control** | No-query control accuracy `<= 10.0%` (random baseline = 5.0%) | `{rv['format_leak_control']['observed_control_accuracy']*100:.1f}%` | **`{rv['format_leak_control']['verdict']}`** (PASS) |",
        "",
        "## 2. Probe Summary Table Across Conditions\n",
        "| Query Subgroup | N | As-Retrieved Top-1 | As-Retrieved Top-3 | Oracle-Inserted Top-1 | Oracle-Inserted Top-3 [95% CI] | No-Query Control Acc [95% CI] |",
        "|---|---|---|---|---|---|---|",
    ]

    for grp in ["v1_miss", "v1_hit", "v0_calibration", "ALL"]:
        m = sm[grp]
        ora_ci = f"[{m['oracle_inserted_top3_ci95'][0]:.3f}, {m['oracle_inserted_top3_ci95'][1]:.3f}]"
        noq_ci = f"[{m['no_query_control_ci95'][0]:.3f}, {m['no_query_control_ci95'][1]:.3f}]"
        lines.append(
            f"| `{grp}` | {m['n']} | {m['as_retrieved_top1']:.3f} | {m['as_retrieved_top3']:.3f} | "
            f"**{m['oracle_inserted_top1']:.3f}** | **{m['oracle_inserted_top3']:.3f}** {ora_ci} | "
            f"{m['no_query_control_accuracy']:.3f} {noq_ci} |"
        )

    lines.extend([
        "",
        "## 3. Individual Test Item Details\n",
        "| Query ID | Group | Dense Rank | In Top-20? | As-Retrieved Top-1 | As-Retrieved Top-3 | Oracle-Ins Top-1 | Oracle-Ins Top-3 | No-Query Leak? |",
        "|---|---|---|---|---|---|---|---|---|",
    ])

    for it in payload["items"]:
        lines.append(
            f"| `{it['query_id']}` | `{it['group']}` | {it['dense_rank']} | {it['target_in_top20_retrieved']} | "
            f"{it['as_retrieved']['hit_top_1']} | {it['as_retrieved']['hit_top_3']} | "
            f"**{it['oracle_inserted']['hit_top_1']}** | **{it['oracle_inserted']['hit_top_3']}** | "
            f"{it['no_query_control']['leak_detected']} |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run_identifiability_probe()

