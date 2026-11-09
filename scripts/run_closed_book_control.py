"""Closed-Book Contamination Control for English Terminology Dose-Response Pilot (Step E4).

Protocol:
  - Takes all accepted R0 queries from TEST (n=193).
  - Prompts the LLM (openai/gpt-oss-120b) with zero corpus context to identify the exact paper title.
  - Measures:
      1. Exact-title hit rate (normalized string match).
      2. Share of UNKNOWN responses overall and stratified by publication year (2018-2024).
      3. Share of sources published before training cutoff reported as unknown.

EXPLORATORY - agent-proxy labels, human verification pending
"""
from __future__ import annotations

import concurrent.futures
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pilot.evaluate import exact_binomial_interval as clopper_pearson
from src.pilot.llm import complete, load_env

BANNER = "EXPLORATORY - agent-proxy labels, human verification pending"


def normalize_title(title: str) -> str:
    norm = unicodedata.normalize("NFKD", title.lower())
    norm = "".join(c for c in norm if not unicodedata.combining(c))
    norm = re.sub(r"[^a-z0-9\s]", " ", norm)
    return " ".join(norm.split())


def run_closed_book():
    print(f"[{BANNER}]", flush=True)
    print("=== Running Closed-Book Contamination Control (R0 Queries) ===", flush=True)

    load_env()
    model = "openai/gpt-oss-120b"
    queries_path = ROOT / "results/real/en/queries_test_en.jsonl"
    sources_path = ROOT / "data/en/sources_300.jsonl"

    queries = [json.loads(line) for line in open(queries_path, encoding="utf-8")]
    r0_queries = [q for q in queries if q["level"] == "R0"]
    sources = {s["paper_id"]: s for s in [json.loads(line) for line in open(sources_path, encoding="utf-8")]}

    print(f"Total accepted R0 queries to evaluate: {len(r0_queries)}", flush=True)

    system_prompt = (
        "You are an expert NLP researcher. Given an idea-level description of a published NLP research paper, "
        "identify the exact title of the published paper. If you are not completely certain, respond with UNKNOWN. "
        "Output ONLY the exact title or UNKNOWN, with no other words or explanation."
    )

    def evaluate_one(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        idx, q = item
        sid = q["source_id"]
        src = sources.get(sid, {})
        true_title = src.get("title", "")
        pub_year = src.get("year", "unknown")
        q_text = q["query_text"]

        user_prompt = f"Paper description:\n{q_text}\n\nTitle:"
        raw_resp = complete(user_prompt, system=system_prompt, model=model).strip()

        clean_resp = raw_resp.split("\n")[0].strip().strip('"').strip("'")
        is_unknown = (
            "unknown" in clean_resp.lower()
            or clean_resp == ""
            or clean_resp.lower().startswith("i don't know")
            or clean_resp.lower().startswith("i do not know")
        )

        norm_pred = normalize_title(clean_resp)
        norm_true = normalize_title(true_title)
        exact_hit = (norm_pred == norm_true) and not is_unknown

        return {
            "banner": BANNER,
            "query_index": idx,
            "source_id": sid,
            "year": pub_year,
            "true_title": true_title,
            "predicted_title": clean_resp,
            "is_unknown": is_unknown,
            "exact_hit": exact_hit,
        }

    results = []
    start_time = time.time()
    items = list(enumerate(r0_queries, 1))

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(evaluate_one, it): it[0] for it in items}
        done_count = 0
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            results.append(res)
            done_count += 1
            if done_count % 25 == 0 or done_count == len(r0_queries):
                elapsed = time.time() - start_time
                print(f"  Processed {done_count}/{len(r0_queries)} queries ({elapsed:.1f}s)", flush=True)

    # Sort results by query_index
    results.sort(key=lambda r: r["query_index"])

    # Compute Summary Statistics
    n_total = len(results)
    n_hits = sum(1 for r in results if r["exact_hit"])
    n_unknown = sum(1 for r in results if r["is_unknown"])
    hit_rate = n_hits / n_total
    unknown_rate = n_unknown / n_total
    hit_ci = clopper_pearson(n_hits, n_total, 0.95)
    unknown_ci = clopper_pearson(n_unknown, n_total, 0.95)

    # Stratified by publication year
    by_year: dict[Any, dict[str, Any]] = {}
    years = sorted(set(r["year"] for r in results))
    for y in years:
        yr_res = [r for r in results if r["year"] == y]
        ny = len(yr_res)
        hy = sum(1 for r in yr_res if r["exact_hit"])
        uy = sum(1 for r in yr_res if r["is_unknown"])
        by_year[str(y)] = {
            "n": ny,
            "exact_hits": hy,
            "hit_rate": round(hy / ny, 4) if ny else 0.0,
            "unknowns": uy,
            "unknown_rate": round(uy / ny, 4) if ny else 0.0,
        }

    # Pre-2024 share (pre-cutoff)
    pre_2024_res = [r for r in results if isinstance(r["year"], int) and r["year"] < 2024]
    n_pre = len(pre_2024_res)
    u_pre = sum(1 for r in pre_2024_res if r["is_unknown"])
    pre_unknown_rate = u_pre / n_pre if n_pre else 0.0

    summary = {
        "banner": BANNER,
        "model": model,
        "n_evaluated_r0": n_total,
        "exact_title_hits": n_hits,
        "exact_title_hit_rate": round(hit_rate, 4),
        "exact_title_hit_rate_ci95": [round(hit_ci[0], 4), round(hit_ci[1], 4)],
        "unknown_count": n_unknown,
        "unknown_rate": round(unknown_rate, 4),
        "unknown_rate_ci95": [round(unknown_ci[0], 4), round(unknown_ci[1], 4)],
        "pre_2024_sources_count": n_pre,
        "pre_2024_unknown_count": u_pre,
        "pre_2024_unknown_share": round(pre_unknown_rate, 4),
        "by_year": by_year,
    }

    out_dir = ROOT / "results/real/en"
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "closed_book_results.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary_path = out_dir / "closed_book_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Closed-Book Contamination Results ===", flush=True)
    print(f"Total R0 Queries: {n_total}", flush=True)
    print(f"Exact-Title Hits: {n_hits} / {n_total} ({hit_rate*100:.2f}% [{hit_ci[0]*100:.2f}%, {hit_ci[1]*100:.2f}%])", flush=True)
    print(f"Unknown Responses: {n_unknown} / {n_total} ({unknown_rate*100:.2f}% [{unknown_ci[0]*100:.2f}%, {unknown_ci[1]*100:.2f}%])", flush=True)
    print(f"Pre-2024 Sources Unknown Share: {u_pre} / {n_pre} ({pre_unknown_rate*100:.2f}%)", flush=True)
    print(f"Wrote log: {jsonl_path}", flush=True)
    print(f"Wrote summary: {summary_path}", flush=True)


if __name__ == "__main__":
    run_closed_book()
