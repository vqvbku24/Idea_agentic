"""TEST Run Pipeline & Local BM25 Runner for English Terminology Dose-Response Pilot (Step E3).

Executes:
  1. Key terms extraction & query generation for 270 TEST papers across 4 retention levels (R100, R50, R25, R0)
  2. Deterministic checks & Judge evaluation (openai/gpt-oss-120b)
  3. Incremental checkpointing
  4. Local Okapi BM25 ranking across all test queries against 5,270-document corpus
  5. Computes Recall@10, Recall@50, Recall@100, MRR with Clopper-Pearson 95% CIs
  6. Exports Colab Kit and manifest for dense_bge_m3 / hybrid_rrf GPU execution

EXPLORATORY - agent-proxy labels, human verification pending
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pilot.evaluate import exact_binomial_interval as clopper_pearson
from src.pilot.llm import complete, load_env
from src.pilot.retrievers import BM25

BANNER = "EXPLORATORY - agent-proxy labels, human verification pending"


def stem_token(word: str) -> str:
    w = re.sub(r"[^a-zA-Z]", "", word).lower()
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("es") and len(w) > 3:
        w = w[:-2]
    elif w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        w = w[:-1]

    if w.endswith("ing") and len(w) > 4:
        w = w[:-3]
    elif w.endswith("ed") and len(w) > 4:
        w = w[:-2]
    elif w.endswith("tion") and len(w) > 5:
        w = w[:-4]
    elif w.endswith("ment") and len(w) > 5:
        w = w[:-4]
    elif w.endswith("able") and len(w) > 5:
        w = w[:-4]
    elif w.endswith("al") and len(w) > 4:
        w = w[:-2]
    return w


def is_term_present(term: str, query: str) -> bool:
    query_tokens = [stem_token(w) for w in re.findall(r"\b[a-zA-Z]+\b", query)]
    term_words = [stem_token(w) for w in re.findall(r"\b[a-zA-Z]+\b", term)]
    if not term_words:
        return False
    if len(term_words) == 1:
        return term_words[0] in query_tokens

    for i in range(len(query_tokens) - len(term_words) + 1):
        if query_tokens[i : i + len(term_words)] == term_words:
            return True

    if all(tw in query_tokens for tw in term_words):
        return True
    return False


def title_leak_relative(query_text: str, title: str, kept_terms: list[str]) -> float:
    title_tokens = {token for token in re.findall(r"\w+", title.lower()) if len(token) >= 5}
    kept_tokens = {token for term in kept_terms for token in re.findall(r"\w+", term.lower()) if len(token) >= 4}
    title_tokens_outside_kept = title_tokens - kept_tokens
    if not title_tokens_outside_kept:
        return 0.0
    query_tokens = set(re.findall(r"\w+", query_text.lower()))
    return len(title_tokens_outside_kept & query_tokens) / len(title_tokens_outside_kept)


def extract_key_terms(abstract: str, prompt_cfg: dict[str, Any], model: str) -> list[str]:
    user_prompt = prompt_cfg["user_template"].format(abstract=abstract)
    system = prompt_cfg.get("system")
    resp = complete(user_prompt, system=system, model=model)

    match = re.search(r"\[.*\]", resp, re.DOTALL)
    if not match:
        return []
    try:
        raw_terms = json.loads(match.group(0))
    except Exception:
        return []

    abstract_lower = abstract.lower()
    valid_terms = []
    seen = set()
    for t in raw_terms:
        t_clean = t.strip()
        t_lower = t_clean.lower()
        if t_lower in abstract_lower and t_lower not in seen and len(t_clean) >= 3:
            seen.add(t_lower)
            valid_terms.append(t_clean)

    return valid_terms[:12]


def check_deterministic(
    query_text: str,
    title: str,
    kept_terms: list[str],
    removed_terms: list[str],
    all_terms: list[str],
    target_retention: int,
) -> tuple[bool, str, dict[str, Any]]:
    words = query_text.split()
    word_count = len(words)

    if word_count < 40 or word_count > 80:
        return False, f"word_count_out_of_range ({word_count})", {}

    leak = title_leak_relative(query_text, title, kept_terms)
    if leak > 0.30:
        return False, f"lexical_leak_high ({leak:.3f} > 0.30)", {}

    removed_present = []
    for rt in removed_terms:
        if is_term_present(rt, query_text):
            removed_present.append(rt)

    for rt in removed_terms:
        digits = re.findall(r"\d+", rt)
        for d in digits:
            if re.search(r"\b" + re.escape(d) + r"\b", query_text):
                removed_present.append(f"digit:{d}")
        acronyms = [w for w in rt.split() if w.isupper() and len(w) >= 2]
        for acr in acronyms:
            if re.search(r"\b" + re.escape(acr) + r"\b", query_text):
                removed_present.append(f"acronym:{acr}")

    if removed_present:
        return False, f"removed_terms_present: {', '.join(removed_present[:2])}", {}

    present_all = [t for t in all_terms if is_term_present(t, query_text)]
    realized_count = len(present_all)

    if abs(realized_count - target_retention) > 1:
        return (
            False,
            f"retention_mismatch (realized {realized_count} vs target {target_retention} +- 1)",
            {},
        )

    return True, "PASS", {
        "word_count": word_count,
        "lexical_leak": round(leak, 3),
        "realized_retention": realized_count,
        "target_retention": target_retention,
    }


def judge_query(
    abstract: str,
    query_text: str,
    prompt_cfg: dict[str, Any],
    judge_model: str,
) -> tuple[bool, str]:
    user_prompt = prompt_cfg["user_template"].format(abstract=abstract, query=query_text)
    system = prompt_cfg.get("system")
    resp = complete(user_prompt, system=system, model=judge_model)

    match = re.search(r"\{.*\}", resp, re.DOTALL)
    if not match:
        return False, "judge_parse_error"
    try:
        data = json.loads(match.group(0))
        faithful = bool(data.get("faithful", False))
        fluent = bool(data.get("fluent", False))
        reason = data.get("reasoning", "")
        if faithful and fluent:
            return True, "PASS"
        reasons = []
        if not faithful:
            reasons.append("unfaithful")
        if not fluent:
            reasons.append("non_fluent")
        return False, f"judge_rejected ({', '.join(reasons)}): {reason[:60]}"
    except Exception as e:
        return False, f"judge_json_exception: {e}"


def run_test_generation_and_local_bm25(seed: int = 13) -> None:
    load_env()

    # Load prompts
    prompts_path = ROOT / "config/en_prompts.yaml"
    with open(prompts_path, encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    # Load 270 TEST papers
    test_path = ROOT / "data/en/sources_test_270.jsonl"
    with open(test_path, encoding="utf-8") as f:
        test_papers = [json.loads(line) for line in f if line.strip()]

    print(f"[{BANNER}]", flush=True)
    print(f"Starting TEST Run Pipeline on {len(test_papers)} papers...", flush=True)

    generator_model = "qwen/qwen3.8-27b"
    judge_model = "openai/gpt-oss-120b"
    levels = ["R100", "R50", "R25", "R0"]

    log_dir = ROOT / "results/real/en"
    log_dir.mkdir(parents=True, exist_ok=True)
    test_log_path = log_dir / "test_queries_log.jsonl"

    # Load existing log if restarting
    existing_records = []
    processed_keys = set()
    if test_log_path.exists():
        with open(test_log_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    existing_records.append(rec)
                    processed_keys.add((rec["source_id"], rec["level"]))
        print(f"Resuming: found {len(existing_records)} already processed queries in {test_log_path}", flush=True)

    all_records = list(existing_records)

    start_time = time.time()
    for p_idx, doc in enumerate(test_papers, 1):
        pid = doc["paper_id"]
        title = doc["title"]
        abstract = doc["abstract"]

        # Check if all 4 levels done
        if all((pid, lvl) in processed_keys for lvl in levels):
            continue

        # Extract terms
        terms = extract_key_terms(abstract, prompts["key_term_extraction"], generator_model)
        if len(terms) < 6:
            words = [w for w in re.findall(r"\b[a-zA-Z]{5,}\b", abstract) if w.lower() not in {"these", "their", "which", "about", "could", "would", "paper", "study", "using", "presents", "propose"}]
            for w in words:
                if w.lower() not in [t.lower() for t in terms]:
                    terms.append(w)
                if len(terms) >= 8:
                    break

        K = len(terms)
        rng = random.Random(seed + p_idx * 17)

        # R100
        r100_kept = list(terms)
        r100_removed: list[str] = []
        r100_target = K

        # R50
        k50 = round(0.5 * K)
        r50_kept = sorted(rng.sample(terms, k50))
        r50_removed = [t for t in terms if t not in r50_kept]
        r50_target = k50

        # R25
        k25 = max(1, round(0.25 * K))
        r25_kept = sorted(rng.sample(terms, k25))
        r25_removed = [t for t in terms if t not in r25_kept]
        r25_target = k25

        # R0
        r0_kept: list[str] = []
        r0_removed = list(terms)
        r0_target = 0

        subsets = {
            "R100": (r100_kept, r100_removed, r100_target, prompts["query_generation"]["user_template_r100"]),
            "R50": (r50_kept, r50_removed, r50_target, prompts["query_generation"]["user_template_r50"]),
            "R25": (r25_kept, r25_removed, r25_target, prompts["query_generation"]["user_template_r25"]),
            "R0": (r0_kept, r0_removed, r0_target, prompts["query_generation"]["user_template_r0"]),
        }

        print(f"[{p_idx:03d}/270] {pid:<30} (Terms={K}) | Elapsed: {time.time()-start_time:.1f}s", flush=True)

        for lvl in levels:
            if (pid, lvl) in processed_keys:
                continue

            kept, removed, target_cnt, tmpl = subsets[lvl]
            u_prompt = tmpl.format(
                abstract=abstract,
                kept_terms_json=json.dumps(kept, indent=2),
                removed_terms_json=json.dumps(removed, indent=2),
            )
            q_text = complete(u_prompt, system=prompts["query_generation"].get("system"), model=generator_model).strip()
            if q_text.startswith('"') and q_text.endswith('"'):
                q_text = q_text[1:-1].strip()

            det_ok, det_reason, det_meta = check_deterministic(q_text, title, kept, removed, terms, target_cnt)

            judge_ok = False
            judge_reason = "skipped_due_to_deterministic_fail"
            if det_ok:
                judge_ok, judge_reason = judge_query(abstract, q_text, prompts["judge"], judge_model)

            accepted = det_ok and judge_ok
            final_reason = "ACCEPTED" if accepted else (det_reason if not det_ok else judge_reason)

            record = {
                "source_id": pid,
                "title": title,
                "abstract": abstract,
                "level": lvl,
                "target_retention": target_cnt,
                "total_key_terms": K,
                "kept_terms": kept,
                "removed_terms": removed,
                "query_text": q_text,
                "deterministic_pass": det_ok,
                "deterministic_reason": det_reason,
                "deterministic_meta": det_meta,
                "judge_pass": judge_ok,
                "judge_reason": judge_reason,
                "accepted": accepted,
                "final_reason": final_reason,
            }
            all_records.append(record)
            processed_keys.add((pid, lvl))

            # Incremental append to log
            with open(test_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            sym = "OK" if accepted else "REJ"
            print(f"    {lvl:<4}: [{sym}] words={len(q_text.split())} | {final_reason[:50]}", flush=True)

    # Save accepted queries to data/en/queries_test_en.jsonl and results/real/en/queries_test_en.jsonl
    accepted_records = [r for r in all_records if r["accepted"]]
    print(f"\nTotal TEST Queries: {len(all_records)} | Accepted: {len(accepted_records)} ({len(accepted_records)/max(1, len(all_records))*100:.1f}%)", flush=True)

    out_q1 = ROOT / "data/en/queries_test_en.jsonl"
    out_q2 = ROOT / "results/real/en/queries_test_en.jsonl"
    with open(out_q1, "w", encoding="utf-8") as f1, open(out_q2, "w", encoding="utf-8") as f2:
        for r in accepted_records:
            line = json.dumps(r, ensure_ascii=False) + "\n"
            f1.write(line)
            f2.write(line)

    # 4. Run Local Okapi BM25 Ranking
    print("\n--- Running Local Okapi BM25 Ranking on 5,270-document corpus ---", flush=True)
    sources = [json.loads(line) for line in open(ROOT / "data/en/sources_300.jsonl", encoding="utf-8")]
    distractors = [json.loads(line) for line in open(ROOT / "data/en/distractors_5000.jsonl", encoding="utf-8")]
    corpus = sources + distractors
    corpus_dict = {p["paper_id"]: (p["title"] + " " + p["abstract"]) for p in corpus}

    t_bm0 = time.time()
    bm25 = BM25(corpus_dict)
    print(f"Indexed BM25 corpus ({len(corpus_dict)} docs) in {time.time()-t_bm0:.2f}s", flush=True)

    # Group queries by level
    bm25_results_by_level: dict[str, list[dict[str, Any]]] = {lvl: [] for lvl in levels}
    per_query_ranks = []

    for q_idx, r in enumerate(accepted_records):
        lvl = r["level"]
        sid = r["source_id"]
        q_text = r["query_text"]

        ranked = bm25.rank(q_text)
        rank_pos = next((idx for idx, (did, _) in enumerate(ranked, 1) if did == sid), 9999)

        per_query_ranks.append({
            "query_id": f"{sid}_{lvl}_{q_idx}",
            "source_id": sid,
            "level": lvl,
            "bm25_rank": rank_pos,
            "reciprocal_rank": 1.0 / rank_pos if rank_pos > 0 else 0.0,
            "hit_top10": rank_pos <= 10,
            "hit_top50": rank_pos <= 50,
            "hit_top100": rank_pos <= 100,
        })
        bm25_results_by_level[lvl].append(per_query_ranks[-1])

    # Compute aggregate metrics per level
    bm25_metrics_rows = []
    print("\nLocal BM25 Retrieval Results by Retention Level:")
    print(f"{'Level':<6} | {'N':<4} | {'Recall@10 [95% CI]':<24} | {'Recall@50':<10} | {'Recall@100':<10} | {'MRR':<8}")
    print("-" * 72)

    for lvl in levels:
        sub = bm25_results_by_level[lvl]
        n = len(sub)
        if n == 0:
            continue
        h10 = sum(1 for it in sub if it["hit_top10"])
        h50 = sum(1 for it in sub if it["hit_top50"])
        h100 = sum(1 for it in sub if it["hit_top100"])
        mrr = sum(it["reciprocal_rank"] for it in sub) / n

        r10 = h10 / n
        r50 = h50 / n
        r100 = h100 / n
        ci10 = clopper_pearson(h10, n, 0.95)

        ci_str = f"[{ci10[0]:.3f}, {ci10[1]:.3f}]"
        print(f"{lvl:<6} | {n:<4} | {r10:.3f} {ci_str:<16} | {r50:.3f}      | {r100:.3f}       | {mrr:.4f}")

        bm25_metrics_rows.append({
            "banner": BANNER,
            "retriever": "bm25",
            "level": lvl,
            "n_queries": n,
            "recall_10": round(r10, 4),
            "recall_10_ci95_low": round(ci10[0], 4),
            "recall_10_ci95_high": round(ci10[1], 4),
            "recall_50": round(r50, 4),
            "recall_100": round(r100, 4),
            "mrr": round(mrr, 4),
        })

    # Save BM25 local metrics CSV
    bm25_csv_path = ROOT / "results/real/en/bm25_local_metrics.csv"
    with open(bm25_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "banner", "retriever", "level", "n_queries",
            "recall_10", "recall_10_ci95_low", "recall_10_ci95_high",
            "recall_50", "recall_100", "mrr"
        ])
        writer.writeheader()
        writer.writerows(bm25_metrics_rows)
    print(f"\nWrote local BM25 metrics: {bm25_csv_path}", flush=True)

    # Save rank export for local BM25
    ranks_csv_path = ROOT / "results/real/en/bm25_ranks_local.csv"
    with open(ranks_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "source_id", "level", "bm25_rank", "reciprocal_rank", "hit_top10", "hit_top50", "hit_top100"])
        writer.writeheader()
        writer.writerows(per_query_ranks)
    print(f"Wrote local BM25 query ranks: {ranks_csv_path}", flush=True)

    # 5. Build Colab Rank-Export Kit
    build_colab_kit()


def build_colab_kit() -> None:
    import hashlib

    def get_sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    manifest_files = {
        "sources_300": ROOT / "data/en/sources_300.jsonl",
        "distractors_5000": ROOT / "data/en/distractors_5000.jsonl",
        "queries_test_en": ROOT / "results/real/en/queries_test_en.jsonl",
        "en_dose_rules": ROOT / "config/en_dose_rules.yaml",
        "en_prompts": ROOT / "config/en_prompts.yaml",
    }

    manifest_data = {
        "banner": BANNER,
        "study": "English Terminology Dose-Response Pilot",
        "stage": "E3_colab_kit",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": {
            name: {
                "path": str(p.relative_to(ROOT)),
                "sha256": get_sha256(p) if p.exists() else "MISSING",
                "size_bytes": p.stat().st_size if p.exists() else 0,
            }
            for name, p in manifest_files.items()
        },
    }

    manifest_path = ROOT / "results/real/en/en_colab_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)
    print(f"Wrote Colab manifest: {manifest_path}", flush=True)

    # Generate EN_run_steps.md
    steps_md = f"""> [!IMPORTANT]
> **{BANNER}**

# English Terminology Dose-Response Pilot: Colab GPU Execution Guide (Step E3)

This document provides instructions for executing the neural dense retrieval (`dense_bge_m3`) and hybrid reciprocal rank fusion (`hybrid_rrf`) on Google Colab with GPU acceleration.

## 1. Input Integrity & Verification
Before running, verify that the following files are synchronized with GitHub `origin/main`:
- `results/real/en/queries_test_en.jsonl` (SHA-256: `{manifest_data['files']['queries_test_en']['sha256']}`)
- `data/en/sources_300.jsonl` (SHA-256: `{manifest_data['files']['sources_300']['sha256']}`)
- `data/en/distractors_5000.jsonl` (SHA-256: `{manifest_data['files']['distractors_5000']['sha256']}`)

## 2. Steps on Google Colab

1. **Open Colab Notebook**: Open `agentic_v2.ipynb` in Google Colab.
2. **Select Runtime**: In menu **Runtime -> Change runtime type**, select **T4 GPU** (or A100 GPU).
3. **Mount Google Drive**: Execute the first cell to mount your Google Drive (for embedding caching).
4. **Git Pull / Clone**:
   ```bash
   git pull origin main
   ```
5. **Run Retrieval Script**:
   Execute the EN retrieval runner cell:
   ```bash
   python scripts/run_retrieval_colab_en.py
   ```
   This will:
   - Load the 5,270-document corpus.
   - Compute/load dense embeddings (`BAAI/bge-m3`) with cache stored on Google Drive (`drive_cache_dir`).
   - Run dense retrieval (`dense_bge_m3`) and hybrid RRF (`hybrid_rrf`, $k=60$).
   - Export rank tables:
     - `results/real/en/rank_export_en.csv`
     - `results/real/en/rank_export_en.parquet`
     - `results/real/en/rank_export_en_manifest.json`

## 3. Returning Results
After Colab finishes:
1. Download the exported folder or files:
   - `rank_export_en.csv`
   - `rank_export_en.parquet`
   - `rank_export_en_manifest.json`
2. Place them into your local workspace directory:
   `d:\\BK-Paper\\Idea_agentic\\results\\real\\en\\`
3. Notify the assistant to proceed to Step E4 (dose-response curve analysis, reproduction checks, closed-book controls, and decision rules).
"""

    steps_path = ROOT / "EN_run_steps.md"
    steps_path.write_text(steps_md, encoding="utf-8")
    print(f"Wrote execution steps guide: {steps_path}", flush=True)


if __name__ == "__main__":
    run_test_generation_and_local_bm25()

