"""DEV Run Pipeline for English Terminology Dose-Response Pilot (Step E2).

Processes 30 DEV papers:
  1. Extracts 6-12 verbatim key terms from abstract
  2. Generates queries across 4 retention levels (R100, R50, R25, R0)
  3. Executes deterministic acceptance checks (retention +-1, zero removed terms, title leak <= 0.3, no digits/acronyms, length 40-80)
  4. Executes Judge model evaluation (faithful, fluent)
  5. Prepares 20-sample blind human audit sheet (results/real/en/human_sample_en.csv) with empty human columns

EXPLORATORY - agent-proxy labels, human verification pending
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pilot.llm import complete, load_env
from src.pilot.queries import lexical_leak

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

    # Multi-word phrase: check if phrase occurs as exact sequence of stems
    for i in range(len(query_tokens) - len(term_words) + 1):
        if query_tokens[i : i + len(term_words)] == term_words:
            return True

    # Also check if all words occur together
    if all(tw in query_tokens for tw in term_words):
        return True
    return False


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

    # Cap to 12 max
    return valid_terms[:12]


def title_leak_relative(query_text: str, title: str, kept_terms: list[str]) -> float:
    """Compute title lexical leak relative to title tokens outside kept terms (config/en_amendment_1.yaml)."""
    title_tokens = {token for token in re.findall(r"\w+", title.lower()) if len(token) >= 5}
    kept_tokens = {token for term in kept_terms for token in re.findall(r"\w+", term.lower()) if len(token) >= 4}
    title_tokens_outside_kept = title_tokens - kept_tokens
    if not title_tokens_outside_kept:
        return 0.0
    query_tokens = set(re.findall(r"\w+", query_text.lower()))
    return len(title_tokens_outside_kept & query_tokens) / len(title_tokens_outside_kept)


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

    # Check zero removed terms (stem-insensitive)
    removed_present = []
    for rt in removed_terms:
        if is_term_present(rt, query_text):
            removed_present.append(rt)

    # Check digits and uppercase acronyms from removed terms
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

    # Realized retention count
    present_all = [t for t in all_terms if is_term_present(t, query_text)]
    realized_count = len(present_all)

    # Realized retention equals target +- 1 term
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


def run_en_dev_pipeline(seed: int = 13) -> None:
    load_env()

    # Load prompts
    prompts_path = ROOT / "config/en_prompts.yaml"
    with open(prompts_path, encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    # Load 30 DEV papers
    dev_path = ROOT / "data/en/sources_dev_30.jsonl"
    with open(dev_path, encoding="utf-8") as f:
        dev_papers = [json.loads(line) for line in f if line.strip()]

    print(f"[{BANNER}]")
    print(f"Running DEV Run Pipeline on {len(dev_papers)} papers across 4 retention levels...")

    generator_model = "qwen/qwen3.8-27b"
    judge_model = "openai/gpt-oss-120b"

    levels = ["R100", "R50", "R25", "R0"]
    all_query_records: list[dict[str, Any]] = []

    for p_idx, doc in enumerate(dev_papers, 1):
        pid = doc["paper_id"]
        title = doc["title"]
        abstract = doc["abstract"]

        # 1. Key terms extraction
        terms = extract_key_terms(abstract, prompts["key_term_extraction"], generator_model)
        if len(terms) < 6:
            # Fallback simple n-gram extraction if needed
            print(f"  [{p_idx}/30] {pid}: Extracted {len(terms)} terms (<6). Adding fallback ngrams...")
            words = [w for w in re.findall(r"\b[a-zA-Z]{5,}\b", abstract) if w.lower() not in {"these", "their", "which", "about", "could", "would", "paper", "study", "using", "presents", "propose"}]
            for w in words:
                if w.lower() not in [t.lower() for t in terms]:
                    terms.append(w)
                if len(terms) >= 8:
                    break

        K = len(terms)
        print(f"  [{p_idx}/30] {pid:<30} (Terms={K}):")

        # 2. Retention subsets
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

        for lvl in levels:
            kept, removed, target_cnt, tmpl = subsets[lvl]

            # Generate query
            u_prompt = tmpl.format(
                abstract=abstract,
                kept_terms_json=json.dumps(kept, indent=2),
                removed_terms_json=json.dumps(removed, indent=2),
            )
            q_text = complete(u_prompt, system=prompts["query_generation"].get("system"), model=generator_model).strip()

            # Clean output if wrapped in quotes
            if q_text.startswith('"') and q_text.endswith('"'):
                q_text = q_text[1:-1].strip()

            # Deterministic checks
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
            all_query_records.append(record)

            status_sym = "[OK]" if accepted else "[REJ]"
            print(f"    {lvl:<4}: {status_sym} words={len(q_text.split())} | reason={final_reason}")

    # Save full DEV query log
    log_dir = ROOT / "results/real/en"
    log_dir.mkdir(parents=True, exist_ok=True)
    dev_log_path = log_dir / "dev_queries_log.jsonl"
    with open(dev_log_path, "w", encoding="utf-8") as f:
        for r in all_query_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Generate acceptance breakdown by level and reason
    stats_by_level: dict[str, Counter] = {lvl: Counter() for lvl in levels}
    retention_by_level: dict[str, list[float]] = {lvl: [] for lvl in levels}

    for r in all_query_records:
        lvl = r["level"]
        if r["accepted"]:
            stats_by_level[lvl]["ACCEPTED"] += 1
        else:
            reason_category = r["final_reason"].split()[0].rstrip(":")
            stats_by_level[lvl][reason_category] += 1

        if r["deterministic_meta"]:
            retention_by_level[lvl].append(r["deterministic_meta"]["realized_retention"])

    # Prepare 20-sample blind human review sheet
    accepted_queries = [r for r in all_query_records if r["accepted"]]
    rng_sample = random.Random(seed)
    # Stratified or random sample 20 accepted queries
    sample_pool = list(accepted_queries)
    sample_pool.sort(key=lambda x: (x["source_id"], x["level"]))
    rng_sample.shuffle(sample_pool)
    human_sample_20 = sample_pool[:20]

    sheet_path = log_dir / "human_sample_en.csv"
    sealed_key_path = log_dir / "sealed/human_sample_key.csv"
    sealed_key_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Blind sheet for human (NO level label, empty human review columns)
    with open(sheet_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id",
            "query_text",
            "source_id",
            "abstract",
            "human_faithful",
            "human_hides_terms",
            "human_fluent",
            "human_notes",
        ])
        for idx, item in enumerate(human_sample_20, 1):
            writer.writerow([
                f"EN_DEV_SAMPLE_{idx:02d}",
                item["query_text"],
                item["source_id"],
                item["abstract"],
                "",  # EMPTY for human
                "",  # EMPTY for human
                "",  # EMPTY for human
                "",  # EMPTY for human
            ])

    # 2. Sealed key
    with open(sealed_key_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "source_id", "level", "target_retention", "realized_retention"])
        for idx, item in enumerate(human_sample_20, 1):
            writer.writerow([
                f"EN_DEV_SAMPLE_{idx:02d}",
                item["source_id"],
                item["level"],
                item["target_retention"],
                item["deterministic_meta"].get("realized_retention", ""),
            ])

    # Write summary report markdown
    report_md_path = log_dir / "dev_acceptance_report.md"
    generate_dev_report(all_query_records, stats_by_level, retention_by_level, len(human_sample_20), report_md_path)

    print(f"\n[{BANNER}]")
    print(f"DEV Pipeline Complete.")
    print(f"Wrote full log: {dev_log_path}")
    print(f"Wrote report:   {report_md_path}")
    print(f"Wrote blind human sample sheet: {sheet_path}")
    print(f"Wrote sealed key:               {sealed_key_path}")


def generate_dev_report(
    records: list[dict[str, Any]],
    stats_by_level: dict[str, Counter],
    retention_by_level: dict[str, list[float]],
    sample_size: int,
    path: Path,
) -> None:
    total = len(records)
    total_acc = sum(1 for r in records if r["accepted"])

    lines = [
        f"> [!IMPORTANT]",
        f"> **{BANNER}**\n",
        "# English Terminology Dose-Response Pilot: DEV Run Acceptance Report\n",
        f"Evaluated on 30 DEV papers across 4 retention levels (total queries: {total}).\n",
        "## 1. Acceptance and Rejection Breakdown by Level\n",
        "| Retention Level | Total | Accepted | Rejection Rate | Acceptance Rate | Mean Realized Retention |",
        "|---|---|---|---|---|---|",
    ]

    for lvl in ["R100", "R50", "R25", "R0"]:
        cnt = stats_by_level[lvl]
        lvl_tot = sum(cnt.values())
        acc = cnt.get("ACCEPTED", 0)
        rej = lvl_tot - acc
        acc_rate = (acc / lvl_tot * 100) if lvl_tot else 0.0
        rej_rate = (rej / lvl_tot * 100) if lvl_tot else 0.0
        r_vals = retention_by_level.get(lvl, [])
        mean_ret = sum(r_vals) / len(r_vals) if r_vals else 0.0
        lines.append(f"| `{lvl}` | {lvl_tot} | **{acc}** | {rej} ({rej_rate:.1f}%) | **{acc_rate:.1f}%** | {mean_ret:.2f} terms |")

    lines.extend([
        "",
        "## 2. Granular Failure Reasons by Level\n",
        "| Retention Level | Failure Reason Category | Count | Example Detail |",
        "|---|---|---|---|",
    ])

    for lvl in ["R100", "R50", "R25", "R0"]:
        cnt = stats_by_level[lvl]
        for reason, count in cnt.items():
            if reason != "ACCEPTED":
                # Find an example
                ex = next((r["final_reason"] for r in records if r["level"] == lvl and r["final_reason"].startswith(reason)), "")
                lines.append(f"| `{lvl}` | `{reason}` | {count} | {ex[:70]} |")

    lines.extend([
        "",
        "## 3. Human Sample Sheet Status\n",
        f"- Sampled **{sample_size} blind queries** into `results/real/en/human_sample_en.csv`.",
        "- Human evaluation columns (`human_faithful`, `human_hides_terms`, `human_fluent`) left strictly **EMPTY** per Rule G2.",
        "- Sealed key preserved in `results/real/en/sealed/human_sample_key.csv`.",
        "",
        "## 4. Next Step\n",
        "A human reviewer must inspect the 20 blind samples, record ratings, and execute:",
        "```bash",
        "python scripts/sign_en.py --reviewer <initials>",
        "```",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run_en_dev_pipeline()

