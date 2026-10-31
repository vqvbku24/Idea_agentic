import csv
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Load sealed facets
facets = {}
with open(ROOT / "results/real/sealed/facets_llm.jsonl", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        # Store terms normalized to lowercase without leading/trailing whitespace
        terms = [t.lower().strip() for t in row["facets"].get("key_terms", []) if t.strip()]
        facets[row["paper_id"]] = terms

# Load queries
queries = [json.loads(line) for line in open(ROOT / "results/real/queries.jsonl", encoding="utf-8")]

# Group by variant (for accepted queries)
by_variant = {}
for q in queries:
    var = q["variant"]
    pid = q["source_id"]
    q_text = q["query_text"].lower()
    terms = facets.get(pid, [])
    if not terms:
        continue
    # Count how many terms appear in the query (using word boundaries)
    hits = sum(1 for t in terms if re.search(r"\b" + re.escape(t) + r"\b", q_text))
    share = hits / len(terms) if terms else 0.0
    by_variant.setdefault(var, []).append({
        "source_id": pid,
        "accepted": q["accepted"],
        "share": share,
        "leak": q["lexical_leak"],
    })

out_rows = []
variants = ["V0", "V1", "V2", "V3", "H-agent"]
for var in variants:
    rows = by_variant.get(var, [])
    acc_rows = [r for r in rows if r["accepted"]]
    shares = [r["share"] for r in acc_rows]
    leaks = [r["leak"] for r in acc_rows]

    out_rows.append({
        "variant": var,
        "n_accepted": len(acc_rows),
        "n_total": len(rows),
        "mean_key_term_share": round(statistics.mean(shares), 4) if shares else 0.0,
        "median_key_term_share": round(statistics.median(shares), 4) if shares else 0.0,
        "median_lexical_leak": round(statistics.median(leaks), 4) if leaks else 0.0,
        "mean_lexical_leak": round(statistics.mean(leaks), 4) if leaks else 0.0,
    })

out_path = ROOT / "results/real/disguise_strength.csv"
with out_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    writer.writeheader()
    writer.writerows(out_rows)

print(f"Saved disguise strength to {out_path}")
for r in out_rows:
    print(r)

