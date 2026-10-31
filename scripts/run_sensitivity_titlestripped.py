import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pilot.corpus import Document, build_index_views
from pilot.retrievers import BM25
from scripts.run_retrieval_bm25_preview import load_corpus

BANNER = "EXPLORATORY - agent-proxy labels, human verification pending"


def main() -> None:
    docs, titles = load_corpus(ROOT, distractor_count=500)

    # 1. Load cleaned prefixes
    cleaned_rows = []
    cleaned_dict = {}
    with open(ROOT / "results/real/llm_input_cleaning.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["stripped_prefix"]:
                cleaned_rows.append(row)
                cleaned_dict[row["paper_id"]] = row["cleaned_text"]

    print("=== SENSITIVITY ANALYSIS: en_abstract_titlestripped (BM25) ===")
    print(f"Sources affected ({len(cleaned_dict)}):")
    for r in cleaned_rows:
        print(f"  - {r['paper_id']}: prefix = '{r['stripped_prefix']}'")

    # 2. Build index views
    views_orig = build_index_views(docs)["en_abstract"]

    docs_stripped = []
    for d in docs:
        if d.doc_id in cleaned_dict:
            docs_stripped.append(Document(
                doc_id=d.doc_id, kind=d.kind, lang=d.lang, title=d.title,
                abstract_fr=d.abstract_fr, abstract_en=cleaned_dict[d.doc_id],
                body_fr=d.body_fr, year=d.year, keywords_en=d.keywords_en,
            ))
        else:
            docs_stripped.append(d)

    views_stripped = build_index_views(docs_stripped)["en_abstract"]

    idx_orig = BM25(views_orig)
    idx_stripped = BM25(views_stripped)

    # 3. Evaluate queries
    queries = [
        json.loads(line) for line in open(ROOT / "results/real/queries.jsonl", encoding="utf-8")
        if json.loads(line)["accepted"]
    ]

    out_rows = []
    for var in ["V0", "V1", "V2", "V3"]:
        var_q = [q for q in queries if q["variant"] == var]
        aff_q = [q for q in var_q if q["source_id"] in cleaned_dict]

        # All sources
        r10_all_orig = sum(q["source_id"] in [did for did, _ in idx_orig.rank(q["query_text"])[:10]] for q in var_q) / len(var_q)
        r10_all_strip = sum(q["source_id"] in [did for did, _ in idx_stripped.rank(q["query_text"])[:10]] for q in var_q) / len(var_q)

        # Affected sources only
        r10_aff_orig = sum(q["source_id"] in [did for did, _ in idx_orig.rank(q["query_text"])[:10]] for q in aff_q) / len(aff_q) if aff_q else 0.0
        r10_aff_strip = sum(q["source_id"] in [did for did, _ in idx_stripped.rank(q["query_text"])[:10]] for q in aff_q) / len(aff_q) if aff_q else 0.0

        out_rows.append({
            "variant": var,
            "n_all_accepted": len(var_q),
            "recall@10_all_orig": round(r10_all_orig, 4),
            "recall@10_all_stripped": round(r10_all_strip, 4),
            "diff_all": round(r10_all_strip - r10_all_orig, 4),
            "n_affected_accepted": len(aff_q),
            "recall@10_affected_orig": round(r10_aff_orig, 4),
            "recall@10_affected_stripped": round(r10_aff_strip, 4),
            "diff_affected": round(r10_aff_strip - r10_aff_orig, 4),
        })

    csv_path = ROOT / "results/real/sensitivity_titlestripped.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nSaved sensitivity metrics to {csv_path}")
    print("\nSummary table:")
    print("Variant | N (All) | Orig R@10 | Strip R@10 | Diff   | N (Aff) | Orig R@10 | Strip R@10 | Diff")
    print("--------|---------|-----------|------------|--------|---------|-----------|------------|-----")
    for r in out_rows:
        print(f"{r['variant']:7} | {r['n_all_accepted']:7} | {r['recall@10_all_orig']:.4f}    | {r['recall@10_all_stripped']:.4f}     | {r['diff_all']:+.4f} | {r['n_affected_accepted']:7} | {r['recall@10_affected_orig']:.4f}    | {r['recall@10_affected_stripped']:.4f}     | {r['diff_affected']:+.4f}")


if __name__ == "__main__":
    main()

