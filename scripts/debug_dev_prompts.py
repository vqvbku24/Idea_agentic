import json
import os
import sys
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
from scripts.llm_input_cleaning import detect_and_strip_title


def clean_json_text(text: str) -> str:
    t = text.strip()
    if t.startswith("```json"):
        t = t[7:]
    elif t.startswith("```"):
        t = t[3:]
    if t.endswith("```"):
        t = t[:-3]
    return t.strip()


def main() -> None:
    load_env()
    print("=== DEBUG PROMPTS ON DEV SUBSET ===")
    dev_ids = json.load(open(ROOT / "config/dev_prompt_subset.json", encoding="utf-8"))
    print("Dev papers:", dev_ids)

    dev_docs = {
        d["doc_id"]: d
        for d in [json.loads(line) for line in open(ROOT / "results/real/development_papers_x2.jsonl", encoding="utf-8")]
    }

    cache_dir = ROOT / "cache/llm_dev_debug"

    for idx, pid in enumerate(dev_ids):
        doc = dev_docs[pid]
        print(f"\n--- [{idx + 1}/3] {pid} ---")
        _, clean_en = detect_and_strip_title(doc.get("abstract_en", ""))
        fr_abs = doc.get("abstract_fr", "")
        body_1500 = " ".join(doc.get("body_fr", "").split()[:1500])

        # 1. Facet extraction
        prompt_f = build_facet_extraction_prompt(fr_abs, clean_en, body_1500)
        res_f = complete(prompt_f, cache_dir=cache_dir)
        try:
            facets = json.loads(clean_json_text(res_f))
            print("Extracted facets OK:", list(facets.keys()))
        except Exception as e:
            print("Facet extraction JSON parse error:", e)
            print("Raw response:", res_f[:200])
            continue

        # 2. Variants V0, V1, V2, V3
        variants = {}
        # V0
        variants["V0"] = (complete(build_v0_prompt(facets), cache_dir=cache_dir), ["problem", "mechanism", "setting", "outcome"])
        # V1
        variants["V1"] = (complete(build_v1_prompt(facets), cache_dir=cache_dir), ["problem", "mechanism", "setting", "outcome"])
        # V2
        kept_v2 = select_v2_facets(13, idx)
        variants["V2"] = (complete(build_v2_prompt(facets, kept_v2), cache_dir=cache_dir), kept_v2)
        # V3
        variants["V3"] = (complete(build_v3_roundtrip_prompt(facets), cache_dir=cache_dir), ["problem", "mechanism", "setting", "outcome"])

        # 3. Judge each variant with JUDGE_MODEL
        for v_name, (v_text, targets) in variants.items():
            prompt_j = build_judge_prompt(v_text, targets)
            res_j = complete(prompt_j, model=os.environ.get("JUDGE_MODEL"), cache_dir=cache_dir)
            try:
                judge_eval = json.loads(clean_json_text(res_j))
                leak = lexical_leak(v_text, doc.get("title", ""))
                print(f"  {v_name} ({len(v_text.split())} words) leak={leak:.2f} judge: {judge_eval}")
            except Exception as e:
                print(f"  {v_name} Judge parse error:", e, "Raw:", res_j[:150])

    print("\nDebug complete on all 3 dev papers!")


if __name__ == "__main__":
    main()
