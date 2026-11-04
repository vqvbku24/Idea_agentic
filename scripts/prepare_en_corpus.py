"""Corpus extraction and stratification script for English Dose-Response Pilot.

Extracts 300 sources (30 DEV, 270 TEST) and 5000 distractors from ACL Anthology bulk archive.
Enforces eligibility:
  - Years 2018-2024
  - Abstract 80-300 words
  - No title-like prefix (detect_and_strip_title)
  - Exclude track/types without clear contribution (frontmatter, tutorials, demos)
  - Stratified by year (seed 13)

EXPLORATORY - agent-proxy labels, human verification pending
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.llm_input_cleaning import detect_and_strip_title

BANNER = "EXPLORATORY - agent-proxy labels, human verification pending"
BULK_ARCHIVE = ROOT / "data/cache/acl-anthology-master.zip"
TARGET_YEARS = {str(y) for y in range(2018, 2025)}


def get_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def is_english_text(text: str) -> bool:
    words = set(text.lower().split())
    markers = {"the", "of", "and", "in", "for", "this", "we", "is", "on", "with", "that", "as", "by", "an", "to"}
    return len(words & markers) >= 3


def extract_and_stratify_corpus(seed: int = 13) -> dict[str, Any]:
    if not BULK_ARCHIVE.exists():
        raise FileNotFoundError(f"Missing bulk archive: {BULK_ARCHIVE}")

    print(f"[{BANNER}]")
    print(f"Reading bulk archive: {BULK_ARCHIVE}...")

    stats = Counter()
    papers_by_year: dict[int, list[dict[str, Any]]] = defaultdict(list)
    all_eligible: list[dict[str, Any]] = []

    with zipfile.ZipFile(BULK_ARCHIVE) as z:
        for name in z.namelist():
            if "/data/xml/" in name and name.endswith(".xml"):
                fn = name.rsplit("/", 1)[-1]
                y_str = fn[:4]
                if y_str not in TARGET_YEARS:
                    continue

                root = ET.fromstring(z.read(name))
                for vol in root.findall("volume"):
                    v_id = vol.get("id", "")
                    venue = vol.findtext("./meta/venue") or fn[5:-4]
                    meta_y = vol.findtext("./meta/year")
                    paper_year = int(meta_y) if meta_y and meta_y.isdigit() else int(y_str)

                    v_lower = (v_id or "").lower()
                    is_tutorial = "tutorial" in v_lower
                    is_demo = "demo" in v_lower

                    for paper in vol.findall("paper"):
                        p_id = paper.get("id", "")
                        stats["total_seen"] += 1

                        if p_id == "0" or "frontmatter" in p_id.lower():
                            stats["exclude_frontmatter"] += 1
                            continue
                        if is_tutorial or "tutorial" in p_id.lower():
                            stats["exclude_tutorial"] += 1
                            continue
                        if is_demo or "demo" in p_id.lower():
                            stats["exclude_demo"] += 1
                            continue

                        title = " ".join((paper.findtext("title") or "").split())
                        abstract = " ".join((paper.findtext("abstract") or "").split())
                        lang = paper.findtext("language")

                        if not abstract:
                            stats["exclude_no_abstract"] += 1
                            continue
                        if lang and lang.lower() not in ("eng", "en", "english"):
                            stats["exclude_non_english_lang_tag"] += 1
                            continue
                        if not is_english_text(abstract):
                            stats["exclude_not_english_text"] += 1
                            continue

                        words = abstract.split()
                        if len(words) < 80:
                            stats["exclude_abstract_too_short"] += 1
                            continue
                        if len(words) > 300:
                            stats["exclude_abstract_too_long"] += 1
                            continue

                        # Check for title-like prefix
                        prefix, _ = detect_and_strip_title(abstract)
                        if prefix:
                            stats["exclude_title_prefix_detected"] += 1
                            continue

                        full_id = f"{paper_year}.{venue}-{v_id}.{p_id}"
                        authors = [
                            " ".join(part for part in (a.findtext("first"), a.findtext("last")) if part)
                            for a in paper.findall("author")
                        ]

                        rec = {
                            "paper_id": full_id,
                            "title": title,
                            "abstract": abstract,
                            "authors": authors,
                            "year": paper_year,
                            "venue": venue,
                            "volume_id": v_id,
                            "paper_id_raw": p_id,
                            "word_count": len(words),
                        }
                        stats["eligible_papers"] += 1
                        papers_by_year[paper_year].append(rec)
                        all_eligible.append(rec)

    print(f"Total eligible papers across 2018-2024: {len(all_eligible)}")
    print("Exclusion breakdown:")
    for k, v in stats.items():
        if k != "eligible_papers":
            print(f"  - {k}: {v}")

    # Stratified sampling of 300 sources (30 DEV, 270 TEST) across 7 years
    rng = random.Random(seed)

    # Year quota for 300 sources: 43 for 6 years, 42 for 1 year
    # Dev quota for 30 dev: 4 for 4 years, 5 for 2 years, 4 for 1 year = 30
    years_sorted = sorted(papers_by_year.keys())
    sources_by_year: dict[int, list[dict[str, Any]]] = {}
    dev_sources: list[dict[str, Any]] = []
    test_sources: list[dict[str, Any]] = []

    dev_quotas = {2018: 4, 2019: 4, 2020: 4, 2021: 4, 2022: 5, 2023: 5, 2024: 4}
    source_quotas = {2018: 43, 2019: 43, 2020: 43, 2021: 43, 2022: 43, 2023: 43, 2024: 42}

    chosen_source_ids = set()

    for y in years_sorted:
        pool = list(papers_by_year[y])
        # Sort deterministically before sample
        pool.sort(key=lambda x: x["paper_id"])
        rng.shuffle(pool)

        n_src = source_quotas[y]
        n_dev = dev_quotas[y]
        n_test = n_src - n_dev

        selected_src = pool[:n_src]
        for s in selected_src:
            chosen_source_ids.add(s["paper_id"])

        dev_subset = selected_src[:n_dev]
        test_subset = selected_src[n_dev:n_src]

        for s in dev_subset:
            s_copy = dict(s)
            s_copy["split"] = "DEV"
            dev_sources.append(s_copy)

        for s in test_subset:
            s_copy = dict(s)
            s_copy["split"] = "TEST"
            test_sources.append(s_copy)

    total_sources = dev_sources + test_sources
    print(f"\nSampled Sources: Total={len(total_sources)} (DEV={len(dev_sources)}, TEST={len(test_sources)})")

    # Sample 5000 distractors strictly disjoint from sources
    remaining_eligible = [p for p in all_eligible if p["paper_id"] not in chosen_source_ids]
    remaining_eligible.sort(key=lambda x: x["paper_id"])
    rng.shuffle(remaining_eligible)
    distractors_5000 = remaining_eligible[:5000]

    for d in distractors_5000:
        d["split"] = "DISTRACTOR"

    print(f"Sampled Distractors: Total={len(distractors_5000)} (disjoint from sources: {len(chosen_source_ids.intersection(set(d['paper_id'] for d in distractors_5000))) == 0})")

    # Save to data/en/
    out_dir = ROOT / "data/en"
    out_dir.mkdir(parents=True, exist_ok=True)

    sources_path = out_dir / "sources_300.jsonl"
    with open(sources_path, "w", encoding="utf-8") as f:
        for s in total_sources:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    dev_path = out_dir / "sources_dev_30.jsonl"
    with open(dev_path, "w", encoding="utf-8") as f:
        for s in dev_sources:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    test_path = out_dir / "sources_test_270.jsonl"
    with open(test_path, "w", encoding="utf-8") as f:
        for s in test_sources:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    dist_path = out_dir / "distractors_5000.jsonl"
    with open(dist_path, "w", encoding="utf-8") as f:
        for d in distractors_5000:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # Manifest
    manifest = {
        "banner": BANNER,
        "seed": seed,
        "counts": {
            "total_seen": stats["total_seen"],
            "eligible_papers": stats["eligible_papers"],
            "sources_total": len(total_sources),
            "sources_dev": len(dev_sources),
            "sources_test": len(test_sources),
            "distractors": len(distractors_5000),
        },
        "exclusions": dict(stats),
        "year_stratification": {
            y: {
                "source_quota": source_quotas[y],
                "dev_quota": dev_quotas[y],
                "test_quota": source_quotas[y] - dev_quotas[y],
                "eligible_in_corpus": len(papers_by_year[y]),
            }
            for y in years_sorted
        },
        "sha256": {
            "sources_300": get_sha256(sources_path),
            "sources_dev_30": get_sha256(dev_path),
            "sources_test_270": get_sha256(test_path),
            "distractors_5000": get_sha256(dist_path),
        },
    }

    manifest_path = ROOT / "results/real/en/corpus_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {sources_path}")
    print(f"Wrote {dev_path}")
    print(f"Wrote {test_path}")
    print(f"Wrote {dist_path}")
    print(f"Wrote {manifest_path}")

    return manifest


if __name__ == "__main__":
    extract_and_stratify_corpus()

