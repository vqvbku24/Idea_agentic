from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pilot.data_acl import (
    download_pdf,
    list_english_distractors,
    list_french_distractors,
    list_jeptalnrecital_papers,
    stratified_sample,
    write_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover and download French ACL source PDFs.")
    parser.add_argument("--n-source", type=int, default=10)
    parser.add_argument("--years", nargs=2, type=int, default=(2015, 2024))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--exclude-metadata", type=Path)
    parser.add_argument("--english-distractors", type=int, default=500)
    parser.add_argument("--french-distractors", type=int, default=100)
    args = parser.parse_args()
    candidates = list_jeptalnrecital_papers(tuple(args.years))
    excluded = set()
    if args.exclude_metadata and args.exclude_metadata.exists():
        with args.exclude_metadata.open(encoding="utf-8") as handle:
            excluded = {json.loads(line)["paper_id"] for line in handle if line.strip()}
    candidates = [paper for paper in candidates if paper.paper_id not in excluded]
    selected = stratified_sample(candidates, args.n_source, seed=13)
    processed = args.output_dir.parent.parent / "processed"
    write_metadata(selected, processed / "selected_metadata.jsonl")
    english = list_english_distractors(args.english_distractors, seed=13)
    french = list_french_distractors(args.french_distractors, seed=13, source_ids=excluded | {p.paper_id for p in selected})
    write_metadata(english, processed / "english_distractors_metadata.jsonl")
    write_metadata(french, processed / "french_distractors_metadata.jsonl")
    state: dict[str, float] = {}
    events = args.output_dir.parent.parent / "events.jsonl"
    downloaded = 0
    for paper in selected:
        if download_pdf(paper, args.output_dir, state=state, events_path=events):
            downloaded += 1
    french_downloaded = 0
    distractor_dir = args.output_dir / "french_distractors"
    for paper in french:
        if download_pdf(paper, distractor_dir, state=state, events_path=events):
            french_downloaded += 1
    print(
        f"discovered_new={len(candidates)} selected={len(selected)} downloaded={downloaded} "
        f"english_distractors={len(english)} french_distractors={len(french)} "
        f"french_distractor_pdfs={french_downloaded}"
    )
    counts_path = args.output_dir.parent.parent.parent / "results" / "real" / "acquisition_counts.csv"
    counts_path.parent.mkdir(parents=True, exist_ok=True)
    with counts_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["set", "requested", "discovered", "downloaded", "failure_reason"])
        writer.writerow(["new_sources", args.n_source, len(selected), downloaded, ""])
        writer.writerow([
            "english_distractors", args.english_distractors, len(english), 0,
            "ACL metadata API rate limit (HTTP 403)" if not english else "",
        ])
        writer.writerow([
            "french_distractors", args.french_distractors, len(french), french_downloaded,
            "ACL metadata API rate limit (HTTP 403)" if not french else "",
        ])


if __name__ == "__main__":
    main()
