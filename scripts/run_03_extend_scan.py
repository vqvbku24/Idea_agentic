from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pilot.data_acl import (
    download_bulk_metadata,
    download_pdf,
    list_jeptalnrecital_papers,
    stratified_sample,
    write_metadata,
)


def load_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line)["paper_id"] for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extend the frozen P0 source scan.")
    parser.add_argument("--target-eligible", type=int, default=20)
    parser.add_argument("--max-scan", type=int, default=60)
    parser.add_argument("--first-batch", type=Path, default=Path("data/raw/processed/selected_metadata.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/extended"))
    parser.add_argument("--metadata-out", type=Path, default=Path("data/processed/extended_metadata.jsonl"))
    args = parser.parse_args()
    download_bulk_metadata(Path("data/cache/acl-anthology-master.zip"))
    first_ids = load_ids(args.first_batch)
    all_candidates = list_jeptalnrecital_papers((2015, 2024))
    dev_path = Path("data/processed/development_exclusion.jsonl")
    if dev_path.exists():
        all_candidates = [
            paper for paper in all_candidates
            if paper.paper_id not in set(load_ids(dev_path))
        ]
    ordered = stratified_sample(all_candidates, max(args.max_scan, len(first_ids)), seed=13)
    ordered_ids = [paper.paper_id for paper in ordered]
    comparison = {
        "first_batch_ids": first_ids,
        "ordered_prefix_ids": ordered_ids[:len(first_ids)],
        "identical": first_ids == ordered_ids[:len(first_ids)],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (Path("results/real") / "first_batch_comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not comparison["identical"]:
        raise RuntimeError("first new-batch paper IDs do not match the ordered seeded prefix")
    state: dict[str, float] = {}
    events = Path("events.jsonl")
    scanned: list = ordered[:len(first_ids)]
    eligible = 13
    for paper in ordered[len(first_ids):args.max_scan]:
        scanned.append(paper)
        with events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "type": "SourceSelected", "paper_id": paper.paper_id,
                "scan_index": len(scanned),
            }) + "\n")
        write_metadata(scanned, args.metadata_out)
        download_pdf(paper, args.output_dir, state=state, events_path=events)
        subprocess.run([
            sys.executable, "scripts/run_02_extract_build.py",
            "--raw-dir", str(args.output_dir),
            "--metadata", str(args.metadata_out),
            "--output-dir", "results/real",
        ], check=True)
        with Path("results/real/g1_stats.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        eligible = int(next(row["count"] for row in rows if row["stage"] == "eligible"))
        if eligible >= args.target_eligible:
            break
    print(f"scanned={len(scanned)} eligible={eligible} target={args.target_eligible}")
    if len(scanned) >= args.max_scan and eligible < 15:
        print("STOP: scan cap reached with fewer than 15 eligible papers")


if __name__ == "__main__":
    main()
