"""Sign-off script for English Dose-Response human review sheet.

Validates that results/real/en/human_sample_en.csv exists,
refuses empty or partially filled sheets, computes SHA-256 hash,
and records human approval in results/real/en/signatures.jsonl.

Usage:
  python scripts/sign_en.py --reviewer <initials>

EXPLORATORY - agent-proxy labels, human verification pending
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHEET_PATH = ROOT / "results/real/en/human_sample_en.csv"
SIG_PATH = ROOT / "results/real/en/signatures.jsonl"
BANNER = "EXPLORATORY - agent-proxy labels, human verification pending"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def validate_and_sign(reviewer: str) -> None:
    if not SHEET_PATH.exists():
        raise FileNotFoundError(f"Review sheet not found at {SHEET_PATH}")

    reviewer = reviewer.strip()
    if not reviewer or len(reviewer) < 2:
        raise ValueError("Valid reviewer initials or name must be provided (e.g., --reviewer VK)")

    try:
        with open(SHEET_PATH, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except UnicodeDecodeError:
        with open(SHEET_PATH, encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    if not rows:
        raise ValueError(f"Review sheet {SHEET_PATH} is empty!")

    required_cols = ["sample_id", "query_text", "human_faithful", "human_hides_terms", "human_fluent"]
    for col in required_cols:
        if col not in (reader.fieldnames or []):
            raise ValueError(f"Review sheet missing required column: {col}")

    # Verify no empty fields for human evaluation
    empty_items = []
    for idx, r in enumerate(rows, 1):
        hf = r.get("human_faithful", "").strip()
        hh = r.get("human_hides_terms", "").strip()
        hfl = r.get("human_fluent", "").strip()

        if not hf or not hh or not hfl:
            empty_items.append(f"Row {idx} (sample_id={r.get('sample_id')}): faithful='{hf}', hides_terms='{hh}', fluent='{hfl}'")

    if empty_items:
        print("ERROR: Review sheet contains unfilled rows! A human must complete all 20 rows before signing.\n")
        print("Unfilled entries found:")
        for item in empty_items[:5]:
            print(f"  - {item}")
        if len(empty_items) > 5:
            print(f"  ... and {len(empty_items) - 5} more.")
        sys.exit(1)

    file_hash = compute_sha256(SHEET_PATH)
    now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

    sig_record = {
        "banner": BANNER,
        "sheet": "results/real/en/human_sample_en.csv",
        "sha256": file_hash,
        "reviewer": reviewer,
        "rows_reviewed": len(rows),
        "signed_at_utc": now_utc,
    }

    SIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SIG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(sig_record, ensure_ascii=False) + "\n")

    sig_file = ROOT / "results/real/en/human_sample_en.csv.sig"
    sig_file.write_text(json.dumps(sig_record, indent=2), encoding="utf-8")

    print(f"[{BANNER}]")
    print(f"SUCCESS: Verified all {len(rows)} human review items.")
    print(f"Stamped signature:")
    print(f"  Reviewer:    {reviewer}")
    print(f"  SHA-256:     {file_hash}")
    print(f"  Signed (UTC):{now_utc}")
    print(f"  Saved to:    {SIG_PATH} and {sig_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sign off English Dose-Response human review sheet")
    parser.add_argument("--reviewer", required=True, help="Reviewer initials or ID (e.g. VK)")
    args = parser.parse_args()
    validate_and_sign(args.reviewer)


if __name__ == "__main__":
    main()

