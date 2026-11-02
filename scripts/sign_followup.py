from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANNER = "EXPLORATORY - agent-proxy labels, human verification pending"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stamp file hash + initials + UTC date for follow-up review files.")
    parser.add_argument("--reviewer", required=True, help="Human reviewer initials (e.g. 'VK').")
    parser.add_argument(
        "--file",
        type=Path,
        default=ROOT / "results/real/followup/judge_audit_sheet.csv",
        help="Path to the file to sign (default: results/real/followup/judge_audit_sheet.csv).",
    )
    parser.add_argument(
        "--signatures-file",
        type=Path,
        default=ROOT / "results/real/followup/signatures.jsonl",
        help="Path to the signatures log file.",
    )
    args = parser.parse_args()

    target_path = args.file if args.file.is_absolute() else ROOT / args.file
    if not target_path.exists():
        raise SystemExit(f"ABORT: target file does not exist: {target_path}")

    # Check for empty human fields if signing judge_audit_sheet.csv
    if target_path.name == "judge_audit_sheet.csv":
        with target_path.open(encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        empty = [
            r["item_id"] for r in reader
            if not (r.get("human_faithful", "").strip() and r.get("human_leaks", "").strip() and r.get("human_fluent", "").strip())
        ]
        if empty:
            raise SystemExit(
                f"ABORT: Cannot sign because {len(empty)} of {len(reader)} items have empty human fields.\n"
                f"Items with missing fields: {', '.join(empty[:5])}..."
            )

    file_hash = sha256_file(target_path)
    now_utc = datetime.now(timezone.utc)

    sig_record = {
        "file": target_path.relative_to(ROOT).as_posix() if target_path.is_relative_to(ROOT) else target_path.name,
        "sha256": file_hash,
        "reviewer": args.reviewer.strip(),
        "signed_at_utc": now_utc.isoformat(),
        "date_utc": now_utc.strftime("%Y-%m-%d"),
    }

    sig_path = args.signatures_file if args.signatures_file.is_absolute() else ROOT / args.signatures_file
    sig_path.parent.mkdir(parents=True, exist_ok=True)
    with sig_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(sig_record) + "\n")

    # Also write a companion .sig file
    companion_sig = target_path.with_suffix(target_path.suffix + ".sig")
    companion_sig.write_text(json.dumps(sig_record, indent=2), encoding="utf-8")

    print(f"[{BANNER}]")
    print(f"Successfully signed {target_path.name}:")
    print(f"  SHA-256:       {file_hash}")
    print(f"  Reviewer:      {args.reviewer}")
    print(f"  Signed at:     {sig_record['signed_at_utc']}")
    print(f"  Recorded in:   {sig_path} and {companion_sig}")


if __name__ == "__main__":
    main()
