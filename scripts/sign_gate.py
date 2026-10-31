from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewer", required=True)
    args = parser.parse_args()
    decisions_path = ROOT / "data/human/human_review_decisions.csv"
    if not decisions_path.exists():
        raise SystemExit("ABORT: fill data/human/human_review_decisions.csv first")
    with decisions_path.open(encoding="utf-8", newline="") as handle:
        decisions = list(csv.DictReader(handle))
    empty = [row["item_id"] for row in decisions if not row.get("human_status", "").strip()]
    if empty:
        raise SystemExit(f"ABORT: empty human_status for {len(empty)} shown items: {', '.join(empty)}")
    draft_paths = {
        "leak": ROOT / "data/human/leak_review_draft.csv",
        "v21": ROOT / "data/human/v21_review_draft.csv",
    }
    decision_map = {row["item_id"]: row for row in decisions}
    outputs = {}
    overridden = 0
    for kind, draft_path in draft_paths.items():
        with draft_path.open(encoding="utf-8", newline="") as handle:
            draft_rows = list(csv.DictReader(handle))
        rows = []
        for index, row in enumerate(draft_rows, 1):
            item_id = f"{kind}-{index}"
            decision = decision_map.get(item_id)
            status = row["status"]
            if decision and decision["human_status"].strip() != status:
                status = decision["human_status"].strip()
                overridden += 1
            row = dict(row)
            row["item_id"] = item_id
            row["human_status"] = status if decision else "agent_only"
            row["signed"] = f"{args.reviewer},{datetime.now(timezone.utc):%Y-%m-%d}"
            rows.append(row)
        output = ROOT / "data/human" / ("leak_review.csv" if kind == "leak" else "v21_review.csv")
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        outputs[kind] = output
    agent_status = {}
    for kind, draft_path in draft_paths.items():
        with draft_path.open(encoding="utf-8", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle), 1):
                agent_status[f"{kind}-{index}"] = row["status"]
    agreement = sum(
        1 for row in decisions if row.get("human_status") == agent_status.get(row["item_id"])
    ) / len(decisions)
    gate = {
        "reviewer": args.reviewer,
        "date": datetime.now(timezone.utc).isoformat(),
        "number_of_items_shown": len(decisions),
        "number_of_items_decided": len(decisions),
        "number_overridden": overridden,
        "agreement_rate": agreement,
        "sha256": {kind: sha256(path) for kind, path in outputs.items()},
    }
    (ROOT / "data/human/SIGNED_GATE.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
