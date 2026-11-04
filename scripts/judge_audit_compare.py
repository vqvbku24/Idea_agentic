from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

BANNER = "EXPLORATORY - agent-proxy labels, human verification pending"


def parse_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in {"yes", "true", "1", "y"}:
        return True
    if s in {"no", "false", "0", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean from {val!r}")


def compute_cohen_kappa(y_true: list[bool], y_pred: list[bool]) -> float | None:
    """Computes Cohen's kappa between two binary raters. Returns None if undefined."""
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return None

    n = len(y_true)
    # Contingency table:
    # a: both True
    # b: true False, pred True
    # c: true True, pred False
    # d: both False
    a = sum(1 for yt, yp in zip(y_true, y_pred) if yt and yp)
    b = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and yp)
    c = sum(1 for yt, yp in zip(y_true, y_pred) if yt and not yp)
    d = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and not yp)

    po = (a + d) / n
    p_yes = ((a + b) / n) * ((a + c) / n)
    p_no = ((c + d) / n) * ((b + d) / n)
    pe = p_yes + p_no

    if math.isclose(pe, 1.0, abs_tol=1e-9):
        return 1.0 if math.isclose(po, 1.0, abs_tol=1e-9) else 0.0

    return (po - pe) / (1.0 - pe)


def evaluate_judge_audit(
    audit_rows: list[dict],
    key_rows: list[dict],
) -> dict[str, Any]:
    key_by_id = {r["item_id"]: r for r in key_rows}

    # Verify matching item IDs
    item_ids = [r["item_id"] for r in audit_rows]
    for iid in item_ids:
        if iid not in key_by_id:
            raise KeyError(f"Item ID {iid} from audit sheet not found in sealed key")

    # Check for empty human values
    for r in audit_rows:
        for field in ("human_faithful", "human_leaks", "human_fluent"):
            if not str(r.get(field, "")).strip():
                raise ValueError(
                    f"Human label {field} is empty for {r.get('item_id')}. Real data is not yet labelled."
                )

    criteria = ["faithful", "leaks", "fluent", "overall_verdict"]
    results: dict[str, Any] = {
        "banner": BANNER,
        "n_items": len(audit_rows),
        "criteria": {},
    }

    # Collect pairs
    human_faithful = [parse_bool(r["human_faithful"]) for r in audit_rows]
    human_leaks = [parse_bool(r["human_leaks"]) for r in audit_rows]
    human_fluent = [parse_bool(r["human_fluent"]) for r in audit_rows]

    # Human overall accept: faithful and not leaks and fluent (for V0 leaks may be allowed)
    human_verdicts = []
    for r, h_faith, h_leak, h_flu in zip(audit_rows, human_faithful, human_leaks, human_fluent):
        v = key_by_id[r["item_id"]].get("variant", "")
        acc = h_faith and (v == "V0" or not h_leak) and h_flu
        human_verdicts.append(acc)

    judge_faithful = [parse_bool(key_by_id[r["item_id"]]["judge_faithful"]) for r in audit_rows]
    judge_leaks = [parse_bool(key_by_id[r["item_id"]]["judge_leaks"]) for r in audit_rows]
    judge_fluent = [parse_bool(key_by_id[r["item_id"]]["judge_fluent"]) for r in audit_rows]
    judge_verdicts = [parse_bool(key_by_id[r["item_id"]]["accepted"]) for r in audit_rows]

    pairs_dict = {
        "faithful": (human_faithful, judge_faithful),
        "leaks": (human_leaks, judge_leaks),
        "fluent": (human_fluent, judge_fluent),
        "overall_verdict": (human_verdicts, judge_verdicts),
    }

    for name, (y_true, y_pred) in pairs_dict.items():
        n = len(y_true)
        agreed = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
        pct_agree = (agreed / n) * 100.0
        kappa = compute_cohen_kappa(y_true, y_pred)

        # TP, FP, TN, FN (Truth = Human, Pred = Judge)
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt and yp)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and yp)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and not yp)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt and not yp)

        n_neg = fp + tn
        n_pos = tp + fn
        false_accept_rate = (fp / n_neg) if n_neg > 0 else 0.0
        false_reject_rate = (fn / n_pos) if n_pos > 0 else 0.0

        results["criteria"][name] = {
            "agreement_pct": round(pct_agree, 2),
            "cohen_kappa": round(kappa, 4) if kappa is not None else None,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "false_accept_rate": round(false_accept_rate, 4),
            "false_reject_rate": round(false_reject_rate, 4),
            "decision_rule_judge_usable": pct_agree >= 80.0,
        }

    return results


def run_synthetic_test() -> None:
    print("=== RUNNING SYNTHETIC FIXTURE TEST ===")
    mock_audit = [
        {"item_id": "i01", "human_faithful": "yes", "human_leaks": "no", "human_fluent": "yes"},
        {"item_id": "i02", "human_faithful": "yes", "human_leaks": "yes", "human_fluent": "yes"},
        {"item_id": "i03", "human_faithful": "no", "human_leaks": "no", "human_fluent": "yes"},
        {"item_id": "i04", "human_faithful": "yes", "human_leaks": "no", "human_fluent": "no"},
    ]
    mock_key = [
        {"item_id": "i01", "variant": "V1", "judge_faithful": "True", "judge_leaks": "False", "judge_fluent": "True", "accepted": "True"},
        {"item_id": "i02", "variant": "V1", "judge_faithful": "True", "judge_leaks": "True", "judge_fluent": "True", "accepted": "False"},
        {"item_id": "i03", "variant": "V1", "judge_faithful": "False", "judge_leaks": "False", "judge_fluent": "True", "accepted": "False"},
        {"item_id": "i04", "variant": "V1", "judge_faithful": "True", "judge_leaks": "False", "judge_fluent": "True", "accepted": "True"},  # disagreement on fluent
    ]

    res = evaluate_judge_audit(mock_audit, mock_key)
    print("Synthetic evaluation results:")
    for crit, data in res["criteria"].items():
        print(f"  {crit:15}: Agreement={data['agreement_pct']}% | Kappa={data['cohen_kappa']} | FAR={data['false_accept_rate']} | FRR={data['false_reject_rate']}")

    # Check expectations:
    # faithful: 4/4 agree -> 100%
    assert res["criteria"]["faithful"]["agreement_pct"] == 100.0, "Faithful agreement should be 100%"
    assert res["criteria"]["faithful"]["cohen_kappa"] == 1.0, "Faithful kappa should be 1.0"
    # leaks: 4/4 agree -> 100%
    assert res["criteria"]["leaks"]["agreement_pct"] == 100.0, "Leaks agreement should be 100%"
    # fluent: 3/4 agree -> 75%
    assert res["criteria"]["fluent"]["agreement_pct"] == 75.0, "Fluent agreement should be 75%"
    assert res["criteria"]["fluent"]["fp"] == 1, "Fluent FP should be 1"

    print("\nSynthetic fixture tests PASSED successfully!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit comparison between human ratings and LLM judge.")
    parser.add_argument("--test", action="store_true", help="Run synthetic fixture self-test.")
    parser.add_argument("--audit-sheet", type=Path, default=Path("results/real/followup/judge_audit_sheet.csv"))
    parser.add_argument("--key", type=Path, default=Path("results/real/followup/sealed/judge_key.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("results/real/followup/judge_audit_metrics.json"))
    args = parser.parse_args()

    if args.test:
        run_synthetic_test()
        return

    # Check if audit sheet has been filled
    if not args.audit_sheet.exists():
        print(f"Error: audit sheet {args.audit_sheet} does not exist.")
        sys.exit(1)

    with args.audit_sheet.open(encoding="utf-8") as f:
        audit_rows = list(csv.DictReader(f))

    # Guard: do not run on empty real data
    empty_count = sum(1 for r in audit_rows if not str(r.get("human_faithful", "")).strip())
    if empty_count > 0:
        print(f"ABORT: {empty_count} of {len(audit_rows)} rows have empty human labels.")
        print("Under Rule G2, this script cannot be run on real data until the human reviewer completes and signs the audit sheet.")
        sys.exit(1)

    with args.key.open(encoding="utf-8") as f:
        key_rows = list(csv.DictReader(f))

    metrics = evaluate_judge_audit(audit_rows, key_rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved audit comparison metrics to {args.output_json}")


if __name__ == "__main__":
    main()

