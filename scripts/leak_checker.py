from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def check_blind_entry(
    filename: str,
    row: dict[str, Any],
    doc_title: str,
    doc_abstract_en: str,
    authors: list[str],
) -> list[dict[str, str]]:
    paper_id = row.get("paper_id", "")
    text = " ".join(str(v) for k, v in row.items() if k != "paper_id")
    checks: list[dict[str, str]] = []

    # Common checks for all blind files
    french_title_pass = doc_title.casefold() not in text.casefold() if doc_title else True
    checks.append({"file": filename, "paper_id": paper_id, "check": "french_title", "status": "PASS" if french_title_pass else "FAIL"})

    kw_pass = not bool(re.search(r"\b(?:keywords?|key words)\b", text, re.I))
    checks.append({"file": filename, "paper_id": paper_id, "check": "english_keywords", "status": "PASS" if kw_pass else "FAIL"})

    at_pass = "@" not in text
    checks.append({"file": filename, "paper_id": paper_id, "check": "at_sign", "status": "PASS" if at_pass else "FAIL"})

    author_pass = not any(a.casefold() in text.casefold() for a in authors if a)
    checks.append({"file": filename, "paper_id": paper_id, "check": "author_names", "status": "PASS" if author_pass else "FAIL"})

    affil_pass = not bool(re.search(r"\b(?:université|university|cnrs|laboratoire|laboratory)\b", text, re.I))
    checks.append({"file": filename, "paper_id": paper_id, "check": "affiliations", "status": "PASS" if affil_pass else "FAIL"})

    # Specific checks by file type
    if filename in ("t3_input.jsonl", "t4_passA_input.jsonl"):
        # english_title_like_start applies ONLY to abstract_en fields
        abstract_en = str(row.get("abstract_en", ""))
        title_start_pass = not bool(re.match(r"^(?:[A-Z][a-z]+ ){2,}", abstract_en))
        checks.append({"file": filename, "paper_id": paper_id, "check": "english_title_like_start", "status": "PASS" if title_start_pass else "FAIL"})
    elif filename == "t4_passB_input.jsonl":
        # For t4_passB_input use: English keyword blocks, '@', author names from metadata,
        # affiliation lines, English abstract text, French title. Never english_title_like_start.
        abstract_leak_pass = (doc_abstract_en.casefold() not in text.casefold()) if doc_abstract_en else True
        checks.append({"file": filename, "paper_id": paper_id, "check": "english_abstract_in_passB", "status": "PASS" if abstract_leak_pass else "FAIL"})
        front_matter_pass = affil_pass and at_pass
        checks.append({"file": filename, "paper_id": paper_id, "check": "first_page_front_matter", "status": "PASS" if front_matter_pass else "FAIL"})

    return checks


def load_confirmed_redactions(redactions_path: Path | None = None) -> dict[str, dict[str, str]]:
    path = redactions_path or (ROOT / "data/human/blind_redactions.json")
    if not path.exists():
        return {}
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    confirmed = {}
    for entry in entries:
        if entry.get("human_confirmed") is True:
            confirmed[entry["paper_id"]] = {
                "field": entry.get("field", "abstract_en"),
                "prefix_text": entry.get("prefix_text", ""),
            }
    return confirmed


def apply_redaction(text: str, prefix_text: str) -> str:
    if prefix_text and text.startswith(prefix_text):
        return text[len(prefix_text):].lstrip()
    return text

