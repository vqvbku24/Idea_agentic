from __future__ import annotations

import csv
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REVIEWER = "Copilot SDK AI assistant"


def artifact_root() -> Path:
    for candidate in (ROOT / "results/real", ROOT / "results/real/real", ROOT / "results/real-colab/real"):
        if (candidate / "papers_x2.jsonl").exists():
            return candidate
    raise FileNotFoundError("x2 artefacts not found")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm(text: str) -> str:
    return " ".join(text.casefold().split())


def title_rule(text: str) -> tuple[str, str, str, str] | None:
    value = " ".join(text.split())
    match = re.search(r"\s+(?=(?:In this|We |This |Our |To |For |A |An |The |As ))", value)
    if not match:
        return None
    removed = value[:match.start()].strip(" .:-")
    if not 3 <= len(removed.split()) <= 18:
        return None
    if sum(word[:1].isupper() for word in removed.split() if word) < 2:
        return None
    after = value[match.end() - 1:].lstrip()
    return value, after, removed, "title_prefix_detector"


def build_leak_draft(root: Path, data: Path) -> list[dict]:
    docs = {}
    for name in ("papers_x2.jsonl", "development_papers_x2.jsonl", "french_distractors_native_x2.jsonl"):
        for row in read_jsonl(data / name):
            docs[row["doc_id"]] = row
    failures = [row for row in csv.DictReader((data / "blind_leak_checks_x2.csv").open(encoding="utf-8")) if row["status"] == "FAIL"]
    output = []
    for row in failures:
        doc = docs[row["paper_id"]]
        abstract = " ".join(doc.get("abstract_en", "").split())
        start = norm(abstract)
        challenge = bool(re.match(r"^(?:deft|défi|defi|challenge)\b", start))
        prefix = re.split(r"\s+(?:This|We|Our|In this|To|For|As)\b", abstract, maxsplit=1)[0].strip()
        capitalized = sum(word[:1].isupper() for word in re.findall(r"[A-Za-zÀ-ÿ]+", prefix))
        ordinary_starter = bool(re.match(r"^(?:This|The|A|An|We|Our|In|To|For|As)\b", abstract))
        has_sentence_boundary = prefix != abstract
        starts_title = not ordinary_starter and has_sentence_boundary and (
            ":" in prefix or capitalized >= 4 or len(prefix.split()) >= 7
        )
        status = "true_leak" if starts_title or challenge else "false_positive"
        if not (starts_title or challenge) and not re.match(r"^[A-Z][^.]{20,}\.", abstract):
            status = "unsure"
        output.append({
            "file": row["file"],
            "paper_id": row["paper_id"],
            "status": status,
            "evidence": abstract[:200],
            "reviewer_agent": REVIEWER,
            "self_review": "true",
        })
    path = ROOT / "data/human/leak_review_draft.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]) if output else ["file", "paper_id", "status", "evidence", "reviewer_agent", "self_review"])
        writer.writeheader()
        writer.writerows(output)
    return output


def build_v21_draft(data: Path) -> list[dict]:
    output = []
    for set_name, filename in (("DEV", "development_papers_x2.jsonl"), ("distractor", "french_distractors_native_x2.jsonl")):
        for row in read_jsonl(data / filename):
            result = title_rule(row.get("abstract_en", ""))
            if not result:
                continue
            before, after, removed, branch = result
            looks_title = "." not in removed and (
                all(word[:1].isupper() for word in removed.split() if word[:1].isalpha())
                or len(removed.split()) <= 8
            )
            status = "correct" if looks_title else "false_positive"
            output.append({
                "set": set_name,
                "paper_id": row["doc_id"],
                "status": status,
                "removed_text": removed,
                "branch": branch,
                "reviewer_agent": REVIEWER,
                "self_review": "true",
            })
    path = ROOT / "data/human/v21_review_draft.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    return output


def resume_check(data: Path) -> tuple[list[dict], dict[str, str]]:
    import fitz
    docs = {row["doc_id"]: row for row in read_jsonl(data / "papers_x2.jsonl")}
    source_ids = json.loads((ROOT / "data/processed/sources_final.json").read_text(encoding="utf-8"))
    rows = []
    excerpts = {}
    for paper_id in source_ids:
        pdf = ROOT / "data/raw/extended" / f"{paper_id}.pdf"
        doc = docs[paper_id]
        if not pdf.exists():
            rows.append({"paper_id": paper_id, "fr_ok": False, "en_ok": False, "raw_first_40_chars_after_header": "[PDF missing]", "extracted_first_40_chars": (doc.get("abstract_fr", "") + " | " + doc.get("abstract_en", ""))[:40], "reason": "PDF missing"})
            continue
        with fitz.open(pdf) as handle:
            page = handle[0].get_text("text")
        lines = page.splitlines()
        sections: dict[str, list[str]] = {"fr": [], "en": []}
        for index, line in enumerate(lines):
            kind = "fr" if re.match(r"^\s*(?:Résumé|Resume)\b", line, re.I) else "en" if re.match(r"^\s*Abstract\b", line, re.I) else None
            if kind:
                tail = re.sub(r"^\s*(?:Résumé|Resume|Abstract)\s*[.:]?\s*", "", line, flags=re.I).strip()
                if tail:
                    sections[kind].append(tail)
                for following in lines[index + 1:index + 12]:
                    if re.match(r"^\s*(?:Résumé|Resume|Abstract|Keywords?|Mots[- ]cl[eé]s|1[\s.)])\b", following, re.I):
                        break
                    if following.strip():
                        sections[kind].append(following.strip())
        raw_fr = " ".join(sections["fr"]).strip()
        raw_en = " ".join(sections["en"]).strip()
        fr = " ".join(doc.get("abstract_fr", "").split())
        en = " ".join(doc.get("abstract_en", "").split())
        fr_ok = bool(raw_fr and norm(fr).startswith(norm(raw_fr)[:80]))
        en_ok = bool(raw_en and norm(en).startswith(norm(raw_en)[:80]))
        rows.append({
            "paper_id": paper_id,
            "fr_ok": fr_ok,
            "en_ok": en_ok,
            "raw_first_40_chars_after_header": f"fr={raw_fr[:20]} | en={raw_en[:20]}",
            "extracted_first_40_chars": (fr[:20] + " | " + en[:20])[:40],
            "reason": "header text comparison is heuristic",
        })
        if paper_id in {"2017.jeptalnrecital-court.2", "2021.jeptalnrecital-deft.7"}:
            excerpts[paper_id] = "RAW PAGE 1:\n" + page[:2000] + "\nEXTRACTED:\nFR=" + fr + "\nEN=" + en
    path = ROOT / "results/real/resume_check_x2.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows, excerpts


def human_sheet(leaks: list[dict], v21: list[dict], excerpts: dict[str, str], resume: list[dict]) -> None:
    rng = random.Random(13)
    shown = []
    for index, row in enumerate(leaks, 1):
        if row["status"] in {"true_leak", "unsure", "false_positive"}:
            shown.append((f"leak-{index}", "leak", row["status"], row["evidence"], row["paper_id"]))
    correct_indices = {
        id(row) for row in rng.sample(
            [r for r in v21 if r["status"] == "correct"],
            min(5, sum(r["status"] == "correct" for r in v21)),
        )
    }
    for index, row in enumerate(v21, 1):
        if row["status"] in {"false_positive", "unsure"} or id(row) in correct_indices:
            shown.append((f"v21-{index}", "v21", row["status"], row["removed_text"], row["paper_id"]))
    lines = ["# Human review sheet", "", "Reviewer must fill `data/human/human_review_decisions.csv`; unsure is the adversarial default.", ""]
    for item_id, kind, status, text, paper_id in shown:
        lines.extend([f"## {item_id} ({kind})", f"- paper_id: `{paper_id}`", f"- agent verdict: **{status}**", f"- text/evidence: {text}", ""])
    for paper_id, text in excerpts.items():
        lines.extend([f"## PDF excerpt: {paper_id}", "```text", text, "```", ""])
    for row in resume:
        if not row["fr_ok"]:
            lines.extend([f"## Resume check failure: {row['paper_id']}", f"- fr_ok: false", f"- evidence: {row['raw_first_40_chars_after_header']} / {row['extracted_first_40_chars']}", ""])
    (ROOT / "results/real/HUMAN_REVIEW_SHEET.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (ROOT / "data/human/human_review_decisions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "file", "human_status", "note"])
        for item_id, kind, *_ in shown:
            writer.writerow([item_id, kind, "", ""])


def main() -> None:
    data = artifact_root()
    leaks = build_leak_draft(ROOT, data)
    v21 = build_v21_draft(data)
    resume, excerpts = resume_check(data)
    human_sheet(leaks, v21, excerpts, resume)
    print(json.dumps({
        "artifact_root": str(data),
        "leak_draft_rows": len(leaks),
        "v21_draft_rows": len(v21),
        "resume_rows": len(resume),
        "resume_failures": sum(not row["fr_ok"] for row in resume),
        "human_review_items": sum(1 for line in (ROOT / "data/human/human_review_decisions.csv").read_text(encoding="utf-8").splitlines()) - 1,
    }, indent=2))


if __name__ == "__main__":
    main()
