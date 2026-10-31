from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/real"
DATA_CANDIDATES = (ROOT / "results/real", ROOT / "results/real-colab/real")


def data_root() -> Path:
    for candidate in DATA_CANDIDATES:
        if (candidate / "papers_x2.jsonl").exists():
            return candidate
    expected = ", ".join(str(path) for path in DATA_CANDIDATES)
    raise FileNotFoundError(f"Could not find papers_x2.jsonl in any data root: {expected}")


def data_file(name: str) -> Path:
    for candidate in DATA_CANDIDATES:
        path = candidate / name
        if path.exists():
            return path
    expected = ", ".join(str(path / name) for path in DATA_CANDIDATES)
    raise FileNotFoundError(f"Could not find {name} in any data root: {expected}")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def title_rule(text: str) -> tuple[str, str] | None:
    value = " ".join(text.split())
    markers = r"(?:In this|We |This |Our |To |For |A |An |The |As )"
    match = re.search(r"\s+(?=" + markers + r")", value)
    if not match:
        return None
    prefix = value[:match.start()].strip(" .:-")
    if not (3 <= len(prefix.split()) <= 18):
        return None
    if sum(word[:1].isupper() for word in prefix.split() if word) < 2:
        return None
    after = value[match.end() - 1:].lstrip()
    return value, after


def title_rule_details(text: str) -> tuple[str, str, str, bool] | None:
    result = title_rule(text)
    if not result:
        return None
    before, after = result
    removed = before[: len(before) - len(after)].strip()
    looks_like_title = (
        "." not in removed
        and 3 <= len(removed.split()) <= 18
        and bool(removed[:1].isupper())
        and (
            all(word[:1].isupper() for word in removed.split() if word[:1].isalpha())
            or len(removed.split()) <= 8
        )
    )
    return before, after, removed, looks_like_title


def write_leak_cases() -> int:
    docs = {}
    for name in ("papers_x2.jsonl", "development_papers_x2.jsonl", "french_distractors_native_x2.jsonl"):
        for row in read_jsonl(data_file(name)):
            docs[row["doc_id"]] = row
    rows = list(csv.DictReader(data_file("blind_leak_checks_x2.csv").open(encoding="utf-8")))
    failed = [row for row in rows if row["status"] == "FAIL"]
    lines = [
        "# Blind-input leak failures",
        "",
        f"Total failed cells: {len(failed)}.",
        "",
        "| File | Paper ID | Triggered Field | First 200 chars of offending field | Rule |",
        "|---|---|---|---|---|",
    ]
    blind_cache = {}
    for row in failed:
        filename = row["file"]
        if filename not in blind_cache:
            p = ROOT / "data/human" / filename
            blind_cache[filename] = {r["paper_id"]: r for r in read_jsonl(p)} if p.exists() else {}
        input_row = blind_cache[filename].get(row["paper_id"], {})
        if filename == "t4_passB_input.jsonl":
            triggered_field = "body_fr"
            field_text = input_row.get("body_fr", "")
        elif filename == "t4_passA_input.jsonl":
            triggered_field = "abstract_en"
            field_text = input_row.get("abstract_en", "")
        elif filename == "t3_input.jsonl":
            if row["check"] == "english_title_like_start":
                triggered_field = "abstract_en"
                field_text = input_row.get("abstract_en", "")
            else:
                triggered_field = "abstract_en" if row["check"] in ("english_keywords", "at_sign") else "abstract_fr"
                field_text = input_row.get(triggered_field, "") or input_row.get("abstract_en", "")
        else:
            triggered_field = "unknown"
            field_text = ""
        snippet = " ".join(str(field_text).split())[:200]
        safe_snippet = snippet.replace("|", r"\|")
        lines.append(
            f"| {row['file']} | {row['paper_id']} | {triggered_field} | "
            f"{safe_snippet} | {row['check']} |"
        )
    (OUT / "leak_fail_cases_x2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(failed)


def canonical_report() -> None:
    values = list(csv.DictReader(data_file("dense_reference_x2.csv").open(encoding="utf-8")))
    report_path = data_file("retrieval_v2_diagnostics.md")
    report = report_path.read_text(encoding="utf-8")
    canonical = [
        "## B2. Canonical dense reference summary",
        "",
        "The CSV is the single source of truth for aggregate cosine values. The earlier per-text "
        "8192 native values in section B came from a different reference invocation; this section "
        "replaces that aggregate discrepancy with the final CSV values.",
        "",
        "| Reference max length | Text type | Min cosine | Mean cosine | Count < 0.99 |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in values:
        canonical.append(
            f"| {row['reference_max_length']} | {row['text_type']} | "
            f"{float(row['min_cosine']):.6f} | {float(row['mean_cosine']):.6f} | "
            f"{row['count_below_0.99']} |"
        )
    canonical.append("")
    report = re.sub(r"\n## B2\..*", "", report, flags=re.S)
    report += "\n" + "\n".join(canonical) + "\n"
    report += "\n## D2. doc_translate_bm25 language-matched control\n\n"
    report += (
        "The English-abstract query is the language-matched control for `doc_translate_bm25`; "
        "the French-query result remains informational. The Colab report persisted document-level "
        "language checks, not per-chunk translation outputs, so an exact translated-chunk share "
        "cannot be reconstructed from these artefacts without rerunning translation.\n\n"
    )
    report += "| English distractors | View | English-query Recall@10 | Mostly-French translated docs | Docs checked | Chunk share |\n"
    report += "|---:|---|---:|---:|---:|---|\n"
    report += "| 500 | native_abstract | 1.000 | 0 | 120 | not persisted |\n"
    report += "| 100 | native_abstract | 1.000 | 0 | 120 | not persisted |\n"
    report += "| 500 | en_abstract | 0.950 | 1 | 120 | not persisted |\n"
    report += "| 100 | en_abstract | 0.900 | 1 | 120 | not persisted |\n"
    report_path.write_text(report, encoding="utf-8")


def write_rule_review() -> tuple[int, int]:
    changed = []
    counts = {}
    for set_name, filename in (
        ("development", "development_papers_x2.jsonl"),
        ("french_distractor", "french_distractors_native_x2.jsonl"),
    ):
        rows = read_jsonl(data_file(filename))
        count = 0
        for row in rows:
            result = title_rule_details(row.get("abstract_en", ""))
            if result:
                before, after, removed, looks_like_title = result
                changed.append((
                    set_name,
                    row["doc_id"],
                    before[:150],
                    after[:150],
                    removed,
                    "title_prefix_detector",
                    looks_like_title,
                ))
                count += 1
        counts[set_name] = count
    lines = [
        "# Candidate extractor rule v2.1 (proposal only)",
        "",
        "Rule: in `abstract_en`, remove a leading 3–18-word title-case segment only when it "
        "is immediately followed by a sentence-start marker (`In this`, `We`, `This`, `Our`, "
        "`To`, `For`, `A`, `An`, `The`, or `As`). This was evaluated only on DEV and French "
        "distractors; it was not applied to sources and no tag was created.",
        "",
        f"- Development changed: {counts['development']}",
        f"- French distractors changed: {counts['french_distractor']}",
        f"- Total changed: {len(changed)}",
        "",
        "| Set | Paper ID | Before (first 150 chars) | After (first 150 chars) | Removed text (full line) | Rule/branch | looks_like_title |",
        "|---|---|---|---|---|---|---|",
    ]
    matches = sum(item[-1] for item in changed)
    lines.insert(8, f"- Earlier title-prefix detector matches: {matches}")
    lines.insert(9, f"- Earlier detector non-matches: {len(changed) - matches}")
    for set_name, paper_id, before, after, removed, branch, looks_like_title in changed:
        safe_before = before.replace("|", "\\|")
        safe_after = after.replace("|", "\\|")
        safe_removed = removed.replace("|", "\\|")
        lines.append(
            f"| {set_name} | {paper_id} | {safe_before} | {safe_after} | "
            f"{safe_removed} | {branch} | {'yes' if looks_like_title else 'no'} |"
        )
    (OUT / "candidate_rule_v2_1_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return counts["development"], counts["french_distractor"]


def write_meta() -> None:
    tracked = sorted((ROOT / "src/pilot").glob("*.py"))
    hashes = {}
    mismatches = []
    baseline = git("rev-parse", "HEAD")
    for path in tracked:
        relative = path.relative_to(ROOT)
        git_path = relative.as_posix()
        if subprocess.run(
            ["git", "ls-files", "--error-unmatch", git_path],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode:
            hashes[git_path] = {
                "working_tree_sha256_raw": hashlib.sha256(path.read_bytes()).hexdigest(),
                "commit_sha256": None,
            }
            mismatches.append(git_path)
            continue
        raw_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        committed_bytes = subprocess.check_output(["git", "show", f"HEAD:{git_path}"], cwd=ROOT)
        committed_hash = hashlib.sha256(committed_bytes).hexdigest()
        normalized_digest = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        committed_normalized = hashlib.sha256(committed_bytes.replace(b"\r\n", b"\n")).hexdigest()
        hashes[git_path] = {
            "working_tree_sha256_raw": raw_digest,
            "commit_sha256_raw": committed_hash,
            "working_tree_sha256_lf": normalized_digest,
            "commit_sha256_lf": committed_normalized,
        }
        if normalized_digest != committed_normalized:
            mismatches.append(git_path)
    packages = {}
    for name in ("torch", "transformers", "sentence-transformers", "numpy", "scipy", "PyMuPDF"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    meta = {
        "colab_run": {
            "commit_used": "not recorded in Colab log",
            "nearest_local_commit": baseline,
            "commit_verified": False,
            "tags_at_nearest_local_commit": git("tag", "--points-at", baseline).splitlines(),
        },
        "package_versions": {"source": "local provenance environment; Colab package versions were not logged", **packages},
        "device": "not recorded in Colab log",
        "seeds": {"selection": 13},
        "model_ids": {"dense": "BAAI/bge-m3", "translation": "Helsinki-NLP/opus-mt-fr-en"},
        "max_lengths": {"dense": 1024, "reference": [1024, 8192], "translation_chunk_tokens": 400},
        "src_pilot_hash_comparison": {
            "commit": baseline,
            "mismatches_after_lf_normalization": mismatches,
            "files": hashes,
            "note": "Raw hashes differ for CRLF working-tree files; LF-normalized hashes identify semantic content mismatches.",
        },
        "limitations": ["The Colab log contains model-loading output but no git SHA, device, or package manifest."],
    }
    (OUT / "run_meta_x2.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    failed = write_leak_cases()
    canonical_report()
    dev, distractors = write_rule_review()
    write_meta()
    print(f"leak_failures={failed}; rule_changes_dev={dev}; rule_changes_distractors={distractors}")


if __name__ == "__main__":
    main()
