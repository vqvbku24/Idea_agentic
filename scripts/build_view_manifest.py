from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def artifact_root() -> Path:
    for candidate in (ROOT / "results/real", ROOT / "results/real/real", ROOT / "results/real-colab/real"):
        if (candidate / "papers_x2.jsonl").exists():
            return candidate
    raise FileNotFoundError("papers_x2.jsonl not found")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    root = artifact_root()
    sources = read_jsonl(root / "papers_x2.jsonl")
    french = read_jsonl(root / "french_distractors_native_x2.jsonl")
    english_path = ROOT / "data/processed/english_distractors_metadata.jsonl"
    english = read_jsonl(english_path)[:500]
    documents = []
    for row in sources:
        documents.append((row["doc_id"], "source", row.get("title", ""), row.get("abstract_fr", ""), row.get("abstract_en", ""), row.get("body_fr", ""), row.get("keywords_en", "")))
    for row in french:
        if row.get("eligible", True):
            documents.append((row["doc_id"], "fr_distractor", row.get("title", ""), row.get("abstract_fr", ""), row.get("abstract_en", ""), row.get("body_fr", ""), row.get("keywords_en", "")))
    for row in english:
        documents.append((row["paper_id"], "en_distractor", row.get("title", ""), "", row.get("abstract_meta", ""), "", ""))

    views: dict[str, dict[str, str]] = {name: {} for name in ("native", "native_abstract", "en_abstract", "both", "native_kw_en")}
    reasons: dict[str, dict[str, str]] = {name: {} for name in views}
    for doc_id, kind, title, abstract_fr, abstract_en, body_fr, keywords_en in documents:
        if kind == "en_distractor":
            native = f"{title}\n{abstract_en}".strip()
            for view in views:
                views[view][doc_id] = native
            continue
        native = f"{title}\n{abstract_fr}\n{' '.join(body_fr.split()[:1500])}".strip()
        native_abstract = f"{title}\n{abstract_fr}".strip()
        native_kw = f"{native}\n{keywords_en}".strip()
        english_view = f"{title}\n{abstract_en}".strip()
        views["native"][doc_id] = native
        views["native_abstract"][doc_id] = native_abstract
        views["native_kw_en"][doc_id] = native_kw
        if abstract_en.strip():
            views["en_abstract"][doc_id] = english_view
            views["both"][doc_id] = f"{native}\n{english_view}"
        else:
            reasons["en_abstract"][doc_id] = "missing author-written English abstract"
            reasons["both"][doc_id] = "missing author-written English abstract"

    expected = {doc_id for doc_id, *_ in documents}
    result = {"schema_version": 1, "artifact_root": str(root), "views": {}}
    for name, mapping in views.items():
        missing = [
            {"doc_id": doc_id, "kind": next(kind for item_id, kind, *_ in documents if item_id == doc_id), "reason": reasons[name].get(doc_id, "not indexed")}
            for doc_id in sorted(expected - set(mapping))
        ]
        result["views"][name] = {
            "doc_ids": sorted(mapping),
            "doc_count": len(mapping),
            "missing": missing,
        }
    output = ROOT / "results/real/view_manifest.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: value["doc_count"] for name, value in result["views"].items()}, indent=2))


if __name__ == "__main__":
    main()
