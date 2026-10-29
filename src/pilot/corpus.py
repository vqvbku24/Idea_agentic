from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Document:
    doc_id: str
    kind: str
    lang: str
    title: str
    abstract_fr: str = ""
    abstract_en: str = ""
    body_fr: str = ""
    year: int = 0

    @classmethod
    def from_dict(cls, value: dict) -> "Document":
        required = ("doc_id", "kind", "lang", "title")
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"document missing required fields: {', '.join(missing)}")
        return cls(**{field: value.get(field, "") for field in cls.__dataclass_fields__})


def load_jsonl(path: Path) -> list[Document]:
    documents: list[Document] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                documents.append(Document.from_dict(json.loads(line)))
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                raise ValueError(f"invalid corpus line {line_number}: {exc}") from exc
    if not documents:
        raise ValueError(f"corpus is empty: {path}")
    return documents


def write_jsonl(documents: Iterable[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(asdict(document), ensure_ascii=False) + "\n")


def build_index_views(
    documents: Iterable[Document], body_words: int = 1500
) -> dict[str, dict[str, str]]:
    views = {"native": {}, "en_abstract": {}, "both": {}}
    for document in documents:
        if document.lang.lower().startswith("en"):
            native = f"{document.title}\n{document.abstract_en}".strip()
            views["native"][document.doc_id] = native
            views["en_abstract"][document.doc_id] = native
            views["both"][document.doc_id] = native
            continue
        body = " ".join(document.body_fr.split()[:body_words])
        native = f"{document.title}\n{document.abstract_fr}\n{body}".strip()
        english = f"{document.title}\n{document.abstract_en}".strip()
        views["native"][document.doc_id] = native
        if document.abstract_en.strip():
            views["en_abstract"][document.doc_id] = english
            views["both"][document.doc_id] = f"{native}\n{english}"
    return views

