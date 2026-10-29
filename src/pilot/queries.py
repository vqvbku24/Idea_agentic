from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable

from .corpus import Document


@dataclass(frozen=True)
class Query:
    source_id: str
    variant: str
    text: str
    kept_facets: tuple[str, ...] = ()
    lexical_leak: float = 0.0
    accepted: bool = True
    rejection_reason: str = ""


def lexical_leak(query: str, title: str) -> float:
    title_tokens = {token for token in re.findall(r"\w+", title.lower()) if len(token) >= 5}
    query_tokens = set(re.findall(r"\w+", query.lower()))
    return len(title_tokens & query_tokens) / max(1, len(title_tokens))


def _scrub(text: str, title: str) -> str:
    title_words = {word.lower() for word in re.findall(r"\w+", title) if len(word) >= 4}
    return " ".join(word for word in text.split() if word.lower().strip(".,;:") not in title_words)


def generate_queries(document: Document) -> list[Query]:
    facets = {
        "problem": document.abstract_en or document.abstract_fr,
        "mechanism": document.body_fr.split(".")[0] if document.body_fr else "",
        "setting": document.title,
        "outcome": document.abstract_en,
    }
    v0 = " ".join(value for value in facets.values() if value).strip()
    v1 = _scrub(
        "A study addresses a language technology problem using a structured computational "
        "method in a research setting and evaluates the resulting improvement. " + v0,
        document.title,
    )
    kept = ("problem", "outcome")
    v2 = " ".join(facets[name] for name in kept if facets[name]).strip()
    v3 = _scrub(
        "The work studies a problem and reports an outcome after applying a computational "
        "method to research data. " + v0,
        document.title,
    )
    values = [("V0", v0, ()), ("V1", v1, ()), ("V2", v2, kept), ("V3", v3, ())]
    result: list[Query] = []
    for variant, text, kept_facets in values:
        leak = lexical_leak(text, document.title)
        result.append(Query(document.doc_id, variant, text, kept_facets, leak))
    return result


def query_dict(query: Query) -> dict:
    value = asdict(query)
    value["kept_facets"] = list(query.kept_facets)
    return value

