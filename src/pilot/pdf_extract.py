from __future__ import annotations

import re
from collections import Counter
from pathlib import Path


def extract_text(pdf_path: Path) -> str:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for PDF extraction; install it with pip install pymupdf") from exc
    with fitz.open(pdf_path) as document:
        return "\f".join(page.get_text("text") for page in document)


def detect_lang(text: str) -> tuple[str, float]:
    """Small dependency-free French detector used for eligibility screening."""
    words = re.findall(r"[A-Za-zÀ-ÿ]+", text.lower())
    if not words:
        return "und", 0.0
    french_markers = {
        "les", "des", "une", "dans", "pour", "avec", "par", "sur", "est",
        "sont", "cette", "étude", "nous", "comme", "qui", "que", "du",
    }
    hits = sum(word in french_markers for word in words)
    ratio = hits / max(1, len(words))
    confidence = min(0.99, ratio * 12)
    return ("fr" if confidence >= 0.5 else "en"), confidence


def _header_pattern(name: str) -> re.Pattern[str]:
    if name.lower() == "résumé":
        return re.compile(r"(?im)^\s*(?:résumé|resume)\s*:?.*$")
    return re.compile(r"(?im)^\s*abstract\s*:?.*$")


def _section_boundary(text: str, start: int) -> int:
    boundary = re.search(
        r"(?im)^\s*(?:abstract(?:\.|:)?|mots[- ]cl[eé]s|keywords?|1[\s.)]+|introduction|references?)\b.*$",
        text[start:],
    )
    return start + boundary.start() if boundary else len(text)


def _extract_section(first_pages: str, header: str) -> tuple[str | None, int | None, int | None]:
    pattern = _header_pattern(header)
    match = pattern.search(first_pages)
    if not match:
        return None, None, None
    start = match.end()
    end = _section_boundary(first_pages, start)
    value = " ".join(first_pages[start:end].split())
    return (value or None), match.start(), end


def _norm_line(line: str) -> str:
    return " ".join(line.lower().split())


def clean_body(text: str, title: str | None = None) -> str:
    """Remove repeated page furniture and the references section."""
    pages = text.split("\f")
    reference = re.compile(
        r"(?im)^\s*(?:references?|bibliographie|r[eé]f[eé]rences)\s*:?\s*$"
    )
    cleaned_pages: list[str] = []
    line_counts = Counter(
        _norm_line(line)
        for page in pages
        for line in page.splitlines()
        if 3 < len(_norm_line(line)) <= 120
    )
    for page in pages:
        page = reference.split(page, maxsplit=1)[0]
        lines = page.splitlines()
        kept: list[str] = []
        for line in lines:
            normalized = _norm_line(line)
            if normalized and line_counts[normalized] >= 2 and len(normalized) <= 120:
                continue
            kept.append(line)
        cleaned_pages.append("\n".join(kept))
    body = "\f".join(cleaned_pages)
    if title:
        title_norm = _norm_line(title)
        lines = body.splitlines()
        seen_title = False
        kept = []
        for line in lines:
            if _norm_line(line) == title_norm:
                if seen_title:
                    continue
                seen_title = True
            kept.append(line)
        body = "\n".join(kept)
    return " ".join(body.split())


def split_abstracts(
    text: str,
    *,
    metadata_abstract: str | None = None,
    metadata_language: str | None = None,
    title: str | None = None,
) -> dict:
    """Extract author-written abstracts from the first two pages when labelled."""
    pages = text.split("\f")
    first_pages = "\f".join(pages[:2])
    fr, fr_start, fr_end = _extract_section(first_pages, "Résumé")
    en, en_start, en_end = _extract_section(first_pages, "Abstract")
    fr_provenance = "pdf" if fr else "missing"
    en_provenance = "pdf" if en else "missing"
    if not fr and metadata_abstract and metadata_language in {"fra", "fre", "fr"}:
        fr, fr_provenance = metadata_abstract, "metadata"
    if not en and metadata_abstract and metadata_language in {"eng", "en"}:
        en, en_provenance = metadata_abstract, "metadata"
    removed = [(start, end) for start, end in ((fr_start, fr_end), (en_start, en_end))
               if start is not None and end is not None]
    body_prefix = first_pages
    for start, end in sorted(removed, reverse=True):
        body_prefix = body_prefix[:start] + body_prefix[end:]
    body = clean_body(body_prefix + "\f" + "\f".join(pages[2:]), title=title)
    return {
        "fr_abstract": fr or "",
        "en_abstract": en or "",
        "body": " ".join(body.split()),
        "provenance": {"fr_abstract": fr_provenance, "en_abstract": en_provenance},
    }
