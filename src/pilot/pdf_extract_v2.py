from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

from .pdf_extract import clean_body, detect_lang, extract_text

EXTRACTOR_VERSION = 2


def _header_match(line: str, language: str) -> re.Match[str] | None:
    if language == "fr":
        return re.match(r"^[ \t]*(?:Résumé|Resume)[ \t]*[.:]?[ \t]*(.*)$", line, re.I)
    return re.match(r"^[ \t]*Abstract[ \t]*[.:]?[ \t]*(.*)$", line, re.I)


def _section_end(lines: list[str], start: int) -> int:
    for index in range(start, len(lines)):
        if re.match(
            r"^[ \t]*(?:Résumé|Resume|Abstract|MOTS[- ]CL[ÉE]S|KEYWORDS?|"
            r"(?:1[.)]?[ \t]+)?Introduction|References?|Références?|Bibliographie)\b",
            lines[index],
            re.I,
        ):
            return index
    return len(lines)


def _extract_section(lines: list[str], language: str) -> tuple[str, int | None, int | None]:
    for index, line in enumerate(lines):
        match = _header_match(line, language)
        if not match:
            continue
        inline = match.group(1).strip()
        start = index + 1
        values = [inline] if inline else []
        end = _section_end(lines, start)
        values.extend(lines[start:end])
        return "\n".join(values), index, end
    return "", None, None


def _keywords(lines: list[str], language: str) -> tuple[str, int | None]:
    pattern = r"^[ \t]*(?:MOTS[- ]CL[ÉE]S|KEYWORDS?)[ \t]*:?[ \t]*(.*)$"
    for index, line in enumerate(lines):
        match = re.match(pattern, line, re.I)
        if match:
            return " ".join(match.group(1).split()), index
    return "", None


def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def _strip_title_and_challenge(text: str, title: str) -> tuple[str, list[str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    removed: list[str] = []
    title_norm = _norm(title)
    while lines:
        line_norm = _norm(lines[0])
        compact = re.sub(r"[^a-z0-9à-ÿ]+", " ", line_norm).strip()
        title_compact = re.sub(r"[^a-z0-9à-ÿ]+", " ", title_norm).strip()
        title_like = line_norm == title_norm or (
            len(compact) > 20 and (compact in title_compact or title_compact in compact)
        )
        challenge = bool(re.match(r"^(?:deft|défi|defi|challenge)\b", line_norm, re.I))
        if title_like or challenge:
            removed.append(lines.pop(0))
            continue
        break
    return " ".join(lines).strip(), removed


def _remove_front_matter(text: str, title: str, keywords: tuple[str, str]) -> str:
    lines = text.splitlines()
    intro = next(
        (i for i, line in enumerate(lines)
         if re.match(r"^[ \t]*(?:\d+[.)]?[ \t]+)?Introduction\b", line, re.I)),
        None,
    )
    if intro is not None:
        text = "\n".join(lines[intro:])
    else:
        text = "\n".join(lines)
    cleaned = clean_body(text, title=None)
    for keyword in keywords:
        if keyword:
            cleaned = re.sub(re.escape(keyword), "", cleaned, flags=re.I)
    cleaned = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "", cleaned)
    return " ".join(cleaned.split())


def split_abstracts_v2(
    text: str,
    *,
    metadata_abstract: str | None = None,
    metadata_language: str | None = None,
    title: str = "",
) -> dict:
    pages = text.split("\f")
    first = "\n".join(pages[:2])
    lines = first.splitlines()
    fr, fr_start, fr_end = _extract_section(lines, "fr")
    en, en_start, en_end = _extract_section(lines, "en")
    keywords_fr, _ = _keywords(lines, "fr")
    keywords_en, _ = _keywords(lines, "en")
    fr_provenance = "pdf" if fr else "missing"
    en_provenance = "pdf" if en else "missing"
    if not fr and metadata_abstract and metadata_language in {"fra", "fre", "fr"}:
        fr, fr_provenance = metadata_abstract, "metadata"
    if not en and metadata_abstract and metadata_language in {"eng", "en"}:
        en, en_provenance = metadata_abstract, "metadata"
    en, removed_prefix = _strip_title_and_challenge(en, title)
    removed_lines = []
    for start, end in ((fr_start, fr_end), (en_start, en_end)):
        if start is not None and end is not None:
            removed_lines.extend(lines[start:end])
    body = "\n".join(pages)
    body = _remove_front_matter(body, title, (keywords_fr, keywords_en))
    body = clean_body(body, title=title)
    flags = []
    if removed_prefix:
        flags.append("en_prefix_removed")
    if re.search(r"\b(?:MOTS[- ]CL[ÉE]S|KEYWORDS?)\b", body, re.I):
        flags.append("keywords_in_body")
    return {
        "fr_abstract": " ".join(fr.split()),
        "en_abstract": " ".join(en.split()),
        "body": body,
        "keywords_fr": keywords_fr,
        "keywords_en": keywords_en,
        "provenance": {"fr_abstract": fr_provenance, "en_abstract": en_provenance},
        "flags": flags,
        "extractor_version": EXTRACTOR_VERSION,
        "removed_en_prefix": removed_prefix,
        "body_language": detect_lang(body)[0],
        "body_language_confidence": detect_lang(body)[1],
    }


def extract_pdf_v2(pdf_path: Path, **kwargs) -> dict:
    return split_abstracts_v2(extract_text(pdf_path), **kwargs)
