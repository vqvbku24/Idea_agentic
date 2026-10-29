from __future__ import annotations

import json
import random
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ACL_API = "https://api.github.com/repos/acl-org/acl-anthology/contents/data/xml"
ACL_RAW = "https://raw.githubusercontent.com/acl-org/acl-anthology/master/"
USER_AGENT = "cross-lingual-prior-art-pilot/0.1 (research smoke test)"
BULK_ARCHIVE_URL = "https://codeload.github.com/acl-org/acl-anthology/zip/refs/heads/master"


@dataclass(frozen=True)
class PaperMeta:
    paper_id: str
    title: str
    year: int
    venue: str
    pdf_url: str | None
    abstract_meta: str | None
    authors: list[str]
    language: str | None = None


def _request(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def download_bulk_metadata(cache_path: Path) -> Path:
    """Cache the official ACL repository archive containing authoritative XML."""
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    content = _request(BULK_ARCHIVE_URL)
    if not content.startswith(b"PK"):
        raise ValueError("ACL bulk archive response is not a ZIP file")
    cache_path.write_bytes(content)
    return cache_path


def _metadata_files() -> list[str]:
    try:
        payload = json.loads(_request(ACL_API))
        return [
            item["download_url"]
            for item in payload
            if item.get("type") == "file"
            and item.get("name", "").endswith(".xml")
            and ".jeptalnrecital." in item.get("name", "")
        ]
    except Exception:
        return [
            f"{ACL_RAW}data/xml/{year}.jeptalnrecital.xml"
            for year in range(2001, 2025)
        ]


def _all_metadata_files() -> list[str]:
    files: list[str] = []
    page = 1
    try:
        while True:
            payload = json.loads(_request(f"{ACL_API}?per_page=100&page={page}"))
            if not payload:
                break
            files.extend(
                item["download_url"]
                for item in payload
                if item.get("type") == "file" and item.get("name", "").endswith(".xml")
            )
            if len(payload) < 100:
                break
            page += 1
    except Exception:
        return []
    return files


def _all_papers() -> list[PaperMeta]:
    bulk = Path("data/cache/acl-anthology-master.zip")
    if bulk.exists():
        with zipfile.ZipFile(bulk) as archive:
            urls = [
                name for name in archive.namelist()
                if "/data/xml/" in name and name.endswith(".xml")
            ]
            with ThreadPoolExecutor(max_workers=16) as executor:
                batches = executor.map(
                    lambda name: _parse_metadata_xml_bytes(archive.read(name), name),
                    urls,
                )
                return [paper for batch in batches for paper in batch]
    with ThreadPoolExecutor(max_workers=16) as executor:
        batches = executor.map(_parse_metadata_xml, _all_metadata_files())
        return [paper for batch in batches for paper in batch]


def _parse_metadata_xml_bytes(content: bytes, name: str) -> list[PaperMeta]:
    filename = name.rsplit("/", 1)[-1]
    try:
        year = int(filename[:4])
    except ValueError:
        return []
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return []
    return _parse_root(root, filename, year)


def _parse_metadata_xml(url: str) -> list[PaperMeta]:
    filename = url.rsplit("/", 1)[-1]
    try:
        year = int(filename[:4])
    except ValueError:
        return []
    try:
        root = ET.fromstring(_request(url))
    except Exception:
        return []
    return _parse_root(root, filename, year)


def _parse_root(root: ET.Element, filename: str, year: int) -> list[PaperMeta]:
    papers: list[PaperMeta] = []
    for volume in root.findall("volume"):
        volume_id = volume.get("id")
        venue = volume.findtext("./meta/venue") or filename[5:-4]
        meta_year = volume.findtext("./meta/year")
        volume_year = int(meta_year) if meta_year and meta_year.isdigit() else year
        for paper in volume.findall("paper"):
            paper_number = paper.get("id")
            if not volume_id or not paper_number:
                continue
            paper_id = f"{year}.{venue}-{volume_id}.{paper_number}"
            authors = [
                " ".join(part for part in (author.findtext("first"), author.findtext("last")) if part)
                for author in paper.findall("author")
            ]
            papers.append(PaperMeta(
                paper_id=paper_id,
                title=" ".join((paper.findtext("title") or "").split()),
                year=volume_year,
                venue=venue,
                pdf_url=f"https://aclanthology.org/{paper_id}.pdf"
                if paper.find("pdf") is not None else None,
                abstract_meta=" ".join((paper.findtext("abstract") or "").split()) or None,
                authors=authors,
                language=paper.findtext("language"),
            ))
    return papers


def _english_abstract(text: str | None) -> bool:
    if not text:
        return False
    words = set(text.lower().split())
    return len(words & {"the", "of", "and", "in", "for", "this", "we", "is", "on"}) >= 2


def list_english_distractors(
    n: int, seed: int, exclude_venue: str = "jeptalnrecital"
) -> list[PaperMeta]:
    """Discover English abstract-only distractors across Anthology venues."""
    candidates: list[PaperMeta] = []
    for paper in _all_papers():
        if paper.venue.lower() == exclude_venue.lower():
            continue
        if _english_abstract(paper.abstract_meta):
            candidates.append(paper)
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n]


def list_french_distractors(
    n: int,
    seed: int,
    source_ids: set[str] | None = None,
) -> list[PaperMeta]:
    """Discover French TALN papers outside the selected JEP-TALN-RECITAL sources."""
    return list_french_distractor_candidates(seed, source_ids)[:n]


def list_french_distractor_candidates(
    seed: int,
    source_ids: set[str] | None = None,
) -> list[PaperMeta]:
    """Return all eligible French TALN metadata in one deterministic order."""
    source_ids = source_ids or set()
    candidates: list[PaperMeta] = []
    for paper in _all_papers():
        venue = paper.venue.lower()
        if "taln" not in venue:
            continue
        if paper.paper_id in source_ids or not _french_abstract(paper.abstract_meta):
            continue
        candidates.append(paper)
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates


def _french_abstract(text: str | None) -> bool:
    if not text:
        return False
    words = set(text.lower().split())
    return len(words & {"les", "des", "une", "dans", "pour", "avec", "cette", "nous"}) >= 2


def list_jeptalnrecital_papers(years: tuple[int, int]) -> list[PaperMeta]:
    """Discover JEP-TALN-RECITAL papers from the Anthology XML index."""
    start, end = years
    papers: list[PaperMeta] = []
    for url in _metadata_files():
        filename = url.rsplit("/", 1)[-1]
        try:
            year = int(filename[:4])
        except ValueError:
            continue
        if not start <= year <= end:
            continue
        try:
            root = ET.fromstring(_request(url))
        except Exception:
            continue
        for volume in root.findall("volume"):
            volume_id = volume.get("id")
            meta_year = volume.findtext("./meta/year")
            volume_year = int(meta_year) if meta_year and meta_year.isdigit() else year
            for paper in volume.findall("paper"):
                paper_number = paper.get("id")
                if not volume_id or not paper_number:
                    continue
                paper_id = f"{year}.jeptalnrecital-{volume_id}.{paper_number}"
                authors = [
                    " ".join(part for part in (author.findtext("first"), author.findtext("last")) if part)
                    for author in paper.findall("author")
                ]
                papers.append(PaperMeta(
                    paper_id=paper_id,
                    title=" ".join((paper.findtext("title") or "").split()),
                    year=volume_year,
                    venue="jeptalnrecital",
                    pdf_url=f"https://aclanthology.org/{paper_id}.pdf"
                    if paper.find("pdf") is not None else None,
                    abstract_meta=" ".join((paper.findtext("abstract") or "").split()) or None,
                    authors=authors,
                    language=paper.findtext("language"),
                ))
    return papers


def stratified_sample(papers: list[PaperMeta], n: int, seed: int = 13) -> list[PaperMeta]:
    """Select a deterministic, approximately even number per available year."""
    if n < 1:
        raise ValueError("n must be positive")
    by_year: dict[int, list[PaperMeta]] = {}
    rng = random.Random(seed)
    for paper in papers:
        by_year.setdefault(paper.year, []).append(paper)
    for year_papers in by_year.values():
        rng.shuffle(year_papers)
    selected: list[PaperMeta] = []
    years = sorted(by_year)
    while len(selected) < min(n, len(papers)):
        progressed = False
        for year in years:
            if by_year[year] and len(selected) < n:
                selected.append(by_year[year].pop())
                progressed = True
        if not progressed:
            break
    return selected


def download_pdf(
    meta: PaperMeta,
    dest_dir: Path,
    *,
    retries: int = 3,
    min_interval: float = 1.0,
    state: dict[str, float] | None = None,
    events_path: Path | None = None,
) -> Path | None:
    """Download one PDF with rate limiting, retry backoff, and event logging."""
    if not meta.pdf_url:
        _event(events_path, "DownloadFailed", {"paper_id": meta.paper_id, "reason": "missing_pdf_url"})
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    destination = dest_dir / f"{meta.paper_id}.pdf"
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    state = state if state is not None else {}
    for attempt in range(retries):
        wait = min_interval - (time.monotonic() - state.get("last_request", 0.0))
        if wait > 0:
            time.sleep(wait)
        state["last_request"] = time.monotonic()
        try:
            content = _request(meta.pdf_url)
            if not content.startswith(b"%PDF"):
                raise ValueError("response is not a PDF")
            destination.write_bytes(content)
            _event(events_path, "PdfDownloaded", {
                "paper_id": meta.paper_id, "path": str(destination), "bytes": len(content),
            })
            return destination
        except Exception as exc:
            if attempt == retries - 1:
                _event(events_path, "DownloadFailed", {
                    "paper_id": meta.paper_id, "reason": type(exc).__name__, "detail": str(exc),
                })
            else:
                time.sleep(2**attempt)
    return None


def _event(path: Path | None, event_type: str, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": event_type, **payload}, ensure_ascii=False) + "\n")


def write_metadata(papers: list[PaperMeta], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for paper in papers:
            handle.write(json.dumps(asdict(paper), ensure_ascii=False) + "\n")
