from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Protocol

from .queries import Query
from .retrievers import BM25, rrf


class Ranker(Protocol):
    def rank(self, query: str, candidates: set[str] | None = None) -> list[tuple[str, float]]: ...


def bootstrap_ci(values: list[float], resamples: int = 1000, seed: int = 13) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    state = seed
    means: list[float] = []
    for _ in range(resamples):
        sample = []
        for _ in values:
            state = (1103515245 * state + 12345) % (2**31)
            sample.append(values[state % len(values)])
        means.append(sum(sample) / len(sample))
    means.sort()
    return sum(values) / len(values), means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]


def exact_binomial_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if trials < 1:
        return float("nan"), float("nan")
    try:
        from scipy.stats import beta
    except ImportError as exc:
        raise RuntimeError("exact Clopper-Pearson intervals require scipy") from exc
    alpha = 1.0 - confidence
    low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    high = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return low, high


def load_accepted_ids(path: Path) -> dict[str, set[str]]:
    """Load optional human-accepted alternatives without requiring the file."""
    if not path.exists():
        return {}
    accepted: dict[str, set[str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "source_id" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a source_id column")
        candidate_column = next(
            (name for name in ("accepted_id", "candidate_id", "paper_id") if name in reader.fieldnames),
            None,
        )
        if candidate_column is None:
            raise ValueError(f"{path} must contain accepted_id, candidate_id, or paper_id")
        for row in reader:
            source_id, candidate_id = row["source_id"].strip(), row[candidate_column].strip()
            if source_id and candidate_id:
                accepted.setdefault(source_id, set()).add(candidate_id)
    return accepted


def rank_without_source(index: Ranker, query: Query) -> list[tuple[str, float]]:
    return index.rank(query.text, set(index.doc_ids) - {query.source_id})


def _rank_for_query(index: BM25, query: Query, include_source: bool) -> list[tuple[str, float]]:
    candidates = None if include_source else set(index.doc_ids) - {query.source_id}
    return index.rank(query.text, candidates)


def _rank(index: Ranker, query: Query, include_source: bool) -> list[tuple[str, float]]:
    if include_source:
        return index.rank(query.text)
    return rank_without_source(index, query)


def evaluate(
    queries: Iterable[Query],
    indexes: dict[str, BM25],
    top_k: int = 10,
    resamples: int = 1000,
) -> list[dict]:
    grouped: dict[tuple[str, str], list[Query]] = {}
    for query in queries:
        grouped.setdefault((query.variant, "all"), []).append(query)
    rows: list[dict] = []
    for view, index in indexes.items():
        for (variant, _), group in grouped.items():
            per_paper: list[float] = []
            reciprocal: list[float] = []
            absent_scores: list[tuple[float, int]] = []
            for query in group:
                ranked = _rank_for_query(index, query, True)
                ids = [doc_id for doc_id, _ in ranked]
                rank = ids.index(query.source_id) + 1 if query.source_id in ids else math.inf
                per_paper.append(float(rank <= top_k))
                reciprocal.append(1.0 / rank if math.isfinite(rank) else 0.0)
                absent = _rank_for_query(index, query, False)
                present_top = ranked[0][1] if ranked else 0.0
                absent_top = absent[0][1] if absent else 0.0
                absent_scores.extend(((present_top, 1), (absent_top, 0)))
            recall, recall_lo, recall_hi = bootstrap_ci(per_paper, resamples=resamples)
            mrr, mrr_lo, mrr_hi = bootstrap_ci(reciprocal, resamples=resamples)
            rows.append({
                "view": view, "variant": variant, "retriever": "bm25",
                "n": len(group), "recall@1": _recall(group, indexes[view], 1),
                "recall@5": _recall(group, indexes[view], 5), "recall@10": recall,
                "recall@10_ci_low": recall_lo, "recall@10_ci_high": recall_hi,
                "mrr": mrr, "mrr_ci_low": mrr_lo, "mrr_ci_high": mrr_hi,
                "false_gap@10": 1.0 - recall, "absent_auc": auc(absent_scores),
            })
    return rows


def evaluate_rankers(
    queries: Iterable[Query],
    rankers: dict[tuple[str, str], Ranker],
    top_k: int = 10,
    resamples: int = 1000,
    accepted_ids: dict[str, set[str]] | None = None,
) -> list[dict]:
    """Evaluate retriever/view pairs, bootstrapping over source papers."""
    groups: dict[str, list[Query]] = {}
    for query in queries:
        if query.accepted:
            groups.setdefault(query.variant, []).append(query)
    rows: list[dict] = []
    for (retriever, view), index in rankers.items():
        for variant, group in groups.items():
            per_paper: list[float] = []
            reciprocal: list[float] = []
            score_pairs: list[tuple[float, int]] = []
            accepted = accepted_ids or {}
            for query in group:
                ranked = _rank(index, query, True)
                ids = [doc_id for doc_id, _ in ranked]
                targets = {query.source_id} | accepted.get(query.source_id, set())
                target_positions = [ids.index(doc_id) + 1 for doc_id in targets if doc_id in ids]
                rank = min(target_positions, default=math.inf)
                per_paper.append(float(rank <= top_k))
                reciprocal.append(1.0 / rank if math.isfinite(rank) else 0.0)
                present_top = ranked[0][1] if ranked else 0.0
                absent = rank_without_source(index, query)
                absent_top = absent[0][1] if absent else 0.0
                score_pairs.extend(((present_top, 1), (absent_top, 0)))
            recall10, recall_lo, recall_hi = bootstrap_ci(per_paper, resamples, 13)
            mrr, mrr_lo, mrr_hi = bootstrap_ci(reciprocal, resamples, 13)
            exact_lo, exact_hi = exact_binomial_interval(int(sum(per_paper)), len(per_paper))
            rows.append({
                "retriever": retriever,
                "view": view,
                "variant": variant,
                "n": len(group),
                "accepted_n": len(accepted),
                "recall@1": _recall_ranker(group, index, 1),
                "recall@5": _recall_ranker(group, index, 5),
                "recall@10": recall10,
                "recall@10_ci_low": recall_lo,
                "recall@10_ci_high": recall_hi,
                "recall@10_exact_low": exact_lo,
                "recall@10_exact_high": exact_hi,
                "mrr": mrr,
                "mrr_ci_low": mrr_lo,
                "mrr_ci_high": mrr_hi,
                "false_gap@10": 1.0 - recall10,
                "top1_score_auc_present_absent": auc(score_pairs),
            })
    return rows


def _recall_ranker(queries: list[Query], index: Ranker, cutoff: int) -> float:
    return sum(
        query.source_id in [doc_id for doc_id, _ in _rank(index, query, True)[:cutoff]]
        for query in queries
    ) / max(1, len(queries))


def _recall(queries: list[Query], index: BM25, cutoff: int) -> float:
    return sum(
        query.source_id in [doc_id for doc_id, _ in index.rank(query.text)[:cutoff]]
        for query in queries
    ) / max(1, len(queries))


def auc(scored: list[tuple[float, int]]) -> float:
    positives = [score for score, label in scored if label]
    negatives = [score for score, label in scored if not label]
    if not positives or not negatives:
        return float("nan")
    wins = sum(1.0 if positive > negative else 0.5 if positive == negative else 0.0
               for positive in positives for negative in negatives)
    return wins / (len(positives) * len(negatives))


def write_metrics(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
