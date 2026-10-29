from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Callable, Iterable

from .queries import Query
from .retrievers import BM25, rrf


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


def _rank_for_query(index: BM25, query: Query, include_source: bool) -> list[tuple[str, float]]:
    candidates = None if include_source else set(index.doc_ids) - {query.source_id}
    return index.rank(query.text, candidates)


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

