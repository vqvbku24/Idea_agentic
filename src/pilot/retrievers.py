from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", text.lower())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.findall(r"[a-z0-9]+", normalized)


def rrf(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class BM25:
    def __init__(self, documents: dict[str, str], k1: float = 1.5, b: float = 0.75):
        if not documents:
            raise ValueError("cannot index an empty view")
        self.doc_ids = list(documents)
        self.k1, self.b = k1, b
        self.tokens = {doc_id: tokenize(text) for doc_id, text in documents.items()}
        self.lengths = {doc_id: len(tokens) for doc_id, tokens in self.tokens.items()}
        self.average_length = sum(self.lengths.values()) / len(self.lengths)
        self.term_frequency = {doc_id: Counter(tokens) for doc_id, tokens in self.tokens.items()}
        self.doc_frequency: Counter[str] = Counter()
        for tokens in self.tokens.values():
            self.doc_frequency.update(set(tokens))

    def rank(self, query: str, candidates: set[str] | None = None) -> list[tuple[str, float]]:
        query_terms = tokenize(query)
        candidates = set(self.doc_ids) if candidates is None else candidates
        scores: list[tuple[str, float]] = []
        total = len(self.doc_ids)
        for doc_id in candidates:
            counts = self.term_frequency[doc_id]
            length = self.lengths[doc_id]
            score = 0.0
            for term in query_terms:
                df = self.doc_frequency.get(term, 0)
                if not df:
                    continue
                idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
                tf = counts.get(term, 0)
                score += idf * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * length / self.average_length)
                )
            scores.append((doc_id, score))
        return sorted(scores, key=lambda item: (-item[1], item[0]))

