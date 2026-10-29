from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


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


class DenseBgeM3:
    """Transformers-backed BGE-M3 index with persistent document embeddings."""

    model_id = "BAAI/bge-m3"

    def __init__(
        self,
        views: dict[str, dict[str, str]],
        cache_dir: Path = Path("data/cache/embeddings"),
        max_length: int = 1024,
        batch_size: int = 8,
    ):
        if max_length < 1024:
            raise ValueError("dense_bge_m3 requires max_length >= 1024")
        try:
            import numpy as np
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "dense_bge_m3 requires transformers, torch, and numpy"
            ) from exc
        self._np, self._torch = np, torch
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id)
        self.model.eval()
        self.embeddings: dict[str, dict[str, Any]] = {}
        self.doc_ids: dict[str, list[str]] = {}
        for view, documents in views.items():
            self.doc_ids[view] = list(documents)
            self.embeddings[view] = self._load_or_encode(view, documents, cache_dir)

    def _cache_path(self, view: str, documents: dict[str, str], cache_dir: Path) -> Path:
        digest = hashlib.sha256(
            json.dumps(
                {"model": self.model_id, "max_length": self.max_length, "documents": documents},
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{view}-{digest}.npz"

    def _encode(self, texts: list[str]):
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(
                texts[start:start + self.batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            with self._torch.no_grad():
                output = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu().numpy())
        return self._np.vstack(vectors) if vectors else self._np.empty((0, 1))

    def _load_or_encode(self, view: str, documents: dict[str, str], cache_dir: Path):
        path = self._cache_path(view, documents, cache_dir)
        if path.exists():
            return self._np.load(path)["embeddings"]
        embeddings = self._encode(list(documents.values()))
        self._np.savez_compressed(path, embeddings=embeddings)
        return embeddings

    def rank(self, query: str, view: str, candidates: set[str] | None = None) -> list[tuple[str, float]]:
        query_embedding = self._encode([query])[0]
        scores = self.embeddings[view] @ query_embedding
        allowed = candidates if candidates is not None else set(self.doc_ids[view])
        rows = [
            (doc_id, float(scores[index]))
            for index, doc_id in enumerate(self.doc_ids[view])
            if doc_id in allowed
        ]
        return sorted(rows, key=lambda item: (-item[1], item[0]))

    def for_view(self, view: str) -> "ViewRanker":
        if view not in self.embeddings:
            raise KeyError(f"unknown dense view: {view}")
        return ViewRanker(self, view)


class DocTranslateBM25:
    """BM25 over French native views translated in <=400-token chunks."""

    model_id = "Helsinki-NLP/opus-mt-fr-en"

    def __init__(
        self,
        native_views: dict[str, str],
        cache_dir: Path = Path("data/cache/translations"),
        max_tokens: int = 400,
    ):
        if max_tokens > 400:
            raise ValueError("translation chunks must be <= 400 tokens")
        try:
            from transformers import AutoModelForSeq2SeqLM, MarianTokenizer
        except ImportError as exc:
            raise RuntimeError("doc_translate_bm25 requires transformers") from exc
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
        self.model.eval()
        self.max_tokens = max_tokens
        translated = {
            doc_id: self._translate_cached(doc_id, text, cache_dir)
            for doc_id, text in native_views.items()
        }
        self.index = BM25(translated)
        self.doc_ids = self.index.doc_ids

    def _translate_cached(self, doc_id: str, text: str, cache_dir: Path) -> str:
        digest = hashlib.sha256(
            json.dumps(
                {"model": self.model_id, "max_tokens": self.max_tokens, "doc_id": doc_id, "text": text},
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{digest}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["text"]
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunk_ids = [
            token_ids[start:start + self.max_tokens]
            for start in range(0, len(token_ids), self.max_tokens)
        ]
        chunks = []
        for start in range(0, len(chunk_ids), 8):
            batch = chunk_ids[start:start + 8]
            inputs = self.tokenizer.pad(
                [{"input_ids": ids, "attention_mask": [1] * len(ids)} for ids in batch],
                return_tensors="pt",
            )
            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
            chunks.extend(
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in output_ids
            )
        translated = " ".join(chunks).strip()
        path.write_text(json.dumps({"text": translated}, ensure_ascii=False), encoding="utf-8")
        return translated

    def _torch_inputs(self, token_ids: list[int]) -> dict[str, Any]:
        import torch
        return {"input_ids": torch.tensor([token_ids]), "attention_mask": torch.ones((1, len(token_ids)), dtype=torch.long)}

    def rank(self, query: str, view: str = "native", candidates: set[str] | None = None) -> list[tuple[str, float]]:
        return self.index.rank(query, candidates)


class ViewRanker:
    def __init__(self, ranker: Any, view: str):
        self.ranker = ranker
        self.view = view
        self.doc_ids = ranker.doc_ids[view] if isinstance(ranker.doc_ids, dict) else ranker.doc_ids

    def rank(self, query: str, candidates: set[str] | None = None) -> list[tuple[str, float]]:
        return self.ranker.rank(query, self.view, candidates)


class HybridRRF:
    def __init__(self, bm25: RankerLike, dense: RankerLike, k: int = 60):
        if k != 60:
            raise ValueError("pilot hybrid_rrf requires k=60")
        self.bm25, self.dense, self.k = bm25, dense, k
        self.doc_ids = list(dict.fromkeys(bm25.doc_ids + dense.doc_ids))

    def rank(self, query: str, candidates: set[str] | None = None) -> list[tuple[str, float]]:
        bm = [doc_id for doc_id, _ in self.bm25.rank(query, candidates)]
        dense = [doc_id for doc_id, _ in self.dense.rank(query, candidates)]
        return rrf([bm, dense], self.k)


class RankerLike:
    doc_ids: list[str]
