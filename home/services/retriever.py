"""Hybrid BM25 + Dense Vector retriever with Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np
from rank_bm25 import BM25Okapi


_MODEL_LOCK = threading.Lock()
_CACHED_MODEL = None
_CACHED_MODEL_NAME = None

EmbedFn = Callable[[Sequence[str]], np.ndarray]


def get_default_embedder() -> EmbedFn:
    """Lazy-load multilingual SentenceTransformer once per process."""
    global _CACHED_MODEL, _CACHED_MODEL_NAME
    model_name = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

    with _MODEL_LOCK:
        if _CACHED_MODEL is None or _CACHED_MODEL_NAME != model_name:
            from sentence_transformers import SentenceTransformer

            _CACHED_MODEL = SentenceTransformer(model_name)
            _CACHED_MODEL_NAME = model_name

    def _encode(texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        arr = _CACHED_MODEL.encode(list(texts), normalize_embeddings=True)
        return np.asarray(arr, dtype=np.float32)

    return _encode


def encode_vector_bytes(vec: np.ndarray) -> bytes:
    """Serialize a 1D float32 numpy vector safely without pickle."""
    return np.asarray(vec, dtype=np.float32).tobytes()


def decode_vector_bytes(raw: bytes) -> np.ndarray:
    """Deserialize a float32 vector from bytes."""
    return np.frombuffer(raw, dtype=np.float32)


def _tokenize_vietnamese(text: str) -> List[str]:
    """Lowercase alphanumeric/Vietnamese word tokenization for BM25."""
    return re.findall(r"\w+", text.lower())


@dataclass(frozen=True)
class ChunkCandidate:
    """Stored chunk candidate passed into hybrid ranking."""

    chunk_id: int
    document_id: int
    file_name: str
    page_number: int
    text: str
    embedding: np.ndarray


@dataclass(frozen=True)
class RetrievedChunk:
    """Ranked retrieval result with fusion score."""

    chunk_id: int
    document_id: int
    file_name: str
    page_number: int
    text: str
    score: float


def hybrid_retrieve(
    query: str,
    candidates: Sequence[ChunkCandidate],
    top_k: int = 5,
    embed_fn: EmbedFn | None = None,
    rrf_k: int = 60,
    dense_weight: float = 0.65,
    bm25_weight: float = 0.35,
) -> List[RetrievedChunk]:
    """Rank candidates using Reciprocal Rank Fusion over Dense cosine + BM25."""
    if not query.strip() or not candidates:
        return []

    k = max(1, min(top_k, len(candidates)))
    embedder = embed_fn or get_default_embedder()

    # 1. Dense Cosine Similarity (handles mismatched/empty vectors gracefully)
    query_vec = embedder([query])[0]
    q_norm = np.linalg.norm(query_vec)
    if q_norm > 0:
        query_vec = query_vec / q_norm

    dense_scores: List[float] = []
    for cand in candidates:
        vec = cand.embedding
        if vec.shape != query_vec.shape or vec.size == 0:
            dense_scores.append(0.0)
            continue
        v_norm = np.linalg.norm(vec)
        score = float(np.dot(query_vec, vec / v_norm)) if v_norm > 0 else 0.0
        dense_scores.append(score)

    dense_order = np.argsort(dense_scores)[::-1]
    dense_ranks = {int(idx): rank + 1 for rank, idx in enumerate(dense_order)}

    # 2. Sparse BM25 Lexical Ranking
    tokenized_corpus = [_tokenize_vietnamese(c.text) for c in candidates]
    query_tokens = _tokenize_vietnamese(query)
    if query_tokens and any(tokenized_corpus):
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(query_tokens)
        bm25_order = np.argsort(bm25_scores)[::-1]
        bm25_ranks = {int(idx): rank + 1 for rank, idx in enumerate(bm25_order)}
    else:
        bm25_ranks = {idx: idx + 1 for idx in range(len(candidates))}

    # 3. Weighted Reciprocal Rank Fusion
    fused: List[tuple[int, float]] = []
    for idx in range(len(candidates)):
        rrf_score = (
            dense_weight / (rrf_k + dense_ranks[idx])
            + bm25_weight / (rrf_k + bm25_ranks[idx])
        )
        fused.append((idx, rrf_score))

    fused.sort(key=lambda item: item[1], reverse=True)
    return [
        RetrievedChunk(
            chunk_id=candidates[idx].chunk_id,
            document_id=candidates[idx].document_id,
            file_name=candidates[idx].file_name,
            page_number=candidates[idx].page_number,
            text=candidates[idx].text,
            score=round(score, 6),
        )
        for idx, score in fused[:k]
    ]
