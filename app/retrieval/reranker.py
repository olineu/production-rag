"""
Cross-encoder reranking.

Why cross-encoders beat bi-encoders for final ranking:
- Bi-encoder: encodes query and document separately, then compares vectors.
  Fast but lossy — the two sides never interact during encoding.
- Cross-encoder: takes (query, document) as a single input, outputs one score.
  Slow but accurate — the model can directly compare every word in both.

Strategy: use the fast bi-encoder to retrieve 20-50 candidates, then let
the cross-encoder re-score just those candidates. You get accuracy without
running the slow model on your entire corpus.

The cross-encoder runs synchronous CPU inference, so we run it in a thread
pool (run_in_threadpool) to avoid blocking the async event loop.
"""
from __future__ import annotations
from typing import Optional
from starlette.concurrency import run_in_threadpool
from sentence_transformers import CrossEncoder

from app.core.models import RetrievedChunk

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Lazy singleton — model loads on first call, not at import time.
# This avoids slowing down startup if reranking is never used.
_model: Optional[CrossEncoder] = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME)
    return _model


def _rerank_sync(query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """
    Score each (query, chunk) pair and return the top_k by cross-encoder score.
    Runs synchronously — call via rerank() to stay async-safe.
    """
    model = _get_model()
    pairs = [(query, chunk.text) for chunk in chunks]
    scores = model.predict(pairs)

    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [
        RetrievedChunk(
            text=chunk.text,
            score=round(float(score), 4),
            source=chunk.source,
            category=chunk.category,
        )
        for chunk, score in ranked[:top_k]
    ]


async def rerank(query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """Async wrapper — runs the cross-encoder in a thread pool."""
    return await run_in_threadpool(_rerank_sync, query, chunks, top_k)
