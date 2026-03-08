"""
Hybrid retrieval: combines dense (embedding) + sparse (BM25) scores.

Strategy: Reciprocal Rank Fusion (RRF)
- Each retrieval method returns a ranked list
- RRF merges the ranks: score = sum(1 / (rank + k)) across methods
- k=60 is the standard constant (dampens the effect of very high ranks)
- Result: a single merged ranking that rewards docs that rank well in both

Why RRF instead of score normalisation?
- BM25 scores and cosine similarity scores are on completely different scales
- Normalising them requires knowing the distribution of scores upfront
- RRF only cares about rank position, not raw score — more robust
"""
from app.core.models import RetrievedChunk


def reciprocal_rank_fusion(
    dense_results: list[tuple[int, float]],  # (corpus_index, cosine_score)
    sparse_results: list[tuple[int, float]],  # (corpus_index, bm25_score)
    corpus: list[str],
    k: int = 60,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """
    Merge dense and sparse results using RRF.
    Returns top_k chunks sorted by fused score.
    """
    rrf_scores: dict[int, float] = {}

    for rank, (idx, _score) in enumerate(dense_results):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rank + k)

    for rank, (idx, _score) in enumerate(sparse_results):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rank + k)

    # Sort by fused score descending
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    chunks = []
    for idx, score in sorted_results[:top_k]:
        if idx < len(corpus):
            chunks.append(RetrievedChunk(
                text=corpus[idx],
                score=round(score, 4),
                metadata={"retrieval": "hybrid_rrf", "corpus_idx": idx},
            ))
    return chunks
