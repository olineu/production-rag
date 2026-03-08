"""
Sparse retrieval using BM25.

BM25 (Best Match 25) is a classic keyword-based ranking algorithm.
It scores documents by how often query terms appear, penalising very
long documents and rewarding rare terms (like TF-IDF, but better).

Why keep this alongside embeddings?
- Embeddings are great for semantic similarity ("car" ≈ "automobile")
- BM25 is great for exact matches ("GPT-4o", product codes, names)
- Hybrid = both together. Almost always beats either alone.
"""
from rank_bm25 import BM25Okapi


class SparseIndex:
    def __init__(self):
        self._corpus: list[str] = []
        self._bm25: BM25Okapi | None = None

    def add_documents(self, documents: list[str]) -> None:
        """Add documents and rebuild the index."""
        self._corpus.extend(documents)
        tokenized = [doc.lower().split() for doc in self._corpus]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """
        Returns list of (index, score) sorted by score descending.
        Index refers to position in the corpus list.
        """
        if self._bm25 is None or not self._corpus:
            return []

        query_tokens = query.lower().split()
        scores = self._bm25.get_scores(query_tokens)

        # pair each score with its corpus index, sort by score
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return indexed[:top_k]

    @property
    def corpus(self) -> list[str]:
        return self._corpus


# Module-level singleton — one index per process
sparse_index = SparseIndex()
