from pydantic import BaseModel
from typing import Optional


# --- Request / Response models ---

class IngestRequest(BaseModel):
    """Ingest one or more text documents into the vector store."""
    documents: list[str]
    sources: Optional[list[str]] = None       # one source URL/name per document
    categories: Optional[list[str]] = None    # one category per document
    chunk_strategy: Optional[str] = None      # "fixed", "sentence", or None (no chunking)


class IngestResponse(BaseModel):
    ingested: int
    message: str


class QueryRequest(BaseModel):
    """Query the RAG pipeline."""
    query: str
    top_k: int = 5
    use_hybrid: bool = True             # dense + sparse (BM25). Disabled when filters are set.
    use_reranking: bool = True          # cross-encoder reranking after retrieval
    filter_source: Optional[str] = None     # only search docs from this source
    filter_category: Optional[str] = None   # only search docs in this category


class RetrievedChunk(BaseModel):
    text: str
    score: float
    source: Optional[str] = None
    category: Optional[str] = None
    metadata: dict = {}


class QueryResponse(BaseModel):
    answer: str
    chunks: list[RetrievedChunk]
    retrieval_strategy: str
