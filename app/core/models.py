from pydantic import BaseModel
from typing import Optional


# --- Request / Response models ---

class IngestRequest(BaseModel):
    """Ingest one or more text documents into the vector store."""
    documents: list[str]
    metadata: Optional[list[dict]] = None  # optional per-doc metadata


class IngestResponse(BaseModel):
    ingested: int
    message: str


class QueryRequest(BaseModel):
    """Query the RAG pipeline."""
    query: str
    top_k: int = 5                      # how many chunks to retrieve
    use_hybrid: bool = True             # dense + sparse (BM25) retrieval
    use_reranking: bool = True          # rerank after retrieval


class RetrievedChunk(BaseModel):
    text: str
    score: float
    metadata: dict = {}


class QueryResponse(BaseModel):
    answer: str
    chunks: list[RetrievedChunk]        # source chunks used for the answer
    retrieval_strategy: str             # e.g. "hybrid+reranked"
