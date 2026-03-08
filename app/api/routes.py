"""
API routes — two endpoints:
  POST /ingest  — add documents to the index
  POST /query   — retrieve + generate an answer
"""
import logging
from fastapi import APIRouter, HTTPException
from openai import AsyncOpenAI

from app.core.config import settings
from app.core.models import IngestRequest, IngestResponse, QueryRequest, QueryResponse, RetrievedChunk
from app.retrieval.embedder import embed_texts, embed_query
from app.retrieval.sparse import sparse_index
from app.retrieval.hybrid import reciprocal_rank_fusion

logger = logging.getLogger(__name__)
router = APIRouter()
client = AsyncOpenAI(api_key=settings.openai_api_key)

# In-memory store: parallel lists of (text, vector)
# Phase 1: in-memory. Phase 2: swap in pgvector.
_documents: list[str] = []
_vectors: list[list[float]] = []


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """
    Embed and store documents.
    Also adds them to the BM25 sparse index.
    """
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")

    # Embed all documents in one API call (batched)
    vectors = await embed_texts(request.documents)

    _documents.extend(request.documents)
    _vectors.extend(vectors)
    sparse_index.add_documents(request.documents)

    logger.info(f"Ingested {len(request.documents)} documents. Total: {len(_documents)}")
    return IngestResponse(
        ingested=len(request.documents),
        message=f"Total documents in index: {len(_documents)}",
    )


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Retrieve relevant chunks then generate an answer with an LLM.

    Flow:
    1. Embed the query
    2. Dense retrieval: cosine similarity against stored vectors
    3. Sparse retrieval: BM25 keyword search
    4. Hybrid merge: RRF to combine both ranked lists
    5. Generate: send top chunks + query to LLM, stream the response
    """
    if not _documents:
        raise HTTPException(status_code=400, detail="Index is empty. Ingest documents first.")

    # 1. Embed query
    query_vector = await embed_query(request.query)

    # 2. Dense retrieval: dot product (vectors are L2-normalised so this = cosine sim)
    scores = [
        sum(q * d for q, d in zip(query_vector, doc_vec))
        for doc_vec in _vectors
    ]
    dense_results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:request.top_k * 2]

    # 3. Sparse retrieval
    sparse_results = sparse_index.search(request.query, top_k=request.top_k * 2)

    # 4. Hybrid merge
    if request.use_hybrid:
        chunks = reciprocal_rank_fusion(dense_results, sparse_results, _documents, top_k=request.top_k)
        strategy = "hybrid_rrf"
    else:
        # Dense only
        chunks = [
            RetrievedChunk(text=_documents[idx], score=round(score, 4))
            for idx, score in dense_results[:request.top_k]
        ]
        strategy = "dense_only"

    # 5. Generate answer
    context = "\n\n".join(f"[{i+1}] {chunk.text}" for i, chunk in enumerate(chunks))
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the question using only the "
                "provided context. If the context doesn't contain the answer, say so."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {request.query}",
        },
    ]

    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0.2,
    )
    answer = response.choices[0].message.content

    return QueryResponse(answer=answer, chunks=chunks, retrieval_strategy=strategy)
