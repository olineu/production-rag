"""
API routes:
  POST /ingest        — embed documents and persist them to Postgres (pgvector)
  POST /query         — retrieve + generate an answer (full JSON response)
  POST /query/stream  — same retrieval, but streams answer tokens as they arrive
  GET  /stats         — return index stats
"""
import json
import logging
from typing import AsyncGenerator, cast, Optional
from openai.types.chat import ChatCompletionMessageParam
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.models import IngestRequest, IngestResponse, QueryRequest, QueryResponse, RetrievedChunk
from app.retrieval.embedder import embed_texts, embed_query
from app.retrieval.sparse import sparse_index
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.db.session import get_db
from app.db.store import add_documents, similarity_search, count_documents, get_existing_hashes, compute_hash
from app.retrieval.reranker import rerank
from app.retrieval.chunker import chunk_document

logger = logging.getLogger(__name__)
router = APIRouter()
client = AsyncOpenAI(api_key=settings.openai_api_key)

# In-memory corpus: kept in sync with the DB.
# Used by BM25 and by RRF to look up text by corpus index.
_documents: list[str] = []
_id_to_idx: dict[int, int] = {}          # maps DB row id → position in _documents
_doc_metadata: dict[int, dict] = {}      # maps corpus index → {source, category}


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest, session: AsyncSession = Depends(get_db)):
    """
    Embed and persist documents with optional source and category metadata.
    Metadata is stored in Postgres and used for pre-filtering at query time.
    """
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")

    # 0. Chunk documents if a strategy is specified.
    # Each incoming document may produce multiple chunks — metadata is replicated
    # to all chunks that came from the same source document.
    if request.chunk_strategy:
        all_chunks: list[str] = []
        all_sources: list[Optional[str]] = []
        all_categories: list[Optional[str]] = []
        for i, doc in enumerate(request.documents):
            src = (request.sources or [])[i] if request.sources and i < len(request.sources) else None
            cat = (request.categories or [])[i] if request.categories and i < len(request.categories) else None
            chunks_for_doc = chunk_document(doc, request.chunk_strategy)
            all_chunks.extend(chunks_for_doc)
            all_sources.extend([src] * len(chunks_for_doc))
            all_categories.extend([cat] * len(chunks_for_doc))
        documents_to_ingest = all_chunks
        sources = all_sources
        categories = all_categories
    else:
        documents_to_ingest = request.documents
        n = len(request.documents)
        sources = (request.sources or []) + [None] * (n - len(request.sources or []))
        categories = (request.categories or []) + [None] * (n - len(request.categories or []))

    # Deduplication: compute a hash for each chunk, check which already exist in DB.
    # We filter BEFORE embedding — no wasted OpenAI API calls for duplicates.
    hashes = [compute_hash(d) for d in documents_to_ingest]
    existing = await get_existing_hashes(session, hashes)
    deduped = [
        (doc, src, cat)
        for doc, src, cat, h in zip(documents_to_ingest, sources, categories, hashes)
        if h not in existing
    ]
    if not deduped:
        return IngestResponse(ingested=0, message=f"All documents already in index. Total: {len(_documents)}")

    documents_to_ingest, sources, categories = zip(*deduped)
    documents_to_ingest = list(documents_to_ingest)
    sources = list(sources)
    categories = list(categories)
    n = len(documents_to_ingest)

    # 1. Embed all documents in one API call
    vectors = await embed_texts(documents_to_ingest)

    # 2. Persist to Postgres with metadata
    base_idx = len(_documents)
    new_docs = await add_documents(session, documents_to_ingest, vectors, sources, categories)

    # 3. Update in-memory corpus, id→index mapping, and metadata
    _documents.extend(documents_to_ingest)
    for i, doc in enumerate(new_docs):
        idx = base_idx + i
        _id_to_idx[doc.id] = idx
        _doc_metadata[idx] = {"source": doc.source, "category": doc.category}

    # 4. Update BM25 sparse index
    sparse_index.add_documents(documents_to_ingest)

    logger.info(f"Ingested {n} chunks. Total: {len(_documents)}")
    return IngestResponse(
        ingested=n,
        message=f"Total documents in index: {len(_documents)}",
    )


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, session: AsyncSession = Depends(get_db)):
    """
    Retrieve relevant chunks then generate an answer with an LLM.

    When metadata filters are provided (filter_source or filter_category),
    we use dense-only retrieval — BM25 cannot filter by metadata so hybrid
    mode is disabled. This is a deliberate trade-off: precision over recall.

    Flow (no filters):
    1. Embed query
    2. Dense: pgvector cosine similarity
    3. Sparse: BM25 keyword search
    4. Fuse: RRF
    5. Generate

    Flow (with filters):
    1. Embed query
    2. Dense only: pgvector cosine with WHERE clause (pre-filter)
    3. Generate
    """
    if not _documents:
        raise HTTPException(status_code=400, detail="Index is empty. Ingest documents first.")

    has_filter = bool(request.filter_source or request.filter_category)

    # 1. Embed query
    query_vector = await embed_query(request.query)

    # 2. Dense retrieval (always runs; filters applied here when set)
    db_results = await similarity_search(
        session,
        query_vector,
        top_k=request.top_k * 2,
        filter_source=request.filter_source,
        filter_category=request.filter_category,
    )
    dense_results = [
        (_id_to_idx[db_id], score)
        for db_id, score in db_results
        if db_id in _id_to_idx
    ]

    # 3. Hybrid merge (only when no filters — BM25 cannot filter by metadata)
    if request.use_hybrid and not has_filter:
        sparse_results = sparse_index.search(request.query, top_k=request.top_k * 2)
        chunks = reciprocal_rank_fusion(dense_results, sparse_results, _documents, top_k=request.top_k)
        strategy = "hybrid_rrf"
    else:
        chunks = [
            RetrievedChunk(
                text=_documents[idx],
                score=round(score, 4),
                source=_doc_metadata.get(idx, {}).get("source"),
                category=_doc_metadata.get(idx, {}).get("category"),
            )
            for idx, score in dense_results[:request.top_k]
        ]
        strategy = "dense_only" if has_filter else "dense_only_no_hybrid"

    # 4. Rerank (cross-encoder scores each (query, chunk) pair for true relevance)
    if request.use_reranking and len(chunks) > 1:
        chunks = await rerank(request.query, chunks, top_k=request.top_k)
        strategy += "+reranked"

    # 5. Generate answer
    context = "\n\n".join(f"[{i+1}] {chunk.text}" for i, chunk in enumerate(chunks))
    messages = cast(list[ChatCompletionMessageParam], [
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
    ])

    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0.2,
    )
    answer = response.choices[0].message.content or ""

    return QueryResponse(answer=answer, chunks=chunks, retrieval_strategy=strategy)


@router.post("/query/stream")
async def query_stream(request: QueryRequest, session: AsyncSession = Depends(get_db)):
    """
    Same retrieval pipeline as /query, but streams the answer token by token.

    Returns a text/event-stream response. Each event is a JSON object:
      {"token": "..."} for each answer chunk
      {"done": true, "chunks": [...], "retrieval_strategy": "..."} at the end

    Why split from /query? Streaming can't return a structured Pydantic response —
    the answer is built piece by piece. Keeping both lets callers choose.
    """
    if not _documents:
        raise HTTPException(status_code=400, detail="Index is empty. Ingest documents first.")

    has_filter = bool(request.filter_source or request.filter_category)

    # Steps 1-4 are identical to /query — retrieval happens fully before streaming starts.
    # The LLM is the slow part; retrieval is fast enough to do upfront.
    query_vector = await embed_query(request.query)

    db_results = await similarity_search(
        session,
        query_vector,
        top_k=request.top_k * 2,
        filter_source=request.filter_source,
        filter_category=request.filter_category,
    )
    dense_results = [
        (_id_to_idx[db_id], score)
        for db_id, score in db_results
        if db_id in _id_to_idx
    ]

    if request.use_hybrid and not has_filter:
        sparse_results = sparse_index.search(request.query, top_k=request.top_k * 2)
        chunks = reciprocal_rank_fusion(dense_results, sparse_results, _documents, top_k=request.top_k)
        strategy = "hybrid_rrf"
    else:
        chunks = [
            RetrievedChunk(
                text=_documents[idx],
                score=round(score, 4),
                source=_doc_metadata.get(idx, {}).get("source"),
                category=_doc_metadata.get(idx, {}).get("category"),
            )
            for idx, score in dense_results[:request.top_k]
        ]
        strategy = "dense_only" if has_filter else "dense_only_no_hybrid"

    if request.use_reranking and len(chunks) > 1:
        chunks = await rerank(request.query, chunks, top_k=request.top_k)
        strategy += "+reranked"

    context = "\n\n".join(f"[{i+1}] {chunk.text}" for i, chunk in enumerate(chunks))
    messages = cast(list[ChatCompletionMessageParam], [
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
    ])

    async def token_generator() -> AsyncGenerator[str, None]:
        """
        Async generator that yields Server-Sent Events (SSE).
        Each yield sends one chunk to the client immediately — no buffering.
        """
        stream = await client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=0.2,
            stream=True,              # key difference from /query
        )
        async for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                # SSE format: "data: <json>\n\n"
                yield f"data: {json.dumps({'token': token})}\n\n"

        # Final event: send the retrieved chunks and strategy so the client
        # has source attribution even in streaming mode.
        yield f"data: {json.dumps({'done': True, 'chunks': [c.model_dump() for c in chunks], 'retrieval_strategy': strategy})}\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


@router.get("/stats")
async def stats(session: AsyncSession = Depends(get_db)):
    total = await count_documents(session)
    return {
        "total_documents": total,
        "embedding_model": settings.embedding_model,
        "llm_model": settings.llm_model,
    }
