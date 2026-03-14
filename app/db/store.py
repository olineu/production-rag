"""
Database operations for the document store.

This module is the only place that talks to Postgres directly.
Routes call these functions — they never write SQL themselves.
"""
from __future__ import annotations
import hashlib
from typing import Optional
from sqlalchemy import select, text, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Base, Document
from app.db.session import engine


async def init_db() -> None:
    """
    Create the pgvector extension, all tables, any missing columns,
    and the HNSW index for fast approximate nearest-neighbour search.

    Safe to call on every startup — all statements use IF NOT EXISTS.
    """
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

        # Safe migrations: add metadata columns to existing tables
        await conn.execute(text(
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS source VARCHAR(500)"
        ))
        await conn.execute(text(
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS category VARCHAR(200)"
        ))
        await conn.execute(text(
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW()"
        ))

        # Safe migration: add content_hash column for deduplication
        await conn.execute(text(
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64)"
        ))
        # Unique index on content_hash — prevents duplicate inserts at the DB level
        await conn.execute(text(
            "CREATE UNIQUE INDEX IF NOT EXISTS documents_content_hash_unique "
            "ON documents (content_hash) WHERE content_hash IS NOT NULL"
        ))

        # HNSW index: builds a graph over vectors for fast approximate search.
        # m=16: each node connects to 16 neighbours (higher = more accurate, more memory).
        # ef_construction=64: how deep to search when building the graph (higher = better quality).
        # Without this, every query scans all rows. With it, queries stay fast at millions of docs.
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS documents_embedding_hnsw "
            "ON documents USING hnsw (embedding vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 64)"
        ))


def compute_hash(text: str) -> str:
    """SHA-256 hash of the text, truncated to 64 hex chars. Used for deduplication."""
    return hashlib.sha256(text.encode()).hexdigest()[:64]


async def get_existing_hashes(session: AsyncSession, hashes: list[str]) -> set[str]:
    """Return the subset of hashes that already exist in the DB."""
    result = await session.execute(
        select(Document.content_hash).where(Document.content_hash.in_(hashes))
    )
    return {row[0] for row in result}


async def add_documents(
    session: AsyncSession,
    texts: list[str],
    vectors: list[list[float]],
    sources: list[Optional[str]],
    categories: list[Optional[str]],
) -> list[Document]:
    """Insert documents and return them with their new DB ids populated."""
    docs = [
        Document(text=t, content_hash=compute_hash(t), embedding=v, source=s, category=c)
        for t, v, s, c in zip(texts, vectors, sources, categories)
    ]
    session.add_all(docs)
    await session.flush()
    await session.commit()
    return docs


async def load_all(session: AsyncSession) -> list[Document]:
    """Load all documents ordered by insertion (id ascending)."""
    result = await session.execute(select(Document).order_by(Document.id))
    return list(result.scalars().all())


async def similarity_search(
    session: AsyncSession,
    query_vector: list[float],
    top_k: int,
    filter_source: Optional[str] = None,
    filter_category: Optional[str] = None,
) -> list[tuple[int, float]]:
    """
    Find the top_k most similar documents using pgvector cosine distance.

    Filters are applied as WHERE clauses BEFORE the vector search runs —
    this is pre-filtering. It narrows the search space first, then ranks
    only the matching rows by similarity.

    The <=> operator is cosine distance (0=identical). We convert to
    similarity: score = 1 - distance.
    Returns list of (db_id, similarity_score).
    """
    where_clauses = []
    if filter_source:
        where_clauses.append("source = :source")
    if filter_category:
        where_clauses.append("category = :category")

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    result = await session.execute(
        text(
            f"SELECT id, 1 - (embedding <=> CAST(:qv AS vector)) AS score "
            f"FROM documents "
            f"{where_sql} "
            f"ORDER BY embedding <=> CAST(:qv AS vector) "
            f"LIMIT :k"
        ),
        {
            "qv": str(query_vector),
            "k": top_k,
            "source": filter_source,
            "category": filter_category,
        },
    )
    return [(row.id, float(row.score)) for row in result]


async def count_documents(session: AsyncSession) -> int:
    """Return the total number of stored documents."""
    result = await session.execute(select(func.count()).select_from(Document))
    return result.scalar() or 0
