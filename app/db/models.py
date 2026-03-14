"""
SQLAlchemy ORM model for stored documents.

Each row holds the raw text, its embedding vector, and optional metadata
(source, category) used for pre-filtering before vector search.
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional
from sqlalchemy import Integer, Text, String, DateTime, func, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    embedding: Mapped[list] = mapped_column(Vector(1536), nullable=False)
    source: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    category: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
