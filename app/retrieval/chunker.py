"""
Chunking strategies for splitting raw documents before ingestion.

Two strategies:
  fixed_size  — split by character count with overlap
  sentence    — split on natural sentence/paragraph boundaries

Why chunking matters:
  One document = one vector. If a document is too large, the embedding
  averages over many topics and becomes a poor match for any specific query.
  Splitting into smaller, focused chunks gives each chunk a clear meaning
  and improves retrieval precision.
"""
from __future__ import annotations

import re


def fixed_size_chunk(text: str, size: int = 512, overlap: int = 50) -> list[str]:
    """
    Split text into chunks of `size` characters, with `overlap` characters
    repeated at the start of each new chunk.

    The overlap prevents a sentence from being cut cleanly between two chunks —
    each chunk shares a short tail with the previous one, preserving context.

    Example with size=20, overlap=5:
      text:    "The quick brown fox jumps over the lazy dog"
      chunk 1: "The quick brown fox "
      chunk 2: "ox jumps over the la"   ← starts 5 chars back
      chunk 3: "e lazy dog"
    """
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end].strip())
        start += size - overlap   # step forward by (size - overlap), not full size

    # Drop empty chunks that can appear at the end
    return [c for c in chunks if c]


def sentence_chunk(text: str, max_size: int = 1024) -> list[str]:
    """
    Split text on natural sentence/paragraph boundaries.

    Splits first on double newlines (paragraphs), then on sentence-ending
    punctuation. Groups short sentences together until max_size is reached,
    then starts a new chunk.

    This produces chunks that are complete thoughts — never mid-sentence —
    at the cost of variable chunk sizes.
    """
    if not text:
        return []

    # Split on paragraph breaks first, then sentence endings
    # re.split keeps the delimiter with the preceding text using a lookbehind
    raw_sentences = re.split(r'(?<=[\.\!\?])\s+|\n\n+', text)
    raw_sentences = [s.strip() for s in raw_sentences if s.strip()]

    chunks = []
    current = ""

    for sentence in raw_sentences:
        if not current:
            current = sentence
        elif len(current) + len(sentence) + 1 <= max_size:
            current += " " + sentence
        else:
            chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks


def chunk_document(text: str, strategy: str = "sentence") -> list[str]:
    """
    Dispatch to the right chunking strategy by name.

    Called by the /ingest route for each incoming document.
    Returns a list of chunk strings ready to be embedded.
    """
    if strategy == "fixed":
        return fixed_size_chunk(text)
    elif strategy == "sentence":
        return sentence_chunk(text)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}. Use 'fixed' or 'sentence'.")
