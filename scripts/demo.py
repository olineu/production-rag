"""
Quick demo script — ingests a few sample docs and runs a query.
Run after starting the server: uvicorn main:app --reload

Usage:
    python scripts/demo.py
"""
import httpx

BASE_URL = "http://localhost:8000"

SAMPLE_DOCS = [
    "Hybrid retrieval combines dense embeddings with sparse BM25 keyword search.",
    "Reciprocal Rank Fusion (RRF) merges ranked lists from multiple retrieval methods.",
    "Re-ranking uses a cross-encoder to re-score the top-k retrieved candidates.",
    "Chunking strategy matters: semantic chunking outperforms fixed-size on long documents.",
    "pgvector is a PostgreSQL extension that adds vector similarity search.",
]


def main():
    print("Ingesting sample documents...")
    r = httpx.post(f"{BASE_URL}/ingest", json={"documents": SAMPLE_DOCS})
    r.raise_for_status()
    print(f"  → {r.json()['message']}\n")

    query = "How does hybrid retrieval work?"
    print(f"Query: '{query}'")
    r = httpx.post(f"{BASE_URL}/query", json={"query": query, "top_k": 3})
    r.raise_for_status()
    result = r.json()

    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nStrategy: {result['retrieval_strategy']}")
    print(f"\nTop chunks used:")
    for i, chunk in enumerate(result["chunks"], 1):
        print(f"  [{i}] score={chunk['score']} | {chunk['text'][:80]}...")


if __name__ == "__main__":
    main()
