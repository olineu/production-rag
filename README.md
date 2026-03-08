# production-rag

> State-of-the-art RAG pipeline. Hybrid retrieval, RRF fusion, and eval-ready.
> Built to show what RAG actually needs in production — not what tutorials show.

---

## Why most RAG demos fail in production

Most tutorials do this:

```
embed query → cosine similarity → top-5 chunks → LLM
```

That breaks in practice because:
- **Dense-only retrieval misses exact keyword matches** (product codes, names, dates)
- **No reranking** — the 5th chunk might be better than the 1st after cross-attention
- **No evals** — you have no idea if it's actually working
- **No observability** — you can't debug what was retrieved and why

This project does it properly.

---

## Architecture

```
POST /ingest
  └─ embed_texts (OpenAI)
  └─ store vectors (in-memory → pgvector)
  └─ update BM25 sparse index

POST /query
  ├─ embed query
  ├─ dense retrieval  ──┐
  ├─ BM25 retrieval   ──┤── RRF fusion → top-k chunks
  └─ generate answer (LLM) using retrieved context
```

**Retrieval strategy: Reciprocal Rank Fusion (RRF)**
- Dense retrieval: semantic similarity via embeddings (good for meaning)
- Sparse retrieval: BM25 keyword matching (good for exact terms)
- RRF merges both ranked lists by position, not raw score — more robust than score normalisation

---

## Stack

| Component | Choice | Why |
|---|---|---|
| Framework | FastAPI + asyncio | Async-native, needed for concurrent LLM calls |
| Embeddings | OpenAI `text-embedding-3-small` | Cost-efficient, strong performance |
| Dense store | In-memory → pgvector | Start simple, migrate to Postgres without changing the interface |
| Sparse | BM25 (`rank-bm25`) | Keyword matching for exact terms dense embeddings miss |
| Fusion | Reciprocal Rank Fusion | Rank-based, no score normalisation needed |
| LLM | OpenAI `gpt-4o-mini` (configurable) | Cheap for iteration; swap to Claude or GPT-4o in config |
| Validation | Pydantic v2 | Every request/response has a typed contract |

---

## Getting started

```bash
# 1. Clone and install
git clone https://github.com/olineu/production-rag
cd production-rag
pip install -r requirements.txt

# 2. Configure
cp .env.template .env
# Add your OPENAI_API_KEY

# 3. Run
uvicorn main:app --reload

# 4. Try it
python scripts/demo.py
```

API docs available at `http://localhost:8000/docs`

---

## Run tests

```bash
pytest
```

---

## What's next (in progress)

- [ ] pgvector persistence (replace in-memory store)
- [ ] Cross-encoder reranking (Cohere or local `bge-reranker`)
- [ ] Streaming responses (SSE)
- [ ] Eval suite: faithfulness, relevance, groundedness (RAGAS)
- [ ] Chunking strategies: fixed vs semantic vs late chunking
- [ ] Docker + docker-compose
