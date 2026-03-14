# RAG Pipeline Architecture

```mermaid
flowchart TD
    subgraph STARTUP["On Server Start"]
        DB2[(pgvector)] -->|load all docs| BM2[Rebuild BM25 in-memory]
        DB2 -->|build| IDX[_id_to_idx map]
    end

    subgraph INGEST["Ingest Flow — POST /ingest"]
        A[documents] --> B[embed_texts via OpenAI]
        B --> C[(pgvector Postgres)]
        B --> D[BM25 Index in-memory]
        C --> E[_id_to_idx map]
    end

    subgraph QUERY["Query Flow — POST /query"]
        Q[user question] --> QE[embed_query via OpenAI]

        QE --> DENSE[Dense Retrieval via pgvector cosine]
        Q --> SPARSE[Sparse Retrieval via BM25]

        C --> DENSE
        D --> SPARSE

        DENSE --> RRF[RRF Fusion: score = 1 / rank + k]
        SPARSE --> RRF

        RRF --> TOP[Top-K Chunks]
        TOP --> LLM[LLM gpt-4o-mini]
        Q --> LLM
        LLM --> ANS[Answer + Sources]
    end
```

## Key design decisions

| Decision | Why |
|----------|-----|
| pgvector for dense storage | Vectors persist across restarts; native cosine similarity via SQL |
| BM25 in-memory | No standard way to persist BM25 state; rebuilt from DB on startup |
| RRF instead of score averaging | BM25 and cosine scores are on different scales; RRF uses rank position only |
| `top_k * 2` in each retrieval step | Cast a wide net before fusing — gives RRF more signal to work with |
| Startup lifespan | Bridges persistent DB and ephemeral in-memory structures after every restart |
