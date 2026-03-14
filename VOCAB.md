# Technical Vocabulary — Production RAG

A living reference. Add to this as you learn. Study it like flashcards.

---

## Retrieval

| Term | Definition | Analogy |
|------|-----------|---------|
| **RAG** (Retrieval-Augmented Generation) | Give an LLM relevant context at query time, since its knowledge is frozen at training | Letting someone look something up before answering, instead of relying on memory |
| **Dense retrieval** | Find documents by embedding similarity — every dimension has a value | Matching by "vibe"/meaning |
| **Sparse retrieval** | Find documents by keyword overlap — most dimensions are zero | Matching by exact words |
| **Vector / semantic search** | Searching by closeness of embedding vectors in high-dimensional space | Finding the nearest neighbour in a map |
| **Top-k** | Return the k most similar results | "Give me the top 5 matches" |
| **Hybrid retrieval** | Combine dense + sparse results — almost always beats either alone | Two detectives comparing notes |
| **Inverted index** | Pre-built lookup: word → which documents contain it. Makes sparse search fast | A book's index at the back |
| **BM25** (Best Match 25) | Keyword ranking algorithm. Rewards rare terms, penalises long docs | Smarter keyword search |
| **RRF** (Reciprocal Rank Fusion) | Formula to merge two ranked lists into one, without needing to compare raw scores | See below |
| **Pre-filtering** | Apply WHERE clauses before the vector search runs, narrowing the search space | Searching only in the right filing cabinet before looking through folders |
| **Post-filtering** | Filter results after the vector search — risky, can discard the only relevant results | Picking candidates first, then checking criteria |
| **CRAG** (Corrective RAG) | Evaluate retrieved chunks for relevance; if they're poor, reformulate the query or fall back to web search | A surgeon who checks the X-ray is correct before operating |

---

## Tokenization

| Term | Definition |
|------|-----------|
| **Tokenization** | Splitting text into units (tokens) a model or algorithm can process |
| **Stemming** | Reducing words to their root: "running" → "run" |
| **Stop words** | Common words with no meaning ("the", "a", "is") — often removed before indexing |
| **EOS** (End of Sequence) | Special token in neural tokenizers signalling text end — not relevant to BM25 |
| **BPE** (Byte Pair Encoding) | Subword tokenization used by LLMs — splits rare words into pieces |
| **Context dilution** | Stuffing too much text into a prompt spreads the model's attention thin — worse answers |

---

## Embeddings

| Term | Definition |
|------|-----------|
| **Embedding** | A list of numbers representing the meaning of a piece of text |
| **Embedding model** | A neural network that converts text → embedding vector |
| **Cosine similarity** | How "close" two vectors are in direction — 1.0 = identical meaning, 0 = unrelated |
| **Chunking** | Splitting documents into smaller pieces before embedding, for relevance + grounding |
| **Fixed-size chunking** | Split by character/token count with overlap — simple, fast, ignores sentence boundaries |
| **Sentence chunking** | Split on natural sentence/paragraph boundaries — respects meaning, variable size |
| **Chunk overlap** | Repeating N characters at the start of each new chunk to avoid cutting sentences in half |

---

## Reranking

| Term | Definition |
|------|-----------|
| **Bi-encoder** | Encodes query and document independently into single vectors, compared with cosine similarity — fast but approximate |
| **Cross-encoder** | Takes query + document together as input, produces a single relevance score — slower but more accurate |
| **Reranking** | Retrieve many candidates cheaply (bi-encoder), then re-sort the top results accurately (cross-encoder) |
| **LLM-as-judge** | Use a cheap LLM call to evaluate output quality (faithfulness, relevance) instead of rule-based checks |

---

## Storage & Infrastructure

| Term | Definition |
|------|-----------|
| **pgvector** | Postgres extension that adds a `vector` column type and operators for cosine/dot-product similarity search |
| **HNSW** (Hierarchical Navigable Small World) | Approximate nearest neighbour index — multi-layer graph, finds close vectors in milliseconds without scanning all rows |
| **Content hash** | SHA-256 fingerprint of a document's text, stored in the DB to detect and skip duplicate ingestions |
| **Async event loop** | Single-threaded loop in Python async code — must never be blocked by CPU-heavy work |
| **run_in_threadpool** | FastAPI utility that offloads synchronous CPU work (e.g. model inference) to a worker thread, keeping the event loop free |
| **SSE** (Server-Sent Events) | HTTP streaming format: server sends `data: <json>\n\n` events to the client as they're generated |

---

## Eval

| Term | Definition |
|------|-----------|
| **Golden dataset** | Hand-curated question + expected answer pairs used to measure pipeline quality |
| **Faithfulness** | Does the answer contain only information supported by the retrieved context? (hallucination check) |
| **Retrieval hit rate** | Did the retrieved chunks contain the expected information for each question? |
| **Eval threshold** | Minimum acceptable score (e.g. 80% faithfulness) — fail the test suite if the pipeline regresses below it |

---

## RRF Formula

```
score(doc) = Σ  1 / (k + rank)
```

- `rank` = position in a result list (1 = best)
- `k` = constant (usually 60) — dampens the influence of very top results
- Sum across all retrieval methods (dense + sparse)
- Re-rank by this combined score

---

## Engineering patterns

| Term | Definition |
|------|-----------|
| **Batching** | Sending many items in one API call instead of one at a time — saves network round-trip overhead |
| **Parallel lists** | Two lists where index N in list A corresponds to index N in list B — fragile, a database replaces this |
| **Streaming** | Returning tokens to the client as they're generated, word by word — vs. waiting for the full response |
| **Lazy singleton** | Load a heavy resource (e.g. a model) once on first use, then reuse it — avoids loading at import time |
| **Pydantic model** | Python class that validates and serialises data automatically — used for API request/response contracts |
| **Dependency injection** | FastAPI pattern: `Depends(get_db)` wires a DB session into a route without the route creating it manually |

---

| **TF-IDF** (Term Frequency–Inverse Document Frequency) | Older keyword scoring: rewards words that appear often in a doc but rarely across all docs. BM25 improves on it by also penalising long documents | |
