# Production RAG — Personal Course

> Your self-paced curriculum for this project.
> Goal: understand every file deeply enough to explain it in an FDE interview without notes.
> Approach: read → understand → modify → break → fix → explain out loud.

---

## The rule

**No moving to the next lesson until you can explain the current one out loud — without looking at the code.**

If you can't explain it, you haven't learned it yet. Go back.

---

## Lesson 1 — What is RAG and why does it exist?

**Concept to understand:**
Language models are frozen at training time. They don't know what happened yesterday, what's in your company's database, or what's in a specific document. RAG (Retrieval-Augmented Generation) solves this by retrieving relevant information at query time and giving it to the model as context.

**The basic flow:**
```
User question
  → retrieve relevant text chunks from a knowledge base
  → inject those chunks into the LLM prompt
  → LLM generates an answer grounded in the retrieved context
```

**Your task:**
1. Open `app/api/routes.py`
2. Find the `query()` function
3. Trace the flow step by step — where does the query go first? What happens next?
4. Write the flow in your own words (in a comment or on paper)

**You've passed this lesson when you can answer:**
- What problem does RAG solve that fine-tuning doesn't?
- What are the two main steps in a RAG pipeline?
- Why do we retrieve chunks rather than full documents?

---

## Lesson 2 — Embeddings: turning text into numbers

**Concept to understand:**
An embedding is a list of numbers (a vector) that represents the meaning of a piece of text. Texts with similar meaning have vectors that are close together in space. This lets you find semantically similar text by measuring distance between vectors.

**The analogy:** Imagine every sentence has a location on a map. "The cat sat on the mat" and "A feline rested on a rug" are at almost the same location. "The stock market crashed" is far away.

**Your task:**
1. Open `app/retrieval/embedder.py`
2. Read every line — what does `embed_texts()` do? What does it return?
3. Find where `embed_texts` is called in `routes.py`
4. Ask yourself: why do we embed both the documents (at ingest) AND the query (at search time)?

**You've passed this lesson when you can answer:**
- What is an embedding? Explain it without using the word "vector"
- Why do we need to embed the query at search time?
- What does `response.data` contain in the embedder? What shape is it?

---

## Lesson 3 — Dense retrieval: finding similar chunks

**Concept to understand:**
Once you have embeddings, you find relevant chunks by measuring how similar the query vector is to each stored document vector. The standard measure is cosine similarity (or dot product for normalised vectors).

**The analogy:** You're looking for the nearest location on the map to your current position.

**Your task:**
1. In `routes.py`, find the dense retrieval section (look for the comment `# 2. Dense retrieval`)
2. Read the loop — what is it computing? What does the `zip()` do?
3. Try to explain the dot product calculation in plain English
4. What does `enumerate(scores)` return? What does `sorted(..., reverse=True)` do?

**You've passed this lesson when you can answer:**
- What is cosine similarity? What does a score of 1.0 mean? What does 0.0 mean?
- Why do we retrieve `top_k * 2` in the dense step instead of just `top_k`?
- What is the difference between dense retrieval and keyword search?

---

## Lesson 4 — Sparse retrieval: BM25 keyword search

**Concept to understand:**
Dense retrieval is great for meaning but bad for exact matches. If a user asks about "GPT-4o pricing", the embedding might not match a document that contains "GPT-4o" exactly, because it's a specific term. BM25 (Best Match 25) is a classic algorithm that scores documents by keyword frequency — it's essentially improved TF-IDF.

**The analogy:** Dense retrieval is a smart librarian who understands what you mean. BM25 is a search index that finds exact words. You want both.

**Your task:**
1. Open `app/retrieval/sparse.py`
2. Read `add_documents()` — what does "tokenized" mean here? What does `.lower().split()` do?
3. Read `search()` — what does `get_scores()` return?
4. Why does the index need to be rebuilt every time a new document is added? (look at `add_documents`)

**You've passed this lesson when you can answer:**
- What does BM25 measure?
- What is a "token" in the context of BM25? (hint: it's different from LLM tokens)
- Why is keyword search still useful when we have semantic embeddings?

---

## Lesson 5 — Hybrid retrieval: combining both with RRF

**Concept to understand:**
You have two ranked lists — one from dense retrieval, one from BM25. How do you merge them into one? You can't just add the scores because they're on completely different scales. Instead, Reciprocal Rank Fusion (RRF) uses rank position: a document that ranks 1st in both lists gets a higher combined score than one that ranks 1st in one and 5th in the other.

**The formula:** `score = 1 / (rank + k)` where k=60 is a dampening constant.

**Your task:**
1. Open `app/retrieval/hybrid.py`
2. Trace through `reciprocal_rank_fusion()` with a small example on paper:
   - dense_results: [(doc_A, rank 0), (doc_B, rank 1)]
   - sparse_results: [(doc_B, rank 0), (doc_A, rank 1)]
   - Calculate the RRF score for doc_A and doc_B manually
   - Which one wins?
3. What is `rrf_scores.get(idx, 0)` doing? Why the default of 0?

**You've passed this lesson when you can answer:**
- Why can't you just add the raw BM25 and cosine scores together?
- What does RRF reward? What kind of document gets the highest score?
- What happens to a document that only appears in one of the two result lists?

---

## Lesson 6 — The API layer: FastAPI and Pydantic

**Concept to understand:**
FastAPI is the web framework that exposes your retrieval logic as HTTP endpoints. Pydantic ensures that every request and response has a typed contract — if someone sends the wrong data, it fails immediately with a clear error.

**Your task:**
1. Open `app/core/models.py` — read `QueryRequest` and `QueryResponse`
2. In `routes.py`, find `async def query(request: QueryRequest)` — what happens if someone sends `{"query": 123}` (a number instead of a string)?
3. Go to `http://localhost:8000/docs` while the server is running — what do you see?
4. Try sending a request with a missing field — what error does FastAPI return?

**You've passed this lesson when you can answer:**
- What does `async def` mean? Why is the route async?
- What does Pydantic do when a request arrives?
- What is the `/docs` endpoint and how does it work automatically?

---

## Lesson 7 — Testing: why and how

**Concept to understand:**
Tests let you change code confidently. Without tests, every change is a guess. With tests, if you break something, you know immediately and exactly what broke.

**Your task:**
1. Open `tests/test_routes.py`
2. Find `patch("app.api.routes.embed_texts", ...)` — what does `patch` do? Why do we need it?
3. Run the tests: `.venv/bin/pytest -v`
4. Make one test fail on purpose — change an assertion. Run pytest again. Read the output.
5. Fix it. Run again.

**You've passed this lesson when you can answer:**
- What is mocking? Why do we mock `embed_texts` instead of calling the real OpenAI API in tests?
- What does `@pytest.mark.asyncio` do?
- What is an assertion? What happens when one fails?

---

## Lesson 8 — Your first extension (you write it)

**Task:** Add a `GET /stats` endpoint that returns:
```json
{
  "total_documents": 5,
  "embedding_model": "text-embedding-3-small",
  "llm_model": "gpt-4o-mini"
}
```

**Rules:**
- Do not ask Claude to write it
- You may ask Claude to explain anything you don't understand
- Use `app/api/routes.py` — it's one new function
- Use the existing `settings` object for the model names
- Use `len(_documents)` for the count

**When you're done:**
- Run the server and hit `GET /localhost:8000/stats` — does it return the right data?
- Write a test for it in `tests/test_routes.py`

---

## What comes after this course

Once you've completed all 8 lessons, the project has planned extensions you can build:

1. **pgvector persistence** — replace the in-memory list with a real Postgres database
2. **Cross-encoder reranking** — add a reranking step after RRF
3. **Streaming responses** — stream the LLM answer token by token (SSE)
4. **Eval suite** — measure faithfulness and relevance automatically
5. **Chunking strategies** — compare fixed-size vs semantic chunking

Each of these is a real production feature and a separate learning milestone.
