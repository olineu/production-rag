"""
Eval suite — measures pipeline quality against a golden dataset.

Metrics:
  faithfulness  — is the answer grounded in the retrieved context?
                  scored by LLM-as-judge (0.0 or 1.0 per question)
  retrieval_hit — did any retrieved chunk contain the expected keyword?
                  scored exactly (0 or 1 per question)

Run with:
  OPENAI_API_KEY=... pytest tests/test_eval.py -v -s

The -s flag shows the printed report.

Note: this test hits the real OpenAI API and requires:
  1. A running server (uvicorn main:app --reload)
  2. Documents already ingested into the index
"""
import json
import os
import pytest
import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

GOLDEN_PATH = "tests/golden_dataset.json"
BASE_URL = "http://localhost:8000"
JUDGE_MODEL = "gpt-4o-mini"


async def judge_faithfulness(
    client: AsyncOpenAI,
    question: str,
    answer: str,
    context: str,
) -> float:
    """
    Ask an LLM whether the answer is supported by the context.

    This is the LLM-as-judge pattern: instead of writing a rule-based checker,
    we let a cheap model evaluate the output. Returns 1.0 if faithful, 0.0 if not.

    Why a separate judge model call? Because the pipeline LLM might be biased
    toward its own outputs. A separate call with a strict prompt is more reliable.
    """
    prompt = (
        "You are a strict factual evaluator.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        "Does the answer contain ONLY information that is directly supported by the context above? "
        "Reply with a single word: YES or NO."
    )
    response = await client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,   # deterministic — we want consistent scoring
        max_tokens=5,
    )
    verdict = response.choices[0].message.content or ""
    return 1.0 if "YES" in verdict.upper() else 0.0


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Requires OPENAI_API_KEY and a running server with ingested documents",
)
async def test_eval_pipeline():
    """
    Run the full eval suite against the golden dataset.

    For each question:
    1. Call POST /query
    2. Check retrieval_hit: did any chunk contain the expected keyword?
    3. Check faithfulness: does the answer follow from the retrieved context?

    Prints a per-question breakdown and aggregate scores at the end.
    Asserts that both metrics are above minimum thresholds.
    """
    with open(GOLDEN_PATH) as f:
        dataset = json.load(f)

    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    faithfulness_scores = []
    retrieval_hits = []

    print("\n" + "=" * 60)
    print(f"{'EVAL SUITE':^60}")
    print("=" * 60)

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as http:
        for i, row in enumerate(dataset):
            question = row["question"]
            expected_keyword = row["expected_chunk_contains"]

            # 1. Call the pipeline
            resp = await http.post("/query", json={
                "query": question,
                "use_hybrid": True,
                "top_k": 5,
            })
            assert resp.status_code == 200, f"Query failed: {resp.text}"
            data = resp.json()

            answer = data["answer"]
            chunks = data["chunks"]

            # 2. Retrieval hit — did any retrieved chunk contain the expected keyword?
            # Case-insensitive substring check: simple, interpretable, no LLM needed
            hit = any(
                expected_keyword.lower() in chunk["text"].lower()
                for chunk in chunks
            )
            retrieval_hits.append(1.0 if hit else 0.0)

            # 3. Faithfulness — LLM judges whether the answer follows from the context
            context = "\n\n".join(chunk["text"] for chunk in chunks)
            faith_score = await judge_faithfulness(
                openai_client, question, answer, context
            )
            faithfulness_scores.append(faith_score)

            # Per-question output
            print(f"\nQ{i+1}: {question}")
            print(f"  Retrieval hit : {'✓' if hit else '✗'}  (looking for: '{expected_keyword}')")
            print(f"  Faithfulness  : {'✓' if faith_score == 1.0 else '✗'}")
            print(f"  Answer        : {answer[:120]}{'...' if len(answer) > 120 else ''}")

    # Aggregate
    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
    avg_retrieval = sum(retrieval_hits) / len(retrieval_hits)

    print("\n" + "=" * 60)
    print(f"  Faithfulness : {avg_faithfulness:.0%}  ({sum(faithfulness_scores):.0f}/{len(faithfulness_scores)})")
    print(f"  Retrieval    : {avg_retrieval:.0%}  ({sum(retrieval_hits):.0f}/{len(retrieval_hits)})")
    print("=" * 60)

    # Thresholds — adjust as the pipeline improves
    # These are intentionally lenient for a fresh index.
    # In production you'd tighten these over time as the dataset grows.
    MIN_FAITHFULNESS = 0.6
    MIN_RETRIEVAL = 0.6

    assert avg_faithfulness >= MIN_FAITHFULNESS, (
        f"Faithfulness {avg_faithfulness:.0%} below threshold {MIN_FAITHFULNESS:.0%}"
    )
    assert avg_retrieval >= MIN_RETRIEVAL, (
        f"Retrieval hit rate {avg_retrieval:.0%} below threshold {MIN_RETRIEVAL:.0%}"
    )
