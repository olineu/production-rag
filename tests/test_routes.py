"""
Basic tests — run with: pytest

These test the API layer without hitting real OpenAI.
The embedder and LLM calls are mocked.
"""
import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport
from main import app


# A fake embedding vector (1536 dims, all 0.1)
FAKE_VECTOR = [0.1] * 1536


@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_ingest_and_query():
    with (
        patch("app.api.routes.embed_texts", new_callable=AsyncMock, return_value=[FAKE_VECTOR, FAKE_VECTOR]),
        patch("app.api.routes.embed_query", new_callable=AsyncMock, return_value=FAKE_VECTOR),
        patch("app.api.routes.client") as mock_client,
    ):
        # Mock the LLM response
        mock_choice = AsyncMock()
        mock_choice.message.content = "The answer is 42."
        mock_client.chat.completions.create = AsyncMock(
            return_value=AsyncMock(choices=[mock_choice])
        )

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            # Ingest
            ingest_resp = await ac.post("/ingest", json={
                "documents": ["The answer to life is 42.", "Hybrid retrieval beats dense-only."]
            })
            assert ingest_resp.status_code == 200
            assert ingest_resp.json()["ingested"] == 2

            # Query
            query_resp = await ac.post("/query", json={"query": "What is the answer to life?"})
            assert query_resp.status_code == 200
            data = query_resp.json()
            assert "answer" in data
            assert "chunks" in data
            assert len(data["chunks"]) > 0

@pytest.mark.asyncio
async def test_stats():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/stats")
    assert response.status_code == 200
    assert response.json() == {"total_documents": 2,"embedding_model": "text-embedding-3-small","llm_model": "gpt-4o-mini"}
