"""
Embedder — wraps OpenAI embeddings API.

Why a wrapper?
- Centralises the model choice (change one line to swap models)
- Handles batching: the API has limits on tokens per request
- Returns plain lists of floats — no library-specific types leak out
"""
from openai import AsyncOpenAI
from app.core.config import settings

client = AsyncOpenAI(api_key=settings.openai_api_key)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts. Returns one vector per text."""
    response = await client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    # response.data is ordered the same as input
    return [item.embedding for item in response.data]


async def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    vectors = await embed_texts([query])
    return vectors[0]
