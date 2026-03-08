"""
Entry point. Run with:
    uvicorn main:app --reload
"""
import logging
from fastapi import FastAPI
from app.api.routes import router

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

app = FastAPI(
    title="production-rag",
    description="State-of-the-art RAG pipeline: hybrid retrieval, reranking, evals.",
    version="0.1.0",
)

app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok"}
