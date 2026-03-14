"""
Entry point. Run with:
    uvicorn main:app --reload
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api import routes as routes_module
from app.api.routes import router
from app.db.store import init_db, load_all
from app.db.session import AsyncSessionLocal

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once on startup, once on shutdown.
    Startup: create DB tables, then reload any existing documents into the
    in-memory corpus so BM25 and the id→index map are ready to use.
    """
    await init_db()
    async with AsyncSessionLocal() as session:
        existing = await load_all(session)
        if existing:
            texts = [doc.text for doc in existing]
            routes_module._documents.extend(texts)
            routes_module._id_to_idx.update({doc.id: i for i, doc in enumerate(existing)})
            routes_module._doc_metadata.update({
                i: {"source": doc.source, "category": doc.category}
                for i, doc in enumerate(existing)
            })
            routes_module.sparse_index.add_documents(texts)
            logging.getLogger(__name__).info(f"Loaded {len(existing)} documents from DB on startup.")
    yield


app = FastAPI(
    title="production-rag",
    description="State-of-the-art RAG pipeline: hybrid retrieval, reranking, evals.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok"}
