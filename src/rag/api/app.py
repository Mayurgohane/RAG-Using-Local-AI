"""FastAPI application for the local RAG pipeline."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag.api.routes.ingest import router as ingest_router
from rag.api.routes.query import router as query_router
from rag.pipeline.rag import RagService


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = RagService()
    yield
    app.state.rag.close()


app = FastAPI(
    title="Local RAG",
    description="Retrieve passages from the local Qdrant index and answer with a local model.",
    lifespan=lifespan,
)
app.include_router(query_router)
app.include_router(ingest_router)
