"""Query and health endpoints."""

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from rag.config.settings import EMBEDDING_MODEL
from rag.schemas.query import HealthResponse, QueryRequest, QueryResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    service = request.app.state.rag
    chunks = service.chunk_count()
    return HealthResponse(
        status="ok" if chunks else "empty",
        chunks=chunks,
        embedding_model=EMBEDDING_MODEL,
        llm_model=service.llm.model_name,
    )


@router.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, request: Request) -> QueryResponse:
    service = request.app.state.rag
    if not service.ready():
        raise HTTPException(
            status_code=503,
            detail="No documents are indexed. Run ingestion.py or POST /ingest first.",
        )
    return QueryResponse.model_validate(service.answer(body.question, body.top_k))


@router.post("/query/stream")
def query_stream(body: QueryRequest, request: Request) -> StreamingResponse:
    service = request.app.state.rag
    if not service.ready():
        raise HTTPException(
            status_code=503,
            detail="No documents are indexed. Run ingestion.py or POST /ingest first.",
        )

    def events():
        for name, data in service.stream(body.question, body.top_k):
            payload = json.dumps(data, ensure_ascii=False)
            yield f"event: {name}\ndata: {payload}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
