"""Reindex the local PDFs into Qdrant."""

import sys

from fastapi import APIRouter, HTTPException, Request

from rag.config.settings import ROOT

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion import IngestionError
from rag.schemas.query import IngestResponse

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: Request) -> IngestResponse:
    try:
        result = request.app.state.rag.ingest()
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return IngestResponse.model_validate(result)
