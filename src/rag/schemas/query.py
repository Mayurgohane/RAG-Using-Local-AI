"""API models for query and ingest."""

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=4, ge=1, le=8)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Question is required.")
        return cleaned


class Source(BaseModel):
    source: str
    page: int | str
    similarity: float
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]


class IngestResponse(BaseModel):
    files: int
    pages: int
    chunks: int


class HealthResponse(BaseModel):
    status: str
    chunks: int
    embedding_model: str
    llm_model: str
