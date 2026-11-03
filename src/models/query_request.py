"""Request schema for a RAG query."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    session_id: str = Field(min_length=1, max_length=256)
