
"""Validated request schema for RAG queries."""

import re

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    session_id: str = Field(min_length=1, max_length=128)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Query cannot be empty.")

        # Allow normal newlines and tabs, reject other control characters.
        if any(
            (ord(char) < 32 and char not in "\n\t") or ord(char) == 127
            for char in value
        ):
            raise ValueError("Query contains invalid control characters.")

        return value

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        value = value.strip()

        if not re.fullmatch(r"[a-zA-Z0-9._:-]{1,128}", value):
            raise ValueError("Invalid session ID format.")

        return value
