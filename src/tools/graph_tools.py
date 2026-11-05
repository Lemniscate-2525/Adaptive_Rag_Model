"""Conditional routing for the adaptive RAG workflow."""

import os
from typing import Literal

from src.models.state import State

MAX_RETRIEVAL_ATTEMPTS = max(1, int(os.getenv("MAX_RETRIEVAL_ATTEMPTS", "3")))

def routing_tool(state: State) -> Literal["retriever", "general_llm", "web_search"]:
    """Choose a path based on the classified query intent."""
    route = state.get("route")

    if route == "index":
        return "retriever"
    if route == "general":
        return "general_llm"
    if route == "search":
        return "web_search"

    raise ValueError(f"Unsupported route: {route!r}")


def doc_tool(state: State) -> Literal["rewrite", "generate", "web_search"]:
    """Generate on relevant evidence, retry a bounded number of times, then search."""
    if state.get("binary_score") == "yes":
        return "generate"
    attempts = state.get("retrieval_attempts", 0)
    if attempts < MAX_RETRIEVAL_ATTEMPTS:
        return "rewrite"
    return "web_search"
