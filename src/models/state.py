"""State passed between nodes in the adaptive RAG graph."""

from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class State(TypedDict, total=False):
    """Workflow state; messages are accumulated while control fields are updated."""

    messages: Annotated[list[AnyMessage], add_messages]
    binary_score: Optional[str]
    route: Optional[str]
    latest_query: Optional[str]
    original_question: Optional[str]
    user_id: Optional[str]
    context: Optional[str]
    retrieval_attempts: int
