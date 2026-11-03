"""Authenticated API routes for RAG queries and document uploads."""

import asyncio

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from langchain_core.messages import AIMessage, HumanMessage

from src.api.auth import get_current_user
from src.memory.chat_history_mongo import ChatHistory
from src.models.query_request import QueryRequest
from src.rag.document_upload import documents
from src.rag.graph_builder import builder

router = APIRouter()


@router.post("/rag/query")
async def rag_query(req: QueryRequest, current_user: dict = Depends(get_current_user)):
    """Run the workflow using the authenticated user's private document index."""
    session_id = f"{current_user['user_id']}:{req.session_id}"
    chat_history = ChatHistory.get_session_history(session_id)
    await chat_history.add_message(HumanMessage(content=req.query))
    messages = await chat_history.get_messages()

    result = await asyncio.to_thread(
        builder.invoke,
        {"messages": messages, "user_id": current_user["user_id"]},
    )
    output_message = result["messages"][-1]
    await chat_history.add_message(AIMessage(content=output_message.content))
    return {"result": output_message}


@router.post("/rag/documents/upload")
async def upload_file(
    file: UploadFile = File(...),
    description: str = Header(..., alias="X-Description"),
    current_user: dict = Depends(get_current_user),
):
    """Index an upload under the identity verified from the bearer token."""
    if not description.strip():
        raise HTTPException(status_code=400, detail="X-Description must not be empty.")
    result = await asyncio.to_thread(
        documents, description, file, current_user["user_id"]
    )
    return result
