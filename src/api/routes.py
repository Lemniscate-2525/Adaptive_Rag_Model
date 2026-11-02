"""API routes for RAG queries and document uploads."""

import asyncio

from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from langchain_core.messages import AIMessage, HumanMessage

from src.memory.chat_history_mongo import ChatHistory
from src.models.query_request import QueryRequest
from src.rag.document_upload import documents
from src.rag.graph_builder import builder

router = APIRouter()


@router.post("/rag/query")
async def rag_query(req: QueryRequest):
    """Run the workflow with chat history and a user-specific document index."""
    chat_history = ChatHistory.get_session_history(req.session_id)
    await chat_history.add_message(HumanMessage(content=req.query))
    messages = await chat_history.get_messages()

    result = await asyncio.to_thread(
        builder.invoke,
        {"messages": messages, "user_id": req.user_id or req.session_id},
    )
    output_message = result["messages"][-1]
    await chat_history.add_message(AIMessage(content=output_message.content))
    return {"result": output_message}


@router.post("/rag/documents/upload")
async def upload_file(
    file: UploadFile = File(...),
    description: str = Header(..., alias="X-Description"),
    user_id: str = Header(..., alias="X-User-ID"),
):
    """Upload a PDF/TXT file into the caller-supplied user namespace."""
    if not user_id.strip():
        raise HTTPException(status_code=400, detail="X-User-ID must not be empty.")
    result = await asyncio.to_thread(documents, description, file, user_id)
    return result
