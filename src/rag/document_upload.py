"""Document validation, parsing, chunking, and indexing."""

import os
import tempfile
import uuid

from fastapi import File, HTTPException, UploadFile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.retriever_setup import index_documents
from src.tools.common_tools import enhance_description_with_llm


def documents(description: str, file: UploadFile = File(...), user_id: str = ""):
    """Parse a PDF/TXT upload and append its chunks to the owning user's index."""
    filename = file.filename or "uploaded-document"
    suffix = os.path.splitext(filename)[1].lower()
    if suffix not in {".pdf", ".txt"}:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")
    if not user_id or not user_id.strip():
        raise HTTPException(status_code=400, detail="Authenticated user identity is required.")
    if not description or not description.strip():
        raise HTTPException(status_code=400, detail="X-Description is required.")

    file_bytes = file.file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        loader = PyPDFLoader(temp_path) if suffix == ".pdf" else TextLoader(
            temp_path, encoding="utf-8"
        )
        loaded_docs = loader.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded file: {exc}") from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

    if not loaded_docs or not any(doc.page_content.strip() for doc in loaded_docs):
        raise HTTPException(
            status_code=422,
            detail="No readable text was found. Scanned PDFs need OCR before upload.",
        )

    try:
        enhanced_description = enhance_description_with_llm(description.strip())
    except Exception:
        # Indexing should still work if optional description enhancement is unavailable.
        enhanced_description = description.strip()

    document_id = str(uuid.uuid4())
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(loaded_docs)
    for index, chunk in enumerate(chunks):
        chunk.metadata.update({
            "user_id": user_id.strip(),
            "document_id": document_id,
            "source": filename,
            "description": enhanced_description,
            "chunk_index": index,
        })

    try:
        count = index_documents(user_id, chunks)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Document indexing failed: {exc}") from exc

    return {"status": True, "document_id": document_id, "chunks_indexed": count}
