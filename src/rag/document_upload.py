
"""Document validation, parsing, chunking, and indexing."""

import logging
import os
import tempfile
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.retriever_setup import index_documents
from src.tools.common_tools import enhance_description_with_llm

logger = logging.getLogger(__name__)

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".pdf", ".txt"}


def documents(description: str, file: UploadFile, user_id: str):
    """Validate, parse, chunk, and index a document for one user."""

    if not user_id or not user_id.strip():
        raise HTTPException(status_code=400, detail="Invalid user identity.")

    if not description or not description.strip():
        raise HTTPException(status_code=400, detail="Description must not be empty.")

    if len(description) > 1000:
        raise HTTPException(status_code=400, detail="Description is too long.")

    original_filename = file.filename or "uploaded-document"
    filename = Path(original_filename.replace("\\", "/")).name
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported.",
        )

    # Read at most MAX_UPLOAD_SIZE + 1 bytes to detect oversized uploads.
    file_bytes = file.file.read(MAX_UPLOAD_SIZE + 1)

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(file_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the 10 MB upload limit.",
        )

    # Validate actual content instead of trusting the filename extension.
    if suffix == ".pdf" and not file_bytes.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid PDF.",
        )

    if suffix == ".txt":
        try:
            file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="TXT files must use UTF-8 encoding.",
            )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        loader = (
            PyPDFLoader(temp_path)
            if suffix == ".pdf"
            else TextLoader(temp_path, encoding="utf-8")
        )

        loaded_docs = loader.load()

    except Exception as exc:
        logger.exception("Document parsing failed for user %s", user_id)
        raise HTTPException(
            status_code=422,
            detail="Could not parse the uploaded document.",
        ) from exc

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

    if not loaded_docs or not any(doc.page_content.strip() for doc in loaded_docs):
        raise HTTPException(
            status_code=422,
            detail="The document contains no extractable text.",
        )

    try:
        enhanced_description = enhance_description_with_llm(description.strip())
    except Exception:
        logger.exception("Description enhancement failed; using original description.")
        enhanced_description = description.strip()

    document_id = str(uuid.uuid4())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(loaded_docs)

    for index, chunk in enumerate(chunks):
        chunk.metadata.update(
            {
                "user_id": user_id.strip(),
                "document_id": document_id,
                "source": filename,
                "description": enhanced_description,
                "chunk_index": index,
            }
        )

    try:
        count = index_documents(user_id, chunks)
    except Exception as exc:
        logger.exception("Document indexing failed for user %s", user_id)
        raise HTTPException(
            status_code=500,
            detail="Document indexing failed.",
        ) from exc

    return {
        "status": True,
        "document_id": document_id,
        "chunks_indexed": count,
    }
