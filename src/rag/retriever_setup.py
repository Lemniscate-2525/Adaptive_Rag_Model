"""MongoDB-backed persistent FAISS indexes with per-user isolation and caching."""

import hashlib
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from gridfs import GridFSBucket
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient

MONGO_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGODB_DB_NAME", "adaptive_rag")

# A synchronous PyMongo client is used because indexing runs in a worker thread.
_mongo_client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=10000)
_db = _mongo_client[DB_NAME]
_bucket = GridFSBucket(_db, bucket_name="faiss_indexes")
_index_metadata = _db["vector_index_metadata"]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
_cache: dict[str, dict] = {}
_cache_guard = threading.Lock()
_user_locks: dict[str, threading.RLock] = {}


def _user_key(user_id: str) -> str:
    if not user_id or not user_id.strip():
        raise ValueError("A verified user_id is required.")
    return hashlib.sha256(user_id.strip().encode("utf-8")).hexdigest()


def _lock_for(key: str) -> threading.RLock:
    with _cache_guard:
        return _user_locks.setdefault(key, threading.RLock())


def _load_locked(key: str):
    metadata = _index_metadata.find_one({"_id": key})
    if metadata is None:
        _cache.pop(key, None)
        return None

    file_id = metadata["file_id"]
    cached = _cache.get(key)
    if cached and cached["file_id"] == file_id:
        return cached["store"]

    serialized_index = _bucket.open_download_stream(file_id).read()
    store = FAISS.deserialize_from_bytes(
        serialized_index,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    _cache[key] = {"file_id": file_id, "store": store}
    return store


def index_documents(user_id: str, chunks: list[Document]) -> int:
    """Append document chunks and persist the updated index in MongoDB GridFS."""
    if not chunks:
        raise ValueError("Cannot index an empty document.")
    key = _user_key(user_id)

    with _lock_for(key):
        store = _load_locked(key)
        if store is None:
            store = FAISS.from_documents(chunks, embeddings)
        else:
            store.add_documents(chunks)

        new_file_id = _bucket.upload_from_stream(
            f"{key}.faiss",
            store.serialize_to_bytes(),
            metadata={"user_key": key, "updated_at": datetime.now(timezone.utc)},
        )
        previous = _index_metadata.find_one_and_update(
            {"_id": key},
            {"$set": {"file_id": new_file_id, "updated_at": datetime.now(timezone.utc)}},
            upsert=True,
            return_document=False,
        )
        _cache[key] = {"file_id": new_file_id, "store": store}

        old_file_id = previous.get("file_id") if previous else None
        if old_file_id and old_file_id != new_file_id:
            try:
                _bucket.delete(old_file_id)
            except Exception:
                # The new index is already committed; an orphaned old blob is safe.
                pass

    return len(chunks)


def get_vectorstore(user_id: str):
    """Return the cached index or reload the latest persisted version from MongoDB."""
    key = _user_key(user_id)
    with _lock_for(key):
        return _load_locked(key)


def get_retriever(user_id: str):
    store = get_vectorstore(user_id)
    return store.as_retriever(search_kwargs={"k": 4}) if store else None
