"""MongoDB-backed persistent FAISS indexes with per-user isolation and caching."""

import hashlib
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from io import BytesIO
from uuid import uuid4

from gridfs import GridFSBucket
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError

MONGO_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGODB_DB_NAME", "adaptive_rag")

# A synchronous PyMongo client is used because indexing runs in a worker thread.
_mongo_client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=10000)
_db = _mongo_client[DB_NAME]
_bucket = GridFSBucket(_db, bucket_name="faiss_indexes")
_index_metadata = _db["vector_index_metadata"]
_index_locks = _db["vector_index_locks"]

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


@contextmanager
def _distributed_lock(key: str, timeout_seconds: int = 60):
    """Acquire a MongoDB lease so separate API workers serialize writes per user."""
    owner = str(uuid4())
    deadline = time.monotonic() + timeout_seconds
    while True:
        now = datetime.now(timezone.utc)
        try:
            lock_doc = _index_locks.find_one_and_update(
                {
                    "_id": key,
                    "$or": [
                        {"expires_at": {"$lte": now}},
                        {"owner": owner},
                    ],
                },
                {
                    "$set": {
                        "owner": owner,
                        "expires_at": now + timedelta(minutes=30),
                    }
                },
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
            if lock_doc and lock_doc.get("owner") == owner:
                break
        except DuplicateKeyError:
            # Another worker currently owns the unexpired lease.
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for this user's document index lock.")
        time.sleep(0.1)

    try:
        yield
    finally:
        _index_locks.delete_one({"_id": key, "owner": owner})


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

    with _lock_for(key), _distributed_lock(key):
        store = _load_locked(key)
        if store is None:
            store = FAISS.from_documents(chunks, embeddings)
        else:
            store.add_documents(chunks)

        new_file_id = _bucket.upload_from_stream(
            f"{key}.faiss",
            BytesIO(store.serialize_to_bytes()),
            metadata={"user_key": key, "updated_at": datetime.now(timezone.utc)},
        )
        _index_metadata.update_one(
            {"_id": key},
            {"$set": {"file_id": new_file_id, "updated_at": datetime.now(timezone.utc)}},
            upsert=True,
        )
        _cache[key] = {"file_id": new_file_id, "store": store}

        # Keep prior snapshots so concurrent readers that already resolved an older
        # metadata pointer can finish safely. A retention cleanup job can prune them later.

    return len(chunks)


def get_vectorstore(user_id: str):
    """Return the cached index or reload the latest persisted version from MongoDB."""
    key = _user_key(user_id)
    with _lock_for(key):
        return _load_locked(key)


def similarity_search(user_id: str, query: str, k: int = 4):
    """Search while holding the user's in-process lock against concurrent index writes."""
    key = _user_key(user_id)
    with _lock_for(key):
        store = _load_locked(key)
        return store.similarity_search(query, k=k) if store else []


def get_retriever(user_id: str):
    store = get_vectorstore(user_id)
    return store.as_retriever(search_kwargs={"k": 4}) if store else None
