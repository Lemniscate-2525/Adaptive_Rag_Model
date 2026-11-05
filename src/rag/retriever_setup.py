
"""MongoDB-backed persistent FAISS indexes with per-user isolation and caching."""

import hashlib
import logging
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
from langchain_huggingface import HuggingFaceEmbeddings
from pymongo import MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError

logger = logging.getLogger(__name__)

MONGO_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGODB_DB_NAME", "adaptive_rag")

_mongo_client = MongoClient(
    MONGO_URL,
    serverSelectionTimeoutMS=10000,
)

_db = _mongo_client[DB_NAME]
_bucket = GridFSBucket(_db, bucket_name="faiss_indexes")

# Lock state and active snapshot pointer share one MongoDB document.
_index_metadata = _db["vector_index_metadata"]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

_cache: dict[str, dict] = {}
_cache_guard = threading.Lock()
_user_locks: dict[str, threading.RLock] = {}


def _user_key(user_id: str) -> str:
    """Create a non-reversible namespace for an authenticated user."""
    if not user_id or not user_id.strip():
        raise ValueError("A verified user_id is required.")

    return hashlib.sha256(user_id.strip().encode("utf-8")).hexdigest()


def _lock_for(key: str) -> threading.RLock:
    """Get the process-local lock for a user namespace."""
    with _cache_guard:
        return _user_locks.setdefault(key, threading.RLock())


@contextmanager
def _distributed_lock(key: str, timeout_seconds: int = 60):
    """
    Acquire a renewable MongoDB lease with a monotonically increasing
    fencing token.

    The lease and active snapshot metadata live in the same document,
    allowing snapshot commits to validate ownership atomically.
    """
    owner = str(uuid4())

    lease_seconds = 180
    renewal_interval = 45

    deadline = time.monotonic() + timeout_seconds
    stop_renewal = threading.Event()
    lock_lost = threading.Event()

    fencing_token = None

    while time.monotonic() < deadline:
        now = datetime.now(timezone.utc)

        try:
            lock_doc = _index_metadata.find_one_and_update(
                {
                    "_id": key,
                    "$or": [
                        {"expires_at": {"$lte": now}},
                        {"expires_at": {"$exists": False}},
                        {"owner": None},
                    ],
                },
                {
                    "$inc": {"fencing_token": 1},
                    "$set": {
                        "owner": owner,
                        "user_key": key,
                        "expires_at": now + timedelta(
                            seconds=lease_seconds
                        ),
                    },
                },
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )

            if lock_doc and lock_doc.get("owner") == owner:
                fencing_token = lock_doc["fencing_token"]
                break

        except DuplicateKeyError:
            # Another worker owns the unexpired lease.
            pass

        except Exception:
            logger.exception("Failed to acquire FAISS lock.")
            raise

        time.sleep(0.1)

    if fencing_token is None:
        raise TimeoutError(
            "Timed out waiting for this user's document index lock."
        )

    def renew_lease():
        while not stop_renewal.wait(renewal_interval):
            now = datetime.now(timezone.utc)

            try:
                result = _index_metadata.update_one(
                    {
                        "_id": key,
                        "owner": owner,
                        "fencing_token": fencing_token,
                        "expires_at": {"$gt": now},
                    },
                    {
                        "$set": {
                            "expires_at": now + timedelta(
                                seconds=lease_seconds
                            )
                        }
                    },
                )

                if result.matched_count != 1:
                    lock_lost.set()
                    logger.error("Lost FAISS lease for namespace %s", key)
                    return

            except Exception:
                logger.exception("FAISS lease renewal failed.")
                lock_lost.set()
                return

    renewal_thread = threading.Thread(
        target=renew_lease,
        daemon=True,
    )
    renewal_thread.start()

    try:
        yield owner, fencing_token

        if lock_lost.is_set():
            raise RuntimeError(
                "FAISS distributed lease was lost during indexing."
            )

    finally:
        stop_renewal.set()
        renewal_thread.join(timeout=5)

        # Only the current owner and fencing generation can release.
        _index_metadata.update_one(
            {
                "_id": key,
                "owner": owner,
                "fencing_token": fencing_token,
            },
            {
                "$set": {
                    "owner": None,
                    "expires_at": datetime.now(timezone.utc),
                }
            },
        )


def _load_locked(key: str):
    """Load and validate the latest persisted FAISS snapshot."""
    metadata = _index_metadata.find_one({"_id": key})

    if metadata is None or metadata.get("file_id") is None:
        _cache.pop(key, None)
        return None

    if metadata.get("user_key") != key:
        logger.error("FAISS snapshot ownership mismatch.")
        raise RuntimeError("Invalid FAISS snapshot metadata.")

    file_id = metadata["file_id"]

    cached = _cache.get(key)
    if cached and cached["file_id"] == file_id:
        return cached["store"]

    try:
        grid_file = _bucket.get(file_id)

        if not grid_file.metadata:
            raise RuntimeError("GridFS snapshot metadata is missing.")

        if grid_file.metadata.get("user_key") != key:
            raise RuntimeError("GridFS snapshot ownership mismatch.")

        serialized_index = grid_file.read()

        if not serialized_index:
            raise RuntimeError("FAISS snapshot is empty.")

        store = FAISS.deserialize_from_bytes(
            serialized_index,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    except Exception:
        logger.exception("Failed to load FAISS snapshot.")
        raise

    _cache[key] = {
        "file_id": file_id,
        "store": store,
    }

    return store


def index_documents(user_id: str, chunks: list[Document]) -> int:
    """Append document chunks and atomically commit a new FAISS snapshot."""
    if not chunks:
        raise ValueError("Cannot index an empty document.")

    key = _user_key(user_id)

    with _lock_for(key):
        with _distributed_lock(key) as (owner, fencing_token):

            store = _load_locked(key)

            if store is None:
                store = FAISS.from_documents(chunks, embeddings)
            else:
                store.add_documents(chunks)

            # Upload an immutable snapshot to GridFS.
            new_file_id = _bucket.upload_from_stream(
                f"{key}.faiss",
                BytesIO(store.serialize_to_bytes()),
                metadata={
                    "user_key": key,
                    "fencing_token": fencing_token,
                    "updated_at": datetime.now(timezone.utc),
                },
            )

            # Commit only if this worker still owns the current lease.
            now = datetime.now(timezone.utc)

            result = _index_metadata.update_one(
                {
                    "_id": key,
                    "owner": owner,
                    "fencing_token": fencing_token,
                    "expires_at": {"$gt": now},
                },
                {
                    "$set": {
                        "file_id": new_file_id,
                        "user_key": key,
                        "updated_at": now,
                    }
                },
            )

            if result.matched_count != 1:
                logger.error(
                    "Rejected stale FAISS snapshot commit for namespace %s",
                    key,
                )
                raise RuntimeError(
                    "FAISS lease expired or ownership changed. "
                    "Snapshot commit rejected."
                )

            _cache[key] = {
                "file_id": new_file_id,
                "store": store,
            }

    return len(chunks)



def get_vectorstore(user_id: str):
    """Return the cached index or reload the latest persisted snapshot."""
    key = _user_key(user_id)

    with _lock_for(key):
        return _load_locked(key)


def similarity_search(user_id: str, query: str, k: int = 4):
    """Search only the authenticated user's vector index."""
    key = _user_key(user_id)

    with _lock_for(key):
        store = _load_locked(key)

        return store.similarity_search(query, k=k) if store else []


def get_retriever(user_id: str):
    """Return a retriever scoped to the authenticated user's index."""
    store = get_vectorstore(user_id)

    return (
        store.as_retriever(search_kwargs={"k": 4})
        if store
        else None
    )
