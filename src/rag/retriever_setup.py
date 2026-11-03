"""Persistent, user-scoped FAISS storage for uploaded documents."""

import hashlib
import os
import threading
from pathlib import Path

from filelock import FileLock
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

embeddings = OpenAIEmbeddings()
STORE_ROOT = Path(os.getenv("VECTORSTORE_DIR", "data/vectorstores")).resolve()
_cache = {}
_cache_guard = threading.Lock()
_user_locks = {}


def _user_key(user_id: str) -> str:
    """Return a filesystem-safe, non-reversible namespace for a user."""
    if not user_id or not user_id.strip():
        raise ValueError("A non-empty user_id is required for document storage.")
    return hashlib.sha256(user_id.strip().encode("utf-8")).hexdigest()


def _paths(user_id: str):
    key = _user_key(user_id)
    STORE_ROOT.mkdir(parents=True, exist_ok=True)
    store_dir = STORE_ROOT / key
    lock = FileLock(str(STORE_ROOT / f"{key}.lock"))
    return key, store_dir, lock


def _signature(store_dir: Path):
    files = [store_dir / "index.faiss", store_dir / "index.pkl"]
    if not all(path.exists() for path in files):
        return None
    return tuple((path.stat().st_mtime_ns, path.stat().st_size) for path in files)


def _load_locked(key: str, store_dir: Path):
    signature = _signature(store_dir)
    if signature is None:
        _cache.pop(key, None)
        return None
    cached = _cache.get(key)
    if cached and cached["signature"] == signature:
        return cached["store"]
    # Only load indexes created by this application in its own storage directory.
    store = FAISS.load_local(
        str(store_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    _cache[key] = {"store": store, "signature": signature}
    return store


def index_documents(user_id: str, chunks: list[Document]) -> int:
    """Append chunks to the user's persistent index and return the chunk count."""
    if not chunks:
        raise ValueError("Cannot index an empty document.")
    key, store_dir, lock = _paths(user_id)
    with lock:
        store = _load_locked(key, store_dir)
        if store is None:
            store = FAISS.from_documents(chunks, embeddings)
        else:
            store.add_documents(chunks)
        # Readers and writers use the same cross-process file lock.
        store.save_local(str(store_dir))
        signature = _signature(store_dir)
        _cache[key] = {"store": store, "signature": signature}
    return len(chunks)


def get_vectorstore(user_id: str):
    """Load the latest persisted index for this user, or return None if absent."""
    key, store_dir, lock = _paths(user_id)
    with lock:
        return _load_locked(key, store_dir)


def get_retriever(user_id: str = "default"):
    """Return a retriever for a user's own index; None means no upload yet."""
    store = get_vectorstore(user_id)
    return store.as_retriever(search_kwargs={"k": 4}) if store else None
