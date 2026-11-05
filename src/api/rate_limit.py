"""MongoDB-backed rate limiting shared across backend workers."""

import asyncio
import hashlib
import logging
import time
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException
from pymongo import ReturnDocument

from src.db.mongo_client import db

logger = logging.getLogger(__name__)

limits = db["rate_limits"]
_index_ready = False
_index_lock = asyncio.Lock()


async def _ensure_ttl_index():
    global _index_ready

    if _index_ready:
        return

    async with _index_lock:
        if not _index_ready:
            await limits.create_index("expires_at", expireAfterSeconds=0)
            _index_ready = True


async def enforce_rate_limit(
    scope: str,
    identity: str,
    max_requests: int,
    window_seconds: int,
):
    """Enforce a fixed-window limit using an atomic MongoDB counter."""

    try:
        await _ensure_ttl_index()

        window = int(time.time()) // window_seconds
        raw_key = f"{scope}:{identity}:{window}"
        bucket_id = hashlib.sha256(raw_key.encode()).hexdigest()

        now = datetime.now(timezone.utc)
        expires_at = datetime.fromtimestamp(
            (window + 1) * window_seconds,
            tz=timezone.utc,
        ) + timedelta(minutes=2)

        result = await limits.find_one_and_update(
            {"_id": bucket_id},
            {
                "$inc": {"count": 1},
                "$setOnInsert": {"expires_at": expires_at},
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

        if result["count"] > max_requests:
            retry_after = max(
                1,
                int((expires_at - now).total_seconds()),
            )
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later.",
                headers={"Retry-After": str(retry_after)},
            )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Rate limiter unavailable.")
        raise HTTPException(
            status_code=503,
            detail="Request protection is temporarily unavailable.",
        ) from exc