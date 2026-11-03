"""User registration, login, and JWT authentication."""

import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import bcrypt
import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from pymongo.errors import DuplicateKeyError

from src.db.mongo_client import db

router = APIRouter(prefix="/auth", tags=["authentication"])
users = db["users"]
bearer = HTTPBearer(auto_error=False)


class AuthRequest(BaseModel):
    username: str = Field(min_length=3, max_length=40, pattern=r"^[a-zA-Z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=72)


def _secret() -> str:
    secret = os.getenv("JWT_SECRET_KEY", "")
    if len(secret) < 32:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="JWT_SECRET_KEY must be configured with at least 32 characters.",
        )
    return secret


def _issue_token(user: dict) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user["_id"],
        "username": user["username"],
        "iat": now,
        "exp": now + timedelta(hours=24),
    }
    return jwt.encode(payload, _secret(), algorithm="HS256")


async def initialize_auth_indexes() -> None:
    """Enforce unique usernames in MongoDB."""
    await users.create_index("username", unique=True)


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(payload: AuthRequest):
    username = payload.username.strip().lower()
    if len(payload.password.encode("utf-8")) > 72:
        raise HTTPException(status_code=400, detail="Password must be at most 72 UTF-8 bytes.")
    user = {
        "_id": str(uuid4()),
        "username": username,
        "password_hash": bcrypt.hashpw(
            payload.password.encode("utf-8"), bcrypt.gensalt()
        ).decode("utf-8"),
        "created_at": datetime.now(timezone.utc),
    }
    try:
        await users.insert_one(user)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="That username is already registered.")
    return {"message": "Account created. Please log in."}


@router.post("/login")
async def login(payload: AuthRequest):
    username = payload.username.strip().lower()
    user = await users.find_one({"username": username})
    valid = bool(
        user
        and bcrypt.checkpw(
            payload.password.encode("utf-8"),
            user["password_hash"].encode("utf-8"),
        )
    )
    if not valid:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    return {
        "access_token": _issue_token(user),
        "token_type": "bearer",
        "user_id": user["_id"],
        "username": user["username"],
    }


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> dict:
    """Validate the bearer token and return the authenticated identity."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="Bearer token required.")
    try:
        payload = jwt.decode(
            credentials.credentials,
            _secret(),
            algorithms=["HS256"],
            options={"require": ["sub", "exp"]},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid authentication token.")
    return {"user_id": payload["sub"], "username": payload.get("username", "")}
