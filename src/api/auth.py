
"""User registration, login, and JWT authentication."""

import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import bcrypt
import jwt
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from pymongo.errors import DuplicateKeyError

from src.api.rate_limit import enforce_rate_limit
from src.db.mongo_client import db

router = APIRouter(prefix="/auth", tags=["authentication"])

users = db["users"]
bearer = HTTPBearer(auto_error=False)

JWT_ALGORITHM = "HS256"
JWT_ISSUER = "adaptive-rag-api"
JWT_AUDIENCE = "adaptive-rag-client"
JWT_EXPIRY_HOURS = 24


class AuthRequest(BaseModel):
    username: str = Field(
        min_length=3,
        max_length=40,
        pattern=r"^[a-zA-Z0-9_.-]+$",
    )
    password: str = Field(min_length=8, max_length=72)


def _secret() -> str:
    """Retrieve and validate the JWT signing secret."""
    secret = os.getenv("JWT_SECRET_KEY", "")

    if len(secret.encode("utf-8")) < 32:
        raise RuntimeError(
            "JWT_SECRET_KEY must contain at least 32 bytes."
        )

    return secret


def _issue_token(user: dict) -> str:
    """Issue a signed, time-limited access token."""
    now = datetime.now(timezone.utc)

    payload = {
        "sub": user["_id"],
        "username": user["username"],
        "iat": now,
        "exp": now + timedelta(hours=JWT_EXPIRY_HOURS),
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "jti": str(uuid4()),
        "token_type": "access",
    }

    return jwt.encode(
        payload,
        _secret(),
        algorithm=JWT_ALGORITHM,
    )


async def initialize_auth_indexes() -> None:
    """Validate authentication configuration and initialize indexes."""
    _secret()
    await users.create_index("username", unique=True)


@router.post("/register", status_code=201)
async def register(request: Request, payload: AuthRequest):
    """Register a new user."""
    await enforce_rate_limit(
        "register",
        request.client.host if request.client else "unknown",
        max_requests=3,
        window_seconds=3600,
    )

    username = payload.username.strip().lower()
    password_bytes = payload.password.encode("utf-8")

    if len(password_bytes) > 72:
        raise HTTPException(
            status_code=400,
            detail="Password exceeds the maximum supported length.",
        )

    user = {
        "_id": str(uuid4()),
        "username": username,
        "password_hash": bcrypt.hashpw(
            password_bytes,
            bcrypt.gensalt(),
        ).decode("utf-8"),
        "created_at": datetime.now(timezone.utc),
    }

    try:
        await users.insert_one(user)
    except DuplicateKeyError:
        raise HTTPException(
            status_code=409,
            detail="That username is already registered.",
        )

    return {"message": "Account created. Please log in."}


@router.post("/login")
async def login(request: Request, payload: AuthRequest):
    """Authenticate credentials and issue an access token."""
    await enforce_rate_limit(
        "login",
        request.client.host if request.client else "unknown",
        max_requests=10,
        window_seconds=300,
    )

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
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password.",
        )

    return {
        "access_token": _issue_token(user),
        "token_type": "bearer",
        "user_id": user["_id"],
        "username": user["username"],
    }


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> dict:
    """Validate a JWT and return the authenticated identity."""

    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Bearer token required.",
        )

    try:
        payload = jwt.decode(
            credentials.credentials,
            _secret(),
            algorithms=[JWT_ALGORITHM],
            issuer=JWT_ISSUER,
            audience=JWT_AUDIENCE,
            options={
                "require": [
                    "sub",
                    "exp",
                    "iat",
                    "iss",
                    "aud",
                    "jti",
                    "token_type",
                ]
            },
        )

        if payload["token_type"] != "access":
            raise jwt.InvalidTokenError("Invalid token type.")

        user_id = payload["sub"]

        if not isinstance(user_id, str) or not user_id.strip():
            raise jwt.InvalidTokenError("Invalid subject.")

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Session expired. Please log in again.",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token.",
        )

    return {
        "user_id": user_id,
        "username": payload.get("username", ""),
    }
