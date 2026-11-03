"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.auth import initialize_auth_indexes, router as auth_router
from src.api.routes import router as rag_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await initialize_auth_indexes()
    yield


app = FastAPI(title="Adaptive RAG API", version="1.0.0", lifespan=lifespan)
app.include_router(auth_router)
app.include_router(rag_router)


@app.get("/")
async def root():
    return {"message": "Adaptive RAG API is running"}


@app.get("/health")
async def health():
    return {"status": "ok"}
