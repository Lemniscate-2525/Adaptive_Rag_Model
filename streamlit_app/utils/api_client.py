"""HTTP client for the Streamlit frontend."""

import logging

import requests

logger = logging.getLogger(__name__)

RUST_BASE_URL = "http://localhost:8080/api"
PYTHON_BASE_URL = "http://127.0.0.1:8000"


def create_user(username: str, password: str, api_token: str) -> bool:
    headers = {"X-API-TOKEN": api_token, "Content-Type": "application/json"}
    try:
        response = requests.post(
            f"{RUST_BASE_URL}/create_user",
            json={"username": username, "password": password},
            headers=headers,
            timeout=20,
        )
        return response.status_code == 200
    except requests.RequestException:
        logger.exception("User creation request failed")
        return False


def login_user(username: str, password: str, api_token: str) -> dict | None:
    headers = {"X-API-TOKEN": api_token, "Content-Type": "application/json"}
    try:
        response = requests.post(
            f"{RUST_BASE_URL}/login",
            json={"username": username, "password": password},
            headers=headers,
            timeout=20,
        )
        return response.json() if response.status_code == 200 else None
    except (requests.RequestException, ValueError):
        logger.exception("Login request failed")
        return None


def get_api_token() -> str | None:
    try:
        response = requests.post(f"{RUST_BASE_URL}/init", timeout=20)
        return response.json().get("api_token") if response.status_code == 200 else None
    except (requests.RequestException, ValueError):
        logger.exception("API token initialization failed")
        return None


def query_backend(query: str, session_id: str, user_id: str | None = None) -> str:
    """Send a query with separate conversation and document-owner identifiers."""
    try:
        response = requests.post(
            f"{PYTHON_BASE_URL}/rag/query",
            json={"query": query, "session_id": session_id, "user_id": user_id or session_id},
            timeout=120,
        )
        if response.status_code == 200:
            return response.json()["result"]["content"]
        return f"Error: {response.status_code} - {response.text}"
    except requests.RequestException as exc:
        logger.exception("RAG query request failed")
        return f"Request failed: {exc}"


def document_upload_rag(file, description: str, user_id: str) -> bool:
    """Upload a file into the specified user's isolated document index."""
    if not file or not user_id:
        return False
    headers = {"X-Description": description, "X-User-ID": user_id}
    try:
        response = requests.post(
            f"{PYTHON_BASE_URL}/rag/documents/upload",
            files={"file": (file.name, file, file.type)},
            headers=headers,
            timeout=180,
        )
        if response.status_code != 200:
            logger.error("Document upload failed: %s - %s", response.status_code, response.text)
        return response.status_code == 200
    except requests.RequestException:
        logger.exception("Document upload request failed")
        return False
