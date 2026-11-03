"""HTTP client used by the Streamlit frontend."""

import os

import requests
import streamlit as st


def backend_url() -> str:
    """Read the backend URL from Streamlit secrets or local environment."""
    configured = None
    try:
        configured = st.secrets.get("BACKEND_URL")
    except Exception:
        pass
    return (configured or os.getenv("BACKEND_URL") or "http://127.0.0.1:8000").rstrip("/")


def register_user(username: str, password: str) -> tuple[bool, str]:
    try:
        response = requests.post(
            f"{backend_url()}/auth/register",
            json={"username": username, "password": password},
            timeout=20,
        )
        if response.status_code == 201:
            return True, response.json().get("message", "Account created.")
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return False, str(detail)
    except requests.RequestException as exc:
        return False, f"Backend connection failed: {exc}"


def login_user(username: str, password: str) -> tuple[dict | None, str]:
    try:
        response = requests.post(
            f"{backend_url()}/auth/login",
            json={"username": username, "password": password},
            timeout=20,
        )
        if response.status_code == 200:
            return response.json(), ""
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return None, str(detail)
    except requests.RequestException as exc:
        return None, f"Backend connection failed: {exc}"


def query_backend(query: str, session_id: str, access_token: str) -> str:
    try:
        response = requests.post(
            f"{backend_url()}/rag/query",
            json={"query": query, "session_id": session_id},
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["result"]["content"]
    except requests.RequestException as exc:
        try:
            detail = exc.response.json().get("detail", str(exc)) if exc.response else str(exc)
        except ValueError:
            detail = str(exc)
        return f"Request failed: {detail}"


def document_upload_rag(file, description: str, access_token: str) -> tuple[bool, str]:
    if not file or not description.strip():
        return False, "Select a file and provide a short description."
    headers = {
        "X-Description": description.strip(),
        "Authorization": f"Bearer {access_token}",
    }
    try:
        response = requests.post(
            f"{backend_url()}/rag/documents/upload",
            files={"file": (file.name, file.getvalue(), file.type or "application/octet-stream")},
            headers=headers,
            timeout=180,
        )
        response.raise_for_status()
        result = response.json()
        return True, f"Indexed {result['chunks_indexed']} chunks."
    except requests.RequestException as exc:
        try:
            detail = exc.response.json().get("detail", str(exc)) if exc.response else str(exc)
        except ValueError:
            detail = str(exc)
        return False, f"Upload failed: {detail}"
