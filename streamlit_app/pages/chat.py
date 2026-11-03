"""Authenticated chat and document-upload interface."""

import streamlit as st

from streamlit_app.utils.api_client import document_upload_rag, query_backend

st.set_page_config(page_title="Adaptive RAG | Chat", page_icon="🔎", layout="wide")

if not st.session_state.get("access_token"):
    st.warning("Please sign in to continue.")
    st.switch_page("home.py")

access_token = st.session_state["access_token"]
username = st.session_state.get("username", "user")
session_id = st.session_state["session_id"]

with st.sidebar:
    st.subheader(f"Signed in as {username}")
    if st.button("Sign out", use_container_width=True):
        for key in ("access_token", "user_id", "username", "session_id", "chat_history"):
            st.session_state.pop(key, None)
        st.switch_page("home.py")

    st.divider()
    st.header("Upload a document")
    uploaded_file = st.file_uploader("PDF or TXT", type=["pdf", "txt"])
    description = st.text_area(
        "What is this document about?",
        max_chars=300,
        placeholder="e.g. Lecture notes on database indexing",
    )
    if st.button("Upload and index", disabled=not uploaded_file or not description.strip()):
        with st.spinner("Extracting, chunking and indexing..."):
            success, message = document_upload_rag(uploaded_file, description, access_token)
        if success:
            st.success(message)
        else:
            st.error(message)

st.title("Adaptive RAG")
st.caption("Questions are routed to your documents, the general assistant, or web search.")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

for role, message in st.session_state["chat_history"]:
    st.chat_message(role).write(message)

question = st.chat_input("Ask a question...")
if question and question.strip():
    st.session_state["chat_history"].append(("user", question))
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = query_backend(question, session_id, access_token)
        st.write(answer)
    st.session_state["chat_history"].append(("assistant", answer))
