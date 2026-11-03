"""Chat page for the Streamlit application."""

import streamlit as st

from streamlit_app.utils.api_client import document_upload_rag, query_backend

st.set_page_config(
    page_title="Adaptive RAG",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "session_id" not in st.session_state or "jwt_token" not in st.session_state:
    st.warning("Please log in first.")
    st.stop()

user_id = st.session_state.get("username")
if not user_id:
    st.warning("Your user session is missing. Please log in again.")
    st.stop()

col1, col2 = st.columns([10, 2])
with col2:
    if st.button("Log out", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.switch_page("Home.py")

st.title("Adaptive RAG")

with st.sidebar:
    st.header("Upload a document")
    uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])
    if uploaded_file:
        description = st.text_input(
            "What is this document about?",
            max_chars=300,
            placeholder="For example: lecture notes on database indexing",
        )
        if description and st.button("Upload and index"):
            with st.spinner("Reading and indexing document..."):
                success = document_upload_rag(uploaded_file, description, user_id)
            if success:
                st.success(f"Indexed {uploaded_file.name} for {user_id}.")
            else:
                st.error("Upload failed. Check the API response/logs and try again.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for role, text in st.session_state.chat_history:
    st.chat_message(role).write(text)

user_input = st.chat_input("Ask a question...")
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_backend(
                user_input,
                st.session_state["jwt_token"],
                user_id=user_id,
            )
        st.write(response)
    st.session_state.chat_history.append(("assistant", response))
