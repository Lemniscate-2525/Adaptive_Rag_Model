"""Login and registration page."""

import uuid

import streamlit as st

from streamlit_app.utils.api_client import login_user, register_user

st.set_page_config(page_title="Adaptive RAG | Sign in", page_icon="🔎", layout="centered")

st.title("Adaptive RAG")
st.caption("Ask questions across your own uploaded documents.")

if st.session_state.get("access_token"):
    st.success(f"Signed in as {st.session_state.get('username', 'user')}.")
    if st.button("Open chat", type="primary", use_container_width=True):
        st.switch_page("pages/chat.py")
    if st.button("Sign out", use_container_width=True):
        for key in ("access_token", "user_id", "username", "session_id", "chat_history"):
            st.session_state.pop(key, None)
        st.rerun()
    st.stop()

mode = st.radio("Account", ["Sign in", "Create account"], horizontal=True)
with st.form("auth_form"):
    username = st.text_input("Username", max_chars=40)
    password = st.text_input("Password", type="password", max_chars=72)
    submitted = st.form_submit_button(
        "Sign in" if mode == "Sign in" else "Create account",
        type="primary",
        use_container_width=True,
    )

if submitted:
    if not username.strip() or not password:
        st.error("Enter both username and password.")
    elif mode == "Create account":
        if len(password) < 8:
            st.error("Use a password with at least 8 characters.")
        else:
            success, message = register_user(username, password)
            if success:
                st.success(message + " You can now sign in.")
            else:
                st.error(message)
    else:
        response, error = login_user(username, password)
        if response:
            st.session_state["access_token"] = response["access_token"]
            st.session_state["user_id"] = response["user_id"]
            st.session_state["username"] = response["username"]
            st.session_state["session_id"] = str(uuid.uuid4())
            st.session_state["chat_history"] = []
            st.switch_page("pages/chat.py")
        else:
            st.error(error)
