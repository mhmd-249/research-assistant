import os
import sys
import time
import subprocess
from typing import List, Dict

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_HOST = os.environ.get("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.environ.get("BACKEND_PORT", "8000"))
BASE_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"

st.set_page_config(page_title="Socratic Paper Explorer", page_icon="📄")


def ensure_backend_running() -> None:
    health_url = f"{BASE_URL}/health"
    try:
        r = requests.get(health_url, timeout=1.5)
        if r.ok:
            return
    except Exception:
        pass

    st.sidebar.info("Starting backend server…")
    # Start uvicorn in a background process
    cmd = [
        sys.executable, "-m", "uvicorn", "backend.main:app",
        "--host", BACKEND_HOST, "--port", str(BACKEND_PORT), "--reload"
    ]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Wait briefly for it to start
    for _ in range(20):
        try:
            r = requests.get(health_url, timeout=1.5)
            if r.ok:
                return
        except Exception:
            time.sleep(0.3)
    st.sidebar.warning("Backend may still be starting. If issues persist, run it manually: uvicorn backend.main:app --reload")


def api_upload_pdf(file_name: str, file_bytes: bytes) -> Dict:
    url = f"{BASE_URL}/api/upload_pdf"
    files = {"file": (file_name, file_bytes, "application/pdf")}
    resp = requests.post(url, files=files, timeout=120)
    if not resp.ok:
        # Try to surface backend error details (JSON or text)
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise requests.HTTPError(f"Upload failed: {resp.status_code} {detail}", response=resp)
    return resp.json()


def api_chat(session_id: str, user_message: str, chat_history: List[Dict], lead: bool = False) -> Dict:
    url = f"{BASE_URL}/api/chat"
    payload = {
        "session_id": session_id,
        "user_message": user_message,
        "chat_history": chat_history,
        "lead": lead,
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


# UI
st.title("📄 Socratic Paper Explorer")

with st.sidebar:
    st.markdown("**How it works**")
    st.markdown("1. Upload a PDF paper.")
    st.markdown("2. We extract text, chunk, embed with OpenAI, and store in ChromaDB.")
    st.markdown("3. Get a plain-language summary.")
    st.markdown("4. Chat with a Socratic AI mentor grounded in the paper.")

ensure_backend_running()

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content}
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

uploaded = st.file_uploader("Upload a research paper (PDF)", type=["pdf"], accept_multiple_files=False)
process = st.button("Process Paper", disabled=uploaded is None)

if process and uploaded is not None:
    with st.spinner("Processing, embedding, and summarizing…"):
        try:
            data = api_upload_pdf(uploaded.name, uploaded.getvalue())
        except requests.HTTPError as e:
            st.error("Processing failed. See backend details below:")
            # Show backend response text or JSON for easier debugging
            if getattr(e, "response", None) is not None:
                try:
                    st.code(e.response.json())
                except Exception:
                    st.code(e.response.text)
            else:
                st.code(str(e))
            st.stop()
        st.session_state.session_id = data["session_id"]
        summary = data.get("summary", "")
        page_count = data.get("page_count", 0)
        chunk_count = data.get("chunk_count", 0)

    st.success(f"Processed {page_count} pages into {chunk_count} chunks.")
    with st.expander("Accessible Summary", expanded=True):
        st.write(summary)

    # Kick off Socratic dialogue
    with st.spinner("Starting Socratic discussion…"):
        resp = api_chat(st.session_state.session_id, user_message="", chat_history=st.session_state.messages, lead=True)
        ai_msg = resp.get("ai_message", "")
        st.session_state.messages.append({"role": "assistant", "content": ai_msg})
        st.session_state.last_sources = resp.get("sources", [])

# Chat UI
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"]) 

user_input = st.chat_input("Ask a question or continue the discussion…")
if user_input and st.session_state.session_id:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            resp = api_chat(st.session_state.session_id, user_input, st.session_state.messages, lead=False)
            ai_msg = resp.get("ai_message", "")
            st.session_state.messages.append({"role": "assistant", "content": ai_msg})
            st.session_state.last_sources = resp.get("sources", [])
            st.markdown(ai_msg)

with st.expander("Show sources from last answer"):
    if st.session_state.last_sources:
        for i, src in enumerate(st.session_state.last_sources, 1):
            st.markdown(f"**Source {i} (p.{src.get('page', '?')})**: {src.get('excerpt', '')}")
    else:
        st.caption("No sources available.")
