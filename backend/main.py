import os
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from . import rag
from .openai_client import chat_completion
from .models import ChatRequest, ChatResponse, UploadResponse
from .prompts import BASE_SOCRATIC_SYSTEM_PROMPT, RAG_CONTEXT_HEADER

load_dotenv()

app = FastAPI(title="RAG Socratic Research Mentor")

# CORS for local dev (Streamlit frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.getcwd(), "data", "uploads")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/upload_pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    file_bytes = await file.read()
    pdf_path = rag.persist_pdf(UPLOAD_DIR, file_bytes)

    # Extract and chunk
    try:
        pages = rag.extract_pages(pdf_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {e}")
    chunk_texts, metadatas = rag.chunk_pages(pages)

    if not chunk_texts:
        raise HTTPException(status_code=400, detail="No extractable text found in PDF.")

    # Upsert into ChromaDB
    session_id = os.path.splitext(os.path.basename(pdf_path))[0]
    try:
        rag.upsert_chunks(session_id, chunk_texts, metadatas)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding/DB failure: {e}")

    # Produce accessible summary (using a trimmed full text)
    full_text = "\n\n".join(pages)
    try:
        summary = rag.generate_accessible_summary(full_text)
    except Exception as e:
        summary = f"Summary unavailable due to an error: {e}"

    return UploadResponse(
        session_id=session_id,
        summary=summary,
        page_count=len(pages),
        chunk_count=len(chunk_texts),
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Retrieve RAG context
    results = rag.query_chunks(req.session_id, req.user_message if req.user_message else "paper overview", k=4)
    context_blocks = []
    source_previews = []
    for r in results:
        md = r.get("metadata", {})
        page = md.get("page", "?")
        text = r.get("text", "")
        context_blocks.append(f"[p.{page}] {text}")
        # Keep short preview for UI
        preview = (text[:220] + "…") if len(text) > 220 else text
        source_previews.append({"page": page, "excerpt": preview})

    context_str = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no context retrieved)"

    # Build messages for chat completion
    messages: List[dict] = [{"role": "system", "content": BASE_SOCRATIC_SYSTEM_PROMPT}]
    messages.append({"role": "system", "content": RAG_CONTEXT_HEADER.format(context=context_str)})

    # Include history
    for m in req.chat_history:
        messages.append({"role": m.role, "content": m.content})

    # If lead is requested or user message is empty, ask the model to start the conversation
    user_message = req.user_message.strip()
    if req.lead or not user_message:
        user_message = (
            "Begin the Socratic discussion based on the context. "
            "First, ask one focused, open-ended question that helps identify the paper's central question or motivation."
        )

    messages.append({"role": "user", "content": user_message})

    ai_text = chat_completion(messages, temperature=0.4)

    return ChatResponse(ai_message=ai_text, sources=source_previews)
