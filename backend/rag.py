import os
import uuid
from typing import List, Tuple, Dict, Any
from functools import lru_cache

from pypdf import PdfReader
import chromadb
from chromadb.config import Settings

from .openai_client import embed_texts


@lru_cache(maxsize=1)
def get_chroma_client():
    db_dir = os.environ.get("CHROMA_DB_DIR", "./data/chroma")
    os.makedirs(db_dir, exist_ok=True)
    return chromadb.PersistentClient(path=db_dir, settings=Settings(anonymized_telemetry=False))


def extract_pages(pdf_path: str) -> List[str]:
    """Extract text for each page from PDF using pypdf."""
    pages: List[str] = []
    reader = PdfReader(pdf_path)
    # Attempt to handle encrypted PDFs without password
    if getattr(reader, "is_encrypted", False):
        try:
            result = reader.decrypt("")
            if result == 0:
                raise RuntimeError("PDF is encrypted and requires a password.")
        except Exception as e:
            raise RuntimeError(f"PDF is encrypted or could not be decrypted: {e}")
    for page in reader.pages:
        text = page.extract_text() or ""
        # Normalize whitespace a bit
        text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
        pages.append(text)
    return pages


def split_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """Split a single page text into overlapping character chunks."""
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def chunk_pages(pages: List[str], chunk_size: int = 1200, overlap: int = 200) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Create chunks per page; return list of chunk texts and metadatas."""
    chunk_texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    for idx, page_text in enumerate(pages):
        page_num = idx + 1
        parts = split_text(page_text, chunk_size=chunk_size, overlap=overlap)
        for ci, part in enumerate(parts):
            chunk_texts.append(part)
            metadatas.append({"page": page_num, "chunk": ci})
    return chunk_texts, metadatas


def upsert_chunks(session_id: str, chunk_texts: List[str], metadatas: List[Dict[str, Any]]):
    client = get_chroma_client()
    coll_name = f"paper_{session_id}"
    collection = client.get_or_create_collection(name=coll_name)

    # Generate IDs
    ids = [f"{session_id}_p{m.get('page', 0)}_c{m.get('chunk', 0)}" for m in metadatas]

    # Attach session_id to metadata
    metas = []
    for m in metadatas:
        mi = dict(m)
        mi["session_id"] = session_id
        metas.append(mi)

    # Optionally limit the number of chunks to embed to reduce cost during testing
    max_chunks_env = os.environ.get("MAX_EMBED_CHUNKS")
    if max_chunks_env:
        try:
            max_chunks = int(max_chunks_env)
        except Exception:
            max_chunks = None
        if max_chunks and max_chunks > 0:
            chunk_texts = chunk_texts[:max_chunks]
            ids = ids[:max_chunks]
            metas = metas[:max_chunks]

    # Batch embeddings to avoid large payloads
    BATCH = 64
    n = len(chunk_texts)
    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        batch_docs = chunk_texts[start:end]
        batch_ids = ids[start:end]
        batch_metas = metas[start:end]
        batch_embs = embed_texts(batch_docs)
        collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas, embeddings=batch_embs)


def query_chunks(session_id: str, query: str, k: int = 4) -> List[Dict[str, Any]]:
    client = get_chroma_client()
    coll_name = f"paper_{session_id}"
    collection = client.get_or_create_collection(name=coll_name)

    q_emb = embed_texts([query])[0]
    res = collection.query(query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas", "distances"])

    results: List[Dict[str, Any]] = []
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        results.append({
            "text": doc,
            "metadata": meta,
            "score": float(dist) if dist is not None else None,
        })
    return results


def generate_accessible_summary(full_text: str) -> str:
    """Create a short, accessible summary (not the abstract) of the paper from available text."""
    from .openai_client import chat_completion

    max_context_chars = 30000
    context = full_text[:max_context_chars]

    system = (
        "You are an expert mentor. Summarize the paper for a junior researcher in plain language. "
        "Focus on: problem, motivation, method (high-level), key findings, novelty, and limitations. "
        "Avoid copying the abstract. Use 5–10 concise bullet points, then a one-sentence TL;DR."
    )
    user = f"Here is text extracted from the paper (may be partial):\n\n{context}\n\nPlease produce the accessible summary."

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return chat_completion(messages, temperature=0.3)


def persist_pdf(upload_dir: str, file_bytes: bytes) -> str:
    os.makedirs(upload_dir, exist_ok=True)
    session_id = uuid.uuid4().hex
    pdf_path = os.path.join(upload_dir, f"{session_id}.pdf")
    with open(pdf_path, "wb") as f:
        f.write(file_bytes)
    return pdf_path
