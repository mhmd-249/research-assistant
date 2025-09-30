# Socratic Paper Explorer (MVP)

An MVP app that lets junior researchers explore research papers through a guided, Socratic-style conversation grounded in the paper's content.

- Backend: FastAPI
- Frontend: Streamlit
- LLM: OpenAI gpt-4o-mini
- Embeddings: OpenAI text-embedding-3-small
- Vector DB: ChromaDB (local persistence)
- PDF parsing: pypdf

## Project Structure

```
.
├── app.py                      # Streamlit frontend (run: streamlit run app.py)
├── backend/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app with upload and chat endpoints
│   ├── models.py               # Pydantic schemas
│   ├── openai_client.py        # OpenAI client + helpers
│   ├── prompts.py              # Socratic + RAG context prompts
│   └── rag.py                  # PDF extraction, chunking, embeddings, ChromaDB I/O
├── data/
│   ├── chroma/                 # ChromaDB persistence dir
│   └── uploads/                # Uploaded PDFs
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

1. Python 3.10+ recommended. Create and activate a virtual environment.

```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
```

2. Install dependencies.

```
pip install -r requirements.txt
```

3. Copy environment file and set your OpenAI API key.

```
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

4. Run the app (Streamlit auto-starts the backend).

```
streamlit run app.py
```

If you prefer to run the backend separately:

```
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Open the Streamlit app at the URL printed in your terminal.

## How it Works

1. Upload a PDF.
2. Backend extracts text with pypdf per page.
3. Pages are split into overlapping chunks (~1200 chars, 200 overlap).
4. Each chunk is embedded using OpenAI `text-embedding-3-small` and upserted into ChromaDB (persistent).
5. A plain-language summary (not the abstract) is generated using `gpt-4o-mini`.
6. Chat: Your messages are augmented with the top-k retrieved chunks and guided by a Socratic system prompt.

## Example Snippets

- PDF extraction (see `backend/rag.py`):

```python
pages = extract_pages(pdf_path)  # returns List[str]
chunks, metas = chunk_pages(pages, chunk_size=1200, overlap=200)
```

- Embeddings + Chroma Upsert:

```python
embs = embed_texts(chunks)
collection.add(ids=ids, documents=chunks, metadatas=metas, embeddings=embs)
```

- Retrieval for RAG:

```python
q_emb = embed_texts([query])[0]
res = collection.query(query_embeddings=[q_emb], n_results=4, include=["documents","metadatas"]) 
```

- Socratic Chat (see `backend/main.py`):

```python
messages = [
  {"role": "system", "content": BASE_SOCRATIC_SYSTEM_PROMPT},
  {"role": "system", "content": RAG_CONTEXT_HEADER.format(context=context_str)},
  *history,
  {"role": "user", "content": user_message},
]
ai_text = chat_completion(messages)
```

## Notes & Limitations

- The accessible summary uses the first ~30k characters of extracted text to keep requests tractable.
- Retrieval uses cosine distance provided by Chroma (lower is closer). We show short previews of sources.
- This is an MVP. For production, consider:
  - Tiktoken-based token-aware chunking
  - Better page-aware metadata and highlighting
  - Rate limiting / retries for OpenAI API
  - Persistent session store and authentication
  - Better error handling and logging

## Environment Variables

- `OPENAI_API_KEY` (required)
- `OPENAI_ORG_ID` (optional) — if you use OpenAI Organizations
- `OPENAI_PROJECT_ID` (optional) — if you use OpenAI Projects
- `OPENAI_BASE_URL` (optional) — custom API base (e.g., Azure OpenAI or an API gateway)
- `OPENAI_API_VERSION` (optional) — required by some providers (e.g., Azure)
- `CHAT_MODEL` (optional) — override chat model/deployment (default: `gpt-4o-mini`)
- `EMBEDDING_MODEL` (optional) — override embeddings model/deployment (default: `text-embedding-3-small`)
- `CHROMA_DB_DIR` (default: `./data/chroma`)
- `BACKEND_HOST` (default: `127.0.0.1`)
- `BACKEND_PORT` (default: `8000`)
