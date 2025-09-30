import os
from functools import lru_cache
from typing import List, Dict, Any

from openai import OpenAI

# Allow overriding models via environment (useful for Azure deployments where the model name is the deployment name)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Singleton OpenAI client using environment variable OPENAI_API_KEY."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Please set it in your environment or .env file.")
    org_id = os.environ.get("OPENAI_ORG_ID")
    project_id = os.environ.get("OPENAI_PROJECT_ID")
    # Support custom/base URLs (e.g., Azure OpenAI or gateways) and API versions when required
    base_url = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")  # common alias in some setups
        or os.environ.get("OPENAI_BASE")
    )
    api_version = os.environ.get("OPENAI_API_VERSION")

    kwargs: Dict[str, Any] = {"api_key": api_key}
    if org_id:
        kwargs["organization"] = org_id
    if project_id:
        kwargs["project"] = project_id
    if base_url:
        kwargs["base_url"] = base_url
    # Some providers (e.g., Azure OpenAI) require api-version as a query param
    if api_version:
        kwargs["default_query"] = {"api-version": api_version}
    return OpenAI(**kwargs)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return embeddings for a list of texts using OpenAI embeddings API."""
    client = get_client()
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def chat_completion(messages: List[Dict[str, Any]], temperature: float = 0.4, max_tokens: int | None = None) -> str:
    """Return assistant message text for given chat messages."""
    client = get_client()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""
