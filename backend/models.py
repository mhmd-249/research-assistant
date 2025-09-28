from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class UploadResponse(BaseModel):
    session_id: str
    summary: str
    page_count: int
    chunk_count: int


class ChatRequest(BaseModel):
    session_id: str
    user_message: str = ""
    chat_history: List[ChatMessage] = Field(default_factory=list)
    lead: bool = False


class ChatResponse(BaseModel):
    ai_message: str
    sources: List[dict]
