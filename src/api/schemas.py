"""
API request/response schemas for insurance LLM serving.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., description="Conversation messages")
    max_tokens: int = Field(256, ge=1, le=2048, description="Max tokens to generate")
    temperature: float = Field(0.1, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p sampling")


class ChatResponse(BaseModel):
    response: str = Field(..., description="Generated response")
    model: str = Field(..., description="Model identifier")
    usage: dict = Field(default_factory=dict, description="Token usage stats")


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    model_loaded: bool