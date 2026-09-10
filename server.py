"""
server.py: the FastAPI web server for the Bookly support agent.

This is the entry point. It:
1. Serves the chat UI (static HTML file)
2. Handles /chat/stream, which streams the agent's response as Server-Sent Events
3. Handles /summary, which generates a conversation recap for the CX team

Run with: uvicorn server:app --reload
"""

from typing import List
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from agent import get_agent_response_stream, generate_conversation_summary

app = FastAPI(title="Bookly Support Agent")


# ---------------------------------------------------------------------------
# REQUEST / RESPONSE MODELS
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]  # Full conversation history
    conversation_state: dict = {}  # Tracks verification, current order, etc.


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Serve the chat UI."""
    return FileResponse(
        "static/index.html",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint. Returns Server-Sent Events.
    Tool calls arrive as they happen, then text streams token-by-token.
    """
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    return StreamingResponse(
        get_agent_response_stream(messages, request.conversation_state),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/summary")
async def summary(request: ChatRequest):
    """
    Generate a conversation summary for internal CX team review.
    Uses a separate LLM call to analyze the conversation.
    """
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    result = generate_conversation_summary(messages, request.conversation_state)
    return result


# Serve static files (the chat UI)
app.mount("/static", StaticFiles(directory="static"), name="static")
