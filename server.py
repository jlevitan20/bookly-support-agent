"""
server.py — FastAPI web server for the Bookly support agent.

This is the entry point. It:
1. Serves the chat UI (static HTML file)
2. Handles the /chat API endpoint — receives user messages, calls the agent, returns responses
3. Handles the /chat/stream endpoint — streams the agent's response via Server-Sent Events

Run with: uvicorn server:app --reload
"""

from typing import List, Optional
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from agent import get_agent_response, get_agent_response_stream, generate_conversation_summary

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


class ChatResponse(BaseModel):
    response: str  # The agent's text reply
    tool_calls: list  # Tools that were called (for UI indicators)
    conversation_state: dict  # Updated state to send back next turn
    escalation: Optional[dict] = None  # Escalation details if agent handed off


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


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint (non-streaming).
    Receives the user's message + conversation history,
    calls the agent, returns the response.
    """
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    result = get_agent_response(messages, request.conversation_state)

    return ChatResponse(
        response=result["response"],
        tool_calls=result["tool_calls"],
        conversation_state=result["conversation_state"],
        escalation=result.get("escalation"),
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint — returns Server-Sent Events.
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
