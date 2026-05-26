from __future__ import annotations

from typing import List, Dict, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.ai_agent import chat
from backend.services.pipeline_service import pipeline

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict]] = None


@router.post("")
def chat_endpoint(req: ChatRequest):
    """
    Natural-language interface to the forecasting pipeline.
    The Groq AI agent interprets the question, calls the relevant tool,
    and returns a human-readable response + structured data for charts.
    """
    if not pipeline.ready:
        raise HTTPException(503, "Pipeline not initialised yet — please wait a moment and retry.")
    if not req.message.strip():
        raise HTTPException(400, "message cannot be empty")

    result = chat(req.message, pipeline, history=req.history)
    return result
