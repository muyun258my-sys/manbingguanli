from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .models import ChatRequest, envelope
from .services import Orchestrator

app = FastAPI(title="慢病健康管理助理", version="0.1.0")
_orchestrator = Orchestrator()


class ChatBody(BaseModel):
    session_id: str
    user_id: str
    message: str


class ProfileBody(BaseModel):
    condition_description: Optional[str] = None
    conditions: Optional[List[str]] = None
    medications: Optional[List[str]] = None
    allergies: Optional[List[str]] = None


@app.get("/")
def root() -> JSONResponse:
    return JSONResponse(
        {
            "code": 0,
            "message": "backend is running",
            "data": {
                "service": "app",
                "frontend": "http://localhost:8501",
                "docs": "/docs",
                "health": "/health",
                "chat": "/chat",
            },
            "disclaimer": "",
        }
    )


@app.post("/chat")
def chat(body: ChatBody) -> JSONResponse:
    request = ChatRequest(
        session_id=body.session_id,
        user_id=body.user_id,
        message=body.message,
    )
    return JSONResponse(_orchestrator.chat(request))


@app.get("/profile/{user_id}")
def get_profile(user_id: str) -> JSONResponse:
    return JSONResponse(_orchestrator.get_profile(user_id))


@app.put("/profile/{user_id}")
def update_profile(user_id: str, body: ProfileBody) -> JSONResponse:
    result = _orchestrator.update_profile(
        user_id,
        condition_description=body.condition_description,
        conditions=body.conditions,
        medications=body.medications,
        allergies=body.allergies,
    )
    return JSONResponse(result)


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse(_orchestrator.health())
