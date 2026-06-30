from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

Intent = Literal[
    "symptom_analysis",
    "medication_query",
    "diagnosis_query",
    "mixed_query",
    "profile_management",
    "diet_query",
    "high_risk_input",
    "general_health",
]

Severity = Literal["red", "yellow", "green"]
Confidence = Literal["high", "medium", "low"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def envelope(code: int, message: str, data: Any = None, disclaimer: str = "") -> Dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "data": data,
        "disclaimer": disclaimer,
    }


@dataclass
class Profile:
    user_id: str
    condition_description: str = ""
    conditions: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChatRequest:
    session_id: str
    user_id: str
    message: str


@dataclass
class SourceRef:
    title: str
    excerpt: str = ""
    source: str = ""

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class AgentOutput:
    agent: str
    content: str
    severity: Optional[Severity] = None
    sources: List[SourceRef] = field(default_factory=list)
    confidence: Confidence = "medium"

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["sources"] = [source.to_dict() for source in self.sources]
        return payload


@dataclass
class ChatResponse:
    session_id: str
    user_id: str
    message: str
    reply: str
    intent: Intent
    severity: Optional[Severity]
    emergency: bool
    sources: List[SourceRef] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["sources"] = [source.to_dict() for source in self.sources]
        return payload
