from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


class AgentConfigError(RuntimeError):
    pass


def load_project_env(env_path: Optional[Path] = None) -> None:
    if env_path is None:
        env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class AgentConfig:
    name: str
    base_url: str
    api_key_env: str
    model: str
    temperature: float
    stream: bool
    system_message: str

    @classmethod
    def from_file(cls, path: Path) -> "AgentConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        request = payload.get("request") or {}
        messages = request.get("messages") or []
        system_message = ""
        for message in messages:
            if message.get("role") == "system":
                system_message = str(message.get("content", ""))
                break
        if not system_message:
            raise AgentConfigError(f"Agent config {path} has no system message.")

        return cls(
            name=str(payload["name"]),
            base_url=str(payload["base_url"]).rstrip("/"),
            api_key_env=str(payload.get("api_key_env", "DEEPSEEK_API_KEY")),
            model=str(request.get("model", "deepseek-v4-flash")),
            temperature=float(request.get("temperature", 0.2)),
            stream=bool(request.get("stream", False)),
            system_message=system_message,
        )


class AgentConfigStore:
    def __init__(self, agents_dir: Optional[Path] = None) -> None:
        if agents_dir is None:
            agents_dir = Path(__file__).resolve().parents[1] / ".agents"
        self.agents_dir = agents_dir

    def load(self, filename: str) -> AgentConfig:
        path = self.agents_dir / filename
        if not path.exists():
            raise AgentConfigError(f"Missing agent config: {path}")
        return AgentConfig.from_file(path)


class OpenAICompatibleChatClient:
    def __init__(self, timeout_seconds: float = 30.0) -> None:
        load_project_env()
        self.timeout_seconds = timeout_seconds

    def is_configured(self, config: AgentConfig) -> bool:
        return bool(os.getenv(config.api_key_env))

    def complete(
        self,
        config: AgentConfig,
        *,
        messages: List[Dict[str, str]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise AgentConfigError(f"Environment variable {config.api_key_env} is not set.")

        payload: Dict[str, Any] = {
            "model": config.model,
            "temperature": config.temperature,
            "stream": config.stream,
            "messages": [{"role": "system", "content": config.system_message}, *messages],
        }
        if metadata:
            payload["metadata"] = metadata

        response = requests.post(
            f"{config.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise AgentConfigError("Chat completion response did not include choices.")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not content:
            raise AgentConfigError("Chat completion response did not include message content.")
        return str(content)
