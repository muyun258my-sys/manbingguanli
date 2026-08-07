import json

import pytest
import requests

from app.agent_runtime import (
    AgentConfig,
    AgentConfigError,
    AgentConfigStore,
    OpenAICompatibleChatClient,
    load_project_env,
)


def test_load_project_env_sets_missing_value(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("DEEPSEEK_API_KEY=test-key\n", encoding="utf-8")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    load_project_env(env_file)

    assert __import__("os").environ["DEEPSEEK_API_KEY"] == "test-key"


def test_load_project_env_does_not_override_existing_value(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("DEEPSEEK_API_KEY=file-key\n", encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "shell-key")

    load_project_env(env_file)

    assert __import__("os").environ["DEEPSEEK_API_KEY"] == "shell-key"


def test_load_project_env_skips_comments_and_blank_lines(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n\nFOO=bar\nQUOTED=\"strip me\"\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("FOO", raising=False)
    monkeypatch.delenv("QUOTED", raising=False)

    load_project_env(env_file)

    assert __import__("os").environ["FOO"] == "bar"
    assert __import__("os").environ["QUOTED"] == "strip me"


# ── AgentConfig.from_file ────────────────────────────────────────────────────

def test_agent_config_from_file_parses_request_block(tmp_path):
    config_file = tmp_path / "agent.json"
    config_file.write_text(
        json.dumps(
            {
                "name": "demo-agent",
                "base_url": "https://api.example.com/",
                "api_key_env": "MY_KEY",
                "request": {
                    "model": "model-x",
                    "temperature": 0.7,
                    "stream": True,
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {"role": "system", "content": "You are a demo agent."},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    config = AgentConfig.from_file(config_file)

    assert config.name == "demo-agent"
    assert config.base_url == "https://api.example.com"  # 尾斜杠被去掉
    assert config.api_key_env == "MY_KEY"
    assert config.model == "model-x"
    assert config.temperature == 0.7
    assert config.stream is True
    assert config.system_message == "You are a demo agent."


def test_agent_config_from_file_rejects_missing_system_message(tmp_path):
    config_file = tmp_path / "agent.json"
    config_file.write_text(
        json.dumps(
            {
                "name": "demo-agent",
                "base_url": "https://api.example.com",
                "request": {"messages": [{"role": "user", "content": "hi"}]},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(AgentConfigError, match="no system message"):
        AgentConfig.from_file(config_file)


def test_agent_config_store_loads_from_agents_dir(tmp_path):
    agents_dir = tmp_path / ".agents"
    agents_dir.mkdir()
    (agents_dir / "demo.json").write_text(
        json.dumps(
            {
                "name": "demo-agent",
                "base_url": "https://api.example.com",
                "request": {"messages": [{"role": "system", "content": "sys"}]},
            }
        ),
        encoding="utf-8",
    )

    store = AgentConfigStore(agents_dir=agents_dir)
    assert store.load("demo.json").name == "demo-agent"


def test_agent_config_store_missing_file_raises(tmp_path):
    store = AgentConfigStore(agents_dir=tmp_path)
    with pytest.raises(AgentConfigError, match="Missing agent config"):
        store.load("not-there.json")


# ── OpenAICompatibleChatClient ───────────────────────────────────────────────

# OpenAICompatibleChatClient 构造时会调用 load_project_env()，可能把项目根
# .env 里的真实 DEEPSEEK_API_KEY 重新加载进环境，破坏“无 key”场景的模拟。
# 这些测试统一禁用 load_project_env。

def _disable_env_loading(monkeypatch):
    monkeypatch.setattr("app.agent_runtime.load_project_env", lambda *a, **k: None)


def test_client_is_configured_checks_api_key_env(monkeypatch):
    _disable_env_loading(monkeypatch)
    config = AgentConfig("a", "https://x", "DEEPSEEK_API_KEY", "m", 0.2, False, "sys")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    assert OpenAICompatibleChatClient().is_configured(config) is True

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert OpenAICompatibleChatClient().is_configured(config) is False


def test_client_complete_raises_when_key_missing(monkeypatch):
    _disable_env_loading(monkeypatch)
    config = AgentConfig("a", "https://x", "DEEPSEEK_API_KEY", "m", 0.2, False, "sys")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(AgentConfigError, match="not set"):
        OpenAICompatibleChatClient().complete(config, messages=[])


def test_client_complete_posts_chat_completions(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "你好，请多保重。"}}]}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs["headers"]
        captured["json"] = kwargs["json"]
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)
    _disable_env_loading(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    config = AgentConfig("demo", "https://api.example.com", "DEEPSEEK_API_KEY", "model-x", 0.2, False, "You are sys.")
    content = OpenAICompatibleChatClient().complete(
        config,
        messages=[{"role": "user", "content": "hello"}],
        metadata={"agent": "demo"},
    )

    assert content == "你好，请多保重。"
    assert captured["url"] == "https://api.example.com/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["json"]["model"] == "model-x"
    assert captured["json"]["metadata"] == {"agent": "demo"}
    assert captured["json"]["messages"][0] == {"role": "system", "content": "You are sys."}


def test_client_complete_raises_when_response_empty_choices(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": []}

    monkeypatch.setattr(requests, "post", lambda url, **kwargs: FakeResponse())
    _disable_env_loading(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    config = AgentConfig("a", "https://x", "DEEPSEEK_API_KEY", "m", 0.2, False, "sys")
    with pytest.raises(AgentConfigError, match="choices"):
        OpenAICompatibleChatClient().complete(config, messages=[])


def test_client_complete_propagates_http_error(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            raise requests.HTTPError("401 Unauthorized")

        def json(self):
            return {}

    monkeypatch.setattr(requests, "post", lambda url, **kwargs: FakeResponse())
    _disable_env_loading(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    config = AgentConfig("a", "https://x", "DEEPSEEK_API_KEY", "m", 0.2, False, "sys")
    with pytest.raises(requests.HTTPError):
        OpenAICompatibleChatClient().complete(config, messages=[])
