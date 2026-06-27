from app.agent_runtime import load_project_env


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
