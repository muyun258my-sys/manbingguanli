import pytest


@pytest.fixture(autouse=True)
def _isolated_profile_db(tmp_path, monkeypatch):
    """每个测试使用独立的临时 SQLite 档案库，互不干扰、也不污染项目。"""
    monkeypatch.setenv("APP_PROFILE_DB", str(tmp_path / "profiles.db"))
    monkeypatch.setenv("APP_PROFILE_STORE", "sqlite")
    monkeypatch.setenv("APP_VECTOR_DB_DIR", str(tmp_path / "vector_db"))
    # 默认关闭 LLM，避免测试读到本地 .env 里的真实 key 去调外部接口。
    # 需要走模型路径的测试请自行注入 runtime（见 test_langgraph_workflow.py）。
    monkeypatch.setenv("APP_LLM_ENABLED", "false")
