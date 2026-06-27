import pytest


@pytest.fixture(autouse=True)
def _isolated_profile_db(tmp_path, monkeypatch):
    """每个测试使用独立的临时 SQLite 档案库，互不干扰、也不污染项目。"""
    monkeypatch.setenv("APP_PROFILE_DB", str(tmp_path / "profiles.db"))
