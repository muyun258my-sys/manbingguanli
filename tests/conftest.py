import os
import tempfile
from pathlib import Path

import pytest

# ── 模块级隔离（在导入任何被测模块之前生效）───────────────────────────────
# app.app 在模块导入时会实例化 Orchestrator，若此时环境变量未设置，可能
# 读取项目根 .env、创建真实 LLM 客户端或触碰项目 data/ 目录。
# 这里在 pytest 收集阶段最先兜底，确保测试进程内默认关闭 LLM 与外部依赖。
os.environ.setdefault("APP_LLM_ENABLED", "false")
os.environ.setdefault("APP_PROFILE_STORE", "sqlite")
os.environ.setdefault(
    "APP_PROFILE_DB",
    str(Path(tempfile.gettempdir()) / "xm2_test_profiles.db"),
)
os.environ.setdefault(
    "APP_VECTOR_DB_DIR",
    str(Path(tempfile.gettempdir()) / "xm2_test_vector_db_nonexistent"),
)


@pytest.fixture(autouse=True)
def _isolated_profile_db(tmp_path, monkeypatch):
    """每个测试使用独立的临时 SQLite 档案库，互不干扰、也不污染项目。"""
    monkeypatch.setenv("APP_PROFILE_DB", str(tmp_path / "profiles.db"))
    monkeypatch.setenv("APP_PROFILE_STORE", "sqlite")
    monkeypatch.setenv("APP_VECTOR_DB_DIR", str(tmp_path / "vector_db"))
    # 默认关闭 LLM，避免测试读到本地 .env 里的真实 key 去调外部接口。
    # 需要走模型路径的测试请自行注入 runtime（见 test_langgraph_workflow.py）。
    monkeypatch.setenv("APP_LLM_ENABLED", "false")
