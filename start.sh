#!/usr/bin/env bash
# 慢病健康管理助理 — 一键启动（后端 8080 + 前端 8501）
# 用法: bash start.sh   （Git Bash / WSL / macOS / Linux）
set -e
cd "$(dirname "$0")"

echo "[1/2] 启动后端 API (http://127.0.0.1:8080) ..."
python -m app &
BACKEND_PID=$!

# 前端退出时一并关闭后端
trap "echo; echo '正在关闭后端 (PID $BACKEND_PID)...'; kill $BACKEND_PID 2>/dev/null" EXIT

echo "等待后端就绪..."
sleep 3

echo "[2/2] 启动前端页面 (http://localhost:8501) ..."
python -m streamlit run frontend/app.py --server.port 8501
