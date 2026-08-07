#!/usr/bin/env bash
# 慢病健康管理助理 — 一键启动（后端 8080 + 前端 8501）
# 用法: bash start.sh   （Git Bash / WSL / macOS / Linux）
# v2：启动后端后轮询 /health，就绪后再启动前端，避免“无法读取档案”的时序报错
set -e
cd "$(dirname "$0")"

echo "[1/2] 启动后端 API (http://127.0.0.1:8080) ..."
python -m app &
BACKEND_PID=$!

# 前端退出时一并关闭后端
trap "echo; echo '正在关闭后端 (PID $BACKEND_PID)...'; kill $BACKEND_PID 2>/dev/null" EXIT

echo "等待后端就绪（最长 90 秒）..."
READY=0
for _ in $(seq 1 180); do
    if curl -sf -m 2 "http://127.0.0.1:8080/health" >/dev/null 2>&1; then
        READY=1
        echo "后端已就绪"
        break
    fi
    sleep 0.5
done
if [ "$READY" -ne 1 ]; then
    echo "警告：等待超时，后端可能未正常启动，仍将尝试启动前端。"
fi

echo "[2/2] 启动前端页面 (http://localhost:8501) ..."
python -m streamlit run frontend/app.py --server.port 8501
