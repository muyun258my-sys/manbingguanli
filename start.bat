@echo off
chcp 65001 >nul
REM 慢病健康管理助理 — 一键启动（后端 8080 + 前端 8501）
cd /d "%~dp0"

echo [1/2] 启动后端 API (http://127.0.0.1:8080) ...
start "app-backend" cmd /k python -m app

echo 等待后端就绪...
timeout /t 3 /nobreak >nul

echo [2/2] 启动前端页面 (http://localhost:8501) ...
python -m streamlit run frontend/app.py --server.port 8501

REM 前端窗口关闭后，提示用户后端仍在运行
echo.
echo 前端已退出。后端窗口 (app-backend) 仍在运行，可手动关闭。
pause
