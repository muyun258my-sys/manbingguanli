@echo off
chcp 65001 >nul
REM 慢病健康管理助理 — 一键启动（后端 8080 + 前端 8501）
REM v2：启动后端后轮询 /health，就绪后再启动前端，避免“无法读取档案”的时序报错
cd /d "%~dp0"

echo [1/2] 启动后端 API (http://127.0.0.1:8080) ...
start "app-backend" cmd /k python -m app

echo 等待后端就绪（最长 90 秒）...
powershell -NoProfile -Command "$d=(Get-Date).AddSeconds(90); do { try { $ok=((Invoke-WebRequest -Uri 'http://127.0.0.1:8080/health' -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200) } catch { $ok=$false }; if (-not $ok) { Start-Sleep -Milliseconds 500 } } while (-not $ok -and (Get-Date) -lt $d); if ($ok) { Write-Host '后端已就绪' } else { Write-Host '警告：等待超时，后端可能未正常启动' }"

echo [2/2] 启动前端页面 (http://localhost:8501) ...
python -m streamlit run frontend/app.py --server.port 8501

REM 前端窗口关闭后，提示用户后端仍在运行
echo.
echo 前端已退出。后端窗口 (app-backend) 仍在运行，可手动关闭。
pause
