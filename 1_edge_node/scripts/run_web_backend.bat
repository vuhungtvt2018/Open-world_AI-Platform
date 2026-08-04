@echo off
echo =========================================
echo Starting Vision AI Web Backend
echo =========================================

cd ..
set PYTHONPATH=%PYTHONPATH%;%CD%

echo Activating environment and starting FastAPI...
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe apps\edge-agent\web_backend.py
) else (
    .env\Scripts\python.exe apps\edge-agent\web_backend.py
)

pause
