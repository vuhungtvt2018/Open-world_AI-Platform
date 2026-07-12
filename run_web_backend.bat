@echo off
echo =========================================
echo Starting MTI Vision AI Web Backend
echo =========================================

set PYTHONPATH=%PYTHONPATH%;%CD%

echo Activating environment and starting FastAPI...
.env\Scripts\python.exe apps\edge-agent\web_backend.py

pause
