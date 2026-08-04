@echo off
echo =========================================
echo Starting MTI Vision AI Web Backend
echo =========================================

set PYTHONPATH=%PYTHONPATH%;%CD%

echo Starting FastAPI...
web_backend.py

pause
    