@echo off

echo ================================
echo [1] Activate virtual environment .env
echo ================================
call .env\Scripts\activate.bat

echo ==========================================
echo STARTING EDGE NODE (AI PIPELINE + HMI)
echo ==========================================
cd /d "%~dp0.."
python apps\edge-agent\web_backend.py
pause
