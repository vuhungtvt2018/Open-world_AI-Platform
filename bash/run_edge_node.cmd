@echo off
echo ==========================================
echo STARTING EDGE NODE (AI PIPELINE + HMI)
echo ==========================================
cd /d "%~dp0.."
call "..\.env\Scripts\activate.bat"
python web_backend.py
pause
