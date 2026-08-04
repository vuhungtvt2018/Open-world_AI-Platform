@echo off
echo ==========================================
echo STARTING SYNC SERVICES WORKER
echo ==========================================
cd /d "%~dp0.."
call "..\.env\Scripts\activate.bat"
python data_sync_worker.py
pause
