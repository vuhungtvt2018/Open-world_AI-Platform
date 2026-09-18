echo ================================
echo [1] Activate virtual environment .env
echo ================================
call .env\Scripts\activate.bat

@echo off
echo ==========================================
echo STARTING CLOUD SERVER (DEFECT EMBEDDING)
echo ==========================================
cd /d "%~dp0..\defect_embedding"
uvicorn main:app --host 0.0.0.0 --port 8031
pause
