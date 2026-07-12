@echo off

echo ================================
echo [1] Activate virtual environment .env
echo ================================
call .env\Scripts\activate.bat

echo.
echo ================================
echo [2] Run main.py
echo ================================
python main.py

echo.
echo ================================
echo [DONE] Script finished successfully
echo ================================
pause
