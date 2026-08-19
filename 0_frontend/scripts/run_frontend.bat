@echo off
echo =========================================
echo Starting AI Vision UI (Frontend)
echo =========================================

cd 0_frontend

echo Checking dependencies...
call npm install

echo.
echo Starting Vite dev server...
echo.
npm run dev

pause
