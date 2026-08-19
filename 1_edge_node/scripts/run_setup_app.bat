@echo off

echo =========================================
echo [1] Check and create .env if not exists
echo =========================================
if not exist .env (
    echo Creating app virtual environment...
    python -m venv .env
) else (
    echo Virtual environment ".env" already exists
)

echo.
echo =========================================
echo [2] Activate .env environment
echo =========================================
call .env\Scripts\activate.bat

echo.
echo =========================================
echo [3] Install dependencies from requirements_app.txt
echo =========================================
if exist requirements_app.txt (
    echo Installing requirements...
    pip install -r requirements_app.txt
) else (
    echo [WARNING] requirements_app.txt not found, skipping install
)

echo.
echo =========================================
echo [DONE] Setup completed
echo =========================================
pause
