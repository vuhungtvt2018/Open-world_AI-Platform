@echo off
echo =========================================
echo [1] Check and setup uv environment
echo =========================================
cd ..
if not exist .venv (
    echo Creating uv virtual environment...
    uv venv
) else (
    echo Virtual environment already exists
)

echo.
echo =========================================
echo [2] Sync Workspace Dependencies
echo =========================================
echo Syncing workspace...
uv sync

echo.
echo =========================================
echo [DONE] Setup completed
echo =========================================
pause
