@echo off
setlocal
cd /d %~dp0

:: Check for .env file
if not exist .env (
    echo [WARNING] .env file not found. Fallback text generation will be used.
    echo To enable AI trash talk, rename .env.template to .env and add your GOOGLE_API_KEY.
    echo.
)

:: Set PYTHONPATH to current directory
set PYTHONPATH=%PYTHONPATH%;%CD%

echo Starting PokeLeader...
python ui/app.py

pause
