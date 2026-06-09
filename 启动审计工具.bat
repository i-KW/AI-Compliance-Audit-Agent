@echo off
title GDPR Privacy Auditor

echo ============================================
echo   GDPR Privacy Auditor
echo   Starting Web Server...
echo ============================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+
    echo Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo Installing Flask...
    pip install flask -q
)

echo Checking port 5000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5000.*LISTENING" 2^>nul') do (
    echo Killing old server process PID=%%a...
    taskkill /F /PID %%a >nul 2>&1
    timeout /t 2 /nobreak >nul
)

echo Starting server at http://127.0.0.1:5000
echo Browser will open automatically...
echo Press Ctrl+C to stop
echo.

start "" http://127.0.0.1:5000

python web_server.py
pause
