@echo off
REM Starts DADS on this computer and opens http://localhost:8000
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
    echo Run setup_local.bat first.
    exit /b 1
)
venv\Scripts\python.exe main.py
