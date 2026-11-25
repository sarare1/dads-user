@echo off
REM One-time setup on Windows: creates the virtual environment and installs the dependencies.
cd /d "%~dp0"
if not exist venv (
    python -m venv venv || (echo Python 3.11+ is required & exit /b 1)
)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt -c constraints.txt
echo.
echo Setup finished. Start DADS with run_local.bat
