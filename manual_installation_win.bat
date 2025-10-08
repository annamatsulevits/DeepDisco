@echo off
setlocal

REM Move to the folder where the script is
cd /d %~dp0

echo Checking Python installation...
where python >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not added to PATH.
    pause
    exit /b 1
)

REM Create venv if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo Launching DeepDisco...
python run_app_win.py

endlocal
