@echo off
setlocal
cd /d "%~dp0\.."

if not exist ".venv\Scripts\python.exe" (
  echo Creating local virtual environment...
  where py >nul 2>nul
  if errorlevel 1 (
    python -m venv .venv
  ) else (
    py -3 -m venv .venv
  )
  if errorlevel 1 goto :python_error
)

".venv\Scripts\python.exe" -c "import streamlit, wastewater_snd" >nul 2>nul
if errorlevel 1 (
  echo Installing local dashboard dependencies...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  if errorlevel 1 goto :install_error
  ".venv\Scripts\python.exe" -m pip install -e ".[web,plot]"
  if errorlevel 1 goto :install_error
)

set "SND_APP_MODE=local"
set "SND_LOCAL_IMPORT=1"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
if not defined SND_PORT set "SND_PORT=8501"

echo Opening the local-only importer at http://127.0.0.1:%SND_PORT%
start "" /min powershell -NoProfile -Command "Start-Sleep -Seconds 2; Start-Process 'http://127.0.0.1:%SND_PORT%'"
".venv\Scripts\python.exe" -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port %SND_PORT% --server.headless true
goto :eof

:python_error
echo Python 3.10-3.12 was not found. Install Python, then double-click this file again.
pause
exit /b 2

:install_error
echo Dependency installation failed. Check the messages above and try again.
pause
exit /b 3
