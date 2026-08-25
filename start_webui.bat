@echo off
title 4DAnyone Web UI
cd /d "%~dp0"
echo ============================================================
echo   Starting 4DAnyone Standalone Web UI
echo ============================================================

if exist "env\Scripts\python.exe" (
    set "PY_EXE=env\Scripts\python.exe"
) else if exist "..\env\Scripts\python.exe" (
    set "PY_EXE=..\env\Scripts\python.exe"
) else if exist "%USERPROFILE%\pinokio\api\4DAnyone\app\env\Scripts\python.exe" (
    set "PY_EXE=%USERPROFILE%\pinokio\api\4DAnyone\app\env\Scripts\python.exe"
) else (
    set "PY_EXE=python"
)

echo Using Python: %PY_EXE%
"%PY_EXE%" app.py
pause
