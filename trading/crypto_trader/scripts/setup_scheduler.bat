@echo off
REM Wrapper that launches setup_scheduler.ps1 as Administrator.
REM Double-click this file or run from any terminal.

set SCRIPT_DIR=%~dp0

echo Launching PowerShell setup script (will request admin elevation)...
powershell -ExecutionPolicy Bypass -Command "Start-Process powershell -ArgumentList '-ExecutionPolicy Bypass -File \"%SCRIPT_DIR%setup_scheduler.ps1\"' -Verb RunAs"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Failed to launch. You can run manually as Administrator:
    echo   powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%setup_scheduler.ps1"
)
pause
