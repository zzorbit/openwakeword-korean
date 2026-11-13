@echo off
setlocal
set "KOWW_STOP_WEB=%~dp0Stop-Web.ps1"
if not exist "%KOWW_STOP_WEB%" set "KOWW_STOP_WEB=%~dp0scripts\Stop-Web.ps1"
if not exist "%KOWW_STOP_WEB%" (
    echo Stop-Web.ps1 is missing. Keep this file with the supplied scripts.
    pause
    exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%KOWW_STOP_WEB%" %*
set "KOWW_STOP_EXIT=%ERRORLEVEL%"
if not "%KOWW_STOP_EXIT%"=="0" pause
exit /b %KOWW_STOP_EXIT%
