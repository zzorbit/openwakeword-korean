@echo off
setlocal
set "KOWW_START_WEB=%~dp0Start-Web.ps1"
if not exist "%KOWW_START_WEB%" set "KOWW_START_WEB=%~dp0scripts\Start-Web.ps1"
if not exist "%KOWW_START_WEB%" (
    echo Start-Web.ps1 is missing. Keep this file with the supplied scripts.
    pause
    exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%KOWW_START_WEB%" %*
set "KOWW_START_EXIT=%ERRORLEVEL%"
if not "%KOWW_START_EXIT%"=="0" pause
exit /b %KOWW_START_EXIT%
