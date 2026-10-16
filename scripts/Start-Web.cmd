@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-Web.ps1"
if errorlevel 1 pause
