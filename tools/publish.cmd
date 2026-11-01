@echo off
rem Double-click: pushes to GitHub and publishes to crates.io. See publish.ps1.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0publish.ps1" %*
set status=%errorlevel%
echo.
pause
exit /b %status%
