@echo off
setlocal

cd /d "%~dp0"

where python >nul 2>nul
if %errorlevel% equ 0 (
  python launch_upscale.py
) else (
  py -3 launch_upscale.py
)

if %errorlevel% neq 0 (
  echo.
  echo UpScale could not be launched.
  pause
)