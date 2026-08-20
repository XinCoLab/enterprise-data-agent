@echo off
setlocal
cd /d "%~dp0"
docker compose down
if errorlevel 1 pause
