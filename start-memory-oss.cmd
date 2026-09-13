@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" "%~dp0src\api\app.py"
