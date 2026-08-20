@echo off
setlocal
cd /d "%~dp0"

docker info >nul 2>&1
if errorlevel 1 (
  echo Docker Desktop is not running. Start Docker Desktop and try again.
  pause
  exit /b 1
)

docker compose up --build -d
if errorlevel 1 (
  echo.
  echo DataAgent was not started.
  echo If the error mentions auth.docker.io or registry-1.docker.io,
  echo Docker Hub is unreachable from the current network. Connect an
  echo approved proxy, VPN, or company registry mirror, then run start.cmd again.
  echo No project files or local data were changed by this failed build.
  pause
  exit /b 1
)

echo DataAgent is starting at http://localhost:8080
echo First start may take a short time while the image is built.
pause
