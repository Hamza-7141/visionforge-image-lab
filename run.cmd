@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8001 --no-access-log
