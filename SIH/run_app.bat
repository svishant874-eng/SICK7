@echo off
title Launching YOLOv8 & Number Plate Detection App
cd /d "%~dp0"

echo ===================================================
echo   Starting YOLOv8 Object & Number Plate App...
echo ===================================================

:: Open browser automatically after 3 seconds in the background
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:7860"

:: Run the Python app using the virtual environment
.\.venv\Scripts\python.exe app.py

pause
