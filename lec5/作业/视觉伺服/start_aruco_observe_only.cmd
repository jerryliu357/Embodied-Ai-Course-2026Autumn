@echo off
cd /d "%~dp0"
"D:\miniconda\envs\lerobot\python.exe" "%~dp0aruco_visual_servo_gui.py" --observe-only %*
if errorlevel 1 pause
