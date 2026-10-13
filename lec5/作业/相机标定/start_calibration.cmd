@echo off
chcp 65001 >nul
cd /d "%~dp0"
"D:\miniconda\envs\lerobot\python.exe" "%~dp0calibrate_intrinsics.py" %*
if errorlevel 1 pause
