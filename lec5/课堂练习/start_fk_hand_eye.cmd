@echo off
cd /d "%~dp0"
"D:\miniconda\envs\lerobot\python.exe" "%~dp0fk_hand_eye_gui.py" %*
if errorlevel 1 pause
