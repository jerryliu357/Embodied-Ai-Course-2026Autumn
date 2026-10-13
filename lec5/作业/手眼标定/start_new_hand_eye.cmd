@echo off
cd /d "%~dp0"
"D:\miniconda\envs\lerobot\python.exe" "%~dp0fk_hand_eye_gui.py" --new-session --intrinsics "%~dp0..\相机标定\intrinsics.json" %*
if errorlevel 1 pause
