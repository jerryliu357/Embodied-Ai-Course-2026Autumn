@echo off
chcp 65001 >nul
cd /d "%~dp0"
"D:\miniconda\envs\lerobot\python.exe" "%~dp0hand_eye_calibration.py" --intrinsics "D:\Desktop\CS\具身智能\lec5\课堂练习\data\20261012_200142_186137\intrinsics.json" %*
pause
