# 内参标定

双击 start_calibration.cmd 启动中文预览，关闭其他占用USB相机的软件。

- 只打开 GENERAL WEBCAM，相机原生分辨率1280×720，不控制机器人。
- 棋盘格：7×9个格子、6×8个内角点，共48点；用户已确认单格25 mm。
- 全部棋盘格入镜、停稳、出现彩色角点后，按空格或点击“保存照片”。
- 采集20～25张（至少15张），覆盖中央、四周、远近和左右/上下倾角，避免只拍正面。纸张平整固定在硬板上。
- 点击“计算内参”，或按C。完成后显示误差并自动保存参数。补拍后必须重新计算才能包含新照片。
- 退出关闭预览并释放相机，已采照片保留；每次启动建立新的采集目录，不覆盖旧照片。

代码使用 lerobot：D:\miniconda\envs\lerobot\python.exe。
预览使用Tkinter/Pillow，图像处理使用OpenCV，不依赖cv2.imshow。

## 保存内容

每次采集位于本目录的 data/日期时间/：

- session.json：相机、分辨率、格子尺寸。
- frame_*.png：原始相机图，不含角点或文字。
- corners/：对应角点标注截图。
- intrinsics.json：相机内参K、畸变参数、整体与逐图重投影RMS、棋盘格在相机下的位姿（平移单位米）。这些位姿不是手眼标定结果。
- reprojection_errors.csv：每张图误差，不自动删除误差大的照片。
- original_vs_undistorted.jpg：左原图，右去畸变图，保留原始K及分辨率、不裁剪。
- camera.log：采集诊断。

误差定义：sqrt(mean(du²+dv²))，单位像素，与OpenCV calibrateCamera的RMS一致。讲义要求约1.0像素；小误差不单独代表通过，还需要照片位置/倾角多样、棋盘格平整及去畸变效果合理。程序输出覆盖不足等提示，不自动宣布验收通过。

标定和后续使用保持相同相机、分辨率、缩放和焦距；能固定对焦时请固定。程序不自动调整对焦。此阶段可以移动棋盘格，不需要机械臂运动；小ArUco卡片暂时不用。

## 离线重算

在PowerShell中（替换实际采集目录）：

```powershell
& D:\miniconda\envs\lerobot\python.exe "D:\Desktop\CS\具身智能\lec5\课堂练习\calibrate_intrinsics.py" --calibrate "D:\Desktop\CS\具身智能\lec5\课堂练习\data\实际目录"
```

--probe 读取一帧并检测后释放相机，诊断图保存在 diagnostics/。
--smoke-test 检查Tk/Pillow界面库，不连接相机。

使用依据：https://docs.opencv.org/4.13.0/dc/dbb/tutorial_py_calibration.html
