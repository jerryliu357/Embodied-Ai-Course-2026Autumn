# FK：预设姿态与动作

本部分已完成课堂验收。程序通过六个关节的0～1比例，完成姿态示教、保存和动作回放。

## 运行

在本文件夹打开终端，使用已有的 `lerobot` 环境：

```powershell
D:\miniconda\envs\lerobot\python.exe control_fk.py
```

其他电脑需要 Python 3.12 及以上（含 Tkinter），并安装上一级的 `requirements.txt`。驱动源码已附带，不需要原课堂练习目录。

默认识别原转接板；需要时添加 `--port COM9`，号码以实际串口为准。启动只读当前位置，不自动复位。

无硬件时可查看界面，或仅检查配置：

```powershell
D:\miniconda\envs\lerobot\python.exe control_fk.py --demo
D:\miniconda\envs\lerobot\python.exe control_fk.py --check-files
```

`--demo` 不打开串口，记录只保留在内存。

## 操作

1. 拖动六个滑条，或输入0～1数值并点“设置”，调整关节。
2. 选择姿态名称，停稳后点“记录当前实测姿态”，将六个反馈位置保存到 `actions.json`。
3. 选择姿态并播放；动作区可以编辑各段姿态、运动时间和停留时间。
4. 点击非输入框区域后，按 **1** 播放立正，按 **2** 播放挥手；**空格**暂停并保持，**Esc**全部卸力（先托稳机械臂）。关闭窗口保持已控制的关节。

**当前数据状态：**ID5重新安装后，按此前要求清除了旧姿态，仅保留更新后的 `middle`。`stand / rest / lift / left / right / zero / rotated` 需要重新录制；缺少姿态时，程序会提示，不会播放不完整的动作。代码可以启动、调节滑条及录制，但现有JSON不能直接播放立正或完整挥手。

当前挥手顺序为 `rest → lift → left → right → rest`，每段5秒；以 `actions.json` 为准。

也可以托稳并在界面卸力、退出后，手摆姿态，用终端逐个记录：

```powershell
D:\miniconda\envs\lerobot\python.exe record_poses_cli.py
```

按提示摆好整条手臂并回车。该脚本只读取位置，要求六个关节扭矩已关闭。

## 原理与文件

比例按标定端点线性映射成舵机读数；动作从实测起点出发，用三次平滑插值 `3s²−2s³` 更新六个关节的目标，每段到位后进入下一段。速度上限或跟随落后会延长用时。通信沿用已验收的 LeRobot SCS 适配，控制逻辑未因整理提交而改变。

| 文件 | 用途 |
| --- | --- |
| `control_fk.py` | 图形界面入口 |
| `motion.py` / `arm_backend.py` | 动作插值、串口线程、舵机通信与反馈 |
| `action_store.py` / `record_poses_cli.py` | 姿态校验、保存与终端录制 |
| `calibration.json` / `actions.json` | 当前标定、姿态和动作顺序 |
| `driver_loader.py` / `vendor/` | 提交目录内的驱动及支持模块 |

驱动文件取自已运行的 [LeRobot 源码版本](https://github.com/huggingface/lerobot/tree/89236ea0f4f81a81ca566081e20dd1ff5f823cbe)，保留许可证、来源与文件校验信息。两个包入口仅作精简打包，电机驱动原文未修改。
