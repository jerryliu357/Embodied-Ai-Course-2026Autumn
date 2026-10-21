# 第六课作业

以下命令在本文件所在目录打开 PowerShell 运行，使用 `lerobot` 环境。先设置硅基流动 API Key：

```powershell
$env:SILICONFLOW_API_KEY = "你的API Key"
```

## 4.1 视觉动作模仿

已课堂验收。代码在 `4_1/代码/`：`task1.py` 拍照，`task2_2.py` 调用硅基流动分析图片，`task3.py` 为拍照、动作理解和实机控制主程序；提示词在 `task3.py`，动作映射说明在 `notes.txt`。

```powershell
& D:\miniconda\envs\lerobot\python.exe .\4_1\代码\task3.py
```

## 4.2 工具调用 Agent

主程序：`4_2/agent.py`；提示词：`4_2/prompt.txt`；工具定义：`4_2/tool_specs.py`；用法：`4_2/README.md`。启动后在终端输入指令，`/quit` 退出。

```powershell
& D:\miniconda\envs\lerobot\python.exe .\4_2\agent.py
```

## 4.3 视觉语音 Agent

主程序：`4_3/gui.py`；提示词：`4_3/omni_prompt.txt`、`4_3/planner_prompt.txt`，并复用 4.2 的提示词和工具；用法及示例：`4_3/README.md`；演示视频：`4_3/4_3题演示视频.mp4`。真机运行：

```powershell
& D:\miniconda\envs\lerobot\python.exe .\4_3\gui.py --real
```

在窗口选择麦克风，点击“开始录音”，说完点击“结束录音并发送”；也可输入文字。4.2、4.3 使用机械臂腕部相机，4.1 使用电脑摄像头。

共用的舵机驱动、IK 和关节标定位于 `4_1/代码/dependencies/`；FK 姿态、视觉伺服代码和相机标定位于 `4_2/dependencies/`。请保持三个文件夹的相对位置。
