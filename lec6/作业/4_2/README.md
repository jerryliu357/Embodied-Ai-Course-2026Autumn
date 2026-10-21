# 4.2 机械臂工具调用 Agent

使用硅基流动 `Qwen/Qwen3.5-4B`，深度思考关闭。Key读取环境变量 `SILICONFLOW_API_KEY`，也可填写 `../4_1/代码/task2_2.py` 的 `API_KEY`。

在本目录打开PowerShell：

```powershell
# 真机：启动后在终端输入自然语言指令
& D:\miniconda\envs\lerobot\python.exe .\agent.py

# 真实LLM + 模拟机器人/合成ArUco，不连接硬件
& D:\miniconda\envs\lerobot\python.exe .\agent.py --dry-run

# 验证真实API工具调用 / 批量测试保存的指令
& D:\miniconda\envs\lerobot\python.exe .\agent.py --test-api
& D:\miniconda\envs\lerobot\python.exe .\agent.py --dry-run --commands .\test_commands.txt
```

`/help`查看示例，`Ctrl+C`停止当前运动，`/quit`退出并卸力。运动结束后保持姿态，供后续指令继续使用。使用原串口自动识别，也可指定 `--port COM9`。

## 六个工具

| 工具                                                 | 用途                                                  |
| ---------------------------------------------------- | ----------------------------------------------------- |
| `get_robot_state()`                                | 关节读数、角度、FK末端位置、追踪状态、有效预设和错误  |
| `detect_object(description)`                       | ArUco用OpenCV；普通物体用同一视觉模型识别，但不可追踪 |
| `start_tracking(object_id, target_distance_cm=20)` | 只追踪DICT_4X4_50的ArUco；后台运行                    |
| `stop_tracking()`                                  | 停止追踪并保持当前位置                                |
| `ik_move(dx_cm, dy_cm, dz_cm)`                     | 末端相对位移，单位cm；底座X前、Y左、Z上               |
| `fk_move(pose_name=... 或 joints_raw=[...])`       | 预设或六关节绝对读数，二选一；返回FK末端位置          |

## 流程与验收

自然语言 → LLM返回原生 `tool_calls` → 本地按名称调用六个工具 → 真实结果以 `role=tool` 和对应 `tool_call_id`发回模型 → 模型继续调用或回复。追踪、IK、FK共用一个串口线程；切换运动前先停止追踪。

`prompt.txt`是系统提示词，`tool_specs.py`是工具定义，`robot_tools.py`是执行层，`agent.py`是对话循环。每次运行保存在 `runs/时间戳/`：提示词、工具定义、`events.jsonl`调用记录、`conversation.json`对话、检测图片和运动日志。

纯离线检查和测试：

```powershell
& D:\miniconda\envs\lerobot\python.exe .\agent.py --check
& D:\miniconda\envs\lerobot\python.exe -m unittest discover -s . -p test_agent.py
```

API调用格式参考：[硅基流动Function Calling](https://docs.siliconflow.cn/docs/userguide/guides/function-calling)。
