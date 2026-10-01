# IK：末端直线运动

采用助教的 SCS215 驱动和 NumPy 数值求解器，不使用 Placo。程序控制夹爪坐标系 `gripper_frame_link` 的XYZ位置；ID5保持开始时的手腕旋转角，ID6保持夹爪开合，前四个关节由IK计算。

## 运行

在本文件夹打开 CMD 或 PowerShell：

```powershell
D:\miniconda\envs\lerobot\python.exe control_ik.py
```

使用已有 `lerobot` 环境。其他电脑需 Python 3.12 及以上，并安装上一级的 `requirements.txt`；程序、驱动、URDF和标定均在本目录内。

默认识别原转接板，必要时添加 `--port COM9`。启动只读，从当前位置开始，不先复位。关闭其他占用串口的程序，FK和IK分别运行。

## 命令与按键

**每次输入一行，等运动结束再输入下一行；距离单位为厘米。**方向是固定底座坐标：前+X、左+Y、上+Z，不随夹爪朝向改变。

| 输入 | 含义 |
| --- | --- |
| `forward 5` / `back 5` | 前进 / 后退5cm |
| `left 5` / `right 5` | 向左 / 向右5cm |
| `up 5` / `down 5` | 向上 / 向下5cm |
| `move 1 2 3` | 沿直线前1cm、左2cm、上3cm |
| `speed 1` | 推进速度上限1cm/s；跟不上时自动减速 |
| `status` / `diagnose` | 查看当前位置 / 舵机寄存器反馈 |
| `stop` | 停止并保持当前读数 |
| `off` | 六关节卸力，先托稳机械臂 |
| `quit` | 退出并保留目标、扭矩状态 |

依次输入 `step 1`、`jog` 可进入键盘模式，每次按键移动1cm：**W/S**前后、**A/D**左右、**R/F**上下、**Q**返回。`step` 可设为0.1～5cm；运动中空格、Esc、Q或Ctrl+C停止并保持，积压按键不会回放。

## 原理

`servo_info.json` 将读数转换成模型角度，URDF提供各关节几何变换。NumPy求解带关节范围约束的阻尼最小二乘IK，得到直线路径上各点的关节角。执行时保留连续的关节解，利用冗余自由度避开限位；根据位置反馈减慢推进，并用有上限的积分补偿减小持续误差。只控制末端位置，不固定夹爪完整朝向。

局部求解失败时提示“IK无解”，尝试缩短同方向步长；仍不可达时寻找保留前进分量的可行方向，并明确报告偏离原直线。完全无法前进或持续停滞则停止。反馈模型坐标不等同于尺子测量，程序未实现碰撞检测。

## 检查与文件

```powershell
# 离线检查依赖和配置，不打开串口
D:\miniconda\envs\lerobot\python.exe control_ik.py --check
# 连接后只读检查，不运动
D:\miniconda\envs\lerobot\python.exe control_ik.py --read-only
```

| 文件 | 用途 |
| --- | --- |
| `control_ik.py` | CMD命令与键盘入口 |
| `ik_solver.py` | URDF正向运动学、雅可比、数值IK |
| `trajectory.py` / `tracking.py` | 路径规划、反馈调速和有限积分补偿 |
| `robot_adapter.py` | 标定转换、驱动加载、读写舵机 |
| `config.json` / `servo_info.json` | 运行参数与本机械臂标定 |
| `so101_new_calib.urdf` | SO101几何模型；本程序不加载网格文件 |
| `tracking_validation.json` | 实机测试记录，包含失败尝试及最终结果 |
| `vendor/` | 助教驱动、LeRobot支持模块及许可证 |

运行后只覆盖 `last_motion.json`；模拟时覆盖 `last_simulation.json`。当前标定沿用330°/1024。前后左右各50mm的最终实测反馈估计为50.6、52.5、52.1、51.9mm，最大横向偏差约6.5～9.5mm；这些结果来自位置反馈与模型，尚未用尺子确认物理精度。验收或录视频时同时展示实际运动与终端输出。

驱动来自[助教仓库固定版本](https://github.com/JyiDeng/lerobot_EAI_SCS215/tree/bd2ee0bd961904090873d208fe79dc0ab3a4f863)，原文未修改；必要的LeRobot支持模块一同打包，来源及校验见 `vendor` 中的 `SOURCE.json`、`RUNTIME_SOURCE.json`。IK和反馈控制算法由本程序用NumPy实现。

URDF来自课程指定的 [SO101 模型文件](https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf)，使用此前下载并验证的本地副本。
