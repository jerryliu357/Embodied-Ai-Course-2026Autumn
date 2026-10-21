"""发送给LLM的六个工具定义；不执行模型生成的Python代码。"""
def tool(name, description, properties=None, required=None):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties or {},
                       "required": required or [], "additionalProperties": False}}}


TOOLS = [
    tool("get_robot_state", "读取机械臂实测状态或运动中最新缓存：六关节读数、角度、底座系末端位置(cm)、追踪状态、任务结果、已录制姿态和错误。模拟模式明确标记simulated。"),
    tool("detect_object", "用腕部相机寻找目标。ArUco用OpenCV返回真实ID、中心、距离；普通物体用视觉模型识别，trackable=false。只有ArUco可追踪。",
         {"description": {"type": "string", "description": "例如 ArUco 0、红色杯子。ArUco未指定ID时默认0。"}}, ["description"]),
    tool("start_tracking", "后台启动ArUco视觉伺服，不支持其他目标。必须使用检测结果中的ArUco ID；启动后返回running，不表示已达到目标。",
         {"object_id": {"type": "integer", "minimum": 0, "maximum": 49},
          "target_distance_cm": {"type": "number", "exclusiveMinimum": 0, "description": "相机到ArUco的目标光轴距离，厘米，默认20"}}, ["object_id"]),
    tool("stop_tracking", "取消正在运行的ArUco追踪，等待停止并保持当前姿态。可重复调用。切换到IK/FK前先调用此工具。"),
    tool("ik_move", "执行末端相对位移，固定底座坐标X前/Y左/Z上，单位厘米。使用已有IK与轨迹控制；不猜舵机读数，不固定末端朝向，保留腕旋转和夹爪。返回真实到达或部分完成结果。",
         {"dx_cm": {"type": "number"}, "dy_cm": {"type": "number"}, "dz_cm": {"type": "number"},
          "speed_cm_s": {"type": "number", "exclusiveMinimum": 0, "description": "可选，默认沿用第四课1 cm/s"}}, ["dx_cm", "dy_cm", "dz_cm"]),
    tool("fk_move", "执行关节空间姿态运动并返回FK末端位置。pose_name和joints_raw必须二选一。优先使用get_robot_state列出的有效预设；joints_raw仅用于用户明确给出的六关节原始读数，不编造未录制动作。",
         {"pose_name": {"type": "string", "description": "已保存的姿态名称，例如middle"},
          "joints_raw": {"type": "array", "items": {"type": "integer"}, "minItems": 6, "maxItems": 6,
                         "description": "按ID1~6排序的六个绝对原始计数；不是角度"},
          "duration_s": {"type": "number", "exclusiveMinimum": 0, "description": "请求时长，默认5秒，按已有速度设置必要时延长"}}),
]

TOOL_NAMES = tuple(item["function"]["name"] for item in TOOLS)

