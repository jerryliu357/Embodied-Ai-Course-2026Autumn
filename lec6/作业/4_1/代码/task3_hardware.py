"""任务三的电源诊断与退出卸力；不改 EEPROM、报警阈值或旧课驱动。"""

import time


def read_diagnostic(bus, name, register):
    """仅用于诊断/卸力确认：保留硬件报警位，不把通信失败的数据当作读数。"""
    motor = bus.motors[name]
    address, length = bus.model_ctrl_table[motor.model][register]
    value, comm, alarm = bus._read(address, length, motor.id, num_retry=1, raise_on_error=False)
    return {
        "value": int(value) if bus._is_comm_success(comm) else None,
        "communication_result": int(comm), "servo_alarm": int(alarm),
    }


def power_snapshot(bus):
    readings = []
    registers = ("Present_Voltage", "Min_Voltage_Limit", "Max_Voltage_Limit",
                 "Torque_Enable", "Present_Temperature", "Status")
    for name, motor in bus.motors.items():
        row = {"id": motor.id, "name": name}
        for register in registers:
            try:
                row[register] = read_diagnostic(bus, name, register)
            except Exception as exc:
                row[register] = {"value": None, "error": str(exc)}
        readings.append(row)
    return readings


def print_power_snapshot(readings):
    print("舵机电源诊断（以下是舵机自身反馈）：")
    for row in readings:
        def voltage(register):
            value = row[register]["value"]
            return "读取失败" if value is None else f"{value / 10:.1f}V"
        alarms = sorted({value.get("servo_alarm", 0) for value in row.values() if isinstance(value, dict)} - {0})
        print(f"  ID{row['id']}：电压 {voltage('Present_Voltage')}，"
              f"报警范围 {voltage('Min_Voltage_Limit')}~{voltage('Max_Voltage_Limit')}，"
              f"扭矩 {row['Torque_Enable']['value']}，状态 {row['Status']['value']}，响应报警 {alarms or [0]}")


def release_torque(bus, sleep=time.sleep):
    """先向六关节同步写 Torque_Enable=0，再独立读回确认。

    电压报警不妨碍先发卸力命令；读回带报警且通信成功时仍可确认扭矩值。
    不能将成功发送广播等同于已确认卸力。
    """
    result = {"confirmed": False, "motors": {}, "send_errors": []}
    if bus is None or not bus.is_connected:
        result["send_errors"].append("串口未连接，无法发送或确认卸力。")
        return result
    for _ in range(3):
        try:
            bus.sync_write("Torque_Enable", dict.fromkeys(bus.motors, 0), normalize=False, num_retry=2)
        except Exception as exc:
            result["send_errors"].append(str(exc))
        sleep(0.05)
        for name in bus.motors:
            try:
                result["motors"][name] = read_diagnostic(bus, name, "Torque_Enable")
            except Exception as exc:
                result["motors"][name] = {"value": None, "error": str(exc)}
        if all(row["value"] == 0 for row in result["motors"].values()) and result["motors"]:
            result["confirmed"] = True
            return result
    return result


def print_release_result(result):
    if result["confirmed"]:
        print("已确认六个舵机 Torque_Enable=0：退出卸力，不再保持锁定姿态。")
    else:
        missing = [name for name, row in result["motors"].items() if row["value"] != 0]
        print(f"卸力未能全部确认：{missing or result['send_errors']}。请托稳机械臂后断开舵机电源。")
