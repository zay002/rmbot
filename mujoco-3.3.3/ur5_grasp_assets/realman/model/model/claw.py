############控制夹爪程序#############
from Robotic_Arm.rm_robot_interface import *
def control_gripper_open(arm, port, baudrate, timeout, modbus_address, amplitude):
    """
    控制夹爪张开的函数
    :param arm: RoboticArm 类的实例
    :param port: 通讯端口号（0: 控制器RS485端口为主站）
    :param baudrate: 波特率
    :param timeout: 超时时间（单位：百毫秒）
    :param modbus_address: Modbus设备地址
    :param amplitude: 夹爪幅度值（范围0-100）
    :return: None
    """
    # 配置控制器RS485端口为RTU主站
    if arm.rm_set_modbus_mode(port, baudrate, timeout) != 0:
        print("配置Modbus RTU模式失败")
        return

    # 关闭自动找行程指令
    write_params_auto = rm_peripheral_read_write_params_t(port, 0x9C9A, modbus_address)  # 寄存器地址为0x9C9A
    if arm.rm_write_single_register(write_params_auto, 0) != 0:
        print("关闭自动找行程失败")
        return

    # 创建写入参数结构体
    write_params = rm_peripheral_read_write_params_t(port, 0x9C40, modbus_address)  # 寄存器地址为0x9C40

    # 写入夹爪幅度控制寄存器
    result = arm.rm_write_single_register(write_params, amplitude)
    if result == 0:
        print(f"夹爪控制成功，幅度设置为：{amplitude}")
    else:
        print("夹爪控制失败")

def main():
    # 实例化RoboticArm类
    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)

    # 创建机械臂连接
    handle = arm.rm_create_robot_arm("192.168.1.19", 8080)#机器人的ip和端口号
    print(f"连接ID: {handle.id}")

    # 控制夹爪张开
    control_gripper_open(arm, port=1, baudrate=115200, timeout=1, modbus_address=1, amplitude=1)#一定是1，最开始搞错了；amlitude设置为0闭合，100张开

    # 关闭机械臂连接
    arm.rm_delete_robot_arm()

if __name__ == "__main__":
    main()



# #####状态查询######
# from Robotic_Arm.rm_robot_interface import *
#
# # 实例化RoboticArm类
# arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
#
# # 创建机械臂连接，打印连接id
# handle = arm.rm_create_robot_arm("192.168.1.19", 8080)
# print(handle.id)
#
# print(arm.rm_get_current_arm_state())
#
# arm.rm_delete_robot_arm()





################夹爪###########
# from Robotic_Arm.rm_robot_interface import *
# import time
#
# TARGET_POSE = [
#     -0.14491,
#     0.334465,
#     0.044263,
#     3.101,
#     -0.432,
#     1.654
# ]
# # "robot_x": -0.01150901126906425,X的值-0.1334
# # "robot_y": -0.2585567959257364,Y的值+0.5930218
# # "robot_z": 0.07135618461178243,Z的值-0.02709318
# def validate_coordinates(coords):
#     """坐标验证函数"""
#     if not all(isinstance(v, (int, float)) for v in coords[:3]):
#         raise ValueError("坐标值必须为数值类型")
#     if any(abs(v) > 10 for v in coords[:3]):
#         raise ValueError("坐标值超出安全范围")
#
# def execute_safe_movement(arm, target_pose, speed_percent):
#     """安全移动封装函数"""
#     validate_coordinates(target_pose)
#     result = arm.rm_movel(
#         pose=target_pose,
#         v=int(speed_percent),  # 接收3%参数
#         r=0,
#         connect=0,
#         block=1
#     )
#     if result != 0:
#         raise RuntimeError(f"移动失败，错误码: {result}")
#     return True
#
# def main():
#     arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
#     try:
#         handle = arm.rm_create_robot_arm("192.168.1.19", 8080)
#         print(f"机械臂连接成功 | 句柄ID: {handle.id}")
#
#         # 移动到目标位姿（速度3%）
#         print("正在移动到初始目标位姿...")
#         if arm.rm_movej_p(pose=TARGET_POSE, v=3, r=0, connect=0, block=1) != 0:  # 速度修改
#             raise RuntimeError("初始位姿设置失败")
#         time.sleep(2)
#
#         # X轴移动（速度3%）
#         offset = 0.08
#         new_pose = [
#             TARGET_POSE[0] + offset,
#             TARGET_POSE[1],
#             TARGET_POSE[2],
#             TARGET_POSE[3],
#             TARGET_POSE[4],
#             TARGET_POSE[5]
#         ]
#         print("执行X轴正向直线运动...")
#         execute_safe_movement(arm, new_pose, speed_percent=3)  # 速度修改
#
#         print("所有运动指令执行完成")
#
#     except RuntimeError as re:
#         print(f"机械臂操作异常: {str(re)}")
#         arm.rm_stop(0)
#     except Exception as e:
#         print(f"系统错误: {str(e)}")
#     finally:
#         arm.rm_delete_robot_arm()
#         print("连接已安全释放")
#
# if __name__ == "__main__":
#     main()

# from Robotic_Arm.rm_robot_interface import *
# import json
# import time
#
#
# def validate_coordinates(coords):
#     """坐标验证函数"""
#     if not all(isinstance(v, (int, float)) for v in coords[:3]):
#         raise ValueError("坐标值必须为数值类型")
#     if any(abs(v) > 10 for v in coords[:3]):
#         raise ValueError("坐标值超出安全范围")
#
#
# def execute_safe_movement(arm, target_pose, speed_percent):
#     """安全移动封装函数"""
#     validate_coordinates(target_pose)
#     result = arm.rm_movel(
#         pose=target_pose,
#         v=int(speed_percent),
#         r=0,
#         connect=0,
#         block=1
#     )
#     if result != 0:
#         raise RuntimeError(f"移动失败，错误码: {result}")
#     return True
#
#
# def main():
#     arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
#     try:
#         # 初始化连接
#         handle = arm.rm_create_robot_arm("192.168.1.19", 8080)
#         print(f"机械臂连接成功 | 句柄ID: {handle.id}")
#
#         # 加载坐标数据
#         data_path = r"C:\Users\11193\Desktop\pic3\final_results.json"  # 修改文件路径
#         with open(data_path, 'r') as f:
#             hole_data = json.load(f)
#
#         # 获取目标孔位
#         target_hole = next((h for h in hole_data['holes'] if h['id'] == 6), None)
#         if not target_hole:
#             raise ValueError("ID=6的螺丝孔未找到")
#
#         # 构建目标坐标（应用偏移量）
#         hole_x = target_hole['robot_x']
#         hole_y = target_hole['robot_y']
#         hole_z = target_hole['robot_z']
#
#         TARGET_POSE = [
#             hole_x - 0.1334,  # X轴偏移
#             hole_y + 0.5930218,  # Y轴偏移
#             hole_z - 0.02709318,  # Z轴偏移
#             3.101,  # Rx保持原有姿态
#             -0.432,  # Ry保持原有姿态
#             1.654  # Rz保持原有姿态
#         ]
#
#         # 移动到目标位姿
#         print("正在移动到计算目标位姿...")
#         if arm.rm_movej_p(pose=TARGET_POSE, v=3, r=0, connect=0, block=1) != 0:
#             raise RuntimeError("目标位姿设置失败")
#         time.sleep(2)
#
#         # 沿基坐标系X轴正方向移动7cm
#         offset = 0.07  # 单位：米
#         new_pose = [
#             TARGET_POSE[0] + offset,  # 仅修改X坐标
#             TARGET_POSE[1],
#             TARGET_POSE[2],
#             TARGET_POSE[3],
#             TARGET_POSE[4],
#             TARGET_POSE[5]
#         ]
#         print("执行X轴正向直线运动...")
#         execute_safe_movement(arm, new_pose, speed_percent=3)
#
#         print("所有操作成功完成")
#
#     except json.JSONDecodeError:
#         print("错误：JSON文件格式无效")
#     except IOError as e:
#         print(f"文件操作错误: {str(e)}")
#     except RuntimeError as re:
#         print(f"机械臂操作异常: {str(re)}")
#         arm.rm_stop(0)
#     except Exception as e:
#         print(f"系统错误: {str(e)}")
#     finally:
#         arm.rm_delete_robot_arm()
#         print("连接已安全释放")
#
#
# if __name__ == "__main__":
#     main()






