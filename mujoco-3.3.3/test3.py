import mujoco
import mujoco.viewer
import numpy as np
import time
import random

# 加载模型
model = mujoco.MjModel.from_xml_path("ur5_grasp_assets/robotiq_2f85/2f85.xml")
data = mujoco.MjData(model)

def get_gripper_position():
    """获取夹爪开合位置（0=完全闭合，1=完全打开）"""
    # 读取右侧驱动关节位置（范围0~0.8）
    right_pos = data.joint("right_driver_joint").qpos[0]
    # 标准化到[0,1]范围
    normalized_pos = right_pos / 0.8
    return normalized_pos

def print_actuator_info():
    """打印执行器和关节状态"""
    print("\n=== 夹爪状态 ===")
    # 执行器控制值（0~255）
    actuator_value = data.actuator("fingers_actuator").ctrl[0]
    print(f"执行器控制值: {actuator_value:.1f} (范围: 0~255)")
    
    # 驱动关节实际位置
    right_pos = np.degrees(data.joint("right_driver_joint").qpos[0])
    left_pos = np.degrees(data.joint("left_driver_joint").qpos[0])
    print(f"驱动关节位置: 右={right_pos:.1f}°, 左={left_pos:.1f}°")
    
    # 标准化开合位置
    print(f"夹爪开合度: {get_gripper_position()*100:.1f}%")

# 交互式控制
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("按ESC退出...")
    last_print = time.time()
    
    while viewer.is_running():
        # 示例：周期性地设置随机开合度
        if time.time() - last_print > 1.0:
            target = random.uniform(0, 255)  # 随机控制值
            data.ctrl[0] = target  # 设置执行器
            print_actuator_info()
            last_print = time.time()
        
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.001)