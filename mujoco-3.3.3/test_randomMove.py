import mujoco
import mujoco.viewer
import numpy as np
import time
import random
import argparse

# --- 1. 优化后的控制参数 (您可以在这里调整以获得不同效果) ---

# 平滑系数：数值越小，动作越平滑、越慢。建议范围: 0.01 ~ 0.1
SMOOTHNESS = 0.03

# 目标更新间隔 (秒)：每隔多少秒产生一个新的随机动作目标。数值越大，动作越“沉稳”。
TARGET_UPDATE_INTERVAL = 0.8

# 最大关节角速度 (弧度/秒)：控制随机动作的“幅度”或“激烈程度”。数值越小，动作幅度越小。
MAX_JOINT_VELOCITY = 0.4


# --- 无需改动的部分 ---

parser = argparse.ArgumentParser(description="Mujoco robot simulation.")
parser.add_argument(
    "--move",
    action="store_true",
    help="Set this flag to make the robot arm move randomly."
)
args = parser.parse_args()

# 加载模型
model = mujoco.MjModel.from_xml_path("ur5_grasp_assets/scenes/scene_rm65b.xml")
data = mujoco.MjData(model)

# 获取所有驱动关节信息
def get_actuated_joints(model):
    joints = []
    for i in range(model.nu):
        joint_id = model.actuator_trnid[i, 0]
        joint_name = model.joint(joint_id).name
        limited = model.jnt_limited[joint_id]
        jnt_range = model.jnt_range[joint_id] if limited else [-np.pi, np.pi]
        joints.append({
            "name": joint_name,
            "id": joint_id,
            "limited": limited,
            "range": jnt_range,
            "type": model.jnt_type[joint_id]
        })
    return joints

actuated_joints = get_actuated_joints(model)

print("=== 驱动关节信息 ===")
for i, joint in enumerate(actuated_joints):
    unit = "rad" if joint["type"] == 1 else "m"
    range_str = f"{joint['range'][0]:.2f}~{joint['range'][1]:.2f}{unit}"
    print(f"{i}: {joint['name']} (运动范围: {range_str})")

selected_joint_indices = [0, 1, 2, 3, 4, 5]
selected_joints = [actuated_joints[i] for i in selected_joint_indices]

# 初始化位置，让目标位置和当前位置都等于模型的初始位置
target_pos = data.qpos.copy()
current_pos = data.qpos.copy()

with mujoco.viewer.launch_passive(model, data) as viewer:
    if args.move:
        print("\n模式: 随机运动中 - 按ESC退出")
    else:
        print("\n模式: 保持静止 - 按ESC退出 (使用 --move 参数启动以使机械臂运动)")
        
    last_target_update_time = time.time()
    last_print_time = time.time()
    
    while viewer.is_running():
        
        # --- 2. 优化后的随机目标生成逻辑 ---
        if args.move:
            # 检查是否到了更新随机目标的时间
            if time.time() - last_target_update_time > TARGET_UPDATE_INTERVAL:
                # 根据最大角速度和更新间隔，计算本次更新的最大步长
                max_step = MAX_JOINT_VELOCITY * TARGET_UPDATE_INTERVAL
                
                # 对每个关节执行“随机游走”
                for joint in actuated_joints:
                    # 生成一个小的、随机的增量
                    delta = random.uniform(-max_step, max_step)
                    
                    # 将增量应用到当前目标位置上
                    new_target = target_pos[joint["id"]] + delta
                    
                    # 使用np.clip确保新目标不会超出关节的物理极限
                    target_pos[joint["id"]] = np.clip(new_target, joint["range"][0], joint["range"][1])
                
                last_target_update_time = time.time()
        
        # --- 3. 优化后的平滑控制 ---
        # 使用更小的平滑系数，让实际控制位置缓慢地向目标位置靠近
        current_pos = current_pos * (1 - SMOOTHNESS) + target_pos * SMOOTHNESS
        
        # 应用控制指令
        for i in range(model.nu):
            joint_id = model.actuator_trnid[i, 0]
            data.ctrl[i] = current_pos[joint_id]
        
        # 每秒打印选定关节位置
        if time.time() - last_print_time >= 1.0:
            print("\n=== 选定关节位置 ===")
            for joint in selected_joints:
                pos = data.qpos[joint["id"]]
                if joint["type"] == 1:
                    print(f"{joint['name']}: {np.degrees(pos):.2f}° (原始值: {pos:.3f}rad)")
                else:
                    print(f"{joint['name']}: {pos:.4f}m")
            last_print_time = time.time()
        
        # 步进仿真
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.001)