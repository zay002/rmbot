import mujoco
import mujoco.viewer
import numpy as np
import time
import random
import argparse

# --- 控制参数 ---
# 平滑系数：数值越小，动作越平滑、越慢。
SMOOTHNESS = 0.03
# 机械臂目标更新间隔 (秒)
ARM_TARGET_UPDATE_INTERVAL = 0.8
# 夹爪目标更新间隔 (秒)
GRIPPER_TARGET_UPDATE_INTERVAL = 2.5
# 最大关节角速度 (弧度/秒)
MAX_JOINT_VELOCITY = 0.4


# --- 参数解析 ---
parser = argparse.ArgumentParser(description="Mujoco robot simulation.")
parser.add_argument(
    "--move",
    action="store_true",
    help="Set this flag to make the robot arm move randomly."
)
args = parser.parse_args()

# --- 模型加载 ---
try:
    # 注意：这里加载的是不带mocap的场景文件
    model = mujoco.MjModel.from_xml_path("mujoco-3.3.3/ur5_grasp_assets/scenes/scene_aubo.xml")
    data = mujoco.MjData(model)
except Exception as e:
    print(f"模型加载失败: {e}")
    exit()


# --- 修正后的关节/执行器信息获取函数 ---
def get_arm_joints(model):
    """动态获取所有被执行器直接驱动的【机械臂关节】信息。"""
    joints = []
    for i in range(model.nu):
        if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
            joint_id = model.actuator_trnid[i, 0]
            joint_name = model.joint(joint_id).name
            limited = model.jnt_limited[joint_id]
            jnt_range = model.jnt_range[joint_id] if limited else [-np.pi, np.pi]
            joints.append({
                "name": joint_name, "id": joint_id, "limited": limited,
                "range": jnt_range, "actuator_id": i
            })
    return joints

def find_gripper_actuator(model):
    """专门查找驱动tendon的【夹爪执行器】。"""
    for i in range(model.nu):
        if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_TENDON:
            return i, model.actuator(i).ctrlrange
    return None, None

# 获取机械臂关节和夹爪执行器
arm_joints = get_arm_joints(model)
gripper_actuator_id, gripper_ctrl_range = find_gripper_actuator(model)
print(f"初始化成功：发现 {len(arm_joints)} 个臂部关节和 {1 if gripper_actuator_id is not None else 0} 个夹爪执行器。")


# 初始化位置
home_qpos = data.qpos.copy()
current_pos = home_qpos.copy()
target_pos = home_qpos.copy()
# 为夹爪单独设置一个控制目标
gripper_target_ctrl = 0


# --- 仿真主循环 ---
with mujoco.viewer.launch_passive(model, data) as viewer:
    if args.move:
        print("\n模式: 随机运动中 - 按ESC退出")
    else:
        print("\n模式: 保持静止 - 按ESC退出")
        
    last_arm_update_time = time.time()
    last_gripper_update_time = time.time()
    last_print_time = time.time()
    
    while viewer.is_running():
        step_start = time.time()

        # 随机运动逻辑
        if args.move:
            # 1. 随机更新【机械臂】的目标位置
            if time.time() - last_arm_update_time > ARM_TARGET_UPDATE_INTERVAL:
                max_step = MAX_JOINT_VELOCITY * ARM_TARGET_UPDATE_INTERVAL
                for joint in arm_joints:
                    delta = random.uniform(-max_step, max_step)
                    new_target = target_pos[joint["id"]] + delta
                    target_pos[joint["id"]] = np.clip(new_target, joint["range"][0], joint["range"][1])
                last_arm_update_time = time.time()
            
            # 2. 随机更新【夹爪】的目标状态 (开/合)
            if gripper_actuator_id is not None and time.time() - last_gripper_update_time > GRIPPER_TARGET_UPDATE_INTERVAL:
                # 在张开(min)和闭合(max)之间随机选择一个
                gripper_target_ctrl = random.choice(gripper_ctrl_range)
                print(f"夹爪新目标: {'闭合' if gripper_target_ctrl == gripper_ctrl_range[1] else '张开'}")
                last_gripper_update_time = time.time()

        # 平滑过渡 (只针对机械臂关节)
        current_pos = current_pos * (1 - SMOOTHNESS) + target_pos * SMOOTHNESS
        
        # 应用控制指令
        # 1. 应用机械臂的控制指令
        for joint in arm_joints:
            data.ctrl[joint["actuator_id"]] = current_pos[joint["id"]]
        
        # 2. 应用夹爪的控制指令
        if gripper_actuator_id is not None:
            data.ctrl[gripper_actuator_id] = gripper_target_ctrl
        
        # 每秒打印选定关节位置
        if time.time() - last_print_time >= 1.0:
            print("\n=== 选定关节位置 ===")
            for joint in arm_joints:
                pos = data.qpos[joint["id"]]
                print(f"{joint['name']}: {np.degrees(pos):.2f}° (原始值: {pos:.3f}rad)")
            last_print_time = time.time()

        # 步进与同步
        mujoco.mj_step(model, data)
        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)