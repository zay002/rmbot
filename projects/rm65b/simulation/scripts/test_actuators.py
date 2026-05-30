import time

import mujoco
import mujoco.viewer
import numpy as np

# --- 测试控制参数 ---
# 每个测试阶段的移动时间（秒）
MOVE_DURATION = 3.0
# 平滑系数：数值越小，动作越平滑。
SMOOTHNESS = 0.05

# --- 模型加载 ---
# 加载我们最终配置好的场景文件
try:
    # 注意：这里我们加载的是包含mocap体的场景文件
    model = mujoco.MjModel.from_xml_path("projects/rm65b/simulation/assets/scenes/scene_rm65b.xml")
    data = mujoco.MjData(model)
except Exception as e:
    print(f"模型加载失败: {e}")
    print("请确保您正在加载 'scene_rm65b.xml' 并且所有相关文件路径正确。")
    exit()

# --- 获取Mocap控制所需的ID ---
try:
    # 机械臂的末端法兰盘
    eef_body_id = model.body("flange").id
    # 我们用于控制的“幽灵”mocap体
    mocap_body_id = model.body("gripper_mocap").id
except KeyError as e:
    print(f"XML中找不到必要的body: {e}。请确保XML文件正确。")
    exit()


# --- 修正后的关节/执行器信息获取函数 ---
def get_arm_joints(model):
    """动态获取所有被执行器直接驱动的【机械臂关节】信息。"""
    joints = []
    print("正在扫描机械臂关节...")
    for i in range(model.nu):
        if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
            joint_id = model.actuator_trnid[i, 0]
            joint_name = model.joint(joint_id).name
            limited = model.jnt_limited[joint_id]
            jnt_range = model.jnt_range[joint_id] if limited else [-np.pi, np.pi]
            joints.append(
                {"name": joint_name, "id": joint_id, "limited": limited, "range": jnt_range, "actuator_id": i}
            )
            print(f"  发现关节: {joint_name}, Actuator ID: {i}")
    return joints


def find_gripper_actuator(model):
    """专门查找驱动tendon的【夹爪执行器】。"""
    print("正在扫描夹爪执行器...")
    for i in range(model.nu):
        if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_TENDON:
            actuator_name = model.actuator(i).name
            print(f"  发现夹爪执行器: {actuator_name}, Actuator ID: {i}")
            return i, model.actuator(i).ctrlrange
    return None, None


# 获取机械臂关节和夹爪执行器
arm_joints = get_arm_joints(model)
gripper_actuator_id, gripper_ctrl_range = find_gripper_actuator(model)

# 保存初始姿态作为“归位”点
home_qpos = data.qpos.copy()
# 初始化当前姿态和目标姿态
current_pos = home_qpos.copy()
target_pos = home_qpos.copy()


# --- 仿真主循环 ---
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("\n=== 开始关节顺序测试 (Mocap控制已激活)，程序将循环执行，按ESC退出 ===")

    while viewer.is_running():
        # === 第1部分：测试机械臂的6个关节 ===
        for joint_to_test in arm_joints:
            if not viewer.is_running():
                break
            print(f"\n--- 现在测试机械臂关节: [ {joint_to_test['name']} ] ---")

            # 移动到最小值
            print("--> 移动到 [最小值]")
            target_pos = home_qpos.copy()
            target_pos[joint_to_test["id"]] = joint_to_test["range"][0]
            start_time = time.time()
            while time.time() - start_time < MOVE_DURATION and viewer.is_running():
                # Mocap逻辑必须在每一步都执行
                eef_pos = data.body(eef_body_id).xpos
                eef_quat = data.body(eef_body_id).xquat
                data.mocap_pos[0][:] = eef_pos
                data.mocap_quat[0][:] = eef_quat

                current_pos = current_pos * (1 - SMOOTHNESS) + target_pos * SMOOTHNESS
                for i in range(len(arm_joints)):
                    data.ctrl[arm_joints[i]["actuator_id"]] = current_pos[arm_joints[i]["id"]]

                mujoco.mj_step(model, data)
                viewer.sync()

            # 移动到最大值
            if not viewer.is_running():
                break
            print("--> 移动到 [最大值]")
            target_pos[joint_to_test["id"]] = joint_to_test["range"][1]
            start_time = time.time()
            while time.time() - start_time < MOVE_DURATION * 1.5 and viewer.is_running():
                # Mocap逻辑必须在每一步都执行
                eef_pos = data.body(eef_body_id).xpos
                eef_quat = data.body(eef_body_id).xquat
                data.mocap_pos[0][:] = eef_pos
                data.mocap_quat[0][:] = eef_quat

                current_pos = current_pos * (1 - SMOOTHNESS) + target_pos * SMOOTHNESS
                for i in range(len(arm_joints)):
                    data.ctrl[arm_joints[i]["actuator_id"]] = current_pos[arm_joints[i]["id"]]

                mujoco.mj_step(model, data)
                viewer.sync()

        # === 第2部分：测试夹爪 ===
        if gripper_actuator_id is not None:
            if not viewer.is_running():
                break
            print("\n--- 现在测试夹爪: [ fingers_actuator ] ---")

            # 归位所有臂部关节
            print("--> 臂部归位，准备测试夹爪")
            target_pos = home_qpos.copy()
            start_time = time.time()
            while time.time() - start_time < MOVE_DURATION and viewer.is_running():
                # Mocap逻辑必须在每一步都执行
                eef_pos = data.body(eef_body_id).xpos
                eef_quat = data.body(eef_body_id).xquat
                data.mocap_pos[0][:] = eef_pos
                data.mocap_quat[0][:] = eef_quat

                current_pos = current_pos * (1 - SMOOTHNESS) + target_pos * SMOOTHNESS
                for i in range(len(arm_joints)):
                    data.ctrl[arm_joints[i]["actuator_id"]] = current_pos[arm_joints[i]["id"]]

                mujoco.mj_step(model, data)
                viewer.sync()

            # 闭合夹爪 (直接控制)
            if not viewer.is_running():
                break
            print("--> [闭合] 夹爪")
            data.ctrl[gripper_actuator_id] = gripper_ctrl_range[1]  # 设置为控制范围的最大值
            start_time = time.time()
            while time.time() - start_time < MOVE_DURATION and viewer.is_running():
                # Mocap逻辑必须在每一步都执行
                eef_pos = data.body(eef_body_id).xpos
                eef_quat = data.body(eef_body_id).xquat
                data.mocap_pos[0][:] = eef_pos
                data.mocap_quat[0][:] = eef_quat

                mujoco.mj_step(model, data)
                viewer.sync()

            # 张开夹爪 (直接控制)
            if not viewer.is_running():
                break
            print("--> [张开] 夹爪")
            data.ctrl[gripper_actuator_id] = gripper_ctrl_range[0]  # 设置为控制范围的最小值
            start_time = time.time()
            while time.time() - start_time < MOVE_DURATION and viewer.is_running():
                # Mocap逻辑必须在每一步都执行
                eef_pos = data.body(eef_body_id).xpos
                eef_quat = data.body(eef_body_id).xquat
                data.mocap_pos[0][:] = eef_pos
                data.mocap_quat[0][:] = eef_quat

                mujoco.mj_step(model, data)
                viewer.sync()

        print("\n=== 所有部件测试完毕，2秒后开始下一轮循环 ===")
        time.sleep(2)
