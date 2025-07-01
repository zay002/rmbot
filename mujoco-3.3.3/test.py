import mujoco
import mujoco.viewer
import numpy as np

def print_all_joint_positions(model, data):
    """打印所有关节的当前位置（紧凑格式）"""
    print(f"\n时间: {data.time:.1f}s | ", end="")
    for i in range(model.nq):
        joint_name = model.joint(i).name if i < model.njnt else f"qpos_{i}"
        pos = data.qpos[i]
        unit = "rad" if model.jnt_type[i] == 1 else "m"
        print(f"{joint_name}: {pos:6.3f}{unit} | ", end="")

def print_specific_joints(model, data, target_joint_names):
    """只打印目标关节的信息"""
    print(f"\n时间: {data.time:.2f}s")
    for name in target_joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id == -1:
            print(f"警告: 未找到关节 {name}")
            continue
            
        qpos_adr = model.jnt_qposadr[joint_id]  # 获取该关节在qpos中的地址
        jnt_type = model.jnt_type[joint_id]
        
        if jnt_type == 1:  # 旋转关节
            pos = data.qpos[qpos_adr]
            print(f"{name}: {np.degrees(pos):.2f}°")  # 转换为角度
        elif jnt_type == 2:  # 滑动关节
            pos = data.qpos[qpos_adr]
            print(f"{name}: {pos:.4f} m")
        else:
            pos = data.qpos[qpos_adr]
            print(f"{name}: {pos:.4f}")
            # print(f"{name}: 非驱动关节 (类型: {jnt_type})")

# 示例用法
model = mujoco.MjModel.from_xml_path("ur5_grasp_assets/scenes/scene_test.xml")
data = mujoco.MjData(model)

# 指定需要监控的关节名称列表
target_joint_names = [
    "shoulder_pan_joint",
    "shoulder_lift_joint", 
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "fingers_actuator"# 驱动器打印错误
]

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        
        if data.time % 1.0 < 0.001:  # 每秒打印一次
            print_specific_joints(model, data, target_joint_names)
        
        viewer.sync()

