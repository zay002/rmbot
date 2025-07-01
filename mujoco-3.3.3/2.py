import mujoco
import mujoco.viewer
import numpy as np
import time
import random

# 加载模型
model = mujoco.MjModel.from_xml_path("mujoco-3.3.3/ur5_grasp_assets/scenes/scene_test.xml") #"ur5_grasp_assets/scenes/scene_test.xml"
data = mujoco.MjData(model)

# # 初始化执行器控制数组
# ctrl = np.zeros(model.nu)

# # 设置随机运动参数
# min_fps = 10  # 最小帧率
# max_fps = 50  # 最大帧率
# smoothness = 0.2  # 平滑系数 (0~1, 越大越平滑)

# # 获取所有驱动关节的初始位置
# initial_qpos = data.qpos.copy()
# target_pos = initial_qpos.copy()
# current_pos = initial_qpos.copy()

# # 启动查看器
# with mujoco.viewer.launch_passive(model, data) as viewer:
#     print("控制UR5e随机运动中 - 按ESC退出")
#     last_time = time.time()
#     last_print_time = time.time()
    
#     while viewer.is_running():
#         # 计算当前帧率 (10~50 FPS随机)
#         current_fps = random.randint(min_fps, max_fps)
#         frame_delay = 1.0 / current_fps
        
#         # 随机生成新目标位置 (限制在关节范围内)
#         if time.time() - last_time > frame_delay:
#             for i in range(model.nu):
#                 joint_id = model.actuator_trnid[i, 0]
#                 joint_range = model.jnt_range[joint_id]
                
#                 # 如果关节有范围限制，则生成范围内随机值
#                 if model.jnt_limited[joint_id]:
#                     target_pos[joint_id] = random.uniform(joint_range[0], joint_range[1])
#                 else:
#                     target_pos[joint_id] = random.uniform(-np.pi, np.pi)
            
#             last_time = time.time()
        
#         # 平滑过渡到目标位置
#         current_pos = current_pos * (1 - smoothness) + target_pos * smoothness
        
#         # 将位置赋值给执行器 (使用位置控制)
#         for i in range(model.nu):
#             joint_id = model.actuator_trnid[i, 0]
#             data.ctrl[i] = current_pos[joint_id]
        
#         # 每秒打印一次关节位置
#         if time.time() - last_print_time >= 1.0:
#             print("\n=== 关节位置 ===")
#             for i in range(model.nu):
#                 joint_id = model.actuator_trnid[i, 0]
#                 joint_name = model.joint(joint_id).name
#                 pos = data.qpos[joint_id]
#                 print(f"{joint_name}: {np.degrees(pos):.2f}°" if model.jnt_type[joint_id] == 1 
#                       else f"{joint_name}: {pos:.4f} m")
#             last_print_time = time.time()
        
#         # 实时控制值打印
#         print("\r控制值: " + " ".join([f"{x:.3f}" for x in data.ctrl]), end="", flush=True)
        
#         # 步进仿真
#         mujoco.mj_step(model, data)
#         viewer.sync()
        
#         # 控制帧率
#         time.sleep(0.001)

# 初始化执行器控制数组
ctrl = np.zeros(model.nu)

# 设置随机运动参数
min_fps = 10  # 最小帧率
max_fps = 50  # 最大帧率
smoothness = 0.2  # 平滑系数

# 获取所有驱动关节信息
def get_actuated_joints(model):
    joints = []
    for i in range(model.nu):
        joint_id = model.actuator_trnid[i, 0]
        joint_name = model.joint(joint_id).name
        limited = model.jnt_limited[joint_id]
        jnt_range = model.jnt_range[joint_id] if limited else [-np.pi, np.pi]  # 默认±π
        joints.append({
            "name": joint_name,
            "id": joint_id,
            "limited": limited,
            "range": jnt_range,
            "type": model.jnt_type[joint_id]  # 1=旋转, 2=滑动
        })
    return joints

# 获取所有驱动关节
actuated_joints = get_actuated_joints(model)

# 打印所有驱动关节信息
print("=== 驱动关节信息 ===")
for i, joint in enumerate(actuated_joints):
    unit = "rad" if joint["type"] == 1 else "m"
    range_str = f"{joint['range'][0]:.2f}~{joint['range'][1]:.2f}{unit}"
    print(f"{i}: {joint['name']} (运动范围: {range_str})")

# 选择要监控的关节 (示例选择前3个关节)
selected_joint_indices = [0, 1, 2, 3, 4, 5]  # 修改这里选择要监控的关节索引
selected_joints = [actuated_joints[i] for i in selected_joint_indices]

# 初始化位置
target_pos = data.qpos.copy()
current_pos = data.qpos.copy()

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("\n控制UR5e随机运动中 - 按ESC退出")
    last_time = time.time()
    last_print_time = time.time()
    
    while viewer.is_running():
        # 计算当前帧率
        current_fps = random.randint(min_fps, max_fps)
        frame_delay = 1.0 / current_fps
        
        # 生成新目标位置
        if time.time() - last_time > frame_delay:
            for joint in actuated_joints:
                target_pos[joint["id"]] = random.uniform(joint["range"][0], joint["range"][1])
            last_time = time.time()
        
        # 平滑过渡
        current_pos = current_pos * (1 - smoothness) + target_pos * smoothness
        
        # 应用控制
        for i in range(model.nu):
            joint_id = model.actuator_trnid[i, 0]
            data.ctrl[i] = current_pos[joint_id]
        
        # 每秒打印选定关节位置
        if time.time() - last_print_time >= 1.0:
            print("\n=== 选定关节位置 ===")
            for joint in selected_joints:
                pos = data.qpos[joint["id"]]
                if joint["type"] == 1:  # 旋转关节
                    print(f"{joint['name']}: {np.degrees(pos):.2f}° (原始值: {pos:.3f}rad)")
                else:  # 滑动关节
                    print(f"{joint['name']}: {pos:.4f}m")
            last_print_time = time.time()
        
        # 步进仿真
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.001)