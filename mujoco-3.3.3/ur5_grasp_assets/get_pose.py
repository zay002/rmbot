import mujoco
import numpy as np

# --- 1. 请在这里配置您要查询的参数 ---

# 要加载的场景XML文件路径
# 我们使用之前最终确定的、带2F85夹具的RM-65B场景
MODEL_PATH = "mujoco-3.3.3/ur5_grasp_assets/scenes/scene_rm65b.xml"

# 您要查询的“末端附件”的body名称
# 根据我们之前的约定，这个物体的名字是 'flange'
BODY_NAME_TO_INSPECT = "flange"


# --- 2. 主程序：加载模型并获取信息 ---

def get_initial_pose(model_path, body_name):
    """
    加载一个MuJoCo模型，并返回指定body在初始状态下的世界位姿信息。
    """
    print(f"正在加载模型: {model_path}")
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"错误：模型加载失败。\n{e}")
        return

    # 执行一步仿真，以确保所有约束（如焊接）和初始位置都已就绪
    mujoco.mj_step(model, data)

    try:
        # 获取目标物体的ID
        body_id = model.body(body_name).id

        # --- 从 data 中提取位姿信息 ---
        
        # 1. 世界坐标 (x, y, z)
        position = data.body(body_id).xpos

        # 2. 世界姿态 - 四元数 (w, x, y, z)
        quaternion = data.body(body_id).xquat

        # 3. 将四元数转换为欧拉角 (更直观)
        euler_radians = np.empty(3)
        # MuJoCo默认使用 ZYX 顺序计算欧拉角
        #mujoco.mju_quat2euler(euler_radians, quaternion) 
        # 将弧度转换为度，方便阅读
        euler_degrees = np.degrees(euler_radians)

        # --- 打印结果 ---
        print("\n" + "="*50)
        print(f"  物体 '{body_name}' 的初始位姿信息")
        print("="*50)

        print("\n位置 (Position / x, y, z):")
        print(f"  {np.array2string(position, precision=4, suppress_small=True)}")

        print("\n姿态 - 四元数 (Quaternion / w, x, y, z):")
        print(f"  {np.array2string(quaternion, precision=4, suppress_small=True)}")

        print("\n姿态 - 欧拉角 (Euler Angles in Degrees):")
        print(f"  绕X轴(Roll) : {euler_degrees[0]:.2f}°")
        print(f"  绕Y轴(Pitch): {euler_degrees[1]:.2f}°")
        print(f"  绕Z轴(Yaw)  : {euler_degrees[2]:.2f}°")
        print("="*50)

    except KeyError:
        print(f"\n错误：在模型中找不到名为 '{body_name}' 的body。")
        print("请检查XML文件中的body名称是否正确。")
    except Exception as e:
        print(f"获取位姿时发生未知错误: {e}")


if __name__ == "__main__":
    get_initial_pose(MODEL_PATH, BODY_NAME_TO_INSPECT)