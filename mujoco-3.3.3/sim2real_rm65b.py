import dataclasses
import enum
import logging
import pathlib
import time
import sys
import os
from typing import Optional, Union

import numpy as np
import mujoco
import mujoco.viewer
from PIL import Image
from scipy.spatial.transform import Rotation
import tyro
import rich.console
import rich.table

# --- OpenPI Client Imports ---
from openpi_client import websocket_client_policy as _websocket_client_policy

# --- Real Robot Control Code Integration ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    api_path = os.path.join(project_root, "RM_API2", "Python")
    sys.path.insert(0, api_path)
    from Robotic_Arm.rm_robot_interface import *
except ImportError:
    print("Warning: Could not import 'rm_robot_interface'. Real robot control will be disabled.")

logger = logging.getLogger(__name__)


# --- Configuration and Argument Parsing ---
class EnvMode(enum.Enum):
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"

@dataclasses.dataclass
class Args:
    """Command line arguments for running the simulation and/or the real robot."""
    host: str = "202.115.65.101"
    port: Optional[int] = 8000
    api_key: Optional[str] = None
    model_path: str = "mujoco-3.3.3/ur5_grasp_assets/scenes/scene_rm65b.xml" # MODIFIED: Use the scene with the gripper
    hold_steps: int = 50
    run_robot: bool = False
    robot_ip: str = "192.168.1.19"
    robot_port: int = 8080
    robot_speed: int = 10
    env: EnvMode = EnvMode.LIBERO
    average_actions: bool = False
    i: bool = False


# --- Helper function for mapping values (unchanged) ---
def map_value(value: float, from_min: float, from_max: float, to_min: float, to_max: float, inverted: bool = False) -> float:
    # ... (function content unchanged)
    if inverted: to_min, to_max = to_max, to_min
    value = max(from_min, min(value, from_max))
    from_span = from_max - from_min
    if from_span == 0: return to_min
    to_span = to_max - to_min
    scaled_value = (value - from_min) / from_span
    return to_min + (scaled_value * to_span)


# --- Real Robot Control Functions (unchanged) ---
def connect_robot(ip: str, port: int) -> Optional['RoboticArm']:
    # ... (function content unchanged)
    try:
        robot = RoboticArm(rm_thread_mode_e(2))
        handle = robot.rm_create_robot_arm(ip, port, 3)
        if handle.id == -1:
            logger.error("Failed to connect to the real robot.")
            return None
        logger.info(f"Successfully connected to the real robot with handle: {handle.id}")
        return robot
    except (NameError, Exception) as e:
        logger.error(f"Failed to connect to robot: {e}")
        return None

def disconnect_robot(robot: 'RoboticArm'):
    # ... (function content unchanged)
    if not robot: return
    if robot.rm_delete_robot_arm() == 0: logger.info("Successfully disconnected from the real robot.")
    else: logger.error("Failed to disconnect from the real robot.")

def reset_robot_position(robot: 'RoboticArm'):
    # ... (function content unchanged)
    if not robot: return
    logger.info("Moving real robot to safe reset position [0,0,0,0,0,0]...")
    movej(robot, [0.0] * 6, v=15)
    logger.info("Closing real robot gripper...")
    control_gripper_open(robot, 0)
    logger.info("Real robot has been reset.")

def movej(robot: 'RoboticArm', joint_angles: list[float], v: int, r: int = 0, block: int = 1):
    # ... (function content unchanged)
    logger.info(f"  - [Real] MoveJ Sent: {[f'{a:.2f}' for a in joint_angles]} at speed {v}")
    movej_result = robot.rm_movej(joint_angles, v, r, 0, block)
    if movej_result != 0: logger.error(f"Real robot movej command failed with error code: {movej_result}")

def control_gripper_open(arm: 'RoboticArm', amplitude: float):
    # ... (function content unchanged)
    port, baudrate, timeout, modbus_address = 1, 115200, 1, 1
    logger.info(f"  - [Real] Gripper Sent: Amplitude {amplitude:.2f}")
    if arm.rm_set_modbus_mode(port, baudrate, timeout) != 0:
        logger.warning("[Real] Failed to set Modbus RTU mode.")
        return
    write_params_auto = rm_peripheral_read_write_params_t(port, 0x9C9A, modbus_address)
    if arm.rm_write_single_register(write_params_auto, 0) != 0:
        logger.warning("[Real] Failed to disable automatic stroke search.")
        return
    write_params = rm_peripheral_read_write_params_t(port, 0x9C40, modbus_address)
    if arm.rm_write_single_register(write_params, int(amplitude)) != 0:
        logger.error("[Real] Gripper control command failed.")
    time.sleep(0.008)

# --- MuJoCo (Simulation) Control Functions ---

def get_actuator_info(model: mujoco.MjModel) -> dict:
    # ... (function content unchanged)
    info = {}
    for i in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        info[i] = { "name": actuator_name, "range": model.actuator_ctrlrange[i].copy(), "is_limited": model.actuator_ctrllimited[i] == 1 }
    logger.info(f"Found simulation actuators: {info}")
    return info

def movej_mujoco(data: mujoco.MjData, target_angles_deg: list[float], arm_actuator_indices: range):
    # ... (function content unchanged)
    if len(target_angles_deg) != len(arm_actuator_indices):
        logging.warning(f"[Sim] Mismatch between target angles ({len(target_angles_deg)}) and arm actuators ({len(arm_actuator_indices)}).")
        return
    target_angles_rad = [np.radians(deg) for deg in target_angles_deg]
    data.ctrl[arm_actuator_indices] = target_angles_rad
    logging.info(f"  - [Sim] MoveJ Sent (Degrees): {[f'{a:.2f}' for a in target_angles_deg]}")

def set_gripper_mujoco_ctrl(data: mujoco.MjData, gripper_actuator_index: int, ctrl_value: float):
    # ... (function content unchanged)
    data.ctrl[gripper_actuator_index] = ctrl_value
    logging.info(f"  - [Sim] Gripper Ctrl Set: {ctrl_value:.3f}")

# --- MOCAP修改 (1/3): 增加一个辅助函数来执行Mocap位姿更新 ---
def update_mocap_pose(data: mujoco.MjData, eef_body_id: int, mocap_id: int):
    """读取末端位姿并应用到mocap体上，这是运动学连接的核心。"""
    data.mocap_pos[mocap_id][:] = data.body(eef_body_id).xpos
    data.mocap_quat[mocap_id][:] = data.body(eef_body_id).xquat

# --- Observation Function (unchanged) ---
def get_mujoco_observation(model: mujoco.MjModel, data: mujoco.MjData, renderer: mujoco.Renderer) -> Optional[dict]:
    # ... (function content unchanged)
    try:
        state = np.zeros(8, dtype=np.float64)
        flange_pos = data.body('flange').xpos
        flange_quat = data.body('flange').xquat
        rotation = Rotation.from_quat([flange_quat[1], flange_quat[2], flange_quat[3], flange_quat[0]]) # Scipy expects (x,y,z,w)
        euler_angles_rad = rotation.as_euler('zyx', degrees=False)
        state[0:3], state[3:6] = flange_pos, euler_angles_rad
        state[6] = np.degrees(data.joint("left_driver_joint").qpos[0])
        state[7] = np.degrees(data.joint("right_driver_joint").qpos[0])
        renderer.update_scene(data, camera="cam1")
        main_image = renderer.render()
        renderer.update_scene(data, camera="flange_cam")
        wrist_image = renderer.render()
        return { "observation/state": state, "observation/image": main_image, "observation/wrist_image": wrist_image, "prompt": "pick up the purple block" }
    except Exception as e:
        logging.error(f"Failed to get MuJoCo observation: {e}")
        return None


# --- Main Execution ---

def main(args: Args) -> None:
    """Main function to run the simulation and optionally control the real robot."""
    try:
        model = mujoco.MjModel.from_xml_path(args.model_path)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=224, width=224)
    except Exception as e:
        logging.error(f"Fatal: Failed to load MuJoCo model from '{args.model_path}'. Error: {e}")
        return

    actuator_info = get_actuator_info(model)
    if model.nu < 7:
        logging.error(f"Model requires at least 7 actuators (6 arm, 1 gripper), but found {model.nu}.")
        return

    ARM_ACTUATOR_INDICES = range(6)
    GRIPPER_ACTUATOR_INDEX = 6
    sim_gripper_min, sim_gripper_max = 0.0, 255.0

    # --- MOCAP修改 (2/3): 在主函数开始时获取Mocap和末端ID ---
    try:
        eef_body_id = model.body('flange').id
        # mocap体的索引是0，因为我们在XML中只定义了一个
        mocap_id = 0 
    except KeyError as e:
        logging.error(f"Mocap setup error: XML is missing a required body ('flange' or 'gripper_mocap'). Details: {e}")
        return

    robot_arm = None
    if args.run_robot:
        # ... (real robot connection logic unchanged) ...
        logger.info("The --run-robot flag is set. Attempting to connect to the real robot.")
        robot_arm = connect_robot(args.robot_ip, args.robot_port)
        if not robot_arm:
            logger.warning("Failed to connect to real robot. Continuing in simulation-only mode.")
            args.run_robot = False
    else:
        logger.info("The --run-robot flag is not set. Running in simulation-only mode.")

    try:
        policy = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port, api_key=args.api_key)
        logging.info(f"Connected to policy server. Metadata: {policy.get_server_metadata()}")
    except Exception as e:
        logging.error(f"Fatal: Could not connect to policy server at {args.host}:{args.port}. Error: {e}")
        if robot_arm: disconnect_robot(robot_arm)
        return

    logging.info("Warming up the policy server...")
    obs_fn = lambda: get_mujoco_observation(model, data, renderer)
    update_mocap_pose(data, eef_body_id, mocap_id) # Mocap update before step
    mujoco.mj_step(model, data)
    warmup_obs = obs_fn()
    if warmup_obs is None:
        logging.error("Fatal: Failed to get observation during warmup. Exiting.")
        if robot_arm: disconnect_robot(robot_arm)
        return
    for _ in range(2):
        policy.infer(warmup_obs)

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            logging.info("\n*** Simulation and Control Loop Started ***")
            # ... (logging messages unchanged) ...

            while viewer.is_running():
                current_obs = obs_fn()
                if current_obs is None:
                    time.sleep(0.008)
                    continue

                response = policy.infer(current_obs)
                action_sequence = response.get('actions')

                if not isinstance(action_sequence, (list, np.ndarray)) or len(action_sequence) == 0 or not all(len(sub) == 7 for sub in action_sequence):
                    logging.warning(f"Invalid or empty 'actions' received, skipping: {action_sequence}")
                    update_mocap_pose(data, eef_body_id, mocap_id) # Mocap update before step
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    continue

                processed_sequence = [np.mean(action_sequence, axis=0)] if args.average_actions else action_sequence

                for sub_action in processed_sequence:
                    if not viewer.is_running(): break
                    
                    joint_angles_rad = list(sub_action[:6])
                    if args.i: joint_angles_rad.reverse()
                    
                    final_joint_angles_deg = [np.degrees(rad) for rad in joint_angles_rad]
                    clamped_joint_angles = [max(-180.0, min(angle, 180.0)) for angle in final_joint_angles_deg]
                    
                    raw_gripper_action = sub_action[6]
                    sim_gripper_ctrl_value = max(0.0, min(raw_gripper_action, 1.0)) * 255
                    
                    movej_mujoco(data, clamped_joint_angles, ARM_ACTUATOR_INDICES)
                    set_gripper_mujoco_ctrl(data, GRIPPER_ACTUATOR_INDEX, sim_gripper_ctrl_value)

                    if args.run_robot and robot_arm:
                        real_gripper_amp = map_value(sim_gripper_ctrl_value, sim_gripper_min, sim_gripper_max, 0, 100)
                        movej(robot_arm, clamped_joint_angles, v=args.robot_speed)
                        control_gripper_open(robot_arm, real_gripper_amp)
                    
                    for _ in range(args.hold_steps):
                        if not viewer.is_running(): break
                        # --- MOCAP修改 (3/3): 在每一步物理计算前，都调用Mocap更新函数 ---
                        update_mocap_pose(data, eef_body_id, mocap_id)
                        mujoco.mj_step(model, data)
                        viewer.sync()
    finally:
        # --- Cleanup section (unchanged) ---
        logging.info("Program exiting. Cleaning up resources...")
        # ... (cleanup code unchanged) ...
        movej(robot_arm,[0,0,0,0,0,0],v=args.robot_speed)
        control_gripper_open(robot_arm,0)
        if robot_arm: disconnect_robot(robot_arm)
        logging.info("Cleanup complete. ✅")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
    main(tyro.cli(Args))