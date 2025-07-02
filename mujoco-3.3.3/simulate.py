import mujoco
import mujoco.viewer
import numpy as np
import time
import dataclasses
import enum
import logging
import pathlib
import tyro
import os
from PIL import Image

# scipy is the most common and reliable library for rotation transformations
# If not already installed, run: pip install scipy
from scipy.spatial.transform import Rotation

# --- Assume openpi_client is installed ---
# You can install it via: pip install openpi-client
from openpi_client import websocket_client_policy as _websocket_client_policy

# --- Configuration and Argument Parsing (from main.py) ---

class EnvMode(enum.Enum):
    """Supported environments."""
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"

@dataclasses.dataclass
class Args:
    """Command line arguments for the simulation."""
    # --- Server Connection Parameters ---
    host: str = "202.115.65.101"
    port: int | None = 8000
    api_key: str | None = None

    # --- Simulation and Control Parameters ---
    model_path: str = "mujoco-3.3.3/ur5_grasp_assets/scenes/scene_test.xml"
    hold_steps: int = 50
    env: EnvMode = EnvMode.LIBERO
    norm: float | None = None
    average_actions: bool = False

# --- MuJoCo Specific Control Functions ---

def get_actuator_info(model: mujoco.MjModel) -> dict:
    """Extracts information about the model's actuators."""
    info = {}
    for i in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        info[i] = {
            "name": actuator_name,
            "range": model.actuator_ctrlrange[i].copy(),
            "is_limited": model.actuator_ctrllimited[i] == 1
        }
    logging.info(f"Found actuators: {info}")
    return info

def movej_mujoco(data: mujoco.MjData, target_angles: list[float], arm_actuator_indices: range):
    """Executes a 'movej' equivalent command in MuJoCo."""
    if len(target_angles) != len(arm_actuator_indices):
        logging.warning(f"Mismatch between target angles ({len(target_angles)}) and arm actuators ({len(arm_actuator_indices)}). Skipping move.")
        return
    data.ctrl[arm_actuator_indices] = target_angles
    logging.info(f"  - [Sim] MoveJ Sent: {[f'{a:.2f}' for a in target_angles]}")

def control_gripper_mujoco(data: mujoco.MjData, gripper_actuator_index: int, gripper_range: np.ndarray, amplitude: float):
    """Controls the gripper in MuJoCo by mapping a 0-100 amplitude value."""
    min_ctrl, max_ctrl = gripper_range
    ctrl_value = min_ctrl + (amplitude / 100.0) * (max_ctrl - min_ctrl)
    data.ctrl[gripper_actuator_index] = ctrl_value
    logging.info(f"  - [Sim] Gripper Sent: Amplitude {amplitude:.2f} -> Ctrl Value {ctrl_value:.3f}")

# --- Observation Function ---

def get_mujoco_observation(
    model: mujoco.MjModel, 
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    tcp_body_name: str = "flange", 
    left_joint_name: str = "left_driver_joint",
    right_joint_name: str = "right_driver_joint",
    main_camera_name: str = "cam1", 
    wrist_camera_name: str = "flange_cam" 
) -> dict | None:
    """
    Reads observation data from the MuJoCo simulation environment.
    The state vector's first 6 dimensions are the end-effector's pose: [x, y, z, rz, ry, rx] (in radians).
    """
    try:
        # === Part 1: State Vector Calculation ===
        state = np.zeros(8, dtype=np.float64)

        flange_pos = data.body(tcp_body_name).xpos
        flange_quat = data.body(tcp_body_name).xquat
        
        rotation = Rotation.from_quat(flange_quat)
        euler_angles_rad = rotation.as_euler('zyx', degrees=False)
        
        state[0:3] = flange_pos
        state[3:6] = euler_angles_rad

        right_pos = np.degrees(data.joint(right_joint_name).qpos[0])
        left_pos = np.degrees(data.joint(left_joint_name).qpos[0])
        state[6] = left_pos
        state[7] = right_pos

        # === Part 2: Image Rendering ===
        renderer.update_scene(data, camera=main_camera_name)
        main_image = renderer.render()
        renderer.update_scene(data, camera=wrist_camera_name)
        wrist_image = renderer.render()
        
        # === Part 3: Assemble Observation Dictionary ===
        observation = {
            "observation/state": state,
            "observation/image": main_image,
            "observation/wrist_image": wrist_image,
            "prompt": "pick up everything you see"
        }
        return observation

    except KeyError as e:
        logging.error(f"Error: A specified name was not found in the model: {e}. Check camera, body, or joint names in your XML files.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred in get_mujoco_observation: {e}")
        return None

def main(args: Args) -> None:
    """Main function to run the MuJoCo simulation controlled by the policy server."""
    # --- 1. Load MuJoCo Model and Components ---
    try:
        model = mujoco.MjModel.from_xml_path(args.model_path)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=224, width=224)
    except FileNotFoundError:
        logging.error(f"Error: MuJoCo model file not found at '{args.model_path}'")
        return
    except Exception as e:
        logging.error(f"Failed to load model or create renderer: {e}")
        return
        
    actuator_info = get_actuator_info(model)
    if model.nu < 7:
        logging.error(f"Model must have at least 7 actuators (6 for arm, 1 for gripper), but found {model.nu}.")
        return
        
    ARM_ACTUATOR_INDICES = range(6)
    GRIPPER_ACTUATOR_INDEX = 6
    gripper_range = actuator_info[GRIPPER_ACTUATOR_INDEX]["range"]

    # --- 2. Define the Observation Function ---
    obs_fn = lambda: get_mujoco_observation(model, data, renderer)
    
    # --- 3. Connect to Policy Server ---
    try:
        policy = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port, api_key=args.api_key)
        logging.info(f"Initial Server metadata: {policy.get_server_metadata()}")
    except Exception as e:
        logging.error(f"Failed to connect to the policy server at {args.host}:{args.port}. Error: {e}")
        return

    # --- 4. Warm up the server ---
    logging.info("Warming up the server...")

    # 【关键修复】在第一次获取观测前，先运行一步模拟来初始化所有物理状态
    mujoco.mj_step(model, data)

    warmup_obs = obs_fn()
    if warmup_obs is None:
        logging.error("Failed to get initial observation during warmup. Exiting.")
        return
    for _ in range(2):
        policy.infer(warmup_obs)
        
    # --- 5. Setup for Image Saving ---
    IMAGE_SAVE_DIR = "saved_images"
    os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
    action_counter = 0
    main_cam_saved_count = 0
    wrist_cam_saved_count = 0
    IMAGE_SAVE_LIMIT = 5
    SAVE_INTERVAL = 10

    # --- 6. Main Simulation Loop ---
    with mujoco.viewer.launch_passive(model, data) as viewer:
        logging.info("\nSimulation started. Receiving actions from server to control the robot.")
        logging.info(f"Images will be saved every {SAVE_INTERVAL} actions to '{IMAGE_SAVE_DIR}/' (limit: {IMAGE_SAVE_LIMIT} per camera).")
        logging.info("Press ESC to quit.")
        
        while viewer.is_running():
            # a. Get observation
            current_obs = obs_fn()
            if current_obs is None:
                logging.warning("Failed to get observation, skipping this step.")
                time.sleep(0.1)
                continue

            # b. Get action from the policy server
            response = policy.infer(current_obs)
            action_sequence = response.get('actions')
            
            # c. Check if the received action is valid
            if not isinstance(action_sequence, (list, np.ndarray)) or len(action_sequence) == 0 or not all(len(sub) == 7 for sub in action_sequence):
                logging.warning(f"Received invalid or empty 'actions', skipping: {action_sequence}")
                mujoco.mj_step(model, data)
                viewer.sync()
                continue
                
            # d. Increment action counter and save images if interval is reached
            action_counter += 1
            if action_counter >= SAVE_INTERVAL:
                if main_cam_saved_count < IMAGE_SAVE_LIMIT:
                    main_img_array = current_obs["observation/image"]
                    img = Image.fromarray(main_img_array)
                    filename = os.path.join(IMAGE_SAVE_DIR, f"main_camera_{main_cam_saved_count + 1}.png")
                    img.save(filename)
                    logging.info(f"Saved main camera image: {filename}")
                    main_cam_saved_count += 1
                
                if wrist_cam_saved_count < IMAGE_SAVE_LIMIT:
                    wrist_img_array = current_obs["observation/wrist_image"]
                    img = Image.fromarray(wrist_img_array)
                    filename = os.path.join(IMAGE_SAVE_DIR, f"wrist_camera_{wrist_cam_saved_count + 1}.png")
                    img.save(filename)
                    logging.info(f"Saved wrist camera image: {filename}")
                    wrist_cam_saved_count += 1
                
                action_counter = 0

            # e. Process action sequence
            processed_sequence = action_sequence
            if args.average_actions:
                logging.info(f"Averaging {len(action_sequence)} sub-actions into a single action.")
                averaged_action = np.mean(action_sequence, axis=0)
                processed_sequence = [averaged_action]

            # f. Execute the action sequence in the simulation
            for sub_action in processed_sequence:
                if not viewer.is_running(): break
                
                joint_angles = list(sub_action[:6])
                scaled_joint_angles = [angle * 15 for angle in joint_angles]
                
                norm_value = 0.0
                if args.norm is not None:
                    norm_value = min(args.norm, 45.0)
                
                final_joint_angles = [angle + norm_value for angle in scaled_joint_angles]
                movej_mujoco(data, final_joint_angles, ARM_ACTUATOR_INDICES)

                gripper_value = sub_action[6]
                gripper_amplitude = max(0, min(100, abs(gripper_value)))
                control_gripper_mujoco(data, GRIPPER_ACTUATOR_INDEX, gripper_range, gripper_amplitude)

                # Hold the control signal for a number of simulation steps
                for _ in range(args.hold_steps):
                    if not viewer.is_running(): break
                    mujoco.mj_step(model, data)
                    viewer.sync()
            
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main(tyro.cli(Args))