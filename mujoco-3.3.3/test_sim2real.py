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
    # This logic assumes the script is run from a location where it can find the RM_API2 directory.
    # Adjust the path if your project structure is different.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    api_path = os.path.join(project_root, "RM_API2", "Python")
    sys.path.insert(0, api_path)
    from Robotic_Arm.rm_robot_interface import *
except ImportError:
    print("Error: Could not import 'rm_robot_interface'.")
    print("Please ensure the 'RM_API2' folder is correctly placed in your project structure.")
    print(f"Attempted to import from: {api_path}")
    # We don't exit here, to allow simulation-only mode to function.

logger = logging.getLogger(__name__)


# --- Configuration and Argument Parsing ---

class EnvMode(enum.Enum):
    """Supported environments for policy server observation generation."""
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Args:
    """Command line arguments for running the simulation and/or the real robot."""
    # --- Server Connection Parameters ---
    host: str = "202.115.65.101"
    port: Optional[int] = 8000
    api_key: Optional[str] = None

    # --- Simulation Parameters ---
    model_path: str = "mujoco-3.3.3/ur5_grasp_assets/scenes/scene_test.xml"
    hold_steps: int = 50

    # --- Real Robot Parameters ---
    run_robot: bool = False  # Set to True to enable the real robot
    robot_ip: str = "192.168.1.19"
    robot_port: int = 8080
    ## MODIFIED: Default speed changed to 10
    robot_speed: int = 5 # Velocity for the real robot's movej command

    # --- General Control Parameters ---
    env: EnvMode = EnvMode.LIBERO
    norm: Optional[float] = None
    average_actions: bool = False
    # Set to True to invert the order of the first 6 action dimensions.
    i: bool = False


# --- Helper function for mapping values ---
def map_value(value: float, from_min: float, from_max: float, to_min: float, to_max: float, inverted: bool = False) -> float:
    """Linearly maps a value from one range to another, with an option to invert the output."""
    if inverted:
        to_min, to_max = to_max, to_min
    # Clamp the value to the source range
    value = max(from_min, min(value, from_max))
    from_span = from_max - from_min
    # Avoid division by zero
    if from_span == 0:
        return to_min
    to_span = to_max - to_min
    scaled_value = (value - from_min) / from_span
    return to_min + (scaled_value * to_span)


# --- Real Robot Control Functions ---

def connect_robot(ip: str, port: int) -> Optional['RoboticArm']:
    """Initializes and connects to the real robotic arm."""
    try:
        robot = RoboticArm(rm_thread_mode_e(2))
        handle = robot.rm_create_robot_arm(ip, port, 3)
        if handle.id == -1:
            logger.error("Failed to connect to the real robot.")
            return None
        logger.info(f"Successfully connected to the real robot with handle: {handle.id}")
        return robot
    except NameError:
        logger.error("RoboticArm class not found. Real robot control is disabled.")
        return None
    except Exception as e:
        logger.error(f"An error occurred while connecting to the robot: {e}")
        return None

def disconnect_robot(robot: 'RoboticArm'):
    """Disconnects from the real robotic arm."""
    if not robot:
        return
    if robot.rm_delete_robot_arm() == 0:
        logger.info("Successfully disconnected from the real robot.")
    else:
        logger.error("Failed to disconnect from the real robot.")

def reset_robot_position(robot: 'RoboticArm'):
    """Moves the robot to a zero-joint state and closes the gripper."""
    if not robot:
        return
    logger.info("Moving robot to safe reset position [0,0,0,0,0,0]...")
    movej(robot, [0, 0, 0, 0, 0, 0], v=15)
    logger.info("Closing gripper...")
    control_gripper_open(robot, 0) # 0 corresponds to the closed state
    logger.info("Robot has been reset.")


def movej(robot: 'RoboticArm', joint_angles: list[float], v: int, r: int = 0, block: int = 1):
    """Executes a movej command on the real robot."""
    logger.info(f"  - [Real] MoveJ Sent: {[f'{a:.2f}' for a in joint_angles]} at speed {v}")
    movej_result = robot.rm_movej(joint_angles, v, r, 0, block)
    if movej_result != 0:
        logger.error(f"Real robot movej command failed with error code: {movej_result}")

def control_gripper_open(arm: 'RoboticArm', amplitude: float):
    """Controls the real robot's gripper opening."""
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
    time.sleep(0.1)


# --- MuJoCo (Simulation) Control Functions ---

def get_actuator_info(model: mujoco.MjModel) -> dict:
    """Extracts information about the simulation model's actuators."""
    info = {}
    for i in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        info[i] = {
            "name": actuator_name,
            "range": model.actuator_ctrlrange[i].copy(),
            "is_limited": model.actuator_ctrllimited[i] == 1
        }
    logger.info(f"Found simulation actuators: {info}")
    return info

def movej_mujoco(data: mujoco.MjData, target_angles: list[float], arm_actuator_indices: range):
    """Sets the control for the arm actuators in the MuJoCo simulation."""
    if len(target_angles) != len(arm_actuator_indices):
        logging.warning(f"[Sim] Mismatch between target angles ({len(target_angles)}) and arm actuators ({len(arm_actuator_indices)}).")
        return
    data.ctrl[arm_actuator_indices] = target_angles
    logging.info(f"  - [Sim] MoveJ Sent: {[f'{a:.2f}' for a in target_angles]}")

def set_gripper_mujoco_ctrl(data: mujoco.MjData, gripper_actuator_index: int, ctrl_value: float):
    """Sets the final control value for the gripper in the MuJoCo simulation."""
    data.ctrl[gripper_actuator_index] = ctrl_value
    logging.info(f"  - [Sim] Gripper Ctrl Set: {ctrl_value:.3f}")


# --- Observation Function ---

def get_mujoco_observation(model: mujoco.MjModel, data: mujoco.MjData, renderer: mujoco.Renderer) -> Optional[dict]:
    """Reads state and image data from the MuJoCo simulation to form an observation."""
    try:
        # State vector
        state = np.zeros(8, dtype=np.float64)
        state[:6] = data.qpos[:6]
        state[6] = np.degrees(data.joint("left_driver_joint").qpos[0])
        state[7] = np.degrees(data.joint("right_driver_joint").qpos[0])

        # Image rendering
        renderer.update_scene(data, camera="cam1")
        main_image = renderer.render()
        renderer.update_scene(data, camera="flange_cam")
        wrist_image = renderer.render()

        # Assemble dictionary
        return {
            "observation/state": state,
            "observation/image": main_image,
            "observation/wrist_image": wrist_image,
            "prompt": "You are a robotic arm. Infer policies from the observations to complete tasks. Now, try to reach for an object, pick it up with your gripper, and hold it."
        }
    except Exception as e:
        logging.error(f"Failed to get MuJoCo observation: {e}")
        return None


# --- Main Execution ---

def main(args: Args) -> None:
    """Main function to run the simulation and optionally control the real robot."""

    # --- 1. Load MuJoCo Model ---
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
    
    # >>> MODIFIED CODE START <<<
    # Get the original gripper range from the MuJoCo model file.
    _sim_gripper_min_from_model, _sim_gripper_max_from_model = actuator_info[GRIPPER_ACTUATOR_INDEX]["range"]
    logger.info(f"Gripper range from model file: [{_sim_gripper_min_from_model:.2f}, {_sim_gripper_max_from_model:.2f}]")
    
    # Override the simulation gripper range to the specified 0-255.
    sim_gripper_min, sim_gripper_max = 0.0, 255.0
    logger.info(f"Overriding simulation gripper control range to: [{sim_gripper_min:.2f}, {sim_gripper_max:.2f}]")
    # >>> MODIFIED CODE END <<<

    # --- 2. (Optional) Connect to Real Robot ---
    robot_arm = None
    if args.run_robot:
        logger.info("The --run-robot flag is set. Attempting to connect to the real robot.")
        robot_arm = connect_robot(args.robot_ip, args.robot_port)
        if not robot_arm:
            logger.warning("Failed to connect to the real robot. The script will continue in simulation-only mode.")
            args.run_robot = False
    else:
        logger.info("The --run-robot flag is not set. The script will run in simulation-only mode.")

    # --- 3. Connect to Policy Server ---
    try:
        policy = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port, api_key=args.api_key)
        logging.info(f"Connected to policy server. Initial metadata: {policy.get_server_metadata()}")
    except Exception as e:
        logging.error(f"Fatal: Could not connect to the policy server at {args.host}:{args.port}. Error: {e}")
        if robot_arm:
            disconnect_robot(robot_arm)
        return

    # --- 4. Warm up the server ---
    logging.info("Warming up the policy server...")
    obs_fn = lambda: get_mujoco_observation(model, data, renderer)
    warmup_obs = obs_fn()
    if warmup_obs is None:
        logging.error("Fatal: Failed to get observation during warmup. Exiting.")
        if robot_arm:
            disconnect_robot(robot_arm)
        return
    for _ in range(2):
        policy.infer(warmup_obs)

    # --- 5. Main Simulation and Control Loop ---
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            logging.info("\n*** Simulation and Control Loop Started ***")
            if args.run_robot:
                logging.info(f"--> CONTROLLING REAL ROBOT AND SIMULATION <--")
                logging.info(f"--> Real robot speed set to: {args.robot_speed} <--")
            else:
                logging.info("--> CONTROLLING SIMULATION ONLY <--")
            logging.info("Press ESC in the viewer window to quit.")

            while viewer.is_running():
                # a. Get observation from simulation
                current_obs = obs_fn()
                if current_obs is None:
                    time.sleep(0.1)
                    continue

                # b. Get action from the policy server
                response = policy.infer(current_obs)
                action_sequence = response.get('actions')

                # c. Validate the received action
                if not isinstance(action_sequence, (list, np.ndarray)) or len(action_sequence) == 0 or not all(len(sub) == 7 for sub in action_sequence):
                    logging.warning(f"Invalid or empty 'actions' received, skipping step: {action_sequence}")
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    continue

                # d. Process action sequence (e.g., averaging)
                processed_sequence = action_sequence
                if args.average_actions:
                    logging.info(f"Averaging {len(action_sequence)} sub-actions.")
                    processed_sequence = [np.mean(action_sequence, axis=0)]

                # e. Execute the action sequence on both platforms
                for sub_action in processed_sequence:
                    if not viewer.is_running(): break

                    # i. Calculate joint angles
                    joint_angles = list(sub_action[:6])

                    # Invert the joint angles if the -i flag is set
                    if args.i:
                        logging.info("  - [Action] Inverting joint order due to -i flag.")
                        joint_angles.reverse()

                    scaled_joint_angles = [angle * 15 for angle in joint_angles]
                    norm_value = 0.0
                    if args.norm is not None:
                        norm_value = min(args.norm, 45.0)
                    final_joint_angles = [angle + norm_value for angle in scaled_joint_angles]

                    # The policy's gripper action is now treated as the primary value for the SIMULATION
                    raw_gripper_action = sub_action[6]
                    # Clamp the action to the valid simulation control range (now 0-100)
                    sim_gripper_ctrl_value = max(0.0, min(raw_gripper_action, 1.0))
                    sim_gripper_ctrl_value = sim_gripper_ctrl_value * 255

                    # ii. Send commands to Simulation
                    movej_mujoco(data, final_joint_angles, ARM_ACTUATOR_INDICES)
                    set_gripper_mujoco_ctrl(data, GRIPPER_ACTUATOR_INDEX, sim_gripper_ctrl_value)

                    # iii. Send commands to Real Robot (if enabled)
                    if args.run_robot and robot_arm:
                        # Map the SIMULATION control value (0-255) to the REAL robot's 0-100 range
                        real_gripper_amplitude = map_value(
                            sim_gripper_ctrl_value,
                            from_min=sim_gripper_min, from_max=sim_gripper_max, # from [0, 255]
                            to_min=0, to_max=100,                               # to [0, 100]
                            inverted=False
                        )

                        movej(robot_arm, final_joint_angles, v=args.robot_speed)
                        control_gripper_open(robot_arm, real_gripper_amplitude)

                    # iv. Step the simulation physics
                    for _ in range(args.hold_steps):
                        if not viewer.is_running(): break
                        mujoco.mj_step(model, data)
                        viewer.sync()

    finally:
        # --- 6. Cleanup ---
        logging.info("Program exiting. Cleaning up resources.")
        if args.run_robot and robot_arm:
            reset_robot_position(robot_arm)
            disconnect_robot(robot_arm)
        logging.info("Cleanup complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
    main(tyro.cli(Args))