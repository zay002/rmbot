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
    REAL_ROBOT_API_AVAILABLE = True
except ImportError:
    print("Warning: Could not import 'rm_robot_interface'. Real robot functionality will be disabled.")
    REAL_ROBOT_API_AVAILABLE = False

logger = logging.getLogger(__name__)

# --- Configuration and Argument Parsing ---
# ... (Args and EnvMode classes are unchanged) ...
class EnvMode(enum.Enum):
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"

@dataclasses.dataclass
class Args:
    host: str = "202.115.65.101"
    port: Optional[int] = 8000
    api_key: Optional[str] = None
    model_path: str = "mujoco-3.3.3/ur5_grasp_assets/scenes/scene_test.xml"
    hold_steps: int = 50
    run_robot: bool = False
    robot_ip: str = "192.168.1.19"
    robot_port: int = 8080
    robot_speed: int = 5
    env: EnvMode = EnvMode.LIBERO
    norm: Optional[float] = None
    average_actions: bool = False
    i: bool = False

# --- Helper function for mapping values (unchanged) ---
def map_value(value: float, from_min: float, from_max: float, to_min: float, to_max: float, inverted: bool = False) -> float:
    if inverted:
        to_min, to_max = to_max, to_min
    value = max(from_min, min(value, from_max))
    from_span = from_max - from_min
    if from_span == 0:
        return to_min
    to_span = to_max - to_min
    scaled_value = (value - from_min) / from_span
    return to_min + (scaled_value * to_span)

# --- Real Robot Control (Refactored into a Class) ---

class RobotController:
    """
    【NEW】A dedicated class to manage the real robot connection, commands, and kinematics.
    """
    def __init__(self, ip: str, port: int):
        if not REAL_ROBOT_API_AVAILABLE:
            raise RuntimeError("Real robot API is not available. Cannot initialize RobotController.")
            
        self.robot: 'RoboticArm' = RoboticArm(rm_thread_mode_e(2))
        self.handle = self.robot.rm_create_robot_arm(ip, port, 3)
        self.kinematics_solver: Optional['Algo'] = None
        self.dof = 6

        if self.handle.id == -1:
            raise ConnectionError("Failed to connect to the real robot.")
        
        logger.info(f"Successfully connected to the real robot with handle: {self.handle.id}")
        self._setup_kinematics_solver()

    def _setup_kinematics_solver(self):
        """Initializes the forward kinematics solver based on the connected robot's model."""
        ret, info = self.robot.rm_get_robot_info()
        if ret == 0:
            arm_model_str = info.get('arm_model')
            self.dof = info.get('arm_dof', 6)
            
            if arm_model_str:
                logger.info(f"Robot Model: {arm_model_str}, DOF: {self.dof}")
                MODEL_MAP = {
                    'RM_65': rm_robot_arm_model_e.RM_65_E,
                    'RM_75': rm_robot_arm_model_e.RM_75_E,
                    'RM_63': rm_robot_arm_model_e.RM_63_E,
                    'ECO_65': rm_robot_arm_model_e.ECO_65_E
                }
                arm_model_enum = MODEL_MAP.get(arm_model_str)
                if arm_model_enum is not None:
                    # Initialize Algo with no force sensor
                    self.kinematics_solver = Algo(arm_model_enum, 0)
                    logger.info("Kinematics solver initialized successfully.")
                else:
                    logger.warning(f"Kinematics not supported for model '{arm_model_str}'.")
            else:
                logger.warning("Could not determine robot model.")
        else:
            logger.warning("Failed to get robot info.")

    def disconnect(self):
        """Disconnects from the robot."""
        logger.info("Disconnecting from the real robot...")
        if self.robot.rm_delete_robot_arm() != 0:
            logger.error("Failed to cleanly disconnect from the robot.")
        RoboticArm.rm_destroy()

    def reset_position(self):
        """Moves the robot to a zero-joint state and closes the gripper."""
        logger.info("Moving robot to safe reset position [0,0,0,0,0,0]...")
        self.movej([0] * self.dof, v=15)
        logger.info("Closing gripper...")
        self.control_gripper(0)
        logger.info("Robot has been reset.")

    def movej(self, joint_angles: list[float], v: int, r: int = 0, block: int = 1):
        """Executes a movej command."""
        logger.info(f"  - [Real] MoveJ Sent: {[f'{a:.2f}' for a in joint_angles]} at speed {v}")
        result = self.robot.rm_movej(joint_angles, v, r, 0, block)
        if result != 0:
            logger.error(f"Real robot movej command failed with error code: {result}")
            
    def control_gripper(self, amplitude: float):
        """Controls the gripper opening."""
        port, baudrate, timeout, modbus_address = 1, 115200, 1, 1
        logger.info(f"  - [Real] Gripper Sent: Amplitude {amplitude:.2f}")
        # ... (gripper modbus logic remains the same) ...
        if self.robot.rm_set_modbus_mode(port, baudrate, timeout) != 0:
            return
        write_params_auto = rm_peripheral_read_write_params_t(port, 0x9C9A, modbus_address)
        if self.robot.rm_write_single_register(write_params_auto, 0) != 0:
            return
        write_params = rm_peripheral_read_write_params_t(port, 0x9C40, modbus_address)
        self.robot.rm_write_single_register(write_params, int(amplitude))
        time.sleep(0.1)

    def get_current_end_effector_pose(self) -> Optional[np.ndarray]:
        """
        【NEW】Gets the robot's current joint angles and calculates its end-effector pose.
        Returns a 6D numpy array [x, y, z, rx, ry, rz] or None if calculation fails.
        """
        if not self.kinematics_solver:
            return None
        
        # 1. Get current real joint angles
        ret, joint_angles = self.robot.rm_get_joint_degree()
        if ret != 0:
            logger.warning("Could not get real robot joint degrees.")
            return None
        
        # 2. Calculate forward kinematics
        # flag=1 returns Euler angles [x, y, z, rx, ry, rz]
        pose = self.robot.rm_algo_forward_kinematics(joint_angles, flag=1)
        return np.array(pose)

# --- MuJoCo Simulation Functions (unchanged) ---
# ... (get_actuator_info, movej_mujoco, set_gripper_mujoco_ctrl, get_mujoco_observation) ...
def get_actuator_info(model: mujoco.MjModel) -> dict:
    info = {}
    for i in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        info[i] = {"name": actuator_name, "range": model.actuator_ctrlrange[i].copy(), "is_limited": model.actuator_ctrllimited[i] == 1}
    logger.info(f"Found simulation actuators: {info}")
    return info

def movej_mujoco(data: mujoco.MjData, target_angles: list[float], arm_actuator_indices: range):
    if len(target_angles) != len(arm_actuator_indices):
        logging.warning(f"[Sim] Mismatch between target angles ({len(target_angles)}) and arm actuators ({len(arm_actuator_indices)}).")
        return
    data.ctrl[arm_actuator_indices] = target_angles
    logging.info(f"  - [Sim] MoveJ Sent: {[f'{a:.2f}' for a in target_angles]}")

def set_gripper_mujoco_ctrl(data: mujoco.MjData, gripper_actuator_index: int, ctrl_value: float):
    data.ctrl[gripper_actuator_index] = ctrl_value
    logging.info(f"  - [Sim] Gripper Ctrl Set: {ctrl_value:.3f}")

def get_mujoco_observation(model: mujoco.MjModel, data: mujoco.MjData, renderer: mujoco.Renderer) -> Optional[dict]:
    try:
        state = np.zeros(8, dtype=np.float64)
        state[:6] = data.qpos[:6]
        state[6] = np.degrees(data.joint("left_driver_joint").qpos[0])
        state[7] = np.degrees(data.joint("right_driver_joint").qpos[0])
        renderer.update_scene(data, camera="cam1")
        main_image = renderer.render()
        renderer.update_scene(data, camera="flange_cam")
        wrist_image = renderer.render()
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
    # --- 1. Load MuJoCo Model (unchanged) ---
    try:
        model = mujoco.MjModel.from_xml_path(args.model_path)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=224, width=224)
    except Exception as e:
        logging.error(f"Fatal: Failed to load MuJoCo model from '{args.model_path}'. Error: {e}")
        return
    # ... (actuator setup is unchanged) ...
    actuator_info = get_actuator_info(model)
    if model.nu < 7:
        logging.error(f"Model requires at least 7 actuators (6 arm, 1 gripper), but found {model.nu}.")
        return
    ARM_ACTUATOR_INDICES = range(6)
    GRIPPER_ACTUATOR_INDEX = 6
    _sim_gripper_min_from_model, _sim_gripper_max_from_model = actuator_info[GRIPPER_ACTUATOR_INDEX]["range"]
    sim_gripper_min, sim_gripper_max = 0.0, 255.0
    
    # --- 2. (Optional) Connect to Real Robot ---
    robot_controller: Optional[RobotController] = None
    if args.run_robot:
        logger.info("The --run-robot flag is set. Attempting to connect to the real robot.")
        try:
            robot_controller = RobotController(args.robot_ip, args.robot_port)
        except (RuntimeError, ConnectionError) as e:
            logger.error(f"Failed to initialize robot controller: {e}")
            args.run_robot = False
    else:
        logger.info("The --run-robot flag is not set. The script will run in simulation-only mode.")

    # --- 3. Connect to Policy Server (unchanged) ---
    try:
        policy = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port, api_key=args.api_key)
        logging.info(f"Connected to policy server. Initial metadata: {policy.get_server_metadata()}")
    except Exception as e:
        logging.error(f"Fatal: Could not connect to the policy server at {args.host}:{args.port}. Error: {e}")
        if robot_controller: robot_controller.disconnect()
        return

    # --- 4. Warm up the server (unchanged) ---
    logging.info("Warming up the policy server...")
    obs_fn = lambda: get_mujoco_observation(model, data, renderer)
    warmup_obs = obs_fn()
    if warmup_obs is None:
        logging.error("Fatal: Failed to get observation during warmup. Exiting.")
        if robot_controller: robot_controller.disconnect()
        return
    for _ in range(2):
        policy.infer(warmup_obs)

    # --- 5. Main Simulation and Control Loop ---
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            logging.info("\n*** Simulation and Control Loop Started ***")
            if args.run_robot: logger.info(f"--> CONTROLLING REAL ROBOT AND SIMULATION | Speed: {args.robot_speed} <--")
            else: logger.info("--> CONTROLLING SIMULATION ONLY <--")
            logging.info("Press ESC in the viewer window to quit.")

            while viewer.is_running():
                # a. Get base observation from simulation
                current_obs = obs_fn()
                if current_obs is None:
                    time.sleep(0.1)
                    continue

                # b. 【MODIFIED】 If connected, get real pose and update the observation
                if args.run_robot and robot_controller:
                    real_pose = robot_controller.get_current_end_effector_pose()
                    if real_pose is not None:
                        logger.info(f"  - [Real] FK Pose: x={real_pose[0]:.3f}, y={real_pose[1]:.3f}, z={real_pose[2]:.3f}")
                        # Overwrite the state vector's first 6 dimensions
                        current_obs["observation/state"][:6] = real_pose
                    else:
                        logger.warning("  - [Real] Could not calculate real-time pose.")
                
                # c. Get action from the policy server using the (potentially updated) observation
                response = policy.infer(current_obs)
                action_sequence = response.get('actions')

                # d. Validate and process the received action (unchanged)
                if not isinstance(action_sequence, (list, np.ndarray)) or len(action_sequence) == 0 or not all(len(sub) == 7 for sub in action_sequence):
                    logging.warning(f"Invalid or empty 'actions' received, skipping step: {action_sequence}")
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    continue
                
                processed_sequence = [np.mean(action_sequence, axis=0)] if args.average_actions else action_sequence

                # e. Execute the action sequence on both platforms
                for sub_action in processed_sequence:
                    if not viewer.is_running(): break

                    # ... (action processing, scaling, and gripper logic is unchanged) ...
                    joint_angles = list(sub_action[:6])
                    if args.i:
                        joint_angles.reverse()
                    scaled_joint_angles = [angle * 15 for angle in joint_angles]
                    norm_value = min(args.norm, 45.0) if args.norm is not None else 0.0
                    final_joint_angles = [angle + norm_value for angle in scaled_joint_angles]
                    raw_gripper_action = sub_action[6]
                    sim_gripper_ctrl_value = max(0.0, min(raw_gripper_action, 1.0)) * 255
                    
                    # Send commands to Simulation
                    movej_mujoco(data, final_joint_angles, ARM_ACTUATOR_INDICES)
                    set_gripper_mujoco_ctrl(data, GRIPPER_ACTUATOR_INDEX, sim_gripper_ctrl_value)

                    # Send commands to Real Robot
                    if args.run_robot and robot_controller:
                        real_gripper_amplitude = map_value(sim_gripper_ctrl_value, from_min=sim_gripper_min, from_max=sim_gripper_max, to_min=0, to_max=100)
                        robot_controller.movej(final_joint_angles, v=args.robot_speed)
                        robot_controller.control_gripper(real_gripper_amplitude)

                    # Step the simulation
                    for _ in range(args.hold_steps):
                        if not viewer.is_running(): break
                        mujoco.mj_step(model, data)
                        viewer.sync()
    finally:
        # --- 6. Cleanup ---
        logging.info("Program exiting. Cleaning up resources.")
        if args.run_robot and robot_controller:
            robot_controller.reset_position()
            robot_controller.disconnect()
        logging.info("Cleanup complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
    main(tyro.cli(Args))