import dataclasses
import enum
import logging
import time

import mujoco
import mujoco.viewer
import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy

# scipy is the most common and reliable library for rotation transformations
from scipy.spatial.transform import Rotation
import tyro

# --- Configuration and Argument Parsing ---


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Args:
    """Command line arguments for the simulation."""

    host: str = "localhost"
    port: int | None = 8000
    model_path: str = "projects/rm65b/simulation/assets/scenes/scene_rm65b.xml"
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
            "is_limited": model.actuator_ctrllimited[i] == 1,
        }
    logging.info(f"Found actuators: {info}")
    return info


def movej_mujoco(data: mujoco.MjData, target_angles: list[float], arm_actuator_indices: range):
    """Executes a 'movej' equivalent command in MuJoCo."""
    if len(target_angles) != len(arm_actuator_indices):
        logging.warning(
            f"Mismatch between target angles ({len(target_angles)}) and arm actuators ({len(arm_actuator_indices)}). Skipping move."
        )
        return
    data.ctrl[arm_actuator_indices] = target_angles
    logging.info(f"  - [Sim] MoveJ Sent: {[f'{a:.2f}' for a in target_angles]}")


def control_gripper_mujoco(
    data: mujoco.MjData, gripper_actuator_index: int, gripper_range: np.ndarray, amplitude: float
):
    """Controls the gripper in MuJoCo by mapping a 0-255 amplitude value."""
    min_ctrl, max_ctrl = gripper_range
    ctrl_value = min_ctrl + (amplitude / 255.0) * (max_ctrl - min_ctrl)
    data.ctrl[gripper_actuator_index] = ctrl_value
    logging.info(f"  - [Sim] Gripper Sent: Amplitude {amplitude:.2f} -> Ctrl Value {ctrl_value:.3f}")


def update_mocap_pose(data: mujoco.MjData, eef_body_id: int, mocap_id: int):
    """Reads the end-effector's pose and applies it to the mocap body."""
    eef_pos = data.body(eef_body_id).xpos
    eef_quat = data.body(eef_body_id).xquat
    data.mocap_pos[0][:] = eef_pos
    data.mocap_quat[0][:] = eef_quat


# --- Observation Functions ---


def get_libero_state_vector(
    data: mujoco.MjData, tcp_body_name: str, left_joint_name: str, right_joint_name: str
) -> np.ndarray:
    """Helper function to extract the state vector used by LIBERO."""
    state = np.zeros(8, dtype=np.float64)
    flange_pos = data.body(tcp_body_name).xpos
    flange_quat = data.body(tcp_body_name).xquat

    rotation = Rotation.from_quat([flange_quat[1], flange_quat[2], flange_quat[3], flange_quat[0]])
    euler_angles_rad = rotation.as_euler("zyx", degrees=False)

    state[0:3] = flange_pos
    state[3:6] = euler_angles_rad

    right_pos = np.degrees(data.joint(right_joint_name).qpos[0])
    left_pos = np.degrees(data.joint(left_joint_name).qpos[0])
    state[6] = left_pos
    state[7] = right_pos
    return state


def get_mujoco_observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    tcp_body_name: str = "flange",
    left_joint_name: str = "left_driver_joint",
    right_joint_name: str = "right_driver_joint",
    main_camera_name: str = "cam1",
    wrist_camera_name: str = "flange_cam",
) -> dict | None:
    """Reads observation data from the MuJoCo simulation environment for LIBERO."""
    try:
        libero_state = get_libero_state_vector(data, tcp_body_name, left_joint_name, right_joint_name)

        renderer.update_scene(data, camera=main_camera_name)
        main_image = renderer.render()
        renderer.update_scene(data, camera=wrist_camera_name)
        wrist_image = renderer.render()

        observation = {
            "observation/state": libero_state,
            "observation/image": main_image,
            "observation/wrist_image": wrist_image,
            "prompt": "move slowly and try your best to pick up something near you",
        }
        return observation
    except KeyError as e:
        logging.error(f"Error: A specified name was not found in the model: {e}.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred in get_mujoco_observation: {e}")
        return None


# MODIFIED: New observation function specifically for ALOHA environment
def get_aloha_observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    tcp_body_name: str = "flange",
    left_joint_name: str = "left_driver_joint",
    right_joint_name: str = "right_driver_joint",
    cam_high_name: str = "cam1",
    cam_low_name: str = "cam2",
    cam_wrist_name: str = "flange_cam",
) -> dict | None:
    """Generates an observation dictionary in the ALOHA format using MuJoCo data."""
    try:
        # 1. State generation
        libero_state = get_libero_state_vector(data, tcp_body_name, left_joint_name, right_joint_name)
        # Per requirements: first 7 dims are 0, last 7 are from libero state
        # We take the first 7 elements of the 8-dim libero state
        aloha_state = np.concatenate([np.zeros(7), libero_state[:7]])

        # 2. Image generation
        # Render images from specified cameras
        renderer.update_scene(data, camera=cam_high_name)
        cam_high_img = renderer.render().transpose(2, 0, 1)  # HWC -> CHW

        renderer.update_scene(data, camera=cam_low_name)
        cam_low_img = renderer.render().transpose(2, 0, 1)  # HWC -> CHW

        renderer.update_scene(data, camera=cam_wrist_name)
        cam_wrist_img = renderer.render().transpose(2, 0, 1)  # HWC -> CHW

        # Create a black image for the left wrist
        black_img = np.zeros((3, renderer.height, renderer.width), dtype=np.uint8)

        # 3. Assemble final observation dictionary
        observation = {
            "state": aloha_state,
            "images": {
                "cam_high": cam_high_img,
                "cam_low": cam_low_img,
                "cam_left_wrist": black_img,
                "cam_right_wrist": cam_wrist_img,
            },
            "prompt": "do something",
        }
        return observation
    except KeyError as e:
        logging.error(f"Error: A specified camera name was not found in the model: {e}.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred in get_aloha_observation: {e}")
        return None


def _random_observation_droid() -> dict:
    """Generates a random observation dictionary in the DROID format."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def main(args: Args) -> None:
    """Main function to run the simulation controlled by the policy server."""
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

    ARM_ACTUATOR_INDICES = range(6)
    GRIPPER_ACTUATOR_INDEX = 6

    limit = 3.1415
    for i in ARM_ACTUATOR_INDICES:
        if model.actuator_ctrllimited[i]:
            model.actuator_ctrlrange[i][0] = -limit
            model.actuator_ctrlrange[i][1] = limit
    logging.info(f"Arm joint limits programmatically set to [{-limit}, {limit}].")

    actuator_info = get_actuator_info(model)
    if model.nu < 7:
        logging.error(f"Model must have at least 7 actuators (6 for arm, 1 for gripper), but found {model.nu}.")
        return

    gripper_range = actuator_info[GRIPPER_ACTUATOR_INDEX]["range"]

    try:
        eef_body_id = model.body("flange").id
        mocap_id = model.body("gripper_mocap").id
    except KeyError as e:
        logging.error(f"Mocap setup error: XML is missing a required body ('flange' or 'gripper_mocap'). Details: {e}")
        return

    # --- 2. Define the Observation Function based on Environment ---
    # MODIFIED: Select observation function based on the --env argument
    logging.info(f"Setting up observation function for environment: {args.env.value}")
    if args.env in [EnvMode.ALOHA, EnvMode.ALOHA_SIM]:
        obs_fn = lambda: get_aloha_observation(model, data, renderer)
    elif args.env == EnvMode.DROID:
        obs_fn = _random_observation_droid  # Droid uses dummy data for now
    else:  # Default to LIBERO which uses the MuJoCo simulation
        obs_fn = lambda: get_mujoco_observation(model, data, renderer)

    # --- 3. Connect to Policy Server ---
    try:
        policy = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
        logging.info(f"Initial Server metadata: {policy.get_server_metadata()}")
    except Exception as e:
        logging.error(f"Failed to connect to the policy server at {args.host}:{args.port}. Error: {e}")
        return

    # --- 4. Warm up the server ---
    logging.info("Warming up the server...")
    update_mocap_pose(data, eef_body_id, mocap_id)
    mujoco.mj_step(model, data)
    warmup_obs = obs_fn()
    if warmup_obs is None:
        logging.error("Failed to get initial observation during warmup. Exiting.")
        return
    for _ in range(2):
        policy.infer(warmup_obs)

    # --- 5. Main Simulation Loop ---
    with mujoco.viewer.launch_passive(model, data) as viewer:
        logging.info("\nSimulation started. Receiving actions from server to control the robot.")
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
            action_sequence = response.get("actions")

            # MODIFIED: c. Check if the received action is valid based on env
            is_action_valid = True
            if not isinstance(action_sequence, (list, np.ndarray)) or len(action_sequence) == 0:
                is_action_valid = False
            elif args.env in [EnvMode.ALOHA, EnvMode.ALOHA_SIM]:
                if not all(len(sub) == 14 for sub in action_sequence):
                    is_action_valid = False
            elif not all(len(sub) == 7 for sub in action_sequence):
                is_action_valid = False

            if not is_action_valid:
                logging.warning(
                    f"Received invalid or empty 'actions' for env '{args.env.value}', skipping: {action_sequence}"
                )
                update_mocap_pose(data, eef_body_id, mocap_id)
                mujoco.mj_step(model, data)
                viewer.sync()
                continue

            # d. Execute the action sequence in the simulation
            for sub_action in action_sequence:
                if not viewer.is_running():
                    break

                # MODIFIED: Select the correct part of the action vector based on env
                if args.env in [EnvMode.ALOHA, EnvMode.ALOHA_SIM]:
                    # For ALOHA, use the last 7 dimensions of the 14-dim action
                    action_to_execute = sub_action[7:]
                else:
                    # For other envs, use the full 7-dim action
                    action_to_execute = sub_action

                joint_angles = list(action_to_execute[:6])
                gripper_value = action_to_execute[6]

                final_joint_angles = joint_angles
                movej_mujoco(data, final_joint_angles, ARM_ACTUATOR_INDICES)

                gripper_amplitude = max(0, min(255, abs(gripper_value)))
                control_gripper_mujoco(data, GRIPPER_ACTUATOR_INDEX, gripper_range, gripper_amplitude)

                # Hold the control signal for a number of simulation steps
                for _ in range(args.hold_steps):
                    if not viewer.is_running():
                        break
                    update_mocap_pose(data, eef_body_id, mocap_id)
                    mujoco.mj_step(model, data)
                    viewer.sync()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main(tyro.cli(Args))
