import dataclasses
import enum
import logging
import pathlib
import time
import sys
import os

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy 
import polars as pl
import rich
import rich.console
import rich.table
import tqdm
import tyro

# --- 机器人控制代码集成 ---
# 注意：根据用户提供的相对路径 RM_API2/Python/Robotic_Arm/rm_robot_interface.py 修改导入逻辑
try:
    # 假设此脚本位于项目中的某个位置，我们需要找到 RM_API2/Python 这个目录并添加到系统路径
    # 一个常见的项目结构是脚本在 'scripts' 目录，而 'RM_API2' 在项目根目录
    # 我们向上查找两级目录来定位项目根目录
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    api_path = os.path.join(project_root, "RM_API2", "Python")
    
    # 将API路径添加到 sys.path 的最前面，以优先使用
    sys.path.insert(0, api_path)
    
    from Robotic_Arm.rm_robot_interface import *
except ImportError:
    print("错误：无法导入 'rm_robot_interface'。")
    print("请确保您的项目结构正确，并且 'RM_API2' 文件夹位于项目根目录下。")
    print(f"脚本尝试从以下路径导入API: {api_path}")
    sys.exit(1)
# --- 机器人控制代码集成结束 ---


logger = logging.getLogger(__name__)


# --- 机器人控制函数 ---

def connect_robot(ip, port, level=3, mode=2):
    """
    初始化并连接到机械臂。
    """
    thread_mode = rm_thread_mode_e(mode)
    robot = RoboticArm(thread_mode)
    handle = robot.rm_create_robot_arm(ip, port, level)

    if handle.id == -1:
        logger.error("\n连接机械臂失败\n")
        return None
    else:
        logger.info(f"\n成功连接到机械臂: {handle.id}\n")
        return robot

def disconnect_robot(robot):
    """
    断开与机械臂的连接。
    """
    if not robot:
        return
    handle = robot.rm_delete_robot_arm()
    if handle == 0:
        logger.info("\n成功断开与机械臂的连接\n")
    else:
        logger.error("\n断开与机械臂的连接失败\n")

def movej(robot, joint, v=5, r=0, connect=0, block=1):
    """
    执行 movej 运动。
    """
    movej_result = robot.rm_movej(joint, v, r, connect, block)
    if movej_result == 0:
        logger.info("\nmovej 运动成功\n")
    else:
        logger.error(f"\nmovej 运动失败，错误代码: {movej_result}\n")

def control_gripper_open(arm, port, baudrate, timeout, modbus_address, amplitude):
    """
    控制夹爪张开的函数
    """
    # 配置控制器RS485端口为RTU主站
    if arm.rm_set_modbus_mode(port, baudrate, timeout) != 0:
        logger.warning("配置Modbus RTU模式失败")
        return

    # 关闭自动找行程指令
    write_params_auto = rm_peripheral_read_write_params_t(port, 0x9C9A, modbus_address)  # 寄存器地址为0x9C9A
    if arm.rm_write_single_register(write_params_auto, 0) != 0:
        logger.warning("关闭自动找行程失败")
        return

    # 创建写入参数结构体
    write_params = rm_peripheral_read_write_params_t(port, 0x9C40, modbus_address)  # 寄存器地址为0x9C40

    # 写入夹爪幅度控制寄存器
    result = arm.rm_write_single_register(write_params, int(amplitude))
    if result == 0:
        logger.info(f"夹爪控制成功，幅度设置为：{int(amplitude)}")
    else:
        logger.error("夹爪控制失败")
    # 短暂延时以确保指令执行
    time.sleep(0.1)

# --- 机器人控制函数结束 ---


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Args:
    """Command line arguments."""

    # --- 服务器连接参数 ---
    host: str = "202.115.65.101"
    port: int | None = 8000
    api_key: str | None = None

    # --- 机器人连接参数 ---
    robot_ip: str = "192.168.1.19"
    robot_port: int = 8080

    # --- 脚本运行参数 ---
    num_steps: int = 20
    timing_file: pathlib.Path | None = None
    env: EnvMode = EnvMode.LIBERO
    # 额外增加给前六个维度关节角度的归一化值，最大为45。
    norm: float | None = None
    # 设置为True以连接并控制实体机械臂，默认为不运行。
    run_robot: bool = False
    # 设置为True以合并每个step中的子动作序列为单个平均动作。
    average_actions: bool = False


class TimingRecorder:
    """Records timing measurements for different keys."""

    def __init__(self) -> None:
        self._timings: dict[str, list[float]] = {}

    def record(self, key: str, time_ms: float) -> None:
        """Record a timing measurement for the given key."""
        if key not in self._timings:
            self._timings[key] = []
        self._timings[key].append(time_ms)

    def get_stats(self, key: str) -> dict[str, float]:
        """Get statistics for the given key."""
        if key not in self._timings or not self._timings[key]:
            return {"mean": 0.0, "std": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
        times = self._timings[key]
        return {
            "mean": float(np.mean(times)), "std": float(np.std(times)),
            "p25": float(np.quantile(times, 0.25)), "p50": float(np.quantile(times, 0.50)),
            "p75": float(np.quantile(times, 0.75)), "p90": float(np.quantile(times, 0.90)),
            "p95": float(np.quantile(times, 0.95)), "p99": float(np.quantile(times, 0.99)),
        }

    def print_all_stats(self) -> None:
        """Print statistics for all keys in a concise format."""
        table = rich.table.Table(title="[bold blue]Timing Statistics (ms)[/bold blue]", show_header=True, header_style="bold white", border_style="blue", title_justify="center")
        table.add_column("Metric", style="cyan", justify="left", no_wrap=True)
        stat_columns = [("Mean", "yellow", "mean"), ("Std", "yellow", "std"), ("P25", "magenta", "p25"), ("P50", "magenta", "p50"), ("P75", "magenta", "p75"), ("P90", "magenta", "p90"), ("P95", "magenta", "p95"), ("P99", "magenta", "p99")]
        for name, style, _ in stat_columns:
            table.add_column(name, justify="right", style=style, no_wrap=True)
        for key in sorted(self._timings.keys()):
            stats = self.get_stats(key)
            values = [f"{stats[stat_key]:.1f}" for _, _, stat_key in stat_columns]
            table.add_row(key, *values)
        console = rich.console.Console(width=None, highlight=True)
        console.print(table)

    def write_parquet(self, path: pathlib.Path) -> None:
        """Save the timings to a parquet file."""
        logger.info(f"Writing timings to {path}")
        max_len = max(len(v) for v in self._timings.values()) if self._timings else 0
        padded_timings = {k: v + [None] * (max_len - len(v)) for k, v in self._timings.items()}
        frame = pl.DataFrame(padded_timings)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)


def main(args: Args) -> None:
    # --- 1. (可选) 连接机器人 ---
    robot_arm = None
    if args.run_robot:
        robot_arm = connect_robot(args.robot_ip, args.robot_port)
        if not robot_arm:
            logger.error("无法连接机器人，但脚本将继续运行于“干跑”模式。")
    else:
        logger.info("未提供 --run-robot 参数，跳过机器人连接。脚本将运行于“干跑”模式。")

    try:
        # --- 2. 连接策略服务器 ---
        obs_fn = {
            EnvMode.ALOHA: _random_observation_aloha,
            EnvMode.ALOHA_SIM: _random_observation_aloha,
            EnvMode.DROID: _random_observation_droid,
            EnvMode.LIBERO: _random_observation_libero,
        }[args.env]

        policy = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port, api_key=args.api_key)
        logger.info(f"Initial Server metadata: {policy.get_server_metadata()}")

        # --- 3. 预热服务器 ---
        logger.info("Warming up the server...")
        for _ in range(2):
            policy.infer(obs_fn())

        timing_recorder = TimingRecorder()
        console = rich.console.Console()

        # --- 4. 主循环：获取Action并控制机器人 ---
        for i in tqdm.trange(args.num_steps, desc="Running policy"):
            inference_start = time.time()
            response = policy.infer(obs_fn())
            timing_recorder.record("client_infer_ms", 1000 * (time.time() - inference_start))

            console.print(f"\n[bold magenta]实时收到的 Response (步骤 {i + 1}):[/bold magenta]")
            console.print(response)

            action_sequence = response.get('actions') 
            
            if isinstance(action_sequence, (list, np.ndarray)) and len(action_sequence) > 0 and all(len(sub) == 7 for sub in action_sequence):
                
                processed_sequence = action_sequence
                # 如果设置了 average_actions 参数，则计算平均值
                if args.average_actions:
                    console.print(f"[cyan]合并 {len(action_sequence)} 个子动作序列为单个平均动作...[/cyan]")
                    averaged_action = np.mean(action_sequence, axis=0)
                    processed_sequence = [averaged_action]  # 将单个平均动作视为一个长度为1的序列

                console.print(f"[cyan]开始执行包含 {len(processed_sequence)} 个子动作的序列...[/cyan]")
                
                for sub_action in processed_sequence:
                    # a. 处理并执行关节运动
                    joint_angles = list(sub_action[:6])
                    
                    # 新逻辑: 先将数据扩大15倍
                    scaled_joint_angles = [angle * 15 for angle in joint_angles]
                    
                    norm_value = 0.0
                    if args.norm is not None:
                        norm_value = min(args.norm, 45.0)
                        console.print(f"  - [blue]Normalization:[/blue] 应用归一化值: {norm_value:.2f}")
                    
                    # 然后再加上 norm 值
                    final_joint_angles = [angle + norm_value for angle in scaled_joint_angles]

                    console.print(f"  - [yellow]MoveJ (目标):[/yellow] {[f'{a:.2f}' for a in final_joint_angles]}")
                    # 只有在连接了机器人时才执行
                    if args.run_robot and robot_arm:
                        movej(robot_arm, final_joint_angles)

                    # b. 处理并执行夹爪控制
                    gripper_value = sub_action[6]
                    gripper_amplitude = abs(gripper_value)
                    gripper_amplitude = max(0, min(100, gripper_amplitude))
                    
                    console.print(f"  - [green]Gripper (目标):[/green] 幅度: {gripper_amplitude:.2f}")
                    # 只有在连接了机器人时才执行
                    if args.run_robot and robot_arm:
                        control_gripper_open(robot_arm, port=1, baudrate=115200, timeout=1, modbus_address=1, amplitude=gripper_amplitude)
            else:
                logger.warning(f"收到的 'actions' 格式无效或为空，已跳过: {action_sequence}")

            # 记录计时信息
            for key, value in response.get("server_timing", {}).items():
                timing_recorder.record(f"server_{key}", value)
            for key, value in response.get("policy_timing", {}).items():
                timing_recorder.record(f"policy_{key}", value)
        
        # --- 5. 循环结束后，打印统计数据 ---
        timing_recorder.print_all_stats()
        if args.timing_file is not None:
            timing_recorder.write_parquet(args.timing_file)

    finally:
        # --- 6. 确保断开机器人连接 ---
        if args.run_robot and robot_arm:
            logger.info("程序即将退出，断开机器人连接...")
            disconnect_robot(robot_arm)


def _random_observation_aloha() -> dict:
    return {"state": np.ones((14,)), "images": {"cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8), "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8), "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8), "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8)}, "prompt": "do something"}

def _random_observation_droid() -> dict:
    return {"observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8), "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8), "observation/joint_position": np.random.rand(7), "observation/gripper_position": np.random.rand(1), "prompt": "do something"}

def _random_observation_libero() -> dict:
    return {"observation/state": np.random.rand(8), "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8), "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8), "prompt": "do something"}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main(tyro.cli(Args))