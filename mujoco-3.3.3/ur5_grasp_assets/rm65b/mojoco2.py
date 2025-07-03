##########主从关系交换桌面######################
import numpy as np
import mujoco
from mujoco import viewer
import os
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from scipy.linalg import svd
import time
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev  # 用于样条插值

from scipy.interpolate import interp1d

import pandas as pd




# RRT–Connect规划器：同时从起点和目标出发构建双向树，并尝试连接
class RRTConnectPlanner:
    class Node:
        def __init__(self, q: np.ndarray, parent: 'RRTConnectPlanner.Node' = None):
            self.q = q
            self.parent = parent

    def __init__(self, start: np.ndarray, goal: np.ndarray, sample_fn, dist_fn, steer_fn, collision_fn, goal_bias=0.05):
        self.start_tree = [self.Node(start)]
        self.goal_tree = [self.Node(goal)]
        self.sample_fn = sample_fn
        self.dist_fn = dist_fn
        self.steer_fn = steer_fn
        self.collision_fn = collision_fn
        self.goal_bias = goal_bias

    def nearest(self, tree, q_rand):
        dists = [self.dist_fn(node.q, q_rand) for node in tree]
        return tree[np.argmin(dists)]

    def extend(self, tree, q_target, eps):
        nearest_node = self.nearest(tree, q_target)
        q_new = self.steer_fn(nearest_node.q, q_target, eps)
        if self.collision_fn(q_new):
            new_node = self.Node(q_new, parent=nearest_node)
            tree.append(new_node)
            return new_node
        return None

    def connect(self, tree, q_target, eps):
        new_node = self.extend(tree, q_target, eps)
        while new_node is not None and self.dist_fn(new_node.q, q_target) >= eps:
            new_node = self.extend(tree, q_target, eps)
        return new_node

    def extract_path(self, tree, node):
        path = []
        while node is not None:
            path.append(node.q)
            node = node.parent
        return path[::-1]

    def plan(self, max_iter=3000, eps=0.1):
        for i in range(max_iter):
            if np.random.rand() < self.goal_bias:
                q_rand = self.goal_tree[-1].q
            else:
                q_rand = self.sample_fn()
            new_node_master = self.extend(self.start_tree, q_rand, eps)
            if new_node_master is not None:
                new_node_slave = self.connect(self.goal_tree, new_node_master.q, eps)
                if new_node_slave is not None:
                    path_from_master = self.extract_path(self.start_tree, new_node_master)
                    path_from_slave = self.extract_path(self.goal_tree, new_node_slave)
                    path_from_slave.reverse()
                    return path_from_master + path_from_slave
            self.start_tree, self.goal_tree = self.goal_tree, self.start_tree
        return None


class DualArmRRTControl:
    """
    双臂控制系统
    （注意：经过修改后，主臂为左臂，从臂为右臂）

    功能：
      1. 主臂（左臂）通过多次逆向运动学求解后，利用线性插值慢速运动到目标状态；
      2. 当主臂稳定在目标状态后，从臂（右臂）利用 RRT–Connect规划路径，再通过逆向运动学求解目标姿态，
         并沿规划路径慢速运动；
      3. 整个 MuJoCo 动画显示完整的主臂和从臂运动过程（播放速度放慢 5 倍），
         程序结束后等待用户按回车键手动关闭窗口；
      4. 同时利用 matplotlib 绘制主臂轨迹、从臂原始轨迹、RRT 树节点以及样条平滑后的运动轨迹。

    目标设定：
      - 目标位置“反转”：左臂目标取原 target_right；右臂目标取原 target_left；
      - 目标姿态均使末端工具 Z 轴朝向全局 -Z，主臂在此基础上再绕 Y 轴旋转45°；
      - 对于从臂 IK 求解，允许目标姿态最多存在 3° 内偏差。
    """

    def __init__(self, model_path: str, rrt_attempts=10, slowdown_factor=5):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"XML 文件未找到: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        self.slowdown_factor = slowdown_factor
        self._init_arm_config()
        self.rrt_attempts = rrt_attempts

        # 获取左右臂 body id（交换后，主臂为左臂，从臂为右臂）
        self.master_body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                                for name in ['Link1_1', 'Link2_1', 'Link3_1', 'Link4_1', 'Link5_1', 'Link6_1']]
        self.slave_body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                               for name in ['Link1', 'Link2', 'Link3', 'Link4', 'Link5', 'Link6']]
        # 目标位置反转：
        # 主臂目标取原 target_right -> [0, -0.01, 0.08]
        # 从臂目标取原 target_left  -> [0, -0.08, 0.08]
        self.target_positions = {
            'left': np.array([0, -0.01, 0.08]),   # 主臂（左臂）
            'right': np.array([0, -0.08, 0.08])    # 从臂（右臂）
        }
        # 目标姿态：原左臂采用绕 X 轴旋转180°；原右臂在此基础上再绕 Y 轴旋转45°
        target_quat_left = R.from_euler('x', 180, degrees=True).as_quat()
        target_quat_right = (R.from_euler('y', 0, degrees=True) *
                             R.from_euler('x', 180, degrees=True)).as_quat()
        # 主臂（左）采用原右臂的目标姿态， 从臂（右）采用原左臂的目标姿态
        self.target_orientation = {
            'left': target_quat_right,  # 主臂：多了额外旋转（原右臂的配置）
            'right': target_quat_left   # 从臂：保持原来
        }
        # 初始配置（确保在 IK 可达区域内），交换后：主臂（左臂）使用非零初始配置；从臂（右臂）使用零初始配置
        self.initial_q = {
            'left': np.array([1.58, 0.635, 1.37, 0.062, 1.16, -1.38]),
            'right': np.array([0, 0, 0, 0, 0, 0])
        }

    def _init_arm_config(self):
        # 关节配置：模型中 "joint1_1" 属于左臂，"joint1" 属于右臂
        self.joint_ids = {
            'right': [self.model.joint(name).id for name in
                      ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']],
            'left': [self.model.joint(name).id for name in
                     ['joint1_1', 'joint2_1', 'joint3_1', 'joint4_1', 'joint5_1', 'joint6_1']]
        }
        # 站点配置：主臂采用左侧工具（left_tool），从臂采用右侧工具（right_tool）
        self.site_ids = {
            'left': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'left_tool'),
            'right': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'right_tool')
        }
        if any(sid < 0 for sid in self.site_ids.values()):
            raise RuntimeError("末端站点必须正确定义")

    def _safe_rotm(self, M: np.ndarray) -> np.ndarray:
        U, _, Vt = svd(M)
        Rm = U @ Vt
        if np.linalg.det(Rm) < 0:
            Rm[:, 1] *= -1
        return Rm

    def _get_site_pose(self, arm: str) -> dict:
        site = self.data.site(self.site_ids[arm])
        Rm = site.xmat.reshape(3, 3)
        return {'pos': site.xpos.copy(), 'quat': R.from_matrix(self._safe_rotm(Rm)).as_quat()}

    def _check_master_slave_collision(self) -> bool:
        mujoco.mj_forward(self.model, self.data)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            if (b1 in self.master_body_ids and b2 in self.slave_body_ids) or \
                    (b2 in self.master_body_ids and b1 in self.slave_body_ids):
                return True
        return False

    def is_collision_free(self, q: np.ndarray) -> bool:
        # 从臂（右臂）位置 q，与主臂现有配置 self.master_q（左臂）结合检查碰撞
        self.data.qpos[self.joint_ids['left']] = self.master_q
        self.data.qpos[self.joint_ids['right']] = q
        return not self._check_master_slave_collision()

    def plan_rrt_connect(self, start: np.ndarray, goal: np.ndarray, max_iter=3000, eps=0.05):
        sample = lambda: np.random.uniform(
            [self.model.jnt_range[j][0] for j in self.joint_ids['right']],
            [self.model.jnt_range[j][1] for j in self.joint_ids['right']]
        )
        dist = lambda a, b: np.linalg.norm(a - b)
        steer = lambda a, b, eps: b if dist(a, b) < eps else a + (b - a) / dist(a, b) * eps
        planner = RRTConnectPlanner(start, goal, sample, dist, steer, self.is_collision_free)
        path = planner.plan(max_iter=max_iter, eps=eps)
        # 将两棵树的节点合并存储，以便进行绘图
        self.rrt_tree = planner.start_tree + planner.goal_tree
        return path

    def _inverse_kinematics(self, arm: str, target_pos: np.ndarray, target_quat: np.ndarray = None,
                            max_attempts=10, tolerance=None) -> np.ndarray:
        best, err = None, 1e9
        current_q = self.data.qpos[self.joint_ids[arm]].copy()

        def cost(q):
            self.data.qpos[self.joint_ids[arm]] = q
            mujoco.mj_fwdPosition(self.model, self.data)
            cur = self._get_site_pose(arm)
            pe = np.linalg.norm(cur['pos'] - target_pos)
            re = 0
            if target_quat is not None:
                if arm == 'right':  # 从臂允许最多3°偏差
                    q_current = cur['quat']
                    dot_prod = np.abs(np.dot(q_current, target_quat))
                    dot_prod = np.clip(dot_prod, 0.0, 1.0)
                    angle_diff = 2 * np.degrees(np.arccos(dot_prod))
                    re = max(angle_diff - 3, 0)
                else:
                    diff = R.from_quat(cur['quat']) * R.from_quat(target_quat).inv()
                    euler_diff = diff.as_euler('xyz', True)
                    re = sum(abs(angle) for angle in euler_diff)
            return pe + 1.5 * re

        for attempt in range(max_attempts):
            if attempt == 0:
                q0 = current_q
            else:
                q0 = np.random.uniform(
                    [self.model.jnt_range[j][0] for j in self.joint_ids[arm]],
                    [self.model.jnt_range[j][1] for j in self.joint_ids[arm]]
                )
            res = minimize(cost, q0, method='SLSQP',
                           bounds=[self.model.jnt_range[j] for j in self.joint_ids[arm]],
                           options={'ftol': 1e-6, 'maxiter': 500})
            if res.success and res.fun < err:
                best, err = res.x, res.fun
                if tolerance is not None and err < tolerance:
                    break
        if best is None:
            print("未找到满足条件的逆向运动解, 返回当前配置")
            best = current_q
        return best

    def run(self):
        # Phase 1: 主臂（左臂）多次IK求解并慢速插值运动，直到达到目标状态
        max_master_attempts = 5
        err_threshold = 0.005  # 允许误差5毫米
        target_master = None
        for attempt in range(max_master_attempts):
            candidate = self._inverse_kinematics(
                arm='left',
                target_pos=self.target_positions['left'],
                target_quat=self.target_orientation['left'],
                tolerance=1e-3
            )
            self.data.qpos[self.joint_ids['left']] = candidate
            mujoco.mj_forward(self.model, self.data)
            master_pose = self._get_site_pose('left')
            pos_err = np.linalg.norm(master_pose['pos'] - self.target_positions['left'])
            print(f"主臂 IK 尝试 {attempt + 1}, 末端位置误差：{pos_err:.4f}")
            if pos_err < err_threshold:
                target_master = candidate
                break
        if target_master is None:
            print("主臂多次求解未能达到目标状态，采用最后一次结果")
            target_master = candidate

        # 主臂慢速插值运动动画 (左臂运动)
        init_master = self.initial_q['left']
        steps = 50
        master_traj = []
        for i in range(steps):
            p = (i + 1) / steps
            q = (1 - p) * init_master + p * target_master
            self.data.qpos[self.joint_ids['left']] = q
            self.data.qpos[self.joint_ids['right']] = self.initial_q['right']
            mujoco.mj_step(self.model, self.data)
            time.sleep(self.dt * self.slowdown_factor)
            master_traj.append(self._get_site_pose('left')['pos'].copy())
        self.master_q = target_master

        # 等待确保主臂稳定在目标状态
        master_pose = self._get_site_pose('left')
        pos_err = np.linalg.norm(master_pose['pos'] - self.target_positions['left'])
        max_wait_time = 3.0
        start_time = time.time()
        while pos_err > err_threshold and (time.time() - start_time) < max_wait_time:
            mujoco.mj_step(self.model, self.data)
            master_pose = self._get_site_pose('left')
            pos_err = np.linalg.norm(master_pose['pos'] - self.target_positions['left'])
            time.sleep(0.001)
        if pos_err > err_threshold:
            print(f"警告：主臂未完全达到目标状态，误差 {pos_err:.4f}")
        else:
            print(f"主臂达到目标状态，误差 {pos_err:.4f}")

        # Phase 2: 当主臂稳定后，从臂（右臂）利用 RRT–Connect 规划路径与 IK求解
        slave_goal = self._inverse_kinematics(
            arm='right',
            target_pos=self.target_positions['right'],
            target_quat=self.target_orientation['right']
        )
        path = None
        for attempt in range(self.rrt_attempts):
            path = self.plan_rrt_connect(self.initial_q['right'], slave_goal, max_iter=3000, eps=0.05)
            if path is not None:
                print(f"RRT–Connect 第 {attempt + 1} 次尝试成功找到路径")
                break
            else:
                print(f"RRT–Connect 第 {attempt + 1} 次尝试未找到路径，重试...")
        if path is None:
            raise RuntimeError(f"经过 {self.rrt_attempts} 次尝试，未找到可行路径")

        # Phase 3: 从臂（右臂）沿规划路径慢速插值运动动画，并记录轨迹
        slave_traj = []

        slave_joint_angles = []  # 新增：记录从臂关节角度

        master_joint_angles = []  # 新增：主臂关节角度记录

        v = viewer.launch_passive(self.model, self.data)
        for q in path:
            self.data.qpos[self.joint_ids['left']] = self.master_q
            self.data.qpos[self.joint_ids['right']] = q
            mujoco.mj_step(self.model, self.data)

            slave_traj.append(self._get_site_pose('right')['pos'].copy())

            # 记录当前从臂关节角度（弧度）
            current_angles = self.data.qpos[self.joint_ids['right']].copy()
            slave_joint_angles.append(current_angles)

            master_joint_angles.append(self.data.qpos[self.joint_ids['left']].copy())  # 新增

            v.sync()
            time.sleep(self.dt * self.slowdown_factor)
            if self._check_master_slave_collision():
                print("检测到主从臂碰撞，停止路径执行")
                break

        slave_joint_angles = np.array(slave_joint_angles)  # 转换为NumPy数组
        master_joint_angles = np.array(master_joint_angles)  # 新增

        # Phase 4: 利用样条插值平滑从臂轨迹，并使用 matplotlib 绘制三维轨迹图
        slave_traj_arr = np.array(slave_traj)
        try:
            tck, u = splprep(slave_traj_arr.T, s=2)
            u_new = np.linspace(0, 1, 200)
            smoothed_slave_traj = np.array(splev(u_new, tck)).T
        except Exception as e:
            print("轨迹插值平滑出错：", e)
            smoothed_slave_traj = slave_traj_arr

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        tree_pts = np.array([self._compute_pose_for_q(node.q) for node in self.rrt_tree])
        ax.scatter(tree_pts[:, 0], tree_pts[:, 1], tree_pts[:, 2],
                   s=5, c='gray', alpha=0.5, label='RRT Tree')
        path_pts = np.array([self._compute_pose_for_q(q) for q in path])
        ax.plot(path_pts[:, 0], path_pts[:, 1], path_pts[:, 2],
                c='orange', linewidth=2, label='Planned Path')
        mt = np.array(master_traj)
        st = np.array(slave_traj)
        ax.plot(mt[:, 0], mt[:, 1], mt[:, 2],
                c='blue', linestyle='--', label='Master Trajectory')
        ax.plot(st[:, 0], st[:, 1], st[:, 2],
                c='red', label='Slave Trajectory (Raw)')
        ax.plot(smoothed_slave_traj[:, 0], smoothed_slave_traj[:, 1], smoothed_slave_traj[:, 2],
                c='green', linewidth=2, label='Slave Trajectory (Smoothed)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend() 
        plt.show()

        # 保存主臂关节角度（弧度）
        df_master_rad = pd.DataFrame({
            'Master_Joint1_rad': master_joint_angles[:, 0],
            'Master_Joint2_rad': master_joint_angles[:, 1],
            'Master_Joint3_rad': master_joint_angles[:, 2],
            'Master_Joint4_rad': master_joint_angles[:, 3],
            'Master_Joint5_rad': master_joint_angles[:, 4],
            'Master_Joint6_rad': master_joint_angles[:, 5]
        })

        # 保存主臂关节角度（角度）
        df_master_deg = pd.DataFrame({
            'Master_Joint1_deg': np.degrees(master_joint_angles[:, 0]),
            'Master_Joint2_deg': np.degrees(master_joint_angles[:, 1]),
            'Master_Joint3_deg': np.degrees(master_joint_angles[:, 2]),
            'Master_Joint4_deg': np.degrees(master_joint_angles[:, 3]),
            'Master_Joint5_deg': np.degrees(master_joint_angles[:, 4]),
            'Master_Joint6_deg': np.degrees(master_joint_angles[:, 5])
        })

        # 导出到独立文件
        master_excel_path = 'master_joint_angles_deg.xlsx'
        df_master_deg.to_excel(master_excel_path, index=False)
        print(f"主臂关节角度已导出至：{master_excel_path}")

        # 构建DataFrame（仅包含从臂6个关节角度）
        df = pd.DataFrame({
            'Joint1_rad': slave_joint_angles[:, 0],
            'Joint2_rad': slave_joint_angles[:, 1],
            'Joint3_rad': slave_joint_angles[:, 2],
            'Joint4_rad': slave_joint_angles[:, 3],
            'Joint5_rad': slave_joint_angles[:, 4],
            'Joint6_rad': slave_joint_angles[:, 5]
        })

        # 新增：创建角度制DataFrame
        df_deg = pd.DataFrame({
            'Joint1_deg': np.degrees(slave_joint_angles[:, 0]),
            'Joint2_deg': np.degrees(slave_joint_angles[:, 1]),
            'Joint3_deg': np.degrees(slave_joint_angles[:, 2]),
            'Joint4_deg': np.degrees(slave_joint_angles[:, 3]),
            'Joint5_deg': np.degrees(slave_joint_angles[:, 4]),
            'Joint6_deg': np.degrees(slave_joint_angles[:, 5])
        })

        # 保存到Excel
        excel_path = 'slave_joint_angles_deg.xlsx'
        df_deg.to_excel(excel_path, index=False)




        print(f"从臂关节角度已导出至：{excel_path}")

        print("运动执行结束。MuJoCo 窗口将保持打开。")
        input("请按回车键退出程序：")  # 手动退出

    def _compute_pose_for_q(self, q: np.ndarray) -> np.ndarray:
        # 对于从臂（右臂）的姿态计算：固定主臂（左臂）配置为 self.master_q
        self.data.qpos[self.joint_ids['right']] = q
        self.data.qpos[self.joint_ids['left']] = self.master_q
        mujoco.mj_forward(self.model, self.data)
        return self.data.site(self.site_ids['right']).xpos.copy()


if __name__ == "__main__":
    controller = DualArmRRTControl(
        r"C:/app/mujoco/mujoco-3.3.1-windows-x86_64/model/table/robotontable.xml",
        rrt_attempts=10,
        slowdown_factor=5
    )
    controller.run()










