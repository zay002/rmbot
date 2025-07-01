# import numpy as np
# import random
#
# class Node:
#     def __init__(self, state):
#         self.state = np.array(state)
#         self.parent = None
#
# class RRT:
#     def __init__(self, start, goal, rand_area, expand_dis=0.1, path_resolution=0.05,
#                  goal_sample_rate=10, max_iter=500, is_collision_free=None):
#         self.start = Node(start)
#         self.goal = Node(goal)
#         self.min_rand, self.max_rand = zip(*rand_area)
#         self.expand_dis = expand_dis
#         self.path_resolution = path_resolution
#         self.goal_sample_rate = goal_sample_rate
#         self.max_iter = max_iter
#         self.node_list = [self.start]
#         self.is_collision_free = is_collision_free or (lambda x: True)
#
#     def planning(self):
#         for _ in range(self.max_iter):
#             if random.randint(0, 100) > self.goal_sample_rate:
#                 rnd = self._sample_random()
#             else:
#                 rnd = self.goal.state
#
#             nearest = self._get_nearest_node(rnd)
#             new_node = self._steer(nearest, rnd)
#
#             if self.is_collision_free(new_node.state):
#                 new_node.parent = nearest
#                 self.node_list.append(new_node)
#
#                 if np.linalg.norm(new_node.state - self.goal.state) < self.expand_dis:
#                     return self._generate_final_path(new_node)
#         return None
#
#     def _sample_random(self):
#         return np.array([random.uniform(a, b) for a, b in zip(self.min_rand, self.max_rand)])
#
#     def _get_nearest_node(self, rnd):
#         dlist = [np.linalg.norm(n.state - rnd) for n in self.node_list]
#         return self.node_list[int(np.argmin(dlist))]
#
#     def _steer(self, from_node, to_state):
#         direction = to_state - from_node.state
#         dist = np.linalg.norm(direction)
#         if dist == 0:
#             return Node(from_node.state)
#         direction = direction / dist
#         move = min(self.expand_dis, dist)
#         new_state = from_node.state + direction * move
#         return Node(new_state)
#
#     def _generate_final_path(self, node):
#         path = [self.goal.state]
#         while node is not None:
#             path.append(node.state)
#             node = node.parent
#         return path[::-1]  # 反转路径
import numpy as np
import mujoco
from mujoco import viewer
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize


class DualArmKinematicsControl:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.debug_mode = True

        # 关节索引初始化
        self.right_joint_names = [f'joint{i+1}' for i in range(6)]
        self.left_joint_names = [f'joint{i+1}' for i in range(6)]
        self.right_joint_indices = [self.model.joint(name).qposadr for name in self.right_joint_names]
        self.left_joint_indices = [self.model.joint(name).qposadr for name in self.left_joint_names]

        # 末端执行器名称
        self.right_ee_site = "right_tool"
        self.left_ee_site = "left_tool"

        # 默认姿态修正角度
        self.target_orientation = {
            'right': self._get_corrected_orientation('right'),
            'left': self._get_corrected_orientation('left')
        }

    def _get_site_position(self, arm):
        site = self.right_ee_site if arm == 'right' else self.left_ee_site
        return self.data.site(site).xpos.copy()

    def _get_corrected_orientation(self, arm):
        # 姿态修正：绕 y 轴旋转 -90°，再绕 z 轴 -90°
        rot_y = R.from_euler('y', -90, degrees=True)
        rot_z = R.from_euler('z', -90, degrees=True)
        return (rot_z * rot_y).inv()

    def _update_joint_positions(self, right_q, left_q):
        for i, idx in enumerate(self.right_joint_indices):
            self.data.qpos[idx] = right_q[i]
        for i, idx in enumerate(self.left_joint_indices):
            self.data.qpos[idx] = left_q[i]

    def _get_site_orientation(self, arm):
        site = self.right_ee_site if arm == 'right' else self.left_ee_site
        mat = self.data.site(site).xmat.reshape(3, 3)
        u, _, vh = np.linalg.svd(mat)
        return R.from_matrix(np.dot(u, vh))

    def _inverse_kinematics(self, arm, target_pos, target_quat, attempts=10):
        indices = self.right_joint_indices if arm == 'right' else self.left_joint_indices

        def objective(q):
            for i, idx in enumerate(indices):
                self.data.qpos[idx] = q[i]
            mujoco.mj_forward(self.model, self.data)

            pos_err = np.linalg.norm(self._get_site_position(arm) - target_pos)
            current_quat = self._get_site_orientation(arm).as_quat()
            q_diff = R.from_quat(target_quat).inv() * R.from_quat(current_quat)
            ori_err = np.linalg.norm(q_diff.as_rotvec())
            return pos_err + ori_err

        best_q = None
        best_score = np.inf

        for _ in range(attempts):
            initial_q = np.random.uniform(low=-np.pi, high=np.pi, size=6)
            res = minimize(objective, initial_q, method='BFGS')
            if res.success and res.fun < best_score:
                best_q = res.x
                best_score = res.fun

        if best_q is None:
            raise RuntimeError(f"{arm.capitalize()} arm IK failed.")
        return best_q

    def _check_collision(self, threshold=0.1):
        """检测左右末端是否靠得太近（潜在碰撞）"""
        right_pos = self._get_site_position('right')
        left_pos = self._get_site_position('left')
        distance = np.linalg.norm(right_pos - left_pos)
        return distance < threshold

    def run(self):
        try:
            steps = 100
            # 主臂（左臂）插值轨迹
            left_traj = np.linspace([0.0, -0.6, 0], [0.0, -0.7, 0], steps)

            # 从臂（右臂）分段轨迹
            right_traj1 = np.tile([0.0, -0.7, 0.1], (steps // 2, 1))
            right_traj2 = np.linspace([0.0, -0.7, 0.1], [0.0, -0.7, 0.05], steps // 2)
            right_traj = np.vstack((right_traj1, right_traj2))

            with viewer.launch_passive(self.model, self.data) as v:
                print("主从协作轨迹控制启动...(ESC退出)")
                for i in range(steps):
                    left_target = left_traj[i]
                    right_target = right_traj[i]

                    # 解 IK
                    left_q = self._inverse_kinematics('left', left_target, self.target_orientation['left'].as_quat())
                    right_q = self._inverse_kinematics('right', right_target, self.target_orientation['right'].as_quat())

                    # 设置并更新关节
                    self._update_joint_positions(right_q, left_q)
                    mujoco.mj_step(self.model, self.data)

                    # 检查碰撞
                    if self._check_collision():
                        print(f"\n⚠️ 第{i+1}步发生潜在碰撞，跳过该步。")
                        continue

                    v.sync()
                    print(f"\r第 {i+1}/{steps} 步", end="")

        except Exception as e:
            print(f"\n运行出错: {str(e)}")
            if self.debug_mode:
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    controller = DualArmKinematicsControl("C:/Users/11193/.mujoco/mujoco3.3.1/model/fishbot_base.xml")
    controller.run()
