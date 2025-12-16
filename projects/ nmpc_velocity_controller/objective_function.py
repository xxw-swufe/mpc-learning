"""
四足机器人MPC目标函数模块

从 Quadruped-PyMPC 的 centroidal_nmpc_nominal.py 提取的独立目标函数实现。
该模块提供了清晰、可配置的目标函数定义，可用于任何四足机器人MPC控制器。

作者：从 centroidal_nmpc_nominal.py 提取
"""

import numpy as np
import casadi as cs
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class ObjectiveWeights:
    """目标函数权重配置类"""

    # 状态权重 Q
    Q_position: np.ndarray = None          # [x, y, z] 位置权重
    Q_velocity: np.ndarray = None          # [vx, vy, vz] 速度权重
    Q_base_angle: np.ndarray = None        # [roll, pitch, yaw] 姿态角权重
    Q_base_angle_rates: np.ndarray = None  # [roll_rate, pitch_rate, yaw_rate] 角速度权重
    Q_foot_pos: np.ndarray = None          # 足端位置权重（每只脚3个）

    # 积分项权重（用于消除稳态误差）
    Q_com_position_z_integral: float = 50.0
    Q_com_velocity_x_integral: float = 10.0
    Q_com_velocity_y_integral: float = 10.0
    Q_com_velocity_z_integral: float = 10.0
    Q_roll_integral: float = 10.0
    Q_pitch_integral: float = 10.0

    # 控制输入权重 R
    R_foot_vel: np.ndarray = None          # 足端速度权重（每只脚3个）
    R_foot_force: np.ndarray = None        # 足端力权重（每只脚3个）

    def __post_init__(self):
        """初始化默认权重"""
        if self.Q_position is None:
            self.Q_position = np.array([0, 0, 1500])  # 重点是z轴高度
        if self.Q_velocity is None:
            self.Q_velocity = np.array([200, 200, 200])
        if self.Q_base_angle is None:
            self.Q_base_angle = np.array([500, 500, 0])  # yaw不控制
        if self.Q_base_angle_rates is None:
            self.Q_base_angle_rates = np.array([20, 20, 50])
        if self.Q_foot_pos is None:
            self.Q_foot_pos = np.array([300, 300, 300])
        if self.R_foot_vel is None:
            self.R_foot_vel = np.array([0.0001, 0.0001, 0.00001])
        if self.R_foot_force is None:
            self.R_foot_force = np.array([0.001, 0.001, 0.001])


class QuadrupedObjectiveFunction:
    """
    四足机器人MPC目标函数类

    该类封装了四足机器人模型预测控制的目标函数，包括：
    - 状态跟踪误差最小化
    - 控制输入惩罚
    - 积分动作（消除稳态误差）
    """

    def __init__(self, weights: Optional[ObjectiveWeights] = None):
        """
        初始化目标函数

        Args:
            weights: 目标函数权重配置，如果为None则使用默认权重
        """
        self.weights = weights or ObjectiveWeights()

        # 状态和输入维度（基于质心模型）
        self.nx = 24  # 状态维度
        self.nu = 24  # 输入维度（12足端速度 + 12足端力）

    def create_weight_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建权重矩阵 Q 和 R

        Returns:
            Q: 状态权重矩阵 (nx x nx)
            R: 控制输入权重矩阵 (nu x nu)
        """
        # 构建状态权重矩阵 Q
        Q_diag = np.concatenate([
            self.weights.Q_position,                    # 位置 [3]
            self.weights.Q_velocity,                    # 速度 [3]
            self.weights.Q_base_angle,                  # 姿态角 [3]
            self.weights.Q_base_angle_rates,            # 角速度 [3]
            # 足端位置 (4只脚 x 3个) [12]
            self.weights.Q_foot_pos,  # FL
            self.weights.Q_foot_pos,  # FR
            self.weights.Q_foot_pos,  # RL
            self.weights.Q_foot_pos,  # RR
            # 积分项 [6]
            [self.weights.Q_com_position_z_integral],
            [self.weights.Q_com_velocity_x_integral],
            [self.weights.Q_com_velocity_y_integral],
            [self.weights.Q_com_velocity_z_integral],
            [self.weights.Q_roll_integral],
            [self.weights.Q_pitch_integral],
        ])

        Q = np.diag(Q_diag)

        # 构建控制输入权重矩阵 R
        R_diag = np.concatenate([
            # 足端速度 (4只脚 x 3个) [12]
            self.weights.R_foot_vel,   # FL
            self.weights.R_foot_vel,   # FR
            self.weights.R_foot_vel,   # RL
            self.weights.R_foot_vel,   # RR
            # 足端力 (4只脚 x 3个) [12]
            self.weights.R_foot_force,  # FL
            self.weights.R_foot_force,  # FR
            self.weights.R_foot_force,  # RL
            self.weights.R_foot_force,  # RR
        ])

        R = np.diag(R_diag)

        return Q, R

    def create_cost_expression(self, x: cs.SX, u: cs.SX, x_ref: cs.SX, u_ref: cs.SX) -> cs.SX:
        """
        创建目标函数表达式（二次型）

        Args:
            x: 状态变量 (nx x 1)
            u: 控制输入变量 (nu x 1)
            x_ref: 状态参考 (nx x 1)
            u_ref: 控制输入参考 (nu x 1)

        Returns:
            cost: 目标函数标量值
        """
        Q, R = self.create_weight_matrices()

        # 状态误差成本
        x_error = x - x_ref
        state_cost = cs.bilin(x_error.T @ Q @ x_error)

        # 控制输入成本
        u_error = u - u_ref
        control_cost = cs.bilin(u_error.T @ R @ u_error)

        # 总成本
        total_cost = state_cost + control_cost

        return total_cost

    def setup_acados_cost(self, ocp, model_x: cs.SX, model_u: cs.SX):
        """
        为Acados OCP设置线性最小二乘成本

        Args:
            ocp: Acados OCP对象
            model_x: 模型状态变量
            model_u: 模型输入变量
        """
        Q, R = self.create_weight_matrices()

        # 设置为线性最小二乘形式
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"

        # 设置权重矩阵
        ocp.cost.W_e = Q  # 终端权重
        ocp.cost.W = scipy.linalg.block_diag(Q, R)  # 阶段权重

        # 输出变量定义 [x; u]
        ny = self.nx + self.nu
        ocp.cost.Vx = np.zeros((ny, self.nx))
        ocp.cost.Vx[:self.nx, :self.nx] = np.eye(self.nx)

        ocp.cost.Vu = np.zeros((ny, self.nu))
        ocp.cost.Vu[self.nx:self.nx+self.nu, 0:self.nu] = np.eye(self.nu)

        ocp.cost.Vx_e = np.eye(self.nx)

        # 初始化参考
        ocp.cost.yref = np.zeros((ny,))
        ocp.cost.yref_e = np.zeros((self.nx,))

    def get_reference_trajectory(self, reference_dict: Dict, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        从参考字典构建完整的参考轨迹

        Args:
            reference_dict: 包含所有参考值的字典
            horizon: 预测时域长度

        Returns:
            yref: 阶段参考轨迹 (horizon x (nx + nu))
            yref_e: 终端参考轨迹 (nx,)
        """
        # 构建单个时间步的参考
        yref_step = np.zeros(self.nx + self.nu)
        yref_step[0:3] = reference_dict.get("ref_position", np.zeros(3))
        yref_step[3:6] = reference_dict.get("ref_linear_velocity", np.zeros(3))
        yref_step[6:9] = reference_dict.get("ref_orientation", np.zeros(3))
        yref_step[9:12] = reference_dict.get("ref_angular_velocity", np.zeros(3))

        # 足端位置参考
        foot_refs = ["ref_foot_FL", "ref_foot_FR", "ref_foot_RL", "ref_foot_RR"]
        for i, foot_ref in enumerate(foot_refs):
            foot_pos = reference_dict.get(foot_ref, np.zeros((1, 3)))
            yref_step[12+3*i:15+3*i] = foot_pos[0] if len(foot_pos.shape) > 1 else foot_pos

        # 足端力参考（通常为零，除了支撑腿的z方向力）
        yref_step[44:] = 0  # 足端力参考（全设为0，实际会在compute_control中更新）

        # 构建完整参考轨迹
        yref = np.tile(yref_step, (horizon, 1))
        yref_e = yref_step[:self.nx]  # 终端参考（只包含状态）

        return yref, yref_e

    def update_reference_forces(self, yref: np.ndarray, contact_sequence: np.ndarray,
                                mass: float, gravity: float, horizon: int) -> np.ndarray:
        """
        根据接触序列更新参考足端力

        Args:
            yref: 原始参考轨迹
            contact_sequence: 接触序列 (4 x horizon)
            mass: 机器人质量
            gravity: 重力加速度
            horizon: 预测时域

        Returns:
            yref_updated: 更新后的参考轨迹
        """
        yref_updated = yref.copy()

        for k in range(horizon):
            # 计算当前时间步的支撑腿数量
            stance_legs = contact_sequence[:, k].sum()

            if stance_legs > 0:
                # 每个支撑腿承担的z方向力
                ref_force_z = (mass * gravity) / stance_legs
            else:
                ref_force_z = 0

            # 更新每条腿的z方向力参考
            yref_updated[k, 44] = ref_force_z * contact_sequence[0, k]  # FL
            yref_updated[k, 47] = ref_force_z * contact_sequence[1, k]  # FR
            yref_updated[k, 50] = ref_force_z * contact_sequence[2, k]  # RL
            yref_updated[k, 53] = ref_force_z * contact_sequence[3, k]  # RR

        return yref_updated


# 预定义的权重配置
DEFAULT_WEIGHTS = ObjectiveWeights()

# 针对不同场景的权重配置
BALANCE_WEIGHTS = ObjectiveWeights(
    Q_position=np.array([0, 0, 2000]),      # 更强的z轴控制
    Q_velocity=np.array([300, 300, 300]),  # 更强的速度控制
    Q_base_angle=np.array([800, 800, 0]),  # 更强的姿态控制
)

AGILE_WEIGHTS = ObjectiveWeights(
    Q_position=np.array([100, 100, 500]),   # 更灵活的位置控制
    Q_velocity=np.array([100, 100, 100]),  # 较弱的速度控制以便快速运动
    R_foot_vel=np.array([0.001, 0.001, 0.0001]),  # 更大的足端速度容忍度
)

ENERGY_EFFICIENT_WEIGHTS = ObjectiveWeights(
    Q_velocity=np.array([50, 50, 50]),      # 较弱的速度跟踪以节能
    R_foot_force=np.array([0.01, 0.01, 0.01]),  # 更大的力惩罚以减少能耗
    R_foot_vel=np.array([0.001, 0.001, 0.0001]),
)


if __name__ == "__main__":
    """示例：如何使用目标函数模块"""

    # 创建目标函数
    obj_func = QuadrupedObjectiveFunction(BALANCE_WEIGHTS)

    # 获取权重矩阵
    Q, R = obj_func.create_weight_matrices()
    print(f"状态权重矩阵Q的形状: {Q.shape}")
    print(f"控制权重矩阵R的形状: {R.shape}")

    # 示例：创建CasADi符号变量
    x = cs.SX.sym('x', obj_func.nx)
    u = cs.SX.sym('u', obj_func.nu)
    x_ref = cs.SX.sym('x_ref', obj_func.nx)
    u_ref = cs.SX.sym('u_ref', obj_func.nu)

    # 创建目标函数表达式
    cost = obj_func.create_cost_expression(x, u, x_ref, u_ref)
    print(f"\n目标函数表达式: {cost}")

    # 示例参考轨迹
    reference_dict = {
        "ref_position": np.array([0, 0, 0.3]),
        "ref_linear_velocity": np.array([0.5, 0, 0]),
        "ref_orientation": np.array([0, 0, 0]),
        "ref_foot_FL": np.array([[0.2, 0.15, -0.3]]),
        "ref_foot_FR": np.array([[0.2, -0.15, -0.3]]),
        "ref_foot_RL": np.array([[-0.2, 0.15, -0.3]]),
        "ref_foot_RR": np.array([[-0.2, -0.15, -0.3]]),
    }

    yref, yref_e = obj_func.get_reference_trajectory(reference_dict, horizon=10)
    print(f"\n阶段参考轨迹形状: {yref.shape}")
    print(f"终端参考轨迹形状: {yref_e.shape}")
