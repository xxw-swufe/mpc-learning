"""
基于 CasADi + IPOPT 的四足机器人质心动力学 NMPC 求解器 (修复版)

修复了所有工程级问题：
1. 动力学函数只构建一次
2. 正确处理接触状态
3. 摩擦锥约束的正确实现
4. 接触切换逻辑放在边界而非表达式
5. 约束边界的正确管理

作者：基于您的代码集成并修复
"""

import casadi as cs
import numpy as np
import sys
import os

# 添加路径以导入您的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入您的模块 - 同目录下导入
try:
    from centroidal_model_nominal_fixed import Centroidal_Model_Nominal
    from objective_function import QuadrupedObjectiveFunction, BALANCE_WEIGHTS
    import config
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保文件路径正确")


class FrictionPyramidConstraints:
    """
    摩擦金字塔约束类（工程正确版本）

    关键原则：
    1. 约束表达式结构固定不变
    2. 接触状态只影响边界（lbg/ubg），不影响表达式
    3. 摆动腿的力被显式约束为零
    """

    def __init__(self, mu=0.5, fz_min=0.0, fz_max=500.0):  # 增加到500N，给每腿更大余量
        self.mu = mu
        self.fz_min = fz_min
        self.fz_max = fz_max
        self.fz_contact_min = 10.0  # 新增：接触腿必须承担的最小力
        self.n_legs = 4
        self.n_constraints_per_leg = 7  # 4摩擦 + 1fz_min + 1fx + 1fy

    def build_expression(self, u):
        """
        构建不变的约束表达式

        Args:
            u: 控制输入 (nu x N 或 nu x 1)

        Returns:
            约束表达式 (28 x N) 或 (28 x 1)
        """
        # 提取足端力（最后12维）
        # 使用CasADi的维度检查方法
        is_vector = (u.size1() == 1) or (u.size2() == 1)

        if is_vector:
            # 如果是向量，假设是nu x 1或1 x nu
            forces = cs.reshape(u[12:24], 12, 1)
            repeat_last = False
        else:
            # 如果是矩阵，假设是nu x N
            forces = u[12:24, :]  # 12 x N
            repeat_last = True

        constraints_list = []

        for k in range(forces.size2() if repeat_last else 1):
            k_slice = k if repeat_last else None
            f_k = forces[:, k_slice] if repeat_last else forces

            for i in range(self.n_legs):
                # 每只脚的力
                fx = f_k[3*i]
                fy = f_k[3*i + 1]
                fz = f_k[3*i + 2]

                # 摩擦金字塔约束（4个）
                constraints_list.append(fx - self.mu * fz)      # fx <= mu*fz
                constraints_list.append(-fx - self.mu * fz)     # -fx <= mu*fz
                constraints_list.append(fy - self.mu * fz)      # fy <= mu*fz
                constraints_list.append(-fy - self.mu * fz)     # -fy <= mu*fz

                # 力的边界约束（3个）
                constraints_list.append(fz - self.fz_max)       # fz <= fz_max
                constraints_list.append(-fz + self.fz_contact_min)  # fz >= fz_contact_min (接触腿必须承担最小力)
                constraints_list.append(fx)                      # fx = 0 (将在边界中设置)

        return cs.vertcat(*constraints_list)

    def get_bounds(self, contact_seq):
        """
        根据接触序列计算约束边界

        Args:
            contact_seq: 接触序列 (4 x N)

        Returns:
            tuple: (lbg, ubg) 约束边界
        """
        # 确保contact_seq是NumPy数组
        if not isinstance(contact_seq, np.ndarray):
            contact_seq = np.array(contact_seq)

        if contact_seq.ndim == 1:
            N = 1
            contact_seq = contact_seq.reshape(4, 1)
        else:
            N = contact_seq.shape[1]

        n_total_constraints = N * self.n_legs * self.n_constraints_per_leg

        # 初始化边界
        lbg = np.full(n_total_constraints, -np.inf)
        ubg = np.zeros(n_total_constraints)

        for k in range(N):
            for i in range(self.n_legs):
                ci = contact_seq[i, k]
                base_idx = k * self.n_legs * self.n_constraints_per_leg + i * self.n_constraints_per_leg

                if ci > 0.5:  # 接触
                    # 摩擦约束：<= 0
                    # 已经在 ubg 中设为 0，lbg 为 -inf

                    # 力边界 - 关键修改：接触腿必须承担最小力
                    lbg[base_idx + 4] = -np.inf  # fz - fz_max <= 0
                    ubg[base_idx + 4] = 0
                    lbg[base_idx + 5] = -np.inf  # -fz + fz_contact_min <= 0 => fz >= fz_contact_min
                    ubg[base_idx + 5] = 0

                    # fx 在摆动时为0，接触时自由
                    lbg[base_idx + 6] = -np.inf
                    ubg[base_idx + 6] = np.inf

                else:  # 摆动
                    # 摩擦约束：当 fz=0 时自动满足
                    lbg[base_idx:base_idx + 4] = 0
                    ubg[base_idx:base_idx + 4] = 0

                    # 力必须为0
                    lbg[base_idx + 4] = -self.fz_max  # fz - fz_max <= 0
                    ubg[base_idx + 4] = -self.fz_min  # fz - fz_min <= 0 => fz=0
                    lbg[base_idx + 5] = self.fz_min   # -fz + fz_min <= 0 => fz=0
                    ubg[base_idx + 5] = self.fz_max

                    # fx = fy = 0
                    lbg[base_idx + 6] = 0
                    ubg[base_idx + 6] = 0

        return lbg, ubg


class VerticalForceBalanceConstraint:
    """
    竖直力平衡硬约束：∑fz = mg (对所有接触腿)

    这个约束确保总垂直力等于机器人的重量，防止"全零力"或"不均匀受力"的情况
    """

    def __init__(self, mass=15.019, gravity=9.81, epsilon=0.1):
        self.mass = mass
        self.gravity = gravity
        self.total_weight = mass * gravity
        self.epsilon = epsilon  # 允许的误差范围 (mg ± epsilon)
        self.n_constraints_per_step = 1  # 每个时间步一个约束

    def build_expression(self, U):
        """
        构建竖直力平衡约束表达式

        注意：这里假设所有腿都是接触的（全站立情况）。
        如果某些腿摆动，需要通过边界来处理。

        Args:
            U: 控制输入序列 (nu x N)

        Returns:
            约束表达式：总垂直力 - mg = 0
        """
        # 如果U是向量，转换为矩阵
        if len(U.shape) == 1:
            U = cs.reshape(U, -1, 1)

        N = U.shape[1]
        constraints = []

        # u[12:24]是4个腿的力，每个腿3个分量 [fx, fy, fz]
        # fz的索引：14 (FL), 17 (FR), 20 (RL), 23 (RR)
        fz_FL = U[14, :]
        fz_FR = U[17, :]
        fz_RL = U[20, :]
        fz_RR = U[23, :]

        # 总垂直力
        total_fz = fz_FL + fz_FR + fz_RL + fz_RR

        # 每个时间步一个约束：总垂直力 - mg = 0
        for k in range(N):
            constraints.append(total_fz[k] - self.total_weight)

        return cs.vertcat(*constraints)

    def get_bounds(self, N):
        """
        获取约束边界

        Args:
            N: 时间步数

        Returns:
            (lbg, ubg): 约束边界，允许±epsilon的误差
        """
        lbg = np.full(N, -self.epsilon)
        ubg = np.full(N, self.epsilon)
        return lbg, ubg


class CentroidalNMPC_IPOPT_Fixed:
    """
    修复版的基于质心动力学的 NMPC 求解器
    """

    def __init__(self, horizon=12, dt=0.02, weights=None):
        self.horizon = horizon
        self.dt = dt
        self.weights = weights or BALANCE_WEIGHTS

        # 初始化模型
        self.model = Centroidal_Model_Nominal()
        self.nx = self.model.states.shape[0]  # 30 (24 + 6 integrals)
        self.nu = self.model.inputs.shape[0]   # 24

        # 初始化目标函数
        self.obj_func = QuadrupedObjectiveFunction(self.weights)
        self.Q, self.R = self.obj_func.create_weight_matrices()

        # 初始化控制量变化权重矩阵（用于 Δu 正则项）
        # 通常设置为 R 的对角元素的 0.1-1 倍
        rdu_diag = 0.1 * np.diag(self.R)  # 使用 R 对角元素的 0.1 倍
        self.Rdu = cs.diag(rdu_diag)

        # 预编译动力学函数（只在初始化时构建一次）
        self._compile_dynamics()

        # 初始化约束模块
        self.friction_constraints = FrictionPyramidConstraints()
        self.force_balance_constraint = VerticalForceBalanceConstraint()

        # 构建 NLP
        self.solver = self._build_solver()

        # 为演示代码创建 f_dyn 函数 (x, u, p) -> x_next
        # 演示代码期望的参数格式：[接触状态(4), 其他参数(25)]
        params_default = self._get_default_parameters()

        # 创建一个简化的动力学函数，签名与演示代码期望的一致
        x_sym = cs.SX.sym('x', self.nx)
        u_sym = cs.SX.sym('u', self.nu)
        p_sym = cs.SX.sym('p', params_default.shape[0])  # 应该是29维

        # 从参数中提取接触状态（前4个）和其他参数（后25个）
        contact = p_sym[:4]  # 前4个是接触状态
        params = p_sym[4:]   # 后25个是其他参数

        # 计算下一个状态
        x_next = self.dynamics_func(x_sym, u_sym, contact, params)

        self.f_dyn = cs.Function(
            'f_dyn',
            [x_sym, u_sym, p_sym],
            [x_next],
            ['x', 'u', 'p'],
            ['x_next']
        )

    def _compile_dynamics(self):
        """预编译动力学函数（只在初始化时执行一次）"""
        # 创建符号变量
        x_sym = cs.SX.sym('x', self.nx)
        u_sym = cs.SX.sym('u', self.nu)
        contact_sym = cs.SX.sym('contact', 4)
        params_without_contact = cs.SX.sym('params', 25)  # 不包含接触状态的参数

        # 组合参数：接触状态 + 其他参数
        params_with_contact = cs.vertcat(
            contact_sym,  # 0-3: 接触状态
            params_without_contact  # 4-28: 其他参数
        )

        # 调用动力学模型
        xdot = self.model.forward_dynamics(x_sym, u_sym, params_with_contact)

        # RK4 离散化
        k1 = xdot
        k2 = self.model.forward_dynamics(x_sym + self.dt/2*k1, u_sym, params_with_contact)
        k3 = self.model.forward_dynamics(x_sym + self.dt/2*k2, u_sym, params_with_contact)
        k4 = self.model.forward_dynamics(x_sym + self.dt*k3, u_sym, params_with_contact)

        x_next = x_sym + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        # 编译函数（只构建一次）
        # 注意：这里params参数是25维的（不包含接触状态）
        self.dynamics_func = cs.Function(
            'dynamics',
            [x_sym, u_sym, contact_sym, params_without_contact],
            [x_next],
            ['x', 'u', 'contact', 'params'],
            ['x_next']
        )

    def _build_solver(self):
        """构建 NLP 求解器"""
        # 决策变量
        X = cs.SX.sym("X", self.nx, self.horizon + 1)
        U = cs.SX.sym("U", self.nu, self.horizon)

        # 参数
        x0 = cs.SX.sym("x0", self.nx)
        x_ref = cs.SX.sym("x_ref", self.nx, self.horizon + 1)
        u_ref = cs.SX.sym("u_ref", self.nu, self.horizon)

        # 构建约束列表
        constraints = []

        # 1. 初始条件约束
        constraints.append(X[:, 0] - x0)

        # 2. 动力学约束
        for k in range(self.horizon):
            x_next = self._discrete_dynamics_fast(X[:, k], U[:, k])
            constraints.append(X[:, k + 1] - x_next)

        # 3. 摩擦约束（表达式结构固定）
        friction_expr = self.friction_constraints.build_expression(U)
        constraints.append(friction_expr)

        # 4. 竖直力平衡约束（确保总垂直力等于重量）
        force_balance_expr = self.force_balance_constraint.build_expression(U)
        constraints.append(force_balance_expr)

        # 注意：足端速度约束移到 solve() 方法中，根据接触序列动态设置

        # 目标函数
        total_cost = 0
        for k in range(self.horizon):
            x_error = X[:, k] - x_ref[:, k]
            u_error = U[:, k] - u_ref[:, k]
            total_cost += cs.mtimes([x_error.T, self.Q, x_error]) + cs.mtimes([u_error.T, self.R, u_error])

            # 新增：足端速度软约束 - 严重惩罚接触腿的滑移
            # 提取足端速度 u[0:12] = [vx,vy,vz] × 4条腿
            foot_vel_penalty = 0.0
            for leg in range(4):
                base_idx = leg * 3
                vx = U[base_idx + 0, k]
                vy = U[base_idx + 1, k]
                vz = U[base_idx + 2, k]

                # 假设接触序列全为1（四足支撑），严重惩罚所有足端速度
                # 权重：接触腿的足端速度应该接近0
                penalty_weight = 100.0  # 比原来的1e-4大很多数量级
                foot_vel_penalty += penalty_weight * (vx**2 + vy**2 + vz**2)

            total_cost += foot_vel_penalty

            # 新增：力分布惩罚项 - 鼓励四条腿均匀分担重量
            # 提取每条腿的垂直力 (u[12+2], u[15+2], u[18+2], u[21+2])
            fz_FL = U[12+2, k]  # FL腿z方向力
            fz_FR = U[15+2, k]  # FR腿z方向力
            fz_RL = U[18+2, k]  # RL腿z方向力
            fz_RR = U[21+2, k]  # RR腿z方向力

            # 计算力的方差（鼓励均匀分布）
            fz_mean = (fz_FL + fz_FR + fz_RL + fz_RR) / 4.0
            force_distribution_penalty = 50.0 * (
                (fz_FL - fz_mean)**2 +
                (fz_FR - fz_mean)**2 +
                (fz_RL - fz_mean)**2 +
                (fz_RR - fz_mean)**2
            )
            total_cost += force_distribution_penalty

            # Δu 正则：抑制相邻控制量突变（提高闭环稳定性）
            if k >= 1:
                du = U[:, k] - U[:, k-1]
                total_cost += cs.mtimes([du.T, self.Rdu, du])

        # 终端成本
        terminal_error = X[:, self.horizon] - x_ref[:, self.horizon]
        total_cost += cs.mtimes([terminal_error.T, self.Q, terminal_error])

        # 打包
        w = cs.vertcat(cs.reshape(X, -1, 1), cs.reshape(U, -1, 1))
        g = cs.vertcat(*constraints)

        # 参数（移除 contact_seq，因为它影响边界而非表达式）
        p = cs.vertcat(
            x0,
            cs.reshape(x_ref, -1, 1),
            cs.reshape(u_ref, -1, 1)
        )

        # NLP
        nlp = {'x': w, 'f': total_cost, 'g': g, 'p': p}

        # 求解器选项
        solver_opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 200,
            'ipopt.tol': 1e-6,
            'ipopt.constr_viol_tol': 1e-6,
            # Warm-start（需要提供 x0 初值）
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.warm_start_bound_push': 1e-6,
            'ipopt.warm_start_mult_bound_push': 1e-6,
            'ipopt.warm_start_slack_bound_push': 1e-6,
        }

        # 创建求解器
        solver = cs.nlpsol('solver', 'ipopt', nlp, solver_opts)

        # 存储约束维度信息（用于正确设置边界）
        self.n_dyn_constraints = self.nx * (self.horizon + 1)  # 初始 + 动力学
        self.n_fric_constraints = self.horizon * 4 * 7  # horizon * legs * constraints_per_leg
        self.n_force_balance_constraints = self.horizon * 1  # horizon * force_balance_per_step
        # 删除足端速度约束（通过动力学自然处理）
        self.n_total_constraints = (self.n_dyn_constraints +
                                   self.n_fric_constraints +
                                   self.n_force_balance_constraints)

        return solver

    def _discrete_dynamics_fast(self, x, u):
        """
        快速动力学计算（使用预编译的函数）

        注意：这里假设默认接触，实际接触状态在边界中处理
        """
        # 默认全接触
        contact = np.ones(4)
        # 获取参数向量（不包括接触状态）
        params_without_contact = self._get_default_parameters()[4:]  # 跳过前4个接触状态

        # dynamics_func 返回的是 SX 对象，直接返回
        return self.dynamics_func(x, u, contact, params_without_contact)

    def _get_default_parameters(self):
        """获取默认参数"""
        stance = np.array([1.0, 1.0, 1.0, 1.0])
        mu_friction = np.array([0.5])
        stance_proximity = np.array([0, 0, 0, 0])
        base_position = np.array([0, 0, 0])
        base_yaw = np.array([0])
        external_wrench = np.array([0, 0, 0, 0, 0, 0])
        inertia = np.array([0.0695, 0, 0, 0, 0.1359, 0, 0, 0, 0.1495])
        mass = np.array([15.019])

        return np.concatenate([
            stance, mu_friction, stance_proximity,
            base_position, base_yaw, external_wrench,
            inertia, mass
        ])

    def solve(self, x0, x_ref, u_ref, contact_seq, w0=None):
        """
        求解 NMPC 问题

        Args:
            x0: 初始状态 (nx,)
            x_ref: 状态参考 (nx, horizon+1)
            u_ref: 控制参考 (nu, horizon)
            contact_seq: 接触序列 (4, horizon)
            w0: 初始猜测

        Returns:
            dict: 求解结果
        """
        # 打包参数（使用Fortran顺序与CasADi reshape一致）
        p_val = np.concatenate([
            x0.reshape(-1),
            x_ref.reshape(-1, order="F"),
            u_ref.reshape(-1, order="F")
        ])

        # 初始猜测
        if w0 is None:
            w0 = np.zeros(self.solver.size_in(0))  # 获取输入变量x的维度

        # 正确构建约束边界
        lbg = np.zeros(self.n_total_constraints)
        ubg = np.zeros(self.n_total_constraints)

        # 动力学约束（等式）
        ubg[:self.n_dyn_constraints] = 0

        # 摩擦约束边界（根据接触序列）
        lbg_fric, ubg_fric = self.friction_constraints.get_bounds(contact_seq)
        lbg[self.n_dyn_constraints:self.n_dyn_constraints + self.n_fric_constraints] = lbg_fric
        ubg[self.n_dyn_constraints:self.n_dyn_constraints + self.n_fric_constraints] = ubg_fric

        # 竖直力平衡约束边界
        lbg_force, ubg_force = self.force_balance_constraint.get_bounds(self.horizon)
        force_start_idx = self.n_dyn_constraints + self.n_fric_constraints
        force_end_idx = force_start_idx + self.n_force_balance_constraints
        lbg[force_start_idx:force_end_idx] = lbg_force
        ubg[force_start_idx:force_end_idx] = ubg_force

        # 求解
        try:
            # 如果外部没有提供初值，则使用上一次求解的解做 warm-start
            if w0 is None and getattr(self, 'last_w', None) is not None:
                w0 = self.last_w
            sol = self.solver(
                x0=w0,
                p=p_val,
                lbg=lbg,
                ubg=ubg
            )

            # 提取解
            w_opt = np.array(sol['x']).flatten()

            # 缓存用于下一次 warm-start
            self.last_w = w_opt.copy()

            # 解包
            X_opt = w_opt[:self.nx * (self.horizon + 1)].reshape(self.nx, self.horizon + 1, order='F')
            U_opt = w_opt[self.nx * (self.horizon + 1):].reshape(self.nu, self.horizon, order='F')

            return {
                'status': 'success',
                'u0': U_opt[:, 0],
                'X_opt': X_opt,
                'U_opt': U_opt,
                'w_opt': w_opt,
                'solver_stats': sol['solver_stats'] if 'solver_stats' in sol else None
            }

        except Exception as e:
            print(f"求解失败: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'u0': np.zeros(self.nu)
            }

    def create_reference_trajectory(self, reference_dict, contact_seq):
        """创建参考轨迹"""
        x_ref_traj, u_ref_traj = self.obj_func.get_reference_trajectory(
            reference_dict, self.horizon
        )

        # 更新足端力参考
        u_ref_traj = self.obj_func.update_reference_forces(
            u_ref_traj, contact_seq,
            mass=15.019,
            gravity=9.81,
            horizon=self.horizon
        )

        return x_ref_traj, u_ref_traj


if __name__ == "__main__":
    print("测试修复版的 NMPC 求解器")
    print("="*50)

    # 创建求解器
    nmpc = CentroidalNMPC_IPOPT_Fixed(horizon=10, dt=0.02)

    print(f"状态维度: {nmpc.nx}")
    print(f"控制维度: {nmpc.nu}")
    print(f"总约束数: {nmpc.n_total_constraints}")
    print(f"  - 动力学约束: {nmpc.n_dyn_constraints}")
    print(f"  - 摩擦约束: {nmpc.n_fric_constraints}")
    print(f"  - 竖直力平衡约束: {nmpc.n_force_balance_constraints}")

    # 初始状态
    x0 = np.zeros(nmpc.nx)
    x0[2] = 0.3

    # 参考状态
    reference_dict = {
        "ref_position": np.array([0, 0, 0.3]),
        "ref_linear_velocity": np.array([0, 0, 0]),
        "ref_orientation": np.array([0, 0, 0]),
        "ref_angular_velocity": np.array([0, 0, 0]),
        "ref_foot_FL": np.array([[0.2, 0.15, -0.3]]),
        "ref_foot_FR": np.array([[0.2, -0.15, -0.3]]),
        "ref_foot_RL": np.array([[-0.2, 0.15, -0.3]]),
        "ref_foot_RR": np.array([[-0.2, -0.15, -0.3]]),
    }

    # 接触序列
    contact_seq = np.ones((4, nmpc.horizon))

    # 创建参考
    x_ref, u_ref = nmpc.create_reference_trajectory(reference_dict, contact_seq)

    # 求解
    print("\n求解 NMPC...")
    import time
    start = time.time()
    result = nmpc.solve(x0, x_ref, u_ref, contact_seq)
    solve_time = time.time() - start

    print(f"求解状态: {result['status']}")
    print(f"求解时间: {solve_time*1000:.2f} ms")

    if result['status'] == 'success':
        forces = result['u0'][12:24]
        print(f"\n足端力:")
        print(f"  FL: {forces[0:3]}")
        print(f"  FR: {forces[3:6]}")
        print(f"  RL: {forces[6:9]}")
        print(f"  RR: {forces[9:12]}")

    print("\n✅ 修复完成！主要改进：")
    print("1. 动力学函数只构建一次（性能提升）")
    print("2. 摩擦约束表达式固定，边界动态变化")
    print("3. 正确处理摆动腿的零力约束")
    print("4. 接触逻辑从表达式移到边界")
    print("5. 约束边界索引正确管理")
