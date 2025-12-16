# Description: This file contains the class Centroidal_Model that defines the
# prediction model used by the MPC
#
# Authors: Giulio Turrisi -

import casadi as cs
# import quadruped_pympc.config as config  # 注释掉，避免依赖


# Class that defines the prediction model of the NMPC
class Centroidal_Model_Nominal:
    def __init__(self) -> None:
        """
        This method initializes the prediction model of the NMPC.
        """

        # -------------------------
        # Define state x (CasADi SX)
        # -------------------------
        com_position_x = cs.SX.sym("com_position_x")
        com_position_y = cs.SX.sym("com_position_y")
        com_position_z = cs.SX.sym("com_position_z")

        com_velocity_x = cs.SX.sym("com_velocity_x")
        com_velocity_y = cs.SX.sym("com_velocity_y")
        com_velocity_z = cs.SX.sym("com_velocity_z")

        roll = cs.SX.sym("roll", 1, 1)
        pitch = cs.SX.sym("pitch", 1, 1)
        yaw = cs.SX.sym("yaw", 1, 1)

        omega_x = cs.SX.sym("omega_x", 1, 1)
        omega_y = cs.SX.sym("omega_y", 1, 1)
        omega_z = cs.SX.sym("omega_z", 1, 1)

        foot_position_fl = cs.SX.sym("foot_position_fl", 3, 1)
        foot_position_fr = cs.SX.sym("foot_position_fr", 3, 1)
        foot_position_rl = cs.SX.sym("foot_position_rl", 3, 1)
        foot_position_rr = cs.SX.sym("foot_position_rr", 3, 1)

        # Integral states (used for integral action in MPC)
        com_position_z_integral = cs.SX.sym("com_position_z_integral")
        com_velocity_x_integral = cs.SX.sym("com_velocity_x_integral")
        com_velocity_y_integral = cs.SX.sym("com_velocity_y_integral")
        com_velocity_z_integral = cs.SX.sym("com_velocity_z_integral")
        roll_integral = cs.SX.sym("roll_integral")
        pitch_integral = cs.SX.sym("pitch_integral")

        # NOTE: omega_*_integral were declared in the original code but never used (not in self.states),
        # so we remove them to avoid confusion. If you want to add them properly, tell me and I will
        # provide a consistent "omega-integral" version (states/states_dot/fd/cost all aligned).

        self.states = cs.vertcat(
            com_position_x,
            com_position_y,
            com_position_z,
            com_velocity_x,
            com_velocity_y,
            com_velocity_z,
            roll,
            pitch,
            yaw,
            omega_x,
            omega_y,
            omega_z,
            foot_position_fl,
            foot_position_fr,
            foot_position_rl,
            foot_position_rr,
            com_position_z_integral,
            com_velocity_x_integral,
            com_velocity_y_integral,
            com_velocity_z_integral,
            roll_integral,
            pitch_integral,
        )

        # -------------------------
        # Define xdot (state derivatives)
        # -------------------------
        self.states_dot = cs.vertcat(
            cs.SX.sym("linear_com_vel", 3, 1),            # d/dt com_position
            cs.SX.sym("linear_com_acc", 3, 1),            # d/dt com_velocity
            cs.SX.sym("euler_rates_base", 3, 1),          # d/dt euler angles
            cs.SX.sym("angular_acc_base", 3, 1),          # d/dt omega
            cs.SX.sym("linear_vel_foot_FL", 3, 1),        # d/dt foot_position_fl
            cs.SX.sym("linear_vel_foot_FR", 3, 1),        # d/dt foot_position_fr
            cs.SX.sym("linear_vel_foot_RL", 3, 1),        # d/dt foot_position_rl
            cs.SX.sym("linear_vel_foot_RR", 3, 1),        # d/dt foot_position_rr
            cs.SX.sym("com_position_z_integral_dot", 1, 1),
            cs.SX.sym("com_velocity_integral_dot", 3, 1),
            cs.SX.sym("roll_integral_dot", 1, 1),
            cs.SX.sym("pitch_integral_dot", 1, 1),
        )

        # -------------------------
        # Define input u
        # -------------------------
        foot_velocity_fl = cs.SX.sym("foot_velocity_fl", 3, 1)
        foot_velocity_fr = cs.SX.sym("foot_velocity_fr", 3, 1)
        foot_velocity_rl = cs.SX.sym("foot_velocity_rl", 3, 1)
        foot_velocity_rr = cs.SX.sym("foot_velocity_rr", 3, 1)

        foot_force_fl = cs.SX.sym("foot_force_fl", 3, 1)
        foot_force_fr = cs.SX.sym("foot_force_fr", 3, 1)
        foot_force_rl = cs.SX.sym("foot_force_rl", 3, 1)
        foot_force_rr = cs.SX.sym("foot_force_rr", 3, 1)

        self.inputs = cs.vertcat(
            foot_velocity_fl,
            foot_velocity_fr,
            foot_velocity_rl,
            foot_velocity_rr,
            foot_force_fl,
            foot_force_fr,
            foot_force_rl,
            foot_force_rr,
        )

        # Useful for debugging y_ref mapping (states + inputs)
        self.y_ref = cs.vertcat(self.states, self.inputs)

        # -------------------------
        # Define acados parameters p (runtime changeable)
        # -------------------------
        self.stanceFL = cs.SX.sym("stanceFL", 1, 1)
        self.stanceFR = cs.SX.sym("stanceFR", 1, 1)
        self.stanceRL = cs.SX.sym("stanceRL", 1, 1)
        self.stanceRR = cs.SX.sym("stanceRR", 1, 1)
        self.stance_param = cs.vertcat(self.stanceFL, self.stanceFR, self.stanceRL, self.stanceRR)

        self.mu_friction = cs.SX.sym("mu_friction", 1, 1)
        self.stance_proximity = cs.SX.sym("stanceProximity", 4, 1)
        self.base_position = cs.SX.sym("base_position", 3, 1)
        self.base_yaw = cs.SX.sym("base_yaw", 1, 1)

        self.external_wrench = cs.SX.sym("external_wrench", 6, 1)  # [F_ext(3); tau_ext(3)]
        self.inertia = cs.SX.sym("inertia", 9, 1)                  # flattened 3x3
        self.mass = cs.SX.sym("mass", 1, 1)

        self.gravity_constant = 9.81  # 直接使用重力加速度

        # Create a CasADi function for forward dynamics (debug/utility)
        param = cs.vertcat(
            self.stance_param,      # 0..3
            self.mu_friction,       # 4
            self.stance_proximity,  # 5..8
            self.base_position,     # 9..11
            self.base_yaw,          # 12
            self.external_wrench,   # 13..18
            self.inertia,           # 19..27
            self.mass,              # 28
        )
        fd = self.forward_dynamics(self.states, self.inputs, param)
        self.fun_forward_dynamics = cs.Function(
            "fun_forward_dynamics", [self.states, self.inputs, param], [fd]
        )

    def forward_dynamics(self, states: cs.SX, inputs: cs.SX, param: cs.SX) -> cs.SX:
        """
        Compute symbolic forward dynamics: returns xdot = f(x, u, p).
        IMPORTANT: Must return derivatives, NOT "next state".

        Dimensions (current model):
        - states:  24 (base+feet) + 6 (integrals) = 30?  -> actually here: 12 + 12 + 6 = 30, but this file's states are 12+12+6=30
          (If you count: base 12, feet 12, integrals 6 -> 30)
        - inputs:  24
        - param:   29
        """

        # -------------------------
        # Unpack inputs
        # -------------------------
        foot_velocity_fl = inputs[0:3]
        foot_velocity_fr = inputs[3:6]
        foot_velocity_rl = inputs[6:9]
        foot_velocity_rr = inputs[9:12]

        foot_force_fl = inputs[12:15]
        foot_force_fr = inputs[15:18]
        foot_force_rl = inputs[18:21]
        foot_force_rr = inputs[21:24]

        # -------------------------
        # Unpack states
        # -------------------------
        com_position = states[0:3]
        linear_com_vel = states[3:6]  # = com_velocity

        roll = states[6]
        pitch = states[7]
        yaw = states[8]
        w = states[9:12]  # omega (assumed body frame)

        foot_position_fl = states[12:15]
        foot_position_fr = states[15:18]
        foot_position_rl = states[18:21]
        foot_position_rr = states[21:24]

        # -------------------------
        # Unpack parameters
        # -------------------------
        stanceFL = param[0]
        stanceFR = param[1]
        stanceRL = param[2]
        stanceRR = param[3]

        # mu is param[4] (not used in dynamics here, usually used in constraints)
        stance_proximity_FL = param[5]
        stance_proximity_FR = param[6]
        stance_proximity_RL = param[7]
        stance_proximity_RR = param[8]

        external_wrench_linear = param[13:16]
        external_wrench_angular = param[16:19]

        inertia_flat = param[19:28]
        inertia = cs.reshape(inertia_flat, 3, 3)
        mass = param[28]

        # -------------------------
        # 1) d/dt com_position = com_velocity
        # -------------------------
        # linear_com_vel already extracted

        # -------------------------
        # 2) d/dt com_velocity = (sum stance_i * f_i + f_ext)/m + g
        # -------------------------
        sum_forces = (
            foot_force_fl * stanceFL
            + foot_force_fr * stanceFR
            + foot_force_rl * stanceRL
            + foot_force_rr * stanceRR
            + external_wrench_linear
        )
        gravity = cs.vertcat(0.0, 0.0, -float(self.gravity_constant))
        linear_com_acc = sum_forces / mass + gravity

        # -------------------------
        # 3) Euler rates from omega (ZYX)
        #    omega = E(roll,pitch) * euler_dot  => euler_dot = inv(E) * omega
        # -------------------------
        E = cs.SX.eye(3)
        E[0, 2] = -cs.sin(pitch)
        E[1, 1] = cs.cos(roll)
        E[1, 2] = cs.cos(pitch) * cs.sin(roll)
        E[2, 1] = -cs.sin(roll)
        E[2, 2] = cs.cos(pitch) * cs.cos(roll)

        euler_rates_base = cs.inv(E) @ w

        # -------------------------
        # 4) Angular acceleration (rigid body)
        #    I * wdot = tau_body - w x (I w)
        #    tau_world = sum (r_i - r_com) x f_i + tau_ext_world
        #    tau_body = body_R_world * tau_world
        # -------------------------
        tau_world = (
            cs.skew(foot_position_fl - com_position) @ (foot_force_fl * stanceFL)
            + cs.skew(foot_position_fr - com_position) @ (foot_force_fr * stanceFR)
            + cs.skew(foot_position_rl - com_position) @ (foot_force_rl * stanceRL)
            + cs.skew(foot_position_rr - com_position) @ (foot_force_rr * stanceRR)
            + external_wrench_angular
        )

        # Rotation matrices (ZYX: world_R_body = Rz(yaw)*Ry(pitch)*Rx(roll))
        Rx = cs.SX.eye(3)
        Rx[1, 1] = cs.cos(roll)
        Rx[1, 2] = -cs.sin(roll)
        Rx[2, 1] = cs.sin(roll)
        Rx[2, 2] = cs.cos(roll)

        Ry = cs.SX.eye(3)
        Ry[0, 0] = cs.cos(pitch)
        Ry[0, 2] = cs.sin(pitch)
        Ry[2, 0] = -cs.sin(pitch)
        Ry[2, 2] = cs.cos(pitch)

        Rz = cs.SX.eye(3)
        Rz[0, 0] = cs.cos(yaw)
        Rz[0, 1] = -cs.sin(yaw)
        Rz[1, 0] = cs.sin(yaw)
        Rz[1, 1] = cs.cos(yaw)

        world_R_body = Rz @ Ry @ Rx
        body_R_world = world_R_body.T

        tau_body = body_R_world @ tau_world
        angular_acc_base = cs.inv(inertia) @ (tau_body - cs.skew(w) @ inertia @ w)

        # -------------------------
        # 5) Foot kinematics: only move in swing, and smooth with proximity
        # -------------------------
        # if not config.mpc_params["use_foothold_optimization"]:  # 简化，直接禁用
        if True:  # 简化，不使用足端优化
            foot_velocity_fl = foot_velocity_fl * 0.0
            foot_velocity_fr = foot_velocity_fr * 0.0
            foot_velocity_rl = foot_velocity_rl * 0.0
            foot_velocity_rr = foot_velocity_rr * 0.0

        linear_foot_vel_FL = foot_velocity_fl * (1 - stanceFL) * (1 - stance_proximity_FL)
        linear_foot_vel_FR = foot_velocity_fr * (1 - stanceFR) * (1 - stance_proximity_FR)
        linear_foot_vel_RL = foot_velocity_rl * (1 - stanceRL) * (1 - stance_proximity_RL)
        linear_foot_vel_RR = foot_velocity_rr * (1 - stanceRR) * (1 - stance_proximity_RR)

        # -------------------------
        # 6) Integral states: return DERIVATIVES (NOT "+=" updates)
        #    Here we integrate the raw signals. (Better is integrating tracking error, but needs refs.)
        # -------------------------
        com_position_z_integral_dot = states[2]  # z
        com_velocity_integral_dot = states[3:6]  # vx, vy, vz
        roll_integral_dot = roll
        pitch_integral_dot = pitch

        integral_dot = cs.vertcat(
            com_position_z_integral_dot,
            com_velocity_integral_dot,
            roll_integral_dot,
            pitch_integral_dot,
        )

        # The order of return must match self.states_dot
        return cs.vertcat(
            linear_com_vel,
            linear_com_acc,
            euler_rates_base,
            angular_acc_base,
            linear_foot_vel_FL,
            linear_foot_vel_FR,
            linear_foot_vel_RL,
            linear_foot_vel_RR,
            integral_dot,
        )

    def export_robot_model(self):
        """
        导出模型参数（用于IPOPT，而不是Acados）
        """
        self.param = cs.vertcat(
            self.stance_param,
            self.mu_friction,
            self.stance_proximity,
            self.base_position,
            self.base_yaw,
            self.external_wrench,
            self.inertia,
            self.mass,
        )
        return self.param
