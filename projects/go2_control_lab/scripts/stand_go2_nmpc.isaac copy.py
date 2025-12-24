#!/usr/bin/env python
"""Standalone script to test Go2 standing control with NMPC and Jacobian IK.

This script integrates:
- Isaac Lab simulation environment
- NMPC controller from demo_mpc_no_noise_patched_v3_latency.py
- Jacobian-based velocity IK for foot velocity to joint position mapping
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Go2 NMPC standing control test")
parser.add_argument("--num-envs", type=int, default=1, help="Number of environments")
parser.add_argument("--dt", type=float, default=0.001, help="Simulation timestep")
parser.add_argument("--dt-mpc", type=float, default=0.02, help="MPC timestep (outer loop)")
parser.add_argument("--mpc-decim", type=int, default=20, help="MPC decimation (solve every N steps)")
# ❌ 不要加 --device / --headless（AppLauncher 会自动加）
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

# ✅ 正确启动：把 args_cli 整体传给 AppLauncher
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import sys
import os
import numpy as np
from pathlib import Path
from collections import deque
from dataclasses import dataclass

# 添加包含包的目录到Python路径
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)
print(f"Added to sys.path: {current_dir}")

# Add MPC solver to path
MPC_PATH = Path("/home/xx/quadruped_constraints/GPT")
if str(MPC_PATH) not in sys.path:
    sys.path.insert(0, str(MPC_PATH))

import torch

# 直接导入envs.stand_env
from envs.stand_env import StandEnv

# Import MPC components
try:
    from nmpc_ipopt_centroidal_fixed import CentroidalNMPC_IPOPT_Fixed
    from objective_function import BALANCE_WEIGHTS
    MPC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import MPC components: {e}")
    MPC_AVAILABLE = False


# =========================
# Helper Functions
# =========================

def get_jacobians_all(robot) -> torch.Tensor | None:
    """Try to get PhysX jacobians tensor: (N, num_links, 6, dofs_total)."""
    try:
        view = robot.root_physx_view
        if not hasattr(view, "get_jacobians"):
            return None
        J = view.get_jacobians()
        return J
    except Exception:
        return None


def get_link_jacobian(robot, J_all: torch.Tensor, body_id: int) -> torch.Tensor:
    """Extract Jacobian for a specific link.

    IsaacLab/PhysX tensor convention:
      - fixed base: Jacobian array excludes root body -> index = body_id - 1
      - floating base: index = body_id

    Returns: (N, 6, dofs_total)
    """
    if robot.is_fixed_base:
        idx = body_id - 1
    else:
        idx = body_id
    if idx < 0 or idx >= J_all.shape[1]:
        raise RuntimeError(f"Jacobian link index out of range: idx={idx}, body_id={body_id}, J_all.shape={tuple(J_all.shape)}")
    return J_all[:, idx, :, :]


def damped_pinv(A: torch.Tensor, damping: float = 1e-4) -> torch.Tensor:
    """Damped pseudo-inverse for small matrices.
    A: (..., m, n) -> returns (..., n, m)
    """
    m = A.shape[-2]
    I = torch.eye(m, device=A.device, dtype=A.dtype)
    At = A.transpose(-1, -2)
    M = A @ At + damping * I
    Minv = torch.linalg.inv(M)
    return At @ Minv


def quat_to_rpy_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Quaternion (w,x,y,z) -> roll,pitch,yaw. Works on last dim."""
    w, x, y, z = q.unbind(-1)
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack([roll, pitch, yaw], dim=-1)


def build_state_x30(robot, obs, joint_names_12, feet_names_4) -> torch.Tensor:
    """Build the 30-dimensional state vector for NMPC.

    State layout (30):
      0:3   com_pos (world frame)
      3:6   com_vel (world frame)
      6:9   rpy (roll, pitch, yaw)
      9:12  omega (angular velocity)
      12:15 foot_FL (world frame)
      15:18 foot_FR (world frame)
      18:21 foot_RL (world frame)
      21:24 foot_RR (world frame)
      24:30 integrals (6)
    """
    N = obs["base_pos"].shape[0]
    device = obs["base_pos"].device
    dtype = obs["base_pos"].dtype

    # Extract base state
    base_pos = obs["base_pos"]  # (N, 3)
    base_quat = obs["base_quat"]  # (N, 4) wxyz
    base_vel = obs["base_lin_vel"]  # (N, 3)
    base_ang_vel = obs["base_ang_vel"]  # (N, 3)

    # Convert quaternion to RPY
    base_rpy = quat_to_rpy_wxyz(base_quat)

    # Get foot positions
    body_names = robot.data.body_names
    foot_ids = []
    for foot in feet_names_4:
        if foot in body_names:
            foot_ids.append(body_names.index(foot))
        else:
            print(f"Warning: Foot '{foot}' not found in body names")
            foot_ids.append(0)  # fallback

    feet_pos = robot.data.body_pos_w[:, foot_ids, :]  # (N, 4, 3)

    # Build state vector
    x = torch.zeros(N, 30, device=device, dtype=dtype)

    # 0:3 - COM position (use base position as approximation)
    x[:, 0:3] = base_pos

    # 3:6 - COM velocity
    x[:, 3:6] = base_vel

    # 6:9 - RPY
    x[:, 6:9] = base_rpy

    # 9:12 - Angular velocity
    x[:, 9:12] = base_ang_vel

    # 12:24 - Foot positions (flatten 4x3 -> 12)
    x[:, 12:24] = feet_pos.reshape(N, 12)

    # 24:30 - Integrals (initialized to 0)
    x[:, 24:30] = 0.0

    return x


def foot_vel_to_joint_pos_ik(robot, foot_vel_cmd, joint_names_12, feet_names_4, dt_mpc):
    """Convert foot velocity commands to joint position targets using Jacobian IK.

    Args:
        robot: Isaac Lab robot object
        foot_vel_cmd: (N, 4, 3) foot velocity commands in world frame
        joint_names_12: list of 12 joint names
        feet_names_4: list of 4 foot names
        dt_mpc: MPC timestep for integration

    Returns:
        q_des_12: (N, 12) desired joint positions
    """
    N = foot_vel_cmd.shape[0]
    device = foot_vel_cmd.device
    dtype = foot_vel_cmd.dtype

    # Get current joint positions
    joint_ids_12 = []
    for jn in joint_names_12:
        joint_ids_12.append(robot.data.joint_names.index(jn))
    joint_ids_12 = torch.tensor(joint_ids_12, device=device, dtype=torch.long)

    q_all = robot.data.joint_pos  # (N, total_dofs)
    q12 = q_all[:, joint_ids_12]  # (N, 12)
    qd12 = torch.zeros_like(q12)

    # Get Jacobians
    J_all = get_jacobians_all(robot)
    if J_all is None:
        print("Warning: Jacobians not available, returning current positions")
        return q12

    # Leg joint mappings
    LEG_JOINTS = {
        "FL": ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"],
        "FR": ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"],
        "RL": ["RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"],
        "RR": ["RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"],
    }

    # Process each leg
    for leg_i, leg in enumerate(["FL", "FR", "RL", "RR"]):
        foot_name = feet_names_4[leg_i]
        if foot_name not in robot.data.body_names:
            continue

        foot_body_id = robot.data.body_names.index(foot_name)

        # Get foot Jacobian (translational part)
        J_link = get_link_jacobian(robot, J_all, foot_body_id)
        Jv = J_link[:, 0:3, :]  # (N, 3, total_dofs)

        # Select leg joints
        leg_joint_names = LEG_JOINTS[leg]
        leg_joint_ids = []
        for jn in leg_joint_names:
            if jn in robot.data.joint_names:
                leg_joint_ids.append(robot.data.joint_names.index(jn))

        if len(leg_joint_ids) != 3:
            print(f"Warning: Found {len(leg_joint_ids)} joints for leg {leg}, expected 3")
            continue

        leg_joint_ids = torch.tensor(leg_joint_ids, device=device, dtype=torch.long)
        Jv_leg = Jv[:, :, leg_joint_ids]  # (N, 3, 3)

        # Compute joint velocities using damped pseudo-inverse
        Jv_leg_pinv = damped_pinv(Jv_leg, damping=1e-4)

        v_leg = foot_vel_cmd[:, leg_i, :].unsqueeze(-1)  # (N, 3, 1)
        qd_leg = (Jv_leg_pinv @ v_leg).squeeze(-1)  # (N, 3)

        # Clamp joint velocities for safety
        qd_leg = torch.clamp(qd_leg, -30.0, 30.0)

        # Map to output joint velocities
        for jn_i, jn in enumerate(leg_joint_names):
            if jn in joint_names_12:
                idx_in_12 = joint_names_12.index(jn)
                qd12[:, idx_in_12] = qd_leg[:, jn_i]

    # Integrate to get desired positions
    q_des_12 = q12 + qd12 * dt_mpc
    return q_des_12


# =========================
# MPC Controller Class
# =========================

@dataclass
class MPCController:
    """MPC controller wrapper for Isaac Lab integration."""
    nmpc: object = None
    horizon: int = 10
    dt: float = 0.02
    last_solution: np.ndarray = None
    last_u: np.ndarray = None
    last_contact: np.ndarray = None

    def __post_init__(self):
        if self.nmpc is not None and hasattr(self.nmpc, 'weights'):
            # Tune weights for better standing performance
            self.nmpc.weights.Q_velocity = np.array([350, 300, 300])
            self.nmpc.Q, self.nmpc.R = self.nmpc.obj_func.create_weight_matrices()

            # Increase control regularization
            try:
                import casadi as cs
                factor = 50.0
                self.nmpc.Rdu = cs.diag((0.1 * factor) * np.diag(self.nmpc.R))
                self.nmpc.solver = self.nmpc._build_solver()
            except Exception as e:
                print(f"Warning: Could not update Rdu: {e}")

    def solve(self, x0: np.ndarray, reference_dict: dict,
              contact_seq: np.ndarray = None) -> tuple[np.ndarray, bool, dict]:
        """Solve MPC problem using correct interface.

        Args:
            x0: Current state (30,)
            reference_dict: Reference dictionary with position, foot positions, etc.
            contact_seq: Contact sequence (4, horizon)

        Returns:
            u: Control action (nu,)
            success: Solver success flag
            info: Additional info
        """
        if self.nmpc is None:
            # Return zero action if MPC not available
            nu = 24  # Default: 12 foot vel + 12 forces
            return np.zeros(nu), False, {"error": "MPC not available"}

        if contact_seq is None:
            contact_seq = np.ones((4, self.horizon))

        # --- FIX SHAPES BEFORE create_reference_trajectory() ---
        N = self.nmpc.horizon  # Use NMPC horizon, not MPCController horizon

        # Debug: print contact_seq shape before fix
        print("Debug - contact_seq shape BEFORE fix:", np.asarray(contact_seq).shape, "ndim:", np.asarray(contact_seq).ndim)

        # Ensure contact_seq is (4, N) - simple approach
        contact_seq = np.asarray(contact_seq)
        if contact_seq.shape[1] != N:
            print(f"Warning: contact_seq shape {contact_seq.shape}, padding/truncating to (4, {N})")
            # Simple fix: if wrong shape, create new one with all 1s
            contact_seq = np.ones((4, N))

        # 2) feet refs must be (3,) not (1,3)
        for key in ["ref_foot_FL", "ref_foot_FR", "ref_foot_RL", "ref_foot_RR"]:
            reference_dict[key] = np.asarray(reference_dict[key]).reshape(3,)

        # Debug: print shapes after fix
        print("Debug - contact_seq shape AFTER fix:", contact_seq.shape)
        print("Debug - ref_foot_FL shape AFTER fix:", reference_dict["ref_foot_FL"].shape)

        # Let NMPC create reference trajectories from dictionary
        try:
            # Debug: print reference_dict shapes
            print("Debug - reference_dict shapes:")
            for key, val in reference_dict.items():
                print(f"  {key}: shape={np.asarray(val).shape}")

            x_ref, u_ref = self.nmpc.create_reference_trajectory(reference_dict, contact_seq)

            # Verify reference shapes (debug)
            print("x_ref shape:", np.asarray(x_ref).shape)  # Expect (30, N+1)
            print("u_ref shape:", np.asarray(u_ref).shape)  # Expect (24, N)
        except Exception as e:
            # Get full traceback for better debugging
            import traceback
            print(f"MPC FAIL: Failed to create reference: {str(e)}")
            traceback.print_exc()
            # raise  # Comment out to avoid terminating main loop
            return np.zeros(24), False, {"error": f"Failed to create reference: {str(e)}"}

        # Warm-start
        w0 = self.last_solution
        if w0 is not None:
            print(f"DEBUG - last_solution.shape: {np.asarray(w0).shape}")
            w0 = self._shift_warmstart(w0)
            print(f"DEBUG - shifted w0.shape: {np.asarray(w0).shape}")
        else:
            print("DEBUG - No warm-start available")

        # Debug: print solver dimensions
        print("DEBUG - Solver call info:")
        print(f"  nmpc horizon: {self.nmpc.horizon}")
        print(f"  nmpc nx: {self.nmpc.nx}")
        print(f"  nmpc nu: {self.nmpc.nu}")
        print(f"  x0 shape: {x0.shape}")
        print(f"  x_ref shape: {x_ref.shape}")
        print(f"  u_ref shape: {u_ref.shape}")

        # Check expected buffer sizes
        expected_yref_size = self.nmpc.horizon * (self.nmpc.nx + self.nmpc.nu)
        expected_u_size = self.nmpc.horizon * self.nmpc.nu
        expected_x_size = (self.nmpc.horizon + 1) * self.nmpc.nx
        print(f"  Expected yref buffer size: {expected_yref_size}")
        print(f"  Expected u buffer size: {expected_u_size}")
        print(f"  Expected x buffer size: {expected_x_size}")

        # Solve with correct interface
        try:
            print("DEBUG - About to call solver...")
            res = self.nmpc.solve(x0, x_ref, u_ref, contact_seq, w0=w0)
            print("DEBUG - Solver returned successfully")
            print(f"DEBUG - Solver status: {res.get('status', 'unknown')}")
            print(f"DEBUG - Solver keys: {list(res.keys()) if isinstance(res, dict) else 'not a dict'}")

            if res.get("status", "") == "success":
                u = np.asarray(res["u0"]).reshape(-1)
                self.last_solution = res.get("w_opt", None)
                self.last_u = u.copy()
                print(f"DEBUG - Solution u shape: {u.shape}")
                return u, True, res
            else:
                print(f"DEBUG - Solver failed with status: {res.get('status', 'unknown')}")
                return self.last_u if self.last_u is not None else np.zeros(24), False, res
        except Exception as e:
            print(f"MPC FAIL: Solver error: {str(e)}")
            traceback.print_exc()  # Add this to see full traceback for solver error
            # raise  # Comment out to avoid terminating main loop
            return self.last_u if self.last_u is not None else np.zeros(24), False, {"error": str(e)}

    def _make_reference_traj(self, x_now, v_cmd, z_cmd):
        """Make reference trajectory for MPC."""
        N = self.horizon
        nx = 30
        nu = 24

        x_ref = np.repeat(np.asarray(x_now).reshape(nx, 1), N + 1, axis=1)
        u_ref = np.zeros((nu, N))

        # Current velocity
        v_now = x_now[3:6]

        # Gradual velocity reference
        tau = 1.0
        for k in range(N + 1):
            alpha = 1.0 - np.exp(-k * self.dt / tau)
            v_ref_k = v_now + (v_cmd - v_now) * alpha
            x_ref[3:6, k] = v_ref_k

            if k > 0:
                v_avg = (x_ref[3:6, k-1] + v_ref_k) / 2.0
                x_ref[0:3, k] = x_ref[0:3, k-1] + v_avg * self.dt

            x_ref[2, k] = z_cmd  # Height
            x_ref[6:12, k] = 0.0  # Zero orientation/angular velocity

        # Force reference (standing)
        m, g = 15.019, 9.81
        fz = m * g / 4.0
        for k in range(N):
            for leg in range(4):
                u_ref[12 + 3 * leg + 2, k] = fz

        return x_ref, u_ref

    def _shift_warmstart(self, w_opt):
        """Shift warm-start solution."""
        if w_opt is None:
            return None

        w_opt = np.asarray(w_opt).reshape(-1)
        # Use nmpc dimensions, not controller dimensions
        nx = self.nmpc.nx
        nu = self.nmpc.nu
        N = self.nmpc.horizon
        nx_block = nx * (N + 1)

        if w_opt.size < nx_block:
            return None

        X = w_opt[:nx_block].reshape((nx, N + 1), order="F")
        U = w_opt[nx_block:].reshape((nu, N), order="F") if w_opt.size == nx_block + nu * N else None

        X0 = np.hstack([X[:, 1:], X[:, -1:]])
        if U is None:
            return X0.reshape(-1, order="F")
        U0 = np.hstack([U[:, 1:], U[:, -1:]])
        return np.concatenate([X0.reshape(-1, order="F"), U0.reshape(-1, order="F")])


def main():
    """Main entry point for Go2 NMPC standing test."""

    # Create environment
    print("Creating Go2 standing environment...")
    env = StandEnv.make(
        num_envs=args_cli.num_envs,
        dt=args_cli.dt,
        device=args_cli.device
    )

    # Joint and foot names
    joint_names_12 = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    ]

    feet_names_4 = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

    # Initialize MPC controller if available
    mpc_controller = None
    if MPC_AVAILABLE:
        print("Initializing MPC controller...")
        try:
            # horizon must be integer (number of steps), not time in seconds
            N = int(args_cli.mpc_decim)  # prediction steps
            nmpc = CentroidalNMPC_IPOPT_Fixed(
                horizon=N,
                dt=args_cli.dt_mpc,
                weights=BALANCE_WEIGHTS
            )
            mpc_controller = MPCController(nmpc=nmpc)
            print(f"MPC initialized: horizon={nmpc.horizon} steps, dt={nmpc.dt}s")
            # Debug: print MPC dimensions
            print(f"MPC dims: nx={nmpc.nx}, nu={nmpc.nu}")
        except Exception as e:
            print(f"Failed to initialize MPC: {e}")
            mpc_controller = None

    # Get default joint positions (standing pose)
    q_default = env.robot.get_default_q()
    q_desired = q_default.clone()

    # Use higher standing pose
    q_desired[0] = torch.tensor([
        0.10, -0.10,  0.10, -0.10,   # hips
        0.65,  0.65,  0.90,  0.90,   # thighs
       -1.25, -1.25, -1.25, -1.25,   # calves
    ], device=q_desired.device)

    # Simulation parameters
    max_steps = 5000
    print_interval = 500

    # Control parameters
    mpc_dt = args_cli.dt_mpc
    mpc_steps_interval = args_cli.mpc_decim
    dt_mpc_effective = mpc_steps_interval * args_cli.dt

    # Command: stand still with small forward velocity
    v_cmd = np.array([0.2, 0.0, 0.0])
    z_cmd = 0.30

    # ======= Diagnostic prints =======
    print("\n=== robot.data.joint_names ===")
    print(env.robot.robot.data.joint_names)

    print("\n=== robot.data.body_names ===")
    print(env.robot.robot.data.body_names)

    print("\n=== Jacobian Availability ===")
    print("Has get_jacobians:", hasattr(env.robot.robot.root_physx_view, "get_jacobians"))
    print("================================\n")

    print(f"Running {max_steps} steps @ dt={args_cli.dt}s")
    print(f"MPC every {mpc_steps_interval} steps (~{dt_mpc_effective:.3f}s)")
    print(f"MPC available: {mpc_controller is not None}")

    # Initialize control hold
    u_hold = np.zeros(24)  # Default: zero foot velocity + forces

    try:
        # Main simulation loop
        for step in range(max_steps):
            # Get current observations
            obs = env.get_observations()

            # Build state for MPC
            x30 = build_state_x30(
                env.robot.robot, obs,
                joint_names_12, feet_names_4
            )

            # MPC success flag
            mpc_ok = False

            # --- MPC solve (low frequency) ---
            is_mpc_tick = (step % mpc_steps_interval == 0)
            print(f"DEBUG - sim_step={step}, t={step*args_cli.dt:.3f}s, mpc_tick={is_mpc_tick}")

            if is_mpc_tick:
                mpc_ok = False
                if mpc_controller is not None:
                    # 1. Build reference dictionary for standing
                    # Note: reference_dict expects single-step references, not time series
                    reference_dict = {
                        # CoM / base references - shape (3,)
                        "ref_position": np.array([0.0, 0.0, z_cmd]),
                        "ref_linear_velocity": np.array([0.0, 0.0, 0.0]),
                        "ref_orientation": np.array([0.0, 0.0, 0.0]),
                        "ref_angular_velocity": np.array([0.0, 0.0, 0.0]),

                        # Foot positions - shape (1, 3) as expected by objective_function
                        "ref_foot_FL": np.array([[ 0.20,  0.15, -z_cmd]]),
                        "ref_foot_FR": np.array([[ 0.20, -0.15, -z_cmd]]),
                        "ref_foot_RL": np.array([[-0.20,  0.15, -z_cmd]]),
                        "ref_foot_RR": np.array([[-0.20, -0.15, -z_cmd]]),
                    }

                    # 2. Contact sequence (all standing)
                    # Use nmpc horizon, not mpc_controller horizon
                    contact_seq = np.ones((4, mpc_controller.nmpc.horizon))

                    # 3. Solve MPC with correct interface
                    print("DEBUG - entering controller.solve() ...")
                    try:
                        u, success, info = mpc_controller.solve(
                            x30[0].cpu().numpy(), reference_dict, contact_seq
                        )
                        print("DEBUG - solver returned successfully")
                    except Exception as e:
                        print(f"MPC FAIL at t={step*args_cli.dt:.2f}s: Solver error: {e}")
                        import traceback
                        traceback.print_exc()
                        mpc_ok = False
                        continue  # Skip to next iteration

                    print(f"DEBUG - MPC solve result: success={success}")
                    if success:
                        u_hold = u
                        mpc_ok = True
                        if step % (mpc_steps_interval * 5) == 0:
                            foot_vel = u[:12].reshape(4, 3)
                            forces = u[12:24].reshape(4, 3)
                            print(f"MPC OK at t={step*args_cli.dt:.2f}s")
                            print(f"  Foot vel norm: {np.linalg.norm(foot_vel):.3f}")
                            print(f"  Total force Z: {forces[:, 2].sum():.1f}N")
                    else:
                        # MPC failed: don't use u_hold=0 for IK
                        mpc_ok = False
                        if step % (mpc_steps_interval * 5) == 0:
                            print(f"MPC FAIL at t={step*args_cli.dt:.2f}s: {info.get('error', '')}")
                else:
                    # Stand with zero foot velocity
                    mpc_ok = False

            # --- Map foot velocity to joint positions ---
            if mpc_ok:
                # Extract foot velocity command (first 12 elements of u)
                foot_vel_cmd = torch.from_numpy(u_hold[:12].reshape(1, 4, 3)).to(
                    device=obs["joint_pos"].device, dtype=obs["joint_pos"].dtype
                )
                try:
                    q_des_from_mpc = foot_vel_to_joint_pos_ik(
                        env.robot.robot, foot_vel_cmd,
                        joint_names_12, feet_names_4,
                        dt_mpc_effective
                    )
                    # Use smaller blend factor for standing (verify MPC output first)
                    blend_factor = 0.3
                    actions = blend_factor * q_des_from_mpc + (1 - blend_factor) * q_desired
                except Exception as e:
                    print(f"IK failed at step {step}: {e}")
                    actions = q_desired
            else:
                # True fail-safe: use standing pose directly
                actions = q_desired

            # Apply actions
            env.set_actions(actions)

            # Step simulation
            env.step()

            # Print status periodically
            if step % print_interval == 0:
                base_pos = obs["base_pos"]
                base_height = base_pos[0, 2].item()
                q = obs["joint_pos"]
                qd = obs["joint_vel"]

                print(f"Step {step:4d} | "
                      f"Base height: {base_height:6.3f}m | "
                      f"|q|: {torch.norm(q[0]):6.3f} | "
                      f"|qd|: {torch.norm(qd[0]):6.3f}")

                # Print foot positions
                foot_found = 0
                for foot in feet_names_4:
                    if foot in env.robot.robot.data.body_names:
                        foot_id = env.robot.robot.data.body_names.index(foot)
                        foot_pos = env.robot.robot.data.body_pos_w[0, foot_id]
                        print(f"  {foot}: [{foot_pos[0]:6.3f}, {foot_pos[1]:6.3f}, {foot_pos[2]:6.3f}]")
                        foot_found += 1

                if foot_found < 4:
                    print(f"  Only {foot_found}/4 feet found")
                print()

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")

    finally:
        # Cleanup
        print(f"DEBUG - exiting main loop at sim_step={step if 'step' in locals() else 'unknown'}")
        print("Closing simulation...")
        simulation_app.close()


if __name__ == "__main__":
    main()