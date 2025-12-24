#!/usr/bin/env python
"""Standalone script to test Go2 standing control with AppLauncher."""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Go2 standing control test")
parser.add_argument("--num-envs", type=int, default=1, help="Number of environments")
parser.add_argument("--dt", type=float, default=0.001, help="Simulation timestep")
# ❌ 不要加 --device / --headless（AppLauncher 会自动加）
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

# ✅ 正确启动：把 args_cli 整体传给 AppLauncher
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import sys
import os

# 添加包含包的目录到Python路径（包结构现在是平铺的）
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)
print(f"Added to sys.path: {current_dir}")

import torch
from dataclasses import dataclass

# 直接导入envs.stand_env
from envs.stand_env import StandEnv


@dataclass
class StandController:
    """Simple position controller for standing."""
    position_gain: float = 0.1  # How much of the error to apply each step

    def compute_control(self, q: torch.Tensor, qd: torch.Tensor, q_des: torch.Tensor) -> torch.Tensor:
        """Return desired positions directly.

        Args:
            q: Current joint positions [num_envs, num_joints] (unused)
            qd: Current joint velocities [num_envs, num_joints] (unused)
            q_des: Desired joint positions [num_envs, num_joints]

        Returns:
            Target positions [num_envs, num_joints]
        """
        # Directly return desired positions for standing
        return q_des


def main():
    """Main entry point for Go2 standing test."""

    # Create environment
    print("Creating Go2 standing environment...")
    env = StandEnv.make(
        num_envs=args_cli.num_envs,
        dt=args_cli.dt,
        device=args_cli.device
    )

    # Create standing controller
    controller = StandController(position_gain=0.1)

    # Get default joint positions (standing pose)
    q_default = env.robot.get_default_q()
    q_desired = q_default.clone()

    # 在这里替换为你给的更"高"的站姿
    q_desired[0] = torch.tensor([
        0.10, -0.10,  0.10, -0.10,   # hips
        0.65,  0.65,  0.90,  0.90,   # thighs (rear 稍大一点)
       -1.25, -1.25, -1.25, -1.25,   # calves
    ], device=q_desired.device)

    # Simulation parameters
    max_steps = 5000
    print_interval = 500

    # ======= 关键调试打印（只需要一次） =======
    print("\n=== robot.data.joint_names (order matters!) ===")
    print(env.robot.robot.data.joint_names)

    print("\n=== robot.data.body_names ===")
    print(env.robot.robot.data.body_names)

    print("\n=== q_desired[0] (position target) ===")
    print(q_desired[0].tolist())

    print("========================================\n")
    # ========================================

    # ======= Jacobian诊断代码（获取Isaac Lab版本和方法） =======
    print("\n=== Jacobian Diagnostics ===")
    try:
        print("Type:", type(env.robot.robot.root_physx_view))
        print("Jacobian methods:")
        for item in dir(env.robot.robot.root_physx_view):
            if 'jacob' in item.lower():
                print(f"  {item}")
    except Exception as e:
        print(f"Error accessing root_physx_view: {e}")

    # 获取版本信息
    try:
        import isaaclab
        print("\nIsaac Lab version:", isaaclab.__version__)
    except:
        print("Isaac Lab version: not available")

    try:
        import isaacsim
        print("Isaac Sim version:", isaacsim.core.utils.version.get_version())
    except:
        print("Isaac Sim version: not available")
    print("====================================================\n")
    # ===========================================================

    print("Starting simulation...")
    print(f"Running {max_steps} steps at dt={args_cli.dt}s")

    try:
        # Main simulation loop
        for step in range(max_steps):
            # Get current observations
            obs = env.get_observations()
            q = obs["joint_pos"]
            qd = obs["joint_vel"]
            base_pos = obs["base_pos"]
            base_quat = obs["base_quat"]

            # Compute control actions
            actions = controller.compute_control(q, qd, q_desired)

            # Apply actions
            env.set_actions(actions)

            # Step simulation
            env.step()

            # Print status periodically
            if step % print_interval == 0:
                base_height = base_pos[0, 2].item()
                q_norm = torch.norm(q[0]).item()
                qd_norm = torch.norm(qd[0]).item()
                action_norm = torch.norm(actions[0]).item()

                print(f"Step {step:4d} | "
                      f"Base height: {base_height:6.3f}m | "
                      f"|q|: {q_norm:6.3f} | "
                      f"|qd|: {qd_norm:6.3f} | "
                      f"|action|: {action_norm:6.3f}")

                # === 足端位置验证 ===
                foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
                name_to_id = {n: i for i, n in enumerate(env.robot.robot.data.body_names)}

                # 检查所有脚的名称是否存在
                found_feet = []
                for foot in foot_names:
                    if foot in name_to_id:
                        found_feet.append((foot, name_to_id[foot]))

                if len(found_feet) == 4:
                    ids = [name_to_id[n] for n in foot_names]
                    feet_w = env.robot.robot.data.body_pos_w[0, ids, :]   # (4,3)
                    print("Feet positions (world frame):")
                    for i, (name, pos) in enumerate(zip(foot_names, feet_w)):
                        print(f"  {name}: [{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}]")
                else:
                    print(f"Found {len(found_feet)}/4 feet. Foot names might be different.")
                    print("Available body names (first 20):", env.robot.robot.data.body_names[:20])
                print()

            # Check termination
            terminated = env.check_termination()
            if terminated.any():
                print(f"Episode terminated at step {step}")
                # Reset if needed
                if args_cli.num_envs == 1:
                    print("Resetting environment...")
                    env.reset()

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")

    finally:
        # Cleanup
        print("Closing simulation...")
        simulation_app.close()


if __name__ == "__main__":
    main()