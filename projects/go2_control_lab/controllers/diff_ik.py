"""Differential Inverse Kinematics controller for Go2.

This module implements a differential IK controller that can compute joint velocities
required to achieve desired foot trajectories.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np

from ..robots.go2 import Go2Robot
from ..utils.frames import transform_to_body_frame


class DiffIKController:
    """
    Differential Inverse Kinematics controller for Go2 quadruped.

    Computes joint velocities to achieve desired foot positions using Jacobian-based
    differential IK. Supports position control of individual feet.
    """

    def __init__(
        self,
        robot: Go2Robot,
        gain: float = 10.0,
        damping: float = 0.1,
        max_joint_vel: float = 1.0,
        device: str = "cuda:0"
    ):
        """Initialize differential IK controller.

        Args:
            robot: Go2 robot instance
            gain: Position error gain
            damping: Damping for pseudo-inverse
            max_joint_vel: Maximum joint velocity (rad/s)
            device: Computation device
        """
        self.robot = robot
        self.gain = gain
        self.damping = damping
        self.max_joint_vel = max_joint_vel
        self.device = device

        # Store joint limits
        self._joint_limits = robot._joint_limits

        # Initialize Jacobian cache
        self._jacobian_cache = {}

        # Foot configuration
        self.foot_names = ["FL", "FR", "RL", "RR"]
        self.leg_joint_indices = {
            "FL": [0, 1, 2],   # FL_hip, FL_thigh, FL_calf
            "FR": [3, 4, 5],   # FR_hip, FR_thigh, FR_calf
            "RL": [6, 7, 8],   # RL_hip, RL_thigh, RL_calf
            "RR": [9, 10, 11]  # RR_hip, RR_thigh, RR_calf
        }

        # Leg kinematics parameters (approximate for Go2)
        # These should be calibrated based on actual robot dimensions
        self.leg_params = {
            "hip_offset": 0.08,      # Hip joint lateral offset
            "upper_leg_length": 0.20, # Thigh length
            "lower_leg_length": 0.20, # Calf length
        }

    def compute_leg_jacobian(self, foot_name: str, joint_pos: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian matrix for a single leg.

        Args:
            foot_name: Name of foot ('FL', 'FR', 'RL', 'RR')
            joint_pos: Current joint positions [batch, 12]

        Returns:
            Jacobian matrix [batch, 3, 3] (3 foot DOF, 3 joint DOF)
        """
        batch_size = joint_pos.shape[0]
        device = joint_pos.device

        # Get joint indices for this leg
        joint_idx = self.leg_joint_indices[foot_name]
        q = joint_pos[:, joint_idx]  # [batch, 3]

        # Joint angles
        hip_angle = q[:, 0]
        thigh_angle = q[:, 1]
        calf_angle = q[:, 2]

        # Link lengths
        L1 = self.upper_leg_length
        L2 = self.lower_leg_length

        # Pre-compute sines and cosines
        s1, c1 = torch.sin(hip_angle), torch.cos(hip_angle)
        s2, c2 = torch.sin(thigh_angle), torch.cos(thigh_angle)
        s12 = torch.sin(hip_angle + thigh_angle)
        c12 = torch.cos(hip_angle + thigh_angle)

        # Initialize Jacobian
        J = torch.zeros(batch_size, 3, 3, device=device)

        # Position-based Jacobian (simplified 2D leg kinematics)
        # In a full implementation, this would be 3D with proper rotation matrices

        # dx/dq - foot position change with respect to joint angles
        J[:, 0, 0] = -L1 * s2 - L2 * s12  # dx/d_hip
        J[:, 1, 0] = 0.0                  # dy/d_hip (simplified)
        J[:, 2, 0] = 0.0                  # dz/d_hip (simplified)

        J[:, 0, 1] = -L1 * c2 - L2 * c12  # dx/d_thigh
        J[:, 1, 1] = 0.0                  # dy/d_thigh
        J[:, 2, 1] = 0.0                  # dz/d_thigh

        J[:, 0, 2] = -L2 * c12            # dx/d_calf
        J[:, 1, 2] = 0.0                  # dy/d_calf
        J[:, 2, 2] = 0.0                  # dz/d_calf

        return J

    def compute_joint_velocities(
        self,
        foot_commands: Dict[str, torch.Tensor],
        current_joint_pos: torch.Tensor,
        base_pose: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute joint velocities to achieve desired foot trajectories.

        Args:
            foot_commands: Dictionary mapping foot names to desired velocities
                          {'FL': [batch, 3], 'FR': [batch, 3], ...}
            current_joint_pos: Current joint positions [batch, 12]
            base_pose: Optional base pose for transforming commands

        Returns:
            Joint velocities [batch, 12]
        """
        batch_size = current_joint_pos.shape[0]
        device = current_joint_pos.device

        # Initialize joint velocities
        joint_vel = torch.zeros(batch_size, 12, device=device)

        # Process each foot independently
        for foot_name in self.foot_names:
            if foot_name not in foot_commands:
                continue

            # Get foot velocity command
            foot_vel = foot_commands[foot_name]  # [batch, 3]

            # Transform to body frame if base pose provided
            if base_pose is not None:
                foot_vel = transform_to_body_frame(foot_vel, base_pose)

            # Compute Jacobian for this leg
            J = self.compute_leg_jacobian(foot_name, current_joint_pos)

            # Compute joint velocities using damped least squares
            # J_vel = J^T (J J^T + λI)^-1 foot_vel
            lambda_damping = self.damping * torch.eye(3, device=device).unsqueeze(0)
            JJT = torch.bmm(J, J.transpose(1, 2)) + lambda_damping
            JJT_inv = torch.inverse(JJT)

            # Compute joint velocities for this leg
            leg_joint_idx = self.leg_joint_indices[foot_name]
            leg_joint_vel = torch.bmm(
                J.transpose(1, 2),
                torch.bmm(JJT_inv, foot_vel.unsqueeze(-1))
            ).squeeze(-1)

            # Apply gain
            leg_joint_vel *= self.gain

            # Clamp velocities
            leg_joint_vel = torch.clamp(
                leg_joint_vel, -self.max_joint_vel, self.max_joint_vel
            )

            # Store result
            joint_vel[:, leg_joint_idx] = leg_joint_vel

        return joint_vel

    def compute_stance_control(
        self,
        desired_height: float,
        base_pos: torch.Tensor,
        base_quat: torch.Tensor,
        current_joint_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute joint velocities to maintain stance posture.

        Args:
            desired_height: Desired COM height
            base_pos: Current base position [batch, 3]
            base_quat: Current base orientation [batch, 4]
            current_joint_pos: Current joint positions [batch, 12]

        Returns:
            Joint velocities [batch, 12]
        """
        batch_size = current_joint_pos.shape[0]
        device = current_joint_pos.device

        # Compute height error
        height_error = desired_height - base_pos[:, 2]
        height_error = height_error.unsqueeze(-1)  # [batch, 1]

        # Distribute height correction to all joints equally
        joint_vel = torch.zeros(batch_size, 12, device=device)
        joint_vel[:, :] = height_error * self.gain * 0.1  # Distribute evenly

        return joint_vel

    def ik_position(
        self,
        foot_positions: Dict[str, torch.Tensor],
        current_joint_pos: torch.Tensor,
        max_iterations: int = 10,
        tolerance: float = 1e-4
    ) -> Tuple[torch.Tensor, bool]:
        """
        Solve inverse kinematics for desired foot positions.

        Args:
            foot_positions: Desired foot positions in world frame
            current_joint_pos: Current joint positions
            max_iterations: Maximum optimization iterations
            tolerance: Convergence tolerance

        Returns:
            Tuple of (joint_positions, success_flag)
        """
        # Initialize with current positions
        joint_pos = current_joint_pos.clone()

        for iteration in range(max_iterations):
            # Compute current foot positions (would need FK implementation)
            current_foot_pos = {}  # Placeholder

            # Compute position errors
            all_converged = True
            foot_commands = {}

            for foot_name in self.foot_names:
                if foot_name in foot_positions:
                    target_pos = foot_positions[foot_name]
                    current_pos = current_foot_pos.get(foot_name, target_pos)  # Placeholder
                    error = target_pos - current_pos

                    if torch.norm(error, dim=-1).mean() > tolerance:
                        all_converged = False
                        # Convert position error to velocity command
                        foot_commands[foot_name] = error * self.gain

            if all_converged:
                break

            # Compute joint velocity corrections
            joint_vel = self.compute_joint_velocities(
                foot_commands, joint_pos
            )

            # Update joint positions
            joint_pos += joint_vel * 0.01  # Assuming dt=0.01

            # Clamp to joint limits
            joint_pos = torch.clamp(
                joint_pos,
                self._joint_limits["lower"],
                self._joint_limits["upper"]
            )

        return joint_pos, all_converged

    @property
    def upper_leg_length(self) -> float:
        """Upper leg (thigh) length."""
        return self.leg_params["upper_leg_length"]

    @property
    def lower_leg_length(self) -> float:
        """Lower leg (calf) length."""
        return self.leg_params["lower_leg_length"]