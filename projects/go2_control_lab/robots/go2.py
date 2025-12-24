"""Go2 quadruped robot wrapper for Isaac Lab."""

from dataclasses import dataclass
import torch
from typing import Optional, Tuple

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import DCMotorCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


def make_go2_cfg(prim_path: str) -> ArticulationCfg:
    """Create Go2 robot configuration.

    Args:
        prim_path: Prim path for robot in USD scene

    Returns:
        ArticulationCfg: Go2 robot configuration
    """
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.4),
            joint_pos={
                ".*L_hip_joint": 0.1,
                ".*R_hip_joint": -0.1,
                "F[L,R]_thigh_joint": 0.8,
                "R[L,R]_thigh_joint": 1.0,
                ".*_calf_joint": -1.5,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "base_legs": DCMotorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                effort_limit=23.5,
                saturation_effort=23.5,
                velocity_limit=30.0,
                stiffness=25.0,
                damping=0.5,
                friction=0.0,
            ),
        },
    )


@dataclass
class Go2Robot:
    """Go2 quadruped robot wrapper for Isaac Lab.

    This class provides high-level interface for interacting with Go2 robot in Isaac Lab,
    including state reading, command application, and utilities for control algorithms.
    """
    robot: Articulation
    device: str

    def __post_init__(self):
        """Initialize after dataclass creation."""
        # Store reference to device from robot if not provided
        if self.device is None:
            self.device = self.robot.device

    @classmethod
    def create(
        cls,
        prim_path: str = "/World/Go2",
        device: str = "cuda:0",
        num_envs: int = 1
    ) -> "Go2Robot":
        """Create Go2 robot instance.

        Args:
            prim_path: Prim path for robot in USD scene
            device: Device for computation
            num_envs: Number of environments

        Returns:
            Go2Robot: Robot instance
        """
        # Create configuration
        cfg = make_go2_cfg(prim_path)

        # Create articulation
        robot = Articulation(cfg=cfg)

        return cls(robot=robot, device=device)

    def initialize(self, sim_context):
        """Initialize robot after simulation context is created.

        Args:
            sim_context: Simulation context instance
        """
        # Prepare robot for simulation
        self.robot.initialize(sim_context.device)

    def reset(self, root_pos: Optional[torch.Tensor] = None):
        """Reset robot to default configuration.

        Args:
            root_pos: Optional root position override [num_envs, 3]
        """
        # Reset root state
        root_state = self.robot.data.default_root_state.clone()
        if root_pos is not None:
            root_state[:, :3] = root_pos
        self.robot.write_root_pose_to_sim(root_state[:, :7])
        self.robot.write_root_velocity_to_sim(root_state[:, 7:])

        # Reset joints
        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)

        # Clear internal buffers
        self.robot.reset()

    @property
    def joint_names(self):
        """Get joint names."""
        return self.robot.data.joint_names

    @property
    def body_names(self):
        """Get body names."""
        return self.robot.data.body_names

    @property
    def num_joints(self) -> int:
        """Number of joints."""
        return self.robot.num_joints

    @property
    def num_actions(self) -> int:
        """Dimension of action space."""
        return self.num_joints

    @property
    def num_observations(self) -> int:
        """Dimension of observation space."""
        # Base pose (7) + base velocity (6) + joint positions (12) + joint velocities (12)
        return 37

    def get_default_q(self) -> torch.Tensor:
        """Get default joint positions.

        Returns:
            Default joint positions [num_envs, num_joints]
        """
        return self.robot.data.default_joint_pos.clone()

    def set_q_target(self, q_des: torch.Tensor):
        """Set target joint positions.

        Args:
            q_des: Target joint positions [num_envs, num_joints]
        """
        self.robot.set_joint_position_target(q_des)

    def set_q_vel_target(self, qd_des: torch.Tensor):
        """Set target joint velocities.

        Args:
            qd_des: Target joint velocities [num_envs, num_joints]
        """
        self.robot.set_joint_velocity_target(qd_des)

    def set_q_effort_target(self, tau_des: torch.Tensor):
        """Set target joint efforts.

        Args:
            tau_des: Target joint efforts [num_envs, num_joints]
        """
        self.robot.set_joint_effort_target(tau_des)

    def get_q(self) -> torch.Tensor:
        """Get current joint positions.

        Returns:
            Current joint positions [num_envs, num_joints]
        """
        return self.robot.data.joint_pos.clone()

    def get_qd(self) -> torch.Tensor:
        """Get current joint velocities.

        Returns:
            Current joint velocities [num_envs, num_joints]
        """
        return self.robot.data.joint_vel.clone()

    def get_root_state(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get root state in world frame.

        Returns:
            Tuple of:
            - position [num_envs, 3]
            - quaternion (w,x,y,z) [num_envs, 4]
            - linear velocity [num_envs, 3]
            - angular velocity [num_envs, 3]
        """
        pos = self.robot.data.root_pos_w.clone()
        quat = self.robot.data.root_quat_w.clone()
        lin_vel = self.robot.data.root_lin_vel_w.clone()
        ang_vel = self.robot.data.root_ang_vel_w.clone()
        return pos, quat, lin_vel, ang_vel

    def get_base_pose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get base pose (position and orientation).

        Returns:
            Tuple of (position [num_envs, 3], quaternion [num_envs, 4])
        """
        pos = self.robot.data.root_pos_w.clone()
        quat = self.robot.data.root_quat_w.clone()
        return pos, quat

    def get_base_velocity(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get base velocity (linear and angular).

        Returns:
            Tuple of (linear velocity [num_envs, 3], angular velocity [num_envs, 3])
        """
        lin_vel = self.robot.data.root_lin_vel_w.clone()
        ang_vel = self.robot.data.root_ang_vel_w.clone()
        return lin_vel, ang_vel

    def write_data_to_sim(self):
        """Write queued commands to simulation."""
        self.robot.write_data_to_sim()

    def update(self, dt: float):
        """Update robot data after simulation step.

        Args:
            dt: Simulation time step
        """
        self.robot.update(dt)

    def get_foot_positions(self) -> torch.Tensor:
        """Get current foot positions in world frame.

        Returns:
            Foot positions [num_envs, 4, 3] (FL, FR, RL, RR)
        """
        # This requires implementing forward kinematics
        # For now, return placeholder
        num_envs = self.robot.data.root_pos_w.shape[0]
        return torch.zeros(num_envs, 4, 3, device=self.device)

    def get_foot_forces(self) -> torch.Tensor:
        """Get current foot contact forces.

        Returns:
            Foot forces [num_envs, 4, 3] (FL, FR, RL, RR)
        """
        # This would read from contact sensors
        # For now, return placeholder
        num_envs = self.robot.data.root_pos_w.shape[0]
        return torch.zeros(num_envs, 4, 3, device=self.device)