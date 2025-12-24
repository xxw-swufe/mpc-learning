"""Standing environment for Go2 testing and validation."""

from dataclasses import dataclass
import torch

from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
import isaacsim.core.utils.prims as prim_utils

from robots.go2 import Go2Robot, make_go2_cfg


class StandSceneCfg(InteractiveSceneCfg):
    """Configuration for the standing scene."""
    # Note: robot will be dynamically injected in __init__ with proper prim_path
    # Ground plane and lighting will be created manually in make() function


@dataclass
class StandEnv:
    """
    Standing environment for Go2 robot.

    This environment creates a simple scene with Go2 robot on flat ground
    for testing standing controllers and basic locomotion.
    """
    sim: SimulationContext
    scene: InteractiveScene
    robot: Go2Robot
    dt: float

    @staticmethod
    def make(
        num_envs: int = 1,
        dt: float = 0.001,
        device: str = "cuda"
    ) -> "StandEnv":
        """Create and initialize the standing environment.

        Args:
            num_envs: Number of parallel environments
            dt: Simulation time step
            device: Computing device

        Returns:
            StandEnv: Initialized environment
        """
        # Create simulation context
        sim_cfg = SimulationCfg(dt=dt, device=device)
        sim = SimulationContext(sim_cfg)

        # Set camera view for better visualization
        sim.set_camera_view([2.0, 2.0, 1.0], [0.0, 0.0, 0.3])

        # Create ground plane and lighting manually
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/ground", ground_cfg)

        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0)
        light_cfg.func("/World/Light", light_cfg)

        # Create scene configuration
        scene_cfg = StandSceneCfg(num_envs=num_envs, env_spacing=2.0)

        # Inject Go2 robot configuration into scene
        # Use regex pattern for multiple environments or single path for one environment
        if num_envs > 1:
            robot_prim_path = "/World/envs/env_.*/Robot"
        else:
            robot_prim_path = "/World/Robot"

        scene_cfg.robot = make_go2_cfg(prim_path=robot_prim_path)

        # Create interactive scene
        scene = InteractiveScene(scene_cfg)

        # Initialize simulation and scene
        sim.reset()
        scene.reset()
        scene.update(dt)  # Initialize buffers

        # Get robot articulation from scene
        robot_art: Articulation = scene["robot"]

        # Create Go2 robot wrapper
        go2 = Go2Robot(robot=robot_art, device=device)

        # Initialize standing pose BEFORE gravity takes effect
        q_default = go2.get_default_q()
        go2.set_q_target(q_default)
        scene.write_data_to_sim()
        sim.step()  # Apply one physics step to set the position
        scene.update(dt)

        return StandEnv(sim=sim, scene=scene, robot=go2, dt=dt)

    def step(self):
        """Advance simulation by one time step."""
        # Write all scene data to simulation
        self.scene.write_data_to_sim()

        # Step simulation
        self.sim.step()

        # Update scene buffers
        self.scene.update(self.dt)

    def reset(self):
        """Reset the environment to initial state."""
        # Reset robot to default configuration
        self.robot.reset()

        # Reset scene
        self.scene.reset()

        # Update buffers
        self.scene.update(self.dt)

    def get_observations(self) -> dict:
        """Get current observations from the environment.

        Returns:
            Dictionary containing robot states
        """
        # Get robot state
        q = self.robot.get_q()
        qd = self.robot.get_qd()
        pos, quat, lin_vel, ang_vel = self.robot.get_root_state()

        observations = {
            "joint_pos": q,
            "joint_vel": qd,
            "base_pos": pos,
            "base_quat": quat,
            "base_lin_vel": lin_vel,
            "base_ang_vel": ang_vel,
        }

        return observations

    def set_actions(self, actions: torch.Tensor):
        """Set actions to apply to the robot.

        Args:
            actions: Action tensor [num_envs, num_actions]
        """
        # For standing, we'll use position control
        # This can be extended to velocity or torque control
        self.robot.set_q_target(actions)

    def compute_rewards(self) -> torch.Tensor:
        """Compute rewards for current state.

        Returns:
            Reward tensor [num_envs]
        """
        # Simple reward for maintaining upright posture
        pos, quat, _, _ = self.robot.get_root_state()

        # Extract z-axis from quaternion to measure upright
        # Quaternion order: [w, x, y, z]
        z_axis = quat[:, 3]  # This is simplified, proper calculation needed

        # Reward: higher when more upright (z closer to 1)
        rewards = z_axis

        return rewards

    def check_termination(self) -> torch.Tensor:
        """Check if episode should terminate.

        Returns:
            Boolean tensor [num_envs] indicating termination
        """
        pos, quat, _, _ = self.robot.get_root_state()

        # Terminate if robot falls (height too low)
        terminated = pos[:, 2] < 0.2  # Height threshold

        return terminated