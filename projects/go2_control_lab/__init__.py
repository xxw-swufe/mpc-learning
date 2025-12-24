"""Go2 Control Laboratory.

A package for controlling Unitree Go2 quadruped robot using Isaac Lab.
Provides Model Predictive Control (MPC) and Differential Inverse Kinematics (diff-IK)
implementations for robot locomotion and manipulation tasks.
"""

__version__ = "0.1.0"

# 延迟导入以避免在模块加载时需要 Isaac Sim 环境