# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import ROS_so_100_cube_lift_env_cfg

##
# Pre-defined configs
##
from .ROS_so_100_robot_cfg import SO100_CFG  # isort: skip


@configclass
class SO100CubeLiftIKEnvCfg(ROS_so_100_cube_lift_env_cfg.SO100CubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set SO100 as robot with higher stiffness for IK tracking
        import dataclasses
        from isaaclab.assets import ArticulationCfg
        from isaaclab.actuators import ImplicitActuatorCfg

        # Create a stiffer version of SO100 for better IK tracking
        _robot_cfg = dataclasses.replace(SO100_CFG, prim_path="{ENV_REGEX_NS}/Robot")
        
        # Set initial rotation if needed
        if _robot_cfg.init_state is None:
            _robot_cfg.init_state = ArticulationCfg.InitialStateCfg()
        _robot_cfg.init_state = dataclasses.replace(_robot_cfg.init_state)
        self.scene.robot = _robot_cfg

        # Set actions for IK control (replaces the joint position actions)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
            body_name="gripper",  # End-effector body name
            controller=DifferentialIKControllerCfg(
                command_type="pose", 
                use_relative_mode=True, 
                ik_method="dls"  # Damped Least Squares method
            ),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=(0.01, 0.0, 0.1)  # Offset to the gripper tip (tuple format)
            ),
        )
        
        # Keep the existing gripper action (binary control)
        # self.actions.gripper_action is already set in parent class


@configclass
class SO100CubeLiftIKEnvCfg_PLAY(SO100CubeLiftIKEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
