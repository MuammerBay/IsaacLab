# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import SOARM_stack_joint_pos_env_cfg

##
# Pre-defined configs
##
from .SOARM_robot_cfg import SO100_CFG  # isort: skip


@configclass
class SOARMCubeStackEnvCfg(SOARM_stack_joint_pos_env_cfg.SOARMCubeStackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set SO100 as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = SO100_CFG.copy()
        self.scene.robot.prim_path = "{ENV_REGEX_NS}/Robot"

        # Set actions for the specific robot type (SO100)
        # Override the arm_action to use IK instead of joint position
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
            body_name="gripper",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.1)),
        )
