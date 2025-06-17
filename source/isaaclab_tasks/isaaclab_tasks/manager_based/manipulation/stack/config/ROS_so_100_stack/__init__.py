# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import (
    SOARM_stack_ik_rel_env_cfg,
    SOARM_stack_joint_pos_env_cfg,
    agents,
)

##
# Register Gym environments.
##

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Stack-Cube-SOARM-IK-Rel-v0", # Isaac-Stack-Cube-Franka-IK-Rel-v0
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": SOARM_stack_ik_rel_env_cfg.SOARMCubeStackEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
    },
    disable_env_checker=True,
)