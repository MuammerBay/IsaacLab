# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# Register the SO-100 Cube Lift environment
gym.register(
    id="ROS_Template-So-100-CubeLift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ROS_so_100_cube_lift_env_cfg:SO100CubeLiftEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

# Register the SO-100 Cube Lift IK environment (7D actions: 6D pose + gripper)
gym.register(
    id="ROS_Template-So-100-CubeLift-IK-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ROS_so_100_cube_lift_ik_env_cfg:SO100CubeLiftIKEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)