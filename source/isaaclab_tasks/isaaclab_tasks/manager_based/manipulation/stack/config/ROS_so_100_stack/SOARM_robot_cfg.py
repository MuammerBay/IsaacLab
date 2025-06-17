# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the SO100 5-DOF robot arm for livestream.

The following configurations are available:

* :obj:`SO100_CFG`: SO100 robot arm configuration.

"""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

# Note: Use forward slashes for paths even on Windows
# Construct the absolute path to the USD file relative to this script's location
_THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SO100_USD_PATH = os.path.join(_THIS_SCRIPT_DIR, "asset", "SO-ARM101.usd")  # ROS: STACK_SO-ARM101-ROS2.usd

##
# Configuration
##

SO100_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=SO100_USD_PATH,
        activate_contact_sensors=False,  # Adjust based on need
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,  # Default to False, adjust if needed
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "Rotation": 0.0,
            "Pitch": 0.0,
            "Elbow": 0.0,
            "Wrist_Pitch": 0.0,
            "Wrist_Roll": 0.0,
            "Jaw": 0.3,
        },
        # Set initial joint velocities to zero
        joint_vel={".*": 0.0},
    ),
    # Add actuators for IK control
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"],
            effort_limit=1000,
            velocity_limit=1000,
            stiffness=17.8,
            damping=0.6,
        ),
    },
    # Using default soft limits
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of SO100 robot arm."""
