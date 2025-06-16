# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import os
import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


# Cross-platform path construction
# Just use the absolute path directly
SO100_USD_PATH = "/home/lycheeai/mnt_data/Projects/IsaacLab-SO_100/source/SO_100/SO_100/tasks/manager_based/ROS_so_100/asset/SO-ARM101-ROS2.usd"
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
            "Rotation": 0.1,
            "Pitch": 0.5,
            "Elbow": 0.0,
            "Wrist_Pitch": 0.0,
            "Wrist_Roll": 0.0,
            "Jaw": 0.3,  # Change from 0.5 to 0.3 (middle position) to make movement more apparent
        },
        # Set initial joint velocities to zero
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Grouping arm joints, adjust limits as needed
        # Shoulder rotation moves: ALL mass (~0.8kg total)
        "shoulder_rotation": ImplicitActuatorCfg(
            joint_names_expr=["Rotation"],
            effort_limit=1.9,
            velocity_limit_sim=1.5,
            stiffness=0.0,   # Highest - moves all mass
            damping=0.0,
        ),
        # Shoulder pitch moves: Everything except base (~0.65kg)
        "shoulder_pitch": ImplicitActuatorCfg(
            joint_names_expr=["Pitch"],
            effort_limit=1.9,
            velocity_limit_sim=1.5,
            stiffness=0.0,     # Increased from 25.0 to 60.0 for more reliable closing
            damping=0.0,   
        ),
        # Elbow moves: Lower arm, wrist, gripper (~0.38kg)
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["Elbow"],
            effort_limit=1.9,
            velocity_limit_sim=1.5,
            stiffness=0.0,     # Increased from 25.0 to 60.0 for more reliable closing
            damping=0.0,   
        ),
        # Wrist pitch moves: Wrist and gripper (~0.24kg)
        "wrist_pitch": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Pitch"],
            effort_limit=1.9,
            velocity_limit_sim=1.5,
            stiffness=0.0,     # Increased from 25.0 to 60.0 for more reliable closing
            damping=0.0,   
        ),
        # Wrist roll moves: Gripper assembly (~0.14kg)
        "wrist_roll": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Roll"],
            effort_limit=1.9,
            velocity_limit_sim=1.5,
            stiffness=0.0,     # Increased from 25.0 to 60.0 for more reliable closing
            damping=0.0,   
        ),
        # Gripper moves: Only moving jaw (~0.034kg)
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["Jaw"],
            effort_limit=2.5,    # Increased from 1.9 to 2.5 for stronger grip
            velocity_limit_sim=1.5,
            stiffness=0.0,     # Increased from 25.0 to 60.0 for more reliable closing
            damping=0.0,       # Increased from 10.0 to 20.0 for stability
        ),
    },
)
"""Configuration of SO100 robot arm."""
# Removed FRANKA_PANDA_HIGH_PD_CFG as it's not applicable



class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    SO100 = SO100_CFG.replace(prim_path="{ENV_REGEX_NS}/SO100")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running():

        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # design scene
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
