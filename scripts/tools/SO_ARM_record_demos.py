# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Script to record demonstrations with Isaac Lab environments using human teleoperation.

This script allows users to record demonstrations operated by human teleoperation for a specified task.
The recorded demonstrations are stored as episodes in a hdf5 file. Users can specify the task, teleoperation
device, dataset directory, and environment stepping rate through command-line arguments.

required arguments:
    --task                    Name of the task.

optional arguments:
    -h, --help                Show this help message and exit
    --teleop_device           Device for interacting with environment. (default: keyboard)
    --dataset_file            File path to export recorded demos. (default: "./datasets/dataset.hdf5")
    --step_hz                 Environment stepping rate in Hz. (default: 30)
    --num_demos               Number of demonstrations to record. (default: 0)
    --num_success_steps       Number of continuous steps with task success for concluding a demo as successful. (default: 10)
"""

"""Launch Isaac Sim Simulator first."""

# Standard library imports
import argparse
import contextlib

# Third-party imports
import gymnasium as gym
import numpy as np
import os
import time
import torch

# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Record demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment.")
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--num_demos", type=int, default=0, help="Number of demonstrations to record. Set to 0 for infinite."
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Number of continuous steps with task success for concluding a demo as successful. Default is 10.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Omniverse logger
import omni.log
import omni.ui as ui

# Additional Isaac Lab imports that can only be imported after the simulator is running
from isaaclab.devices.keyboard.se3_keyboard_so_arm import Se3KeyboardSOArm

import isaaclab_mimic.envs  # noqa: F401
from isaaclab_mimic.ui.instruction_display import InstructionDisplay, show_subtask_instructions

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.ui import EmptyWindow
from isaaclab.managers import DatasetExportMode

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def print_keyboard_controls():
    """Print the keyboard control instructions for the SO100 robot."""
    print("\n" + "="*60)
    print("SO100 ROBOT KEYBOARD CONTROLS")
    print("="*60)
    print("\nROBOT ARM CONTROL:")
    print("  W/S    →  Move robot forward/backward")
    print("  A/D    →  Move robot left/right")
    print("  Q/E    →  Move robot up/down")
    print("  Z/X    →  Rotate around X-axis")
    print("  T/G    →  Rotate around Y-axis")
    print("  C/V    →  Rotate around Z-axis")
    print("\nGRIPPER CONTROL:")
    print("  K      →  Toggle gripper (open/close)")
    print("\nDEMO RECORDING:")
    print("  R      →  Reset current demonstration")
    print("  ESC    →  Exit application")
    print("\nNOTE: Hold keys to continue movement/rotation")
    print("="*60 + "\n")


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz: int):
        """Initialize a RateLimiter with specified frequency.

        Args:
            hz: Frequency to enforce in Hertz.
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env: gym.Env):
        """Attempt to sleep at the specified rate in hz.

        Args:
            env: Environment to render during sleep periods.
        """
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def pre_process_actions(
    teleop_data: tuple[np.ndarray, bool], num_envs: int, device: str
) -> torch.Tensor:
    """Convert teleop data to the format expected by the environment action space.

    Args:
        teleop_data: Data from the teleoperation device (6D joint positions: 5 arm + 1 gripper).
        num_envs: Number of environments.
        device: Device to create tensors on.

    Returns:
        Processed actions as a tensor (6D for SO100 - direct joint positions).
    """
    # Extract joint positions from keyboard (interpreting SE3 commands as joint increments)
    delta_pose, _ = teleop_data
    
    # For direct joint control, treat the 6D delta as joint position increments
    # [Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw]
    joint_increments = np.zeros(6)
    joint_increments[:5] = delta_pose[:5]  # Use first 5 elements for arm joints
    joint_increments[5] = delta_pose[5] * 0.1  # Scale the last element for jaw/gripper
    
    # Convert to torch tensor
    actions = torch.tensor(joint_increments, dtype=torch.float, device=device).repeat(num_envs, 1)
    
    return actions


def main():
    """Collect demonstrations from the environment using teleop interfaces."""

    # Print keyboard controls at startup
    print_keyboard_controls()

    rate_limiter = RateLimiter(args_cli.step_hz)

    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.env_name = args_cli.task.split(":")[-1]

    # extract success checking function to invoke in the main loop
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        omni.log.warn(
            "No success termination term was found in the environment."
            " Will not be able to mark recorded demos as successful."
        )

    # modify configuration such that the environment runs indefinitely until
    # the goal is reached or other termination conditions are met
    env_cfg.terminations.time_out = None

    env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Flags for controlling the demonstration recording process
    should_reset_recording_instance = False
    running_recording_instance = True

    def reset_recording_instance():
        """Reset the current recording instance."""
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    # Create keyboard teleoperation device
    teleop_interface = Se3KeyboardSOArm(pos_sensitivity=0.2, rot_sensitivity=0.5)
    teleop_interface.add_callback("R", reset_recording_instance)

    # reset before starting
    env.sim.reset()
    env.reset()
    teleop_interface.reset()

    # simulate environment -- run everything in inference mode
    current_recorded_demo_count = 0
    success_step_count = 0

    label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."

    instruction_display = InstructionDisplay(args_cli.teleop_device)
    window = EmptyWindow(env, "Instruction")
    with window.ui_window_elements["main_vstack"]:
        demo_label = ui.Label(label_text)
        subtask_label = ui.Label("")
        instruction_display.set_labels(subtask_label, demo_label)

    subtasks = {}

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running():
            # get data from teleop device
            teleop_data = teleop_interface.advance()

            # perform action on environment
            if running_recording_instance:
                # compute actions based on environment
                actions = pre_process_actions(teleop_data, env.num_envs, env.device)
                obv = env.step(actions)
                if subtasks is not None:
                    if subtasks == {}:
                        subtasks = obv[0].get("subtask_terms")
                    elif subtasks:
                        show_subtask_instructions(instruction_display, subtasks, obv, env.cfg)
            else:
                env.sim.render()

            if success_term is not None:
                if bool(success_term.func(env, **success_term.params)[0]):
                    success_step_count += 1
                    if success_step_count >= args_cli.num_success_steps:
                        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                        env.recorder_manager.set_success_to_episodes(
                            [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                        )
                        env.recorder_manager.export_episodes([0])
                        should_reset_recording_instance = True
                else:
                    success_step_count = 0

            # print out the current demo count if it has changed
            if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
                print(label_text)

            if should_reset_recording_instance:
                env.sim.reset()
                env.recorder_manager.reset()
                env.reset()
                should_reset_recording_instance = False
                success_step_count = 0
                instruction_display.show_demo(label_text)

            if args_cli.num_demos > 0 and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
                print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                break

            # check that simulation is stopped or not
            if env.sim.is_stopped():
                break

            rate_limiter.sleep(env)

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
