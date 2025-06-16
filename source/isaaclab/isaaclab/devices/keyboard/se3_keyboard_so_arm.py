# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard controller for SE(3) control."""

import numpy as np
import weakref
from collections.abc import Callable
from scipy.spatial.transform import Rotation

import carb
import omni

from ..device_base import DeviceBase


class Se3KeyboardSOArm(DeviceBase):
    """A keyboard controller for sending SE(3) commands as delta poses for SO100 arm.

    This class is designed to provide a keyboard controller for a robotic arm with gripper control.
    It uses the Omniverse keyboard interface to listen to keyboard events and map them to robot's
    task-space commands.

    The command comprises of a 7D vector of (x, y, z, roll, pitch, yaw, gripper) in meters, radians, and velocity.

    Key bindings:
        ============================== ================= =================
        Description                    Key (+ve axis)    Key (-ve axis)
        ============================== ================= =================
        Move along x-axis              W                 S
        Move along y-axis              A                 D
        Move along z-axis              Q                 E
        Rotate along x-axis            Z                 X
        Rotate along y-axis            T                 G
        Rotate along z-axis            C                 V
        Gripper close (gradual)        F (hold)          
        Gripper open (gradual)         R (hold)          
        Reset pose                     L
        ============================== ================= =================

    .. note::
        Gripper control is velocity-based and gradual. Hold F to slowly close, R to slowly open.
        Release the key to stop gripper movement.

    .. seealso::

        The official documentation for the keyboard interface: `Carb Keyboard Interface <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.

    """

    def __init__(self, pos_sensitivity: float = 0.4, rot_sensitivity: float = 0.8):
        """Initialize the keyboard layer.

        Args:
            pos_sensitivity: Magnitude of input position command scaling. Defaults to 0.4.
            rot_sensitivity: Magnitude of scale input rotation commands scaling. Defaults to 0.8.
        """
        # store inputs
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called.
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        # bindings for keyboard to command
        self._create_key_bindings()
        # command buffers
        self._delta_pos = np.zeros(3)  # (x, y, z)
        self._delta_rot = np.zeros(3)  # (roll, pitch, yaw)
        self._gripper_vel = 0.0  # gradual gripper velocity (-0.3 to 0.3)
        # dictionary for additional callbacks
        self._additional_callbacks = dict()

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Keyboard Controller for SE(3): {self.__class__.__name__}\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tMove arm along x-axis: W/S\n"
        msg += "\tMove arm along y-axis: A/D\n"
        msg += "\tMove arm along z-axis: Q/E\n"
        msg += "\tRotate arm along x-axis: Z/X\n"
        msg += "\tRotate arm along y-axis: T/G\n"
        msg += "\tRotate arm along z-axis: C/V\n"
        msg += "\tGripper close (hold): F\n"
        msg += "\tGripper open (hold): R\n"
        msg += "\tReset pose: L"
        return msg

    """
    Operations
    """

    def reset(self):
        # default flags
        self._delta_pos = np.zeros(3)  # (x, y, z)
        self._delta_rot = np.zeros(3)  # (roll, pitch, yaw)
        self._gripper_vel = 0.0  # reset gripper velocity

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind keyboard.

        A list of available keys are present in the
        `carb documentation <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.

        Args:
            key: The keyboard button to check against.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        self._additional_callbacks[key] = func

    def advance(self) -> np.ndarray:
        """Provides the result from keyboard event state.

        Returns:
            A 7D numpy array containing the delta pose command (x, y, z, roll, pitch, yaw, gripper).
        """
        # convert to rotation vector
        rot_vec = Rotation.from_euler("XYZ", self._delta_rot).as_rotvec()
        # return 7D command: 6D pose + gripper velocity
        return np.concatenate([self._delta_pos, rot_vec, [self._gripper_vel]])

    """
    Internal helpers.
    """

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Subscriber callback to when kit is updated.

        Reference:
            https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html
        """
        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "L":
                self.reset()
            elif event.input.name in ["W", "S", "A", "D", "Q", "E"]:
                self._delta_pos += self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["Z", "X", "T", "G", "C", "V"]:
                self._delta_rot += self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name == "F":
                self._gripper_vel = 0.3  # gentle close velocity
            elif event.input.name == "R":
                self._gripper_vel = -0.3  # gentle open velocity
        # remove the command when un-pressed
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in ["W", "S", "A", "D", "Q", "E"]:
                self._delta_pos -= self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["Z", "X", "T", "G", "C", "V"]:
                self._delta_rot -= self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["F", "R"]:
                self._gripper_vel = 0.0  # stop gripper when key released
        # additional callbacks
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._additional_callbacks:
                self._additional_callbacks[event.input.name]()

        # since no error, we are fine :)
        return True

    def _create_key_bindings(self):
        """Creates default key binding."""
        self._INPUT_KEY_MAPPING = {
            # x-axis (forward)
            "W": np.asarray([1.0, 0.0, 0.0]) * self.pos_sensitivity,
            "S": np.asarray([-1.0, 0.0, 0.0]) * self.pos_sensitivity,
            # y-axis (left-right)
            "A": np.asarray([0.0, 1.0, 0.0]) * self.pos_sensitivity,
            "D": np.asarray([0.0, -1.0, 0.0]) * self.pos_sensitivity,
            # z-axis (up-down)
            "Q": np.asarray([0.0, 0.0, 1.0]) * self.pos_sensitivity,
            "E": np.asarray([0.0, 0.0, -1.0]) * self.pos_sensitivity,
            # roll (around x-axis)
            "Z": np.asarray([1.0, 0.0, 0.0]) * self.rot_sensitivity,
            "X": np.asarray([-1.0, 0.0, 0.0]) * self.rot_sensitivity,
            # pitch (around y-axis)
            "T": np.asarray([0.0, 1.0, 0.0]) * self.rot_sensitivity,
            "G": np.asarray([0.0, -1.0, 0.0]) * self.rot_sensitivity,
            # yaw (around z-axis)
            "C": np.asarray([0.0, 0.0, 1.0]) * self.rot_sensitivity,
            "V": np.asarray([0.0, 0.0, -1.0]) * self.rot_sensitivity,
        }
