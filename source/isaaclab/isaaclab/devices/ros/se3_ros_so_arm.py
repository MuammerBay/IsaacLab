# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ROS2 interface for teleoperation with Isaac Lab, compatible with SO100 format."""

import threading
from typing import Any, Callable

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from std_msgs.msg import Bool, Float32

from isaaclab.devices.device_base import DeviceBase


class Se3RosSOArm(DeviceBase):
    """ROS2 interface for SE(3) control, compatible with SO100 keyboard format.
    
    This device subscribes to ROS topics for pose and gripper commands,
    and provides the same 7D output format as the SO100 keyboard device.
    """

    def __init__(self, node_name: str = "isaac_lab_teleop", pos_sensitivity: float = 1.0, rot_sensitivity: float = 1.0):
        """Initialize ROS2 interface for SE(3) control.
        
        Args:
            node_name: Name for the ROS2 node
            pos_sensitivity: Scaling factor for position commands
            rot_sensitivity: Scaling factor for rotation commands
        """
        super().__init__()
        
        # Store parameters
        self.pos_sensitivity = pos_sensitivity 
        self.rot_sensitivity = rot_sensitivity
        self.node_name = node_name
        
        # Initialize ROS
        if not rclpy.ok():
            rclpy.init()
            
        # Create node
        self._node = Node(self.node_name)
        
        # Command storage with thread safety
        self._lock = threading.Lock()
        self._pos_cmd = np.zeros(3)
        self._rot_cmd = np.zeros(3) 
        self._gripper_vel = 0.0  # Gripper velocity like SO100 keyboard
        self._reset_cmd = False
        
        # Store previous joint positions for delta calculation
        self._prev_joint_positions = None
        self._joint_names = ['Rotation', 'Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll', 'Jaw']
        
        # Create subscribers for external ROS commands
        self._pose_sub = self._node.create_subscription(
            Twist, '/robot/cmd_pose', self._pose_callback, 10)
        self._gripper_sub = self._node.create_subscription(
            Float32, '/robot/cmd_gripper', self._gripper_callback, 10)
        self._reset_sub = self._node.create_subscription(
            Bool, '/robot/reset', self._reset_callback, 10)
            
        # Start ROS spinning in background thread
        self._ros_thread = threading.Thread(target=self._ros_spin, daemon=True)
        self._ros_thread.start()
        
        # Additional callbacks
        self._additional_callbacks = {}
        
        # Debug tracking for received commands
        self._debug_print = True
        self._last_print_time = 0
        self._commands_received = 0
        
        print(f"✓ ROS device interface initialized. Node: {node_name}")
        print("  Waiting for ROS commands on topics:")
        print("    - /robot/cmd_pose (geometry_msgs/Twist)")
        print("    - /robot/cmd_gripper (std_msgs/Float32)")  
        print("    - /robot/reset (std_msgs/Bool)")
        print("  Publishing debug info every 5 seconds...")

    def __del__(self):
        """Clean up ROS resources."""
        if hasattr(self, '_node'):
            self._node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def __str__(self) -> str:
        """Return device information."""
        msg = f"ROS2 Controller for SE(3): {self.__class__.__name__}\n"
        msg += f"\tNode name: {self.node_name}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tPose commands: /robot/cmd_pose (geometry_msgs/Twist)\n"
        msg += "\tGripper commands: /robot/cmd_gripper (std_msgs/Float32)\n"
        msg += "\tReset commands: /robot/reset (std_msgs/Bool)\n"
        return msg

    def reset(self):
        """Reset command buffers."""
        with self._lock:
            self._pos_cmd.fill(0.0)
            self._rot_cmd.fill(0.0) 
            self._gripper_vel = 0.0
            self._reset_cmd = False

    def add_callback(self, key: Any, func: Callable):
        """Add additional callback functions.
        
        Args:
            key: Identifier for the callback
            func: Function to call
        """
        self._additional_callbacks[key] = func

    def advance(self) -> np.ndarray:
        """Get current command state.
        
        Returns:
            A 7D numpy array containing the delta pose command (x, y, z, roll, pitch, yaw, gripper_vel).
            This matches the SO100 keyboard device output format.
        """
        with self._lock:
            # Use current commands with sensitivity scaling 
            delta_pos = self._pos_cmd * self.pos_sensitivity
            delta_rot = self._rot_cmd * self.rot_sensitivity  
            gripper_vel = self._gripper_vel
            
            # Convert rotation to rotation vector (like SO100 keyboard does)
            rot_vec = Rotation.from_euler("XYZ", delta_rot).as_rotvec()
            
            # Create 7D command array [dx, dy, dz, drx, dry, drz, gripper_vel]
            command_array = np.concatenate([delta_pos, rot_vec, [gripper_vel]])
            
            # Print debug info for non-zero commands
            if np.any(command_array != 0):
                print(f"📡 Received ROS command: {command_array}")
                self._commands_received += 1
            
            # Status update every 5 seconds
            import time
            current_time = time.time()
            if current_time - self._last_print_time > 5.0:
                print(f"📊 ROS Status: {self._commands_received} commands received in last 5s")
                self._last_print_time = current_time
                self._commands_received = 0
            
            return command_array

    def _pose_callback(self, msg: Twist):
        """Handle pose command messages."""
        with self._lock:
            self._pos_cmd[0] = msg.linear.x
            self._pos_cmd[1] = msg.linear.y  
            self._pos_cmd[2] = msg.linear.z
            self._rot_cmd[0] = msg.angular.x
            self._rot_cmd[1] = msg.angular.y
            self._rot_cmd[2] = msg.angular.z
            print(f"🔍 ROS pose received: pos={self._pos_cmd}, rot={self._rot_cmd}")

    def _gripper_callback(self, msg: Float32):
        """Handle gripper command messages."""
        with self._lock:
            # Store gripper velocity directly (like SO100 keyboard)
            self._gripper_vel = msg.data
            print(f"🔍 ROS gripper received: gripper_vel={self._gripper_vel}")

    def _reset_callback(self, msg: Bool):
        """Handle reset command messages."""
        with self._lock:
            self._reset_cmd = msg.data
            if self._reset_cmd:
                self.reset()
            print(f"🔍 ROS reset received: reset={self._reset_cmd}")

    def stop(self):
        """Stop the robot by setting all commands to zero."""
        with self._lock:
            self._pos_cmd.fill(0.0)
            self._rot_cmd.fill(0.0)
            self._gripper_vel = 0.0
            print("🛑 Robot stopped - all commands set to zero")

    def _ros_spin(self):
        """Background thread for ROS message processing."""
        while rclpy.ok():
            try:
                rclpy.spin_once(self._node, timeout_sec=0.01)
            except Exception as e:
                print(f"ROS spin error: {e}")
                break 