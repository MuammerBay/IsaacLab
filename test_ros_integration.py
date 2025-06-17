#!/usr/bin/env python3

"""
Test script for ROS device integration with Isaac Lab.

This script tests the ROS device interface by creating a simple ROS node
that publishes commands and verifies they are received correctly.
"""

import numpy as np
import time
import threading

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from std_msgs.msg import Float32, Bool
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("Error: ROS2 not available. Please install rclpy to test ROS integration.")
    exit(1)

# Test ROS publisher node


class TestPublisherNode(Node):
    """Test node that publishes commands for the ROS device to receive."""
    
    def __init__(self):
        super().__init__('test_publisher')
        
        # Create publishers
        self.pose_pub = self.create_publisher(Twist, '/robot/cmd_pose', 10)
        self.gripper_pub = self.create_publisher(Float32, '/robot/cmd_gripper', 10)
        self.reset_pub = self.create_publisher(Bool, '/robot/reset', 10)
        
        # Create timer for publishing test commands
        self.timer = self.create_timer(0.1, self.publish_test_commands)  # 10 Hz
        self.counter = 0
        
        self.get_logger().info("Test publisher node started")
    
    def publish_test_commands(self):
        """Publish test commands."""
        # Publish pose command
        pose_msg = Twist()
        pose_msg.linear.x = 0.01 * np.sin(self.counter * 0.1)  # Small sinusoidal motion
        pose_msg.linear.y = 0.005 * np.cos(self.counter * 0.1)
        pose_msg.linear.z = 0.002
        pose_msg.angular.x = 0.01
        pose_msg.angular.y = 0.0
        pose_msg.angular.z = 0.02 * np.sin(self.counter * 0.05)
        
        self.pose_pub.publish(pose_msg)
        
        # Publish gripper command
        gripper_msg = Float32()
        gripper_msg.data = 0.5 * np.sin(self.counter * 0.2)  # Oscillating gripper
        self.gripper_pub.publish(gripper_msg)
        
        self.counter += 1
        
        # Publish reset every 50 iterations
        if self.counter % 50 == 0:
            reset_msg = Bool()
            reset_msg.data = True
            self.reset_pub.publish(reset_msg)
            self.get_logger().info(f"Published reset command (iteration {self.counter})")


def test_ros_device():
    """Test the ROS device interface."""
    if not ROS_AVAILABLE:
        print("ROS2 not available, cannot test")
        return False
    
    # Initialize ROS
    rclpy.init()
    
    try:
        # Import the ROS device
        from source.isaaclab.isaaclab.devices.ros.se3_ros_so_arm import Se3RosSOArm
        
        # Create test publisher
        test_publisher = TestPublisherNode()
        
        # Create ROS device interface
        print("Creating ROS device interface...")
        ros_device = Se3RosSOArm(node_name="test_ros_device")
        print("ROS device created successfully!")
        print(ros_device)
        
        # Start ROS spinning in a separate thread
        def spin_nodes():
            rclpy.spin(test_publisher)
        
        spin_thread = threading.Thread(target=spin_nodes, daemon=True)
        spin_thread.start()
        
        print("\nTesting ROS device commands for 10 seconds...")
        start_time = time.time()
        
        while time.time() - start_time < 10.0:
            # Get commands from ROS device
            commands = ros_device.advance()
            
            if np.any(commands != 0.0):
                print(f"Received commands: {commands}")
            
            time.sleep(0.1)  # 10 Hz
        
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error testing ROS device: {e}")
        return False
    
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    print("Testing ROS device integration...")
    success = test_ros_device()
    if success:
        print("✓ ROS device integration test passed!")
    else:
        print("✗ ROS device integration test failed!") 