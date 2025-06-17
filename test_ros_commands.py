#!/usr/bin/env python3

"""Simple script to send ROS commands for testing the robot."""

import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Bool


class RobotCommander(Node):
    """Node to send robot commands via ROS topics."""
    
    def __init__(self):
        super().__init__('robot_commander')
        
        # Create publishers
        self.pose_pub = self.create_publisher(Twist, '/robot/cmd_pose', 10)
        self.gripper_pub = self.create_publisher(Float32, '/robot/cmd_gripper', 10)
        self.reset_pub = self.create_publisher(Bool, '/robot/reset', 10)
        
        print("Robot commander initialized. Publishing to:")
        print("  - /robot/cmd_pose")
        print("  - /robot/cmd_gripper") 
        print("  - /robot/reset")

    def send_pose_command(self, x=0.0, y=0.0, z=0.0, rx=0.0, ry=0.0, rz=0.0):
        """Send a pose command."""
        msg = Twist()
        msg.linear.x = x
        msg.linear.y = y
        msg.linear.z = z
        msg.angular.x = rx
        msg.angular.y = ry
        msg.angular.z = rz
        self.pose_pub.publish(msg)
        print(f"Sent pose: pos=({x:.3f}, {y:.3f}, {z:.3f}), rot=({rx:.3f}, {ry:.3f}, {rz:.3f})")

    def send_gripper_command(self, value=0.0):
        """Send a gripper command (velocity)."""
        msg = Float32()
        msg.data = value
        self.gripper_pub.publish(msg)
        print(f"Sent gripper velocity: {value:.3f}")

    def send_reset_command(self):
        """Send a reset command."""
        msg = Bool()
        msg.data = True
        self.reset_pub.publish(msg)
        print("Sent reset command")


def main():
    rclpy.init()
    commander = RobotCommander()
    
    print("\nStarting demo sequence in 3 seconds...")
    time.sleep(3)
    
    try:
        # Demo sequence: lift the cube
        print("\n=== Demo Sequence: Lift Cube ===")
        
        # Move forward to cube
        print("1. Moving forward to cube...")
        for i in range(10):
            commander.send_pose_command(x=0.02, y=0.0, z=0.0)
            rclpy.spin_once(commander, timeout_sec=0.01)
            time.sleep(0.1)
        
        # Move down to cube level  
        print("2. Moving down to cube...")
        for i in range(10):
            commander.send_pose_command(x=0.0, y=0.0, z=-0.01)
            rclpy.spin_once(commander, timeout_sec=0.01)
            time.sleep(0.1)
            
        # Close gripper with velocity
        print("3. Closing gripper...")
        for i in range(10):
            commander.send_gripper_command(0.3)  # Positive velocity to close (like SO100 keyboard)
            rclpy.spin_once(commander, timeout_sec=0.01)
            time.sleep(0.1)
            
        # Lift up
        print("4. Lifting cube...")
        for i in range(20):
            commander.send_pose_command(x=0.0, y=0.0, z=0.02)
            rclpy.spin_once(commander, timeout_sec=0.01) 
            time.sleep(0.1)
            
        print("5. Demo complete!")
        
        # Stop all motion
        print("6. Stopping motion...")
        for i in range(5):
            commander.send_pose_command()  # Send zeros
            commander.send_gripper_command(0.0)  # Stop gripper
            rclpy.spin_once(commander, timeout_sec=0.01)
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    
    finally:
        commander.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 