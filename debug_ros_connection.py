#!/usr/bin/env python3

"""Debug script to test ROS connection with Isaac Lab."""

import rclpy
from rclpy.node import Node
import time
import threading
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32


class RosDebugger(Node):
    """Node to debug ROS connection with Isaac Lab."""
    
    def __init__(self):
        super().__init__('ros_debugger')
        
        # Publishers
        self.pose_pub = self.create_publisher(Twist, '/robot/cmd_pose', 10)
        self.gripper_pub = self.create_publisher(Float32, '/robot/cmd_gripper', 10)
        
        # Subscribers to monitor echo
        self.pose_sub = self.create_subscription(
            Twist, '/robot/cmd_pose', self.pose_echo_callback, 10)
        self.gripper_sub = self.create_subscription(
            Float32, '/robot/cmd_gripper', self.gripper_echo_callback, 10)
        
        self.command_count = 0
        self.echo_count = 0
        
        print("🔍 ROS Debugger started!")
        print("  - Publishing commands every 2 seconds")
        print("  - Monitoring if commands are echoed back")
        print("  - Press Ctrl+C to stop")

    def pose_echo_callback(self, msg):
        """Echo callback for pose commands."""
        self.echo_count += 1
        print(f"📡 ECHO #{self.echo_count}: Pose = x:{msg.linear.x:.3f}, y:{msg.linear.y:.3f}, z:{msg.linear.z:.3f}")

    def gripper_echo_callback(self, msg):
        """Echo callback for gripper commands."""
        print(f"📡 ECHO: Gripper = {msg.data:.3f}")

    def send_test_commands(self):
        """Send test commands periodically."""
        time.sleep(2)  # Wait for node to initialize
        
        while rclpy.ok():
            self.command_count += 1
            
            # Send pose command
            pose_msg = Twist()
            pose_msg.linear.x = 0.01 * (self.command_count % 10)  # Varying command
            pose_msg.linear.y = 0.0
            pose_msg.linear.z = 0.0
            self.pose_pub.publish(pose_msg)
            
            print(f"📤 SENT #{self.command_count}: x={pose_msg.linear.x:.3f}")
            
            # Send gripper command occasionally
            if self.command_count % 5 == 0:
                gripper_msg = Float32()
                gripper_msg.data = 0.2 if self.command_count % 10 == 0 else 0.0
                self.gripper_pub.publish(gripper_msg)
                print(f"📤 SENT gripper: {gripper_msg.data:.3f}")
            
            time.sleep(2)


def main():
    rclpy.init()
    
    debugger = RosDebugger()
    
    # Start command sender in background thread
    sender_thread = threading.Thread(target=debugger.send_test_commands, daemon=True)
    sender_thread.start()
    
    try:
        rclpy.spin(debugger)
    except KeyboardInterrupt:
        print("\n🛑 Debug session ended")
        print("📊 Summary:")
        print(f"   Commands sent: {debugger.command_count}")
        print(f"   Commands echoed: {debugger.echo_count}")
        if debugger.echo_count > 0:
            print("✅ ROS connection is working!")
        else:
            print("❌ No commands were echoed - ROS connection issue")
    
    finally:
        debugger.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 