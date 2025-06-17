# ROS2 Integration for Isaac Lab Demo Recording

This document explains how to use the new ROS2 integration to record demonstrations with Isaac Lab while controlling the robot through ROS topics.

## Overview

The ROS integration allows you to:
- Control the SO100 robot using ROS2 topics instead of keyboard/mouse inputs
- Record meaningful action data (not zeros) when using external ROS control
- Maintain all existing success detection and termination functionality

## ROS Topics

The ROS device interface subscribes to the following topics:

| Topic Name | Message Type | Description |
|------------|--------------|-------------|
| `/robot/cmd_pose` | `geometry_msgs/Twist` | Delta pose commands (position + rotation) |
| `/robot/cmd_gripper` | `std_msgs/Float32` | Gripper velocity commands |
| `/robot/reset` | `std_msgs/Bool` | Reset signal |

### Message Format

#### Pose Commands (`/robot/cmd_pose`)
```bash
# geometry_msgs/Twist
linear:
  x: 0.01    # Position delta X (meters)
  y: 0.0     # Position delta Y (meters) 
  z: 0.005   # Position delta Z (meters)
angular:
  x: 0.02    # Roll delta (radians)
  y: 0.0     # Pitch delta (radians)
  z: 0.01    # Yaw delta (radians)
```

#### Gripper Commands (`/robot/cmd_gripper`)
```bash
# std_msgs/Float32
data: 0.5    # Gripper velocity (-1.0 to 1.0)
```

#### Reset Commands (`/robot/reset`)
```bash
# std_msgs/Bool
data: true   # Trigger reset
```

## Usage

### 1. Start Recording with ROS Device

```bash
python scripts/tools/record_demos.py \
    --task Isaac-Lift-Cube-SO100-v0 \
    --teleop_device ros_so_arm \
    --num_demos 10 \
    --dataset_file ./datasets/ros_demos.hdf5
```

### 2. Publish ROS Commands

From another terminal, publish commands to control the robot:

```bash
# Example: Move robot forward and up
ros2 topic pub --once /robot/cmd_pose geometry_msgs/msg/Twist \
  '{linear: {x: 0.02, y: 0.0, z: 0.01}, angular: {x: 0.0, y: 0.0, z: 0.0}}'

# Example: Close gripper
ros2 topic pub --once /robot/cmd_gripper std_msgs/msg/Float32 '{data: 0.8}'

# Example: Reset
ros2 topic pub --once /robot/reset std_msgs/msg/Bool '{data: true}'
```

### 3. Automated Control Script

You can also create a Python script to control the robot:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
import time

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        self.pose_pub = self.create_publisher(Twist, '/robot/cmd_pose', 10)
        self.gripper_pub = self.create_publisher(Float32, '/robot/cmd_gripper', 10)
        
    def move_robot(self, dx, dy, dz, drx=0.0, dry=0.0, drz=0.0):
        msg = Twist()
        msg.linear.x = dx
        msg.linear.y = dy
        msg.linear.z = dz
        msg.angular.x = drx
        msg.angular.y = dry
        msg.angular.z = drz
        self.pose_pub.publish(msg)
        
    def control_gripper(self, velocity):
        msg = Float32()
        msg.data = velocity
        self.gripper_pub.publish(msg)

def main():
    rclpy.init()
    controller = RobotController()
    
    # Example: Simple pick and place sequence
    time.sleep(1)  # Wait for publisher to connect
    
    # Move towards cube
    for _ in range(10):
        controller.move_robot(0.01, 0.0, 0.0)  # Move forward
        time.sleep(0.1)
    
    # Move down to cube
    for _ in range(5):
        controller.move_robot(0.0, 0.0, -0.005)  # Move down
        time.sleep(0.1)
    
    # Close gripper
    controller.control_gripper(0.8)
    time.sleep(0.5)
    
    # Lift cube
    for _ in range(10):
        controller.move_robot(0.0, 0.0, 0.01)  # Move up
        time.sleep(0.1)
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Key Features

### Success Detection
- The success termination condition still works: lifting the cube above 8cm triggers success
- Episodes are automatically saved when success is achieved
- Only successful demonstrations are exported to the dataset

### Action Recording
- **Before**: ROS control resulted in zero actions being recorded
- **After**: ROS commands are captured and recorded as proper action data
- Actions are recorded as 7D vectors: `[dx, dy, dz, drx, dry, drz, gripper_vel]`

### Data Format
The recorded HDF5 file will now contain:
- **Actions**: Real movement commands from ROS (not zeros)
- **Observations**: Same robot state observations as before  
- **Success**: Proper success detection when cube is lifted

## Troubleshooting

### ROS2 Not Available
If you see "ROS device interface not available", install ROS2:
```bash
# Install ROS2 packages
sudo apt install python3-rclpy python3-geometry-msgs python3-std-msgs
```

### No Commands Received
1. Check if ROS topics are being published:
   ```bash
   ros2 topic list
   ros2 topic echo /robot/cmd_pose
   ```

2. Verify the ROS device is running:
   ```bash
   ros2 node list  # Should show isaac_lab_recorder node
   ```

### Zero Actions Still Recorded
This happens when no ROS commands are being published. The device has a 100ms timeout - if no commands are received within this time, it returns zeros.

## Testing

Run the test script to verify ROS integration:
```bash
python test_ros_integration.py
```

This will create a test publisher and verify that commands are received correctly.

## Summary

With this ROS integration, you can now:
1. **Record meaningful demos** using your existing ROS control system
2. **Maintain success detection** - episodes end when the cube is successfully lifted
3. **Export proper action data** for training imitation learning models
4. **Use familiar ROS workflows** while leveraging Isaac Lab's recording capabilities

The key improvement is that your recorded demonstrations will now contain the actual control commands from ROS instead of zeros, making them suitable for imitation learning and policy training. 