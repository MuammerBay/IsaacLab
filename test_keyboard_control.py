#!/usr/bin/env python3
"""
Test script to verify keyboard controls for SO100 robot.
Run this to test that the keyboard inputs are being captured correctly.
"""

from isaaclab.app import AppLauncher
app_launcher = AppLauncher()
simulation_app = app_launcher.app

import numpy as np
from isaaclab.devices.keyboard.se3_keyboard_so_arm import Se3KeyboardSOArm

def main():
    print("=== SO100 Keyboard Control Test ===")
    print()
    print("Use these keys to control the robot:")
    print("Movement: W/S (x-axis), A/D (y-axis), Q/E (z-axis)")
    print("Rotation: Z/X (roll), T/G (pitch), C/V (yaw)")
    print("Gripper: F (close), R (open)")
    print("Reset: L")
    print()
    print("Press keys and see the commands below. Press Ctrl+C to exit.")
    print("-" * 50)
    
    # Create keyboard interface
    keyboard = Se3KeyboardSOArm(pos_sensitivity=0.05, rot_sensitivity=0.05)
    
    try:
        while True:
            # Get current command
            command = keyboard.advance()
            
            # Only print if there's some movement
            if np.any(np.abs(command) > 1e-6):
                pos = command[:3]
                rot = command[3:6] 
                gripper = command[6]
                
                print(f"Position: [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}] | "
                      f"Rotation: [{rot[0]:+.3f}, {rot[1]:+.3f}, {rot[2]:+.3f}] | "
                      f"Gripper: {gripper:+.3f}")
                
            # Small delay to prevent spam
            import time
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\nTest completed!")

if __name__ == "__main__":
    main() 