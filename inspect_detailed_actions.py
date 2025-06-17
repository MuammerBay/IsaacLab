#!/usr/bin/env python3

"""Script to inspect the detailed actions in the recorded dataset."""

import h5py
import numpy as np

def inspect_actions(dataset_path):
    """Inspect the actions recorded in the dataset."""
    print(f"=== DETAILED ACTION ANALYSIS ===")
    print(f"Dataset: {dataset_path}")
    
    with h5py.File(dataset_path, 'r') as f:
        # Get actions from demo_0
        actions = f['data/demo_0/actions'][:]
        
        print(f"Total actions: {actions.shape}")
        print(f"Action dimensions: {actions.shape[1]}")
        
        # Find non-zero actions
        non_zero_mask = np.any(actions != 0, axis=1)
        non_zero_indices = np.where(non_zero_mask)[0]
        non_zero_actions = actions[non_zero_mask]
        
        print(f"\nNon-zero actions found: {len(non_zero_actions)}")
        print(f"Non-zero action indices: {non_zero_indices[:20]}...")  # First 20
        
        print(f"\nDetailed non-zero actions:")
        for i, (idx, action) in enumerate(zip(non_zero_indices, non_zero_actions)):
            print(f"  Step {idx:3d}: {action}")
            if i >= 20:  # Limit output
                print(f"  ... and {len(non_zero_actions) - 21} more")
                break
        
        print(f"\nAction statistics:")
        print(f"  Position commands (x,y,z): min={actions[:,:3].min():.6f}, max={actions[:,:3].max():.6f}")
        print(f"  Rotation commands (rx,ry,rz): min={actions[:,3:6].min():.6f}, max={actions[:,3:6].max():.6f}")
        print(f"  Gripper commands: min={actions[:,6].min():.6f}, max={actions[:,6].max():.6f}")
        
        # Check for your specific commands
        print(f"\nLooking for your ROS commands:")
        
        # Look for forward movement (x=0.02)
        forward_mask = np.abs(actions[:,0] - 0.02) < 1e-6
        if np.any(forward_mask):
            forward_indices = np.where(forward_mask)[0]
            print(f"  Forward movement (x=0.02) found at steps: {forward_indices}")
            
        # Look for downward movement (z=-0.01)  
        down_mask = np.abs(actions[:,2] - (-0.01)) < 1e-6
        if np.any(down_mask):
            down_indices = np.where(down_mask)[0]
            print(f"  Downward movement (z=-0.01) found at steps: {down_indices}")
            
        # Look for gripper closing (0.3)
        gripper_mask = np.abs(actions[:,6] - 0.3) < 1e-6
        if np.any(gripper_mask):
            gripper_indices = np.where(gripper_mask)[0]
            print(f"  Gripper closing (0.3) found at steps: {gripper_indices}")
            
        # Look for upward movement (z=0.02)
        up_mask = np.abs(actions[:,2] - 0.02) < 1e-6
        if np.any(up_mask):
            up_indices = np.where(up_mask)[0]
            print(f"  Upward movement (z=0.02) found at steps: {up_indices}")

if __name__ == "__main__":
    import sys
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "datasets/dataset.hdf5"
    inspect_actions(dataset_path) 