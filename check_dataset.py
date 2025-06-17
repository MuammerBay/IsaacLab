#!/usr/bin/env python3

import h5py
import numpy as np

def analyze_dataset():
    file_path = '/home/lycheeai/mnt_data/IsaacLab/datasets/dataset.hdf5'
    
    with h5py.File(file_path, 'r') as f:
        print("=== DATASET ANALYSIS ===")
        print(f"File: {file_path}")
        print(f"Top-level keys: {list(f.keys())}")
        print()
        
        # Examine data group
        data_group = f['data']
        print(f"Data group contains: {list(data_group.keys())}")
        print()
        
        # Examine demo_0
        demo = data_group['demo_0']
        print("Demo_0 structure:")
        
        for key in demo.keys():
            item = demo[key]
            if hasattr(item, 'shape'):
                print(f"  {key}: shape={item.shape}, dtype={item.dtype}")
                
                # Analyze actions specifically
                if 'action' in key:
                    data = item[()]
                    print(f"    Action range: [{np.min(data):.4f}, {np.max(data):.4f}]")
                    print(f"    Non-zero actions: {np.count_nonzero(np.abs(data) > 1e-6)} / {data.size}")
                    print(f"    Action shape per step: {data.shape[1:] if len(data.shape) > 1 else 'scalar'}")
                    print(f"    First 3 actions:")
                    for i in range(min(3, len(data))):
                        print(f"      Step {i}: {data[i]}")
                    
                # Analyze observations
                elif 'obs' in key:
                    if hasattr(item, 'keys'):
                        print(f"    Observations (group): {list(item.keys())}")
                    else:
                        data = item[()]
                        print(f"    Obs range: [{np.min(data):.4f}, {np.max(data):.4f}]")
                        
            elif hasattr(item, 'keys'):
                print(f"  {key}: (group with {len(list(item.keys()))} items)")
                if 'obs' in key:
                    for obs_key in item.keys():
                        obs_data = item[obs_key]
                        if hasattr(obs_data, 'shape'):
                            print(f"    {obs_key}: shape={obs_data.shape}, dtype={obs_data.dtype}")

if __name__ == "__main__":
    analyze_dataset() 