#!/usr/bin/env python3
"""
Script to inspect HDF5 dataset files and show what's recorded in demonstrations.
"""

import argparse
import h5py
import numpy as np
import os

def inspect_hdf5_dataset(file_path):
    """Inspect an HDF5 dataset file and print its contents structure."""
    
    if not os.path.exists(file_path):
        print(f"ERROR: Dataset file {file_path} does not exist.")
        return
    
    print(f"Inspecting dataset: {file_path}")
    print("=" * 60)
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Print basic file info
            print(f"File size: {os.path.getsize(file_path) / 1024:.1f} KB")
            print(f"HDF5 format version: {f.libver}")
            print()
            
            def print_structure(name, obj, level=0):
                indent = "  " * level
                if isinstance(obj, h5py.Group):
                    print(f"{indent}📁 {name}/ (Group)")
                    if hasattr(obj, 'attrs') and len(obj.attrs) > 0:
                        for attr_name, attr_value in obj.attrs.items():
                            print(f"{indent}  📋 {attr_name}: {attr_value}")
                elif isinstance(obj, h5py.Dataset):
                    shape_str = f"shape={obj.shape}" if obj.shape else "scalar"
                    dtype_str = f"dtype={obj.dtype}"
                    print(f"{indent}📄 {name} ({shape_str}, {dtype_str})")
                    
                    # Show attributes
                    if hasattr(obj, 'attrs') and len(obj.attrs) > 0:
                        for attr_name, attr_value in obj.attrs.items():
                            print(f"{indent}  📋 {attr_name}: {attr_value}")
                    
                    # Show sample data for small datasets
                    if obj.size <= 100:
                        try:
                            data = obj[()]
                            if isinstance(data, bytes):
                                data = data.decode('utf-8')
                            print(f"{indent}  📊 Data: {data}")
                        except:
                            print(f"{indent}  📊 Data: <unable to read>")
                    elif obj.size <= 10000:
                        try:
                            # Show first few elements for larger datasets
                            if len(obj.shape) == 1:
                                sample = obj[:min(5, obj.shape[0])]
                            elif len(obj.shape) == 2:
                                sample = obj[:min(3, obj.shape[0]), :min(5, obj.shape[1])]
                            else:
                                sample = obj[(slice(None, min(3, obj.shape[0])),) + tuple(slice(None, 3) for _ in obj.shape[1:])]
                            print(f"{indent}  📊 Sample: {sample}")
                        except:
                            print(f"{indent}  📊 Sample: <unable to read>")
            
            print("Dataset Structure:")
            print("-" * 40)
            f.visititems(print_structure)
            
            # Look for common episode patterns
            print("\nEpisode Analysis:")
            print("-" * 40)
            
            # Check for episodes
            episode_keys = [key for key in f.keys() if key.startswith('episode_')]
            if episode_keys:
                print(f"Found {len(episode_keys)} episodes: {episode_keys[:5]}{'...' if len(episode_keys) > 5 else ''}")
                
                # Analyze first episode in detail
                first_episode = episode_keys[0]
                print(f"\nDetailed analysis of {first_episode}:")
                episode_group = f[first_episode]
                
                for key in episode_group.keys():
                    dataset = episode_group[key]
                    if isinstance(dataset, h5py.Dataset):
                        print(f"  {key}: {dataset.shape} {dataset.dtype}")
                        
                        # Special analysis for actions
                        if 'action' in key.lower():
                            data = dataset[()]
                            print(f"    Action range: min={np.min(data):.4f}, max={np.max(data):.4f}")
                            print(f"    Action mean: {np.mean(data, axis=0)}")
                            if len(data) > 1:
                                print(f"    Action differences (consecutive steps):")
                                diffs = np.diff(data, axis=0)
                                print(f"      Mean diff: {np.mean(np.abs(diffs), axis=0)}")
                                print(f"      Max diff: {np.max(np.abs(diffs), axis=0)}")
                        
                        # Special analysis for observations
                        elif 'obs' in key.lower() or 'observation' in key.lower():
                            data = dataset[()]
                            print(f"    Obs range: min={np.min(data):.4f}, max={np.max(data):.4f}")
                            print(f"    Obs shape per step: {data.shape[1:] if len(data.shape) > 1 else 'scalar'}")
                            
                        # Show episode length
                        elif 'length' in key.lower() or data.shape[0] > 1:
                            print(f"    Episode length: {data.shape[0]} steps")
            else:
                print("No episodes found with pattern 'episode_*'")
                
                # Check for other data patterns
                top_level_keys = list(f.keys())
                print(f"Top-level keys: {top_level_keys}")
                
                # Look for any array data that might be demonstrations
                for key in top_level_keys:
                    item = f[key]
                    if isinstance(item, h5py.Dataset) and len(item.shape) >= 2:
                        print(f"  Potential trajectory data in '{key}': {item.shape} {item.dtype}")
            
            # Check for metadata
            print(f"\nFile metadata:")
            print("-" * 40)
            if hasattr(f, 'attrs') and len(f.attrs) > 0:
                for attr_name, attr_value in f.attrs.items():
                    print(f"  {attr_name}: {attr_value}")
            else:
                print("  No file-level attributes found")
                
    except Exception as e:
        print(f"ERROR reading HDF5 file: {e}")
        
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Inspect HDF5 dataset files")
    parser.add_argument("dataset_file", help="Path to the HDF5 dataset file")
    args = parser.parse_args()
    
    inspect_hdf5_dataset(args.dataset_file)


if __name__ == "__main__":
    main() 