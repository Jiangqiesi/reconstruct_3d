#!/usr/bin/env python3
"""
3D Reconstruction Script (TSDF Fusion - Route A)

Run this script to perform TSDF fusion on captured RGB-D data.
Uses camera poses from robot arm (via hand-eye calibration) for
accurate reconstruction without SLAM.

Usage:
    python run_reconstruction.py --data data/captures/scan_001 --output output/model.ply

Prerequisites:
    1. Complete data capture with run_capture.py
"""

import argparse
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.capture import DataLoader
from src.reconstruction import TSDFFusion, Visualizer


def main():
    parser = argparse.ArgumentParser(description="TSDF 3D Reconstruction")
    parser.add_argument(
        '--data', type=str, required=True,
        help='Path to captured data directory'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Output path for reconstruction result (.ply, .pcd, .obj)'
    )
    parser.add_argument(
        '--voxel-size', type=float, default=0.002,
        help='Voxel size in meters (default: 2mm)'
    )
    parser.add_argument(
        '--sdf-trunc', type=float, default=0.01,
        help='SDF truncation distance in meters (default: 10mm)'
    )
    parser.add_argument(
        '--depth-trunc', type=float, default=3.0,
        help='Maximum depth in meters (default: 3m)'
    )
    parser.add_argument(
        '--mesh', action='store_true',
        help='Output mesh instead of point cloud'
    )
    parser.add_argument(
        '--simplify', action='store_true',
        help='Simplify mesh (only with --mesh)'
    )
    parser.add_argument(
        '--target-triangles', type=int, default=100000,
        help='Target number of triangles when simplifying'
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Visualize result after reconstruction'
    )
    parser.add_argument(
        '--show-trajectory', action='store_true',
        help='Show camera trajectory in visualization'
    )
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("="*60)
    print("TSDF 3D Reconstruction")
    print("="*60)
    print(f"Data directory: {args.data}")
    print(f"Output: {args.output}")
    print(f"Voxel size: {args.voxel_size*1000:.1f} mm")
    print(f"SDF truncation: {args.sdf_trunc*1000:.1f} mm")
    print(f"Max depth: {args.depth_trunc:.1f} m")
    print("="*60 + "\n")
    
    loader = DataLoader(args.data)
    print(f"Loaded {len(loader)} frames\n")
    
    # Create TSDF fusion pipeline
    fusion = TSDFFusion(
        intrinsic=loader.get_open3d_intrinsic(),
        voxel_length=args.voxel_size,
        sdf_trunc=args.sdf_trunc,
        depth_trunc=args.depth_trunc
    )
    
    # Process all frames
    print("Integrating frames...")
    for i in range(len(loader)):
        try:
            color, depth, pose = loader.get_frame(i)
            fusion.integrate(color, depth, pose, depth_in_meters=True)
        except Exception as e:
            print(f"Warning: Failed to process frame {i}: {e}")
            continue
    
    print(f"\nIntegrated {fusion.frame_count} frames")
    
    # Extract and save result
    if args.mesh:
        print("\nExtracting mesh...")
        fusion.save_mesh(
            str(output_path),
            simplify=args.simplify,
            target_triangles=args.target_triangles
        )
    else:
        print("\nExtracting point cloud...")
        fusion.save_point_cloud(str(output_path), compute_normals=True)
    
    print(f"\nReconstruction saved to: {output_path}")
    
    # Visualization
    if args.visualize:
        import open3d as o3d
        
        print("\nOpening visualization...")
        vis = Visualizer()
        
        # Load the saved result
        if args.mesh:
            geometry = o3d.io.read_triangle_mesh(str(output_path))
            geometry.compute_vertex_normals()
        else:
            geometry = o3d.io.read_point_cloud(str(output_path))
        
        if args.show_trajectory:
            # Load trajectory if available
            trajectory_path = Path(args.data) / "trajectory.npy"
            if trajectory_path.exists():
                poses_array = np.load(str(trajectory_path))
                poses = [poses_array[i] for i in range(poses_array.shape[0])]
                vis.show_reconstruction_with_trajectory(geometry, poses)
            else:
                print("Warning: trajectory.npy not found, showing without trajectory")
                if args.mesh:
                    vis.show_mesh(geometry)
                else:
                    vis.show_point_cloud(geometry)
        else:
            if args.mesh:
                vis.show_mesh(geometry)
            else:
                vis.show_point_cloud(geometry)


if __name__ == "__main__":
    main()
