#!/usr/bin/env python3
"""
Data Capture Script

Run this script to capture RGB-D data along a planned trajectory.
The robot moves through the trajectory while capturing synchronized
images and poses for later reconstruction.

Usage:
    python run_capture.py --config config/camera_config.yaml --output data/captures/scan_001

Prerequisites:
    1. Complete hand-eye calibration
    2. Define object center and scanning parameters
"""

import argparse
import numpy as np
import cv2
import yaml
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.calibration import HandEyeCalibrator
from src.trajectory import TrajectoryPlanner
from src.capture import DataCollector, RealSenseCamera, MockRealSenseCamera
from src.robot import RealmanRobot, MockRealmanRobot


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Data Capture for 3D Reconstruction")
    parser.add_argument(
        '--config', type=str, default='config/camera_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Output directory for captured data'
    )
    parser.add_argument(
        '--calibration', type=str, default='data/calibration/hand_eye_calibration.npy',
        help='Path to hand-eye calibration file'
    )
    parser.add_argument(
        '--center', type=str, default=None,
        help='Object center x,y,z in meters (overrides config)'
    )
    parser.add_argument(
        '--radius', type=float, default=None,
        help='Scanning radius in meters (overrides config)'
    )
    parser.add_argument(
        '--num-views', type=int, default=None,
        help='Number of viewpoints (overrides config)'
    )
    parser.add_argument(
        '--pattern', type=str, default='spiral',
        choices=['spiral', 'layers', 'uniform'],
        help='Trajectory pattern'
    )
    parser.add_argument(
        '--preview', action='store_true',
        help='Preview trajectory without execution'
    )
    parser.add_argument(
        '--mock', action='store_true',
        help='Use mock camera and robot for testing'
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line args
    center = config['trajectory']['center']
    if args.center:
        center = list(map(float, args.center.split(',')))
    
    radius = args.radius or config['trajectory']['radius']
    num_views = args.num_views or config['trajectory']['num_views']
    elevation_range = config['trajectory']['elevation_range']
    
    # Load hand-eye calibration
    calibration_path = Path(args.calibration)
    if calibration_path.exists():
        T_flange_cam = np.load(str(calibration_path))
        print(f"Loaded calibration from {calibration_path}")
    else:
        print(f"Warning: Calibration not found at {calibration_path}")
        print("Using identity matrix (this will give incorrect results!)")
        T_flange_cam = np.eye(4)
    
    # Create trajectory planner
    planner = TrajectoryPlanner(center=center, radius=radius, T_flange_cam=T_flange_cam)
    
    # Generate trajectory
    print(f"\nGenerating {args.pattern} trajectory:")
    print(f"  Object center: {center}")
    print(f"  Radius: {radius} m")
    print(f"  Elevation range: {elevation_range} deg")
    print(f"  Number of views: {num_views}")
    
    camera_poses, flange_poses = planner.generate_spherical_trajectory(
        num_views=num_views,
        elevation_range=tuple(elevation_range),
        pattern=args.pattern,
        return_flange_poses=True
    )
    
    print(f"Generated {len(camera_poses)} waypoints")
    
    # Preview mode
    if args.preview:
        print("\nPreviewing trajectory (no execution)...")
        planner.visualize_trajectory(camera_poses, show=True)
        return
    
    # Initialize devices
    if args.mock:
        camera = MockRealSenseCamera()
        robot = MockRealmanRobot()
    else:
        camera = RealSenseCamera(
            width=config['camera']['width'],
            height=config['camera']['height']
        )
        robot = RealmanRobot(
            ip=config['robot']['ip'],
            port=config['robot']['port'],
            thread_mode=config['robot'].get('thread_mode', 'triple')
        )
    
    # Create output directory and data collector
    output_dir = Path(args.output)
    collector = DataCollector(str(output_dir), depth_scale=config['camera']['depth_scale'])
    
    # Start devices
    camera.start()
    robot.connect()
    
    try:
        # Save camera intrinsics
        intrinsics = camera.intrinsics if hasattr(camera, 'intrinsics') else {
            'fx': config['camera']['fx'],
            'fy': config['camera']['fy'],
            'cx': config['camera']['cx'],
            'cy': config['camera']['cy'],
            'width': config['camera']['width'],
            'height': config['camera']['height']
        }
        
        collector.save_intrinsics(
            fx=intrinsics.get('fx', config['camera']['fx']),
            fy=intrinsics.get('fy', config['camera']['fy']),
            cx=intrinsics.get('cx', config['camera']['cx']),
            cy=intrinsics.get('cy', config['camera']['cy']),
            width=intrinsics.get('width', config['camera']['width']),
            height=intrinsics.get('height', config['camera']['height']),
            depth_scale=config['camera']['depth_scale']
        )
        
        print("\n" + "="*60)
        print("Data Capture")
        print("="*60)
        print(f"Output directory: {output_dir}")
        print(f"Total waypoints: {len(flange_poses)}")
        print("="*60)
        print("\nPress Ctrl+C to abort capture")
        print("="*60 + "\n")
        
        cv2.namedWindow("Capture", cv2.WINDOW_AUTOSIZE)
        
        for i, (cam_pose, flange_pose) in enumerate(zip(camera_poses, flange_poses)):
            print(f"\nWaypoint {i+1}/{len(flange_poses)}")
            
            # Move robot to waypoint
            print(f"  Moving to position: {flange_pose[:3, 3]}")
            success = robot.move_to_pose(flange_pose, speed=0.2, wait=True)
            
            if not success:
                print(f"  Warning: Failed to reach waypoint {i+1}")
                continue
            
            # Wait for robot to settle
            time.sleep(0.3)
            
            # Capture RGB-D frame
            color, depth = camera.capture()
            timestamp = time.time()
            
            # Get actual robot pose (may differ slightly from target)
            try:
                actual_robot_pose = robot.get_current_pose()
                actual_cam_pose = actual_robot_pose @ T_flange_cam
            except:
                actual_cam_pose = cam_pose  # Use planned pose if reading fails
            
            # Save frame
            frame_idx = collector.save_frame(color, depth, actual_cam_pose, timestamp)
            print(f"  Captured frame {frame_idx}")
            
            # Show preview
            depth_normalized = (depth / config['camera']['depth_trunc'] * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            
            preview = np.hstack([color, depth_color])
            cv2.putText(preview, f"Frame {frame_idx+1}/{len(flange_poses)}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Capture", preview)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                print("\nCapture aborted by user")
                break
        
        cv2.destroyAllWindows()
        
        # Finalize and save
        collector.finalize()
        
        # Save trajectory for visualization
        planner.save_trajectory(str(output_dir / "trajectory.npy"), camera_poses)
        
        print(f"\nCapture complete!")
        print(f"  Frames captured: {collector.frame_count}")
        print(f"  Output directory: {output_dir}")
    
    except KeyboardInterrupt:
        print("\nCapture interrupted")
        collector.finalize()
    
    finally:
        camera.stop()
        robot.disconnect()


if __name__ == "__main__":
    main()
