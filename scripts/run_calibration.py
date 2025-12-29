#!/usr/bin/env python3
"""
Hand-Eye Calibration Script

Run this script to perform hand-eye calibration for Eye-in-Hand configuration.
The script guides you through collecting calibration samples and computes
the transformation matrix T_flange_cam.

Usage:
    python run_calibration.py --config config/camera_config.yaml

Prerequisites:
    1. Print a chessboard calibration pattern
    2. Fix the pattern on a stable surface
    3. Connect robot and camera
"""

import argparse
import numpy as np
import cv2
import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.calibration import HandEyeCalibrator, detect_chessboard
from src.capture import RealSenseCamera, MockRealSenseCamera
from src.robot import RealmanRobot, MockRealmanRobot


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Hand-Eye Calibration")
    parser.add_argument(
        '--config', type=str, default='config/camera_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output', type=str, default='data/calibration/hand_eye_calibration.npy',
        help='Output path for calibration result'
    )
    parser.add_argument(
        '--pattern-size', type=str, default='9,6',
        help='Chessboard pattern size (inner corners), e.g., 9,6'
    )
    parser.add_argument(
        '--square-size', type=float, default=0.025,
        help='Chessboard square size in meters'
    )
    parser.add_argument(
        '--num-samples', type=int, default=15,
        help='Number of calibration samples to collect'
    )
    parser.add_argument(
        '--mock', action='store_true',
        help='Use mock camera and robot for testing'
    )
    args = parser.parse_args()
    
    # Parse pattern size
    pattern_size = tuple(map(int, args.pattern_size.split(',')))
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load config
    config = load_config(args.config)
    
    # Initialize camera
    if args.mock:
        camera = MockRealSenseCamera()
    else:
        camera = RealSenseCamera(
            width=config['camera']['width'],
            height=config['camera']['height']
        )
    
    # Initialize robot
    if args.mock:
        robot = MockRealmanRobot()
    else:
        robot = RealmanRobot(
            ip=config['robot']['ip'],
            port=config['robot']['port'],
            thread_mode=config['robot'].get('thread_mode', 'triple')
        )
    
    # Start devices
    camera.start()
    robot.connect()
    
    try:
        # Get camera intrinsics
        K = camera.get_intrinsic_matrix()
        dist_coeffs = np.zeros(5)  # Assume no distortion or use calibrated values
        
        # Create calibrator
        calibrator = HandEyeCalibrator(
            camera_matrix=K,
            dist_coeffs=dist_coeffs,
            pattern_size=pattern_size,
            square_size=args.square_size
        )
        
        print("\n" + "="*60)
        print("Hand-Eye Calibration")
        print("="*60)
        print(f"Pattern size: {pattern_size}")
        print(f"Square size: {args.square_size*1000:.1f} mm")
        print(f"Target samples: {args.num_samples}")
        print("="*60)
        print("\nInstructions:")
        print("1. Move the robot to view the calibration pattern")
        print("2. Press SPACE to capture a sample")
        print("3. Press 'q' to quit and start calibration")
        print("4. Move to different viewpoints between captures")
        print("="*60 + "\n")
        
        cv2.namedWindow("Calibration", cv2.WINDOW_AUTOSIZE)
        
        while calibrator.num_samples < args.num_samples:
            # Capture image
            color, depth = camera.capture()
            
            # Detect chessboard (for preview)
            success, corners, _ = detect_chessboard(color, pattern_size, args.square_size)
            
            # Draw preview
            preview = color.copy()
            if success:
                cv2.drawChessboardCorners(preview, pattern_size, corners, success)
                cv2.putText(preview, "Pattern detected - Press SPACE to capture",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(preview, "Pattern not detected",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show sample count
            cv2.putText(preview, f"Samples: {calibrator.num_samples}/{args.num_samples}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Calibration", preview)
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord(' '):  # Space to capture
                if success:
                    # Get robot pose
                    robot_pose = robot.get_current_pose()
                    
                    # Add sample
                    added, corners_img = calibrator.add_sample(robot_pose, color)
                    
                    if added:
                        print(f"Sample {calibrator.num_samples} captured successfully")
                        # Save calibration image
                        img_path = output_path.parent / f"sample_{calibrator.num_samples:02d}.png"
                        cv2.imwrite(str(img_path), corners_img)
                    else:
                        print("Failed to add sample")
                else:
                    print("Cannot capture - pattern not detected")
            
            elif key == ord('q'):  # Quit
                break
        
        cv2.destroyAllWindows()
        
        # Perform calibration
        if calibrator.num_samples >= 3:
            print("\nPerforming calibration...")
            
            # Try all methods and show results
            results = calibrator.calibrate_all_methods()
            
            print("\nCalibration results:")
            for method, T in results.items():
                valid, error = calibrator.validate()
                print(f"\n{method}:")
                print(f"  Mean error: {error*1000:.2f} mm")
                print(f"  Valid: {valid}")
            
            # Use Tsai method as default
            if 'tsai' in results:
                T_flange_cam = results['tsai']
            else:
                T_flange_cam = list(results.values())[0]
            
            # Save result
            calibrator.save(str(output_path))
            
            # Also save as JSON for readability
            json_path = output_path.with_suffix('.json')
            calibrator.save(str(json_path))
            
            print(f"\nCalibration saved to: {output_path}")
            print("\nT_flange_cam:")
            print(T_flange_cam)
        
        else:
            print(f"\nNot enough samples ({calibrator.num_samples}). Need at least 3.")
    
    finally:
        camera.stop()
        robot.disconnect()


if __name__ == "__main__":
    main()
