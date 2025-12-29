"""
Hand-Eye Calibration Module

Implements AX=XB hand-eye calibration for Eye-in-Hand configuration.
Uses OpenCV's calibrateHandEye function with multiple solver options.

Usage:
    calibrator = HandEyeCalibrator(camera_matrix, dist_coeffs)
    calibrator.add_sample(robot_pose, image)
    T_flange_cam = calibrator.calibrate()
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json

from .calibration_utils import (
    detect_chessboard,
    estimate_board_pose,
    pose_to_matrix,
    matrix_to_pose,
    validate_calibration,
    draw_chessboard_corners
)


class HandEyeCalibrator:
    """
    Hand-Eye Calibration for Eye-in-Hand configuration.
    
    Computes the transformation T_flange_cam from robot flange to camera.
    Formula: T_base_cam = T_base_flange @ T_flange_cam
    
    Attributes:
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        pattern_size: Chessboard inner corner count (width, height)
        square_size: Chessboard square size in meters
    """
    
    # Available calibration methods
    METHODS = {
        'tsai': cv2.CALIB_HAND_EYE_TSAI,
        'park': cv2.CALIB_HAND_EYE_PARK,
        'horaud': cv2.CALIB_HAND_EYE_HORAUD,
        'andreff': cv2.CALIB_HAND_EYE_ANDREFF,
        'daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS
    }
    
    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        pattern_size: Tuple[int, int] = (9, 6),
        square_size: float = 0.025
    ):
        """
        Initialize the calibrator.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Distortion coefficients
            pattern_size: Chessboard inner corners (width, height)
            square_size: Square size in meters
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.pattern_size = pattern_size
        self.square_size = square_size
        
        # Storage for calibration samples
        self.robot_poses: List[np.ndarray] = []  # T_base_flange for each sample
        self.camera_poses: List[np.ndarray] = []  # T_cam_target for each sample
        self.images: List[np.ndarray] = []  # Optional: store images for debugging
        
        # Calibration result
        self.T_flange_cam: Optional[np.ndarray] = None
    
    def add_sample(
        self,
        robot_pose: np.ndarray,
        image: np.ndarray,
        save_image: bool = True
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Add a calibration sample from robot pose and camera image.
        
        Args:
            robot_pose: 4x4 transformation matrix T_base_flange
            image: Camera image containing the calibration target
            save_image: Whether to store the image
            
        Returns:
            success: Whether the sample was successfully added
            corners_image: Image with drawn corners (for visualization)
        """
        # Detect chessboard
        success, corners, object_points = detect_chessboard(
            image, self.pattern_size, self.square_size
        )
        
        if not success:
            print("Failed to detect chessboard in image")
            return False, None
        
        # Estimate board pose in camera frame
        try:
            rvec, tvec = estimate_board_pose(
                corners, object_points, self.camera_matrix, self.dist_coeffs
            )
            T_cam_target = pose_to_matrix(rvec, tvec)
        except RuntimeError as e:
            print(f"Failed to estimate board pose: {e}")
            return False, None
        
        # Store the sample
        self.robot_poses.append(robot_pose.copy())
        self.camera_poses.append(T_cam_target)
        
        if save_image:
            self.images.append(image.copy())
        
        # Draw corners for visualization
        corners_image = draw_chessboard_corners(
            image, self.pattern_size, corners, True
        )
        
        print(f"Sample {len(self.robot_poses)} added successfully")
        return True, corners_image
    
    def add_sample_from_pose(
        self,
        robot_pose: np.ndarray,
        camera_pose: np.ndarray
    ):
        """
        Add a calibration sample directly from poses (no image processing).
        
        Args:
            robot_pose: 4x4 transformation matrix T_base_flange
            camera_pose: 4x4 transformation matrix T_cam_target
        """
        self.robot_poses.append(robot_pose.copy())
        self.camera_poses.append(camera_pose.copy())
        print(f"Sample {len(self.robot_poses)} added successfully")
    
    def calibrate(self, method: str = 'tsai') -> Optional[np.ndarray]:
        """
        Perform hand-eye calibration.
        
        Args:
            method: Calibration method ('tsai', 'park', 'horaud', 'andreff', 'daniilidis')
            
        Returns:
            T_flange_cam: 4x4 transformation matrix from flange to camera
        """
        if len(self.robot_poses) < 3:
            print(f"Need at least 3 samples, got {len(self.robot_poses)}")
            return None
        
        if method.lower() not in self.METHODS:
            print(f"Unknown method: {method}. Available: {list(self.METHODS.keys())}")
            return None
        
        # Prepare input for OpenCV
        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []
        
        for robot_pose, camera_pose in zip(self.robot_poses, self.camera_poses):
            # Robot pose: T_base_flange -> need T_flange_base for OpenCV
            T_flange_base = np.linalg.inv(robot_pose)
            R_gripper2base.append(T_flange_base[:3, :3])
            t_gripper2base.append(T_flange_base[:3, 3].reshape(3, 1))
            
            # Camera pose: T_cam_target
            R_target2cam.append(camera_pose[:3, :3])
            t_target2cam.append(camera_pose[:3, 3].reshape(3, 1))
        
        # Perform calibration
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base,
            R_target2cam, t_target2cam,
            method=self.METHODS[method.lower()]
        )
        
        # Construct T_flange_cam (note: OpenCV returns T_cam_gripper, need inverse)
        T_cam_flange = np.eye(4)
        T_cam_flange[:3, :3] = R_cam2gripper
        T_cam_flange[:3, 3] = t_cam2gripper.flatten()
        
        self.T_flange_cam = np.linalg.inv(T_cam_flange)
        
        print(f"Calibration completed using {method} method")
        print(f"T_flange_cam:\n{self.T_flange_cam}")
        
        return self.T_flange_cam
    
    def calibrate_all_methods(self) -> Dict[str, np.ndarray]:
        """
        Run calibration with all available methods and return results.
        
        Returns:
            results: Dictionary mapping method name to T_flange_cam
        """
        results = {}
        for method in self.METHODS.keys():
            try:
                T = self.calibrate(method)
                if T is not None:
                    results[method] = T.copy()
            except Exception as e:
                print(f"Method {method} failed: {e}")
        
        return results
    
    def validate(self, threshold: float = 0.01) -> Tuple[bool, float]:
        """
        Validate the calibration result.
        
        Args:
            threshold: Maximum acceptable mean error in meters
            
        Returns:
            valid: Whether calibration is valid
            mean_error: Mean reprojection error
        """
        if self.T_flange_cam is None:
            print("No calibration result to validate. Run calibrate() first.")
            return False, float('inf')
        
        return validate_calibration(
            self.T_flange_cam,
            self.robot_poses,
            self.camera_poses,
            threshold
        )
    
    def save(self, filepath: str):
        """
        Save calibration result to file.
        
        Args:
            filepath: Path to save the calibration (.npy or .json)
        """
        if self.T_flange_cam is None:
            print("No calibration result to save")
            return
        
        path = Path(filepath)
        
        if path.suffix == '.npy':
            np.save(filepath, self.T_flange_cam)
        elif path.suffix == '.json':
            data = {
                'T_flange_cam': self.T_flange_cam.tolist(),
                'num_samples': len(self.robot_poses)
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            # Default to numpy format
            np.save(filepath + '.npy', self.T_flange_cam)
        
        print(f"Calibration saved to {filepath}")
    
    def load(self, filepath: str) -> np.ndarray:
        """
        Load calibration result from file.
        
        Args:
            filepath: Path to the calibration file
            
        Returns:
            T_flange_cam: Loaded transformation matrix
        """
        path = Path(filepath)
        
        if path.suffix == '.npy':
            self.T_flange_cam = np.load(filepath)
        elif path.suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.T_flange_cam = np.array(data['T_flange_cam'])
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        print(f"Calibration loaded from {filepath}")
        return self.T_flange_cam
    
    def get_camera_pose(self, robot_pose: np.ndarray) -> np.ndarray:
        """
        Compute camera pose in base frame given robot flange pose.
        
        Formula: T_base_cam = T_base_flange @ T_flange_cam
        
        Args:
            robot_pose: 4x4 transformation matrix T_base_flange
            
        Returns:
            T_base_cam: 4x4 transformation matrix for camera in base frame
        """
        if self.T_flange_cam is None:
            raise RuntimeError("No calibration result. Run calibrate() or load() first.")
        
        return robot_pose @ self.T_flange_cam
    
    def clear_samples(self):
        """Clear all collected samples."""
        self.robot_poses = []
        self.camera_poses = []
        self.images = []
        print("All samples cleared")
    
    @property
    def num_samples(self) -> int:
        """Return the number of collected samples."""
        return len(self.robot_poses)
