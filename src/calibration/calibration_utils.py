"""
Calibration Utility Functions

Provides helper functions for hand-eye calibration including:
- Chessboard detection
- Pose matrix conversions
- Calibration result validation
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List


def detect_chessboard(
    image: np.ndarray,
    pattern_size: Tuple[int, int] = (9, 6),
    square_size: float = 0.025
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Detect chessboard corners in an image.
    
    Args:
        image: Input image (BGR or grayscale)
        pattern_size: Number of inner corners (width, height)
        square_size: Size of each square in meters
        
    Returns:
        success: Whether corners were found
        corners: 2D image coordinates of corners (N, 1, 2)
        object_points: 3D coordinates of corners in chessboard frame (N, 3)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Find chessboard corners
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    success, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    
    if not success:
        return False, None, None
    
    # Refine corners with sub-pixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    
    # Generate 3D object points
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    return True, corners, objp


def estimate_board_pose(
    corners: np.ndarray,
    object_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the pose of the chessboard in camera frame.
    
    Args:
        corners: 2D image coordinates of corners
        object_points: 3D coordinates of corners
        camera_matrix: Camera intrinsic matrix (3x3)
        dist_coeffs: Distortion coefficients
        
    Returns:
        rvec: Rotation vector (3,)
        tvec: Translation vector (3,)
    """
    success, rvec, tvec = cv2.solvePnP(
        object_points, corners, camera_matrix, dist_coeffs
    )
    
    if not success:
        raise RuntimeError("Failed to estimate board pose")
    
    return rvec.flatten(), tvec.flatten()


def pose_to_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """
    Convert rotation vector and translation vector to 4x4 transformation matrix.
    
    Args:
        rvec: Rotation vector (3,) - Rodrigues format
        tvec: Translation vector (3,)
        
    Returns:
        T: 4x4 homogeneous transformation matrix
    """
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


def matrix_to_pose(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert 4x4 transformation matrix to rotation vector and translation vector.
    
    Args:
        T: 4x4 homogeneous transformation matrix
        
    Returns:
        rvec: Rotation vector (3,)
        tvec: Translation vector (3,)
    """
    R = T[:3, :3]
    rvec, _ = cv2.Rodrigues(R)
    tvec = T[:3, 3]
    return rvec.flatten(), tvec.flatten()


def euler_to_rotation_matrix(
    roll: float, pitch: float, yaw: float,
    order: str = 'xyz'
) -> np.ndarray:
    """
    Convert Euler angles to rotation matrix.
    
    Args:
        roll: Rotation around X axis in radians
        pitch: Rotation around Y axis in radians
        yaw: Rotation around Z axis in radians
        order: Rotation order ('xyz', 'zyx', etc.)
        
    Returns:
        R: 3x3 rotation matrix
    """
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Apply rotations in the specified order
    rotations = {'x': Rx, 'y': Ry, 'z': Rz}
    R = np.eye(3)
    for axis in order:
        R = R @ rotations[axis.lower()]
    
    return R


def rotation_matrix_to_euler(R: np.ndarray, order: str = 'xyz') -> Tuple[float, float, float]:
    """
    Convert rotation matrix to Euler angles.
    
    Args:
        R: 3x3 rotation matrix
        order: Rotation order (currently supports 'xyz')
        
    Returns:
        roll, pitch, yaw: Euler angles in radians
    """
    if order.lower() == 'xyz':
        # Extract Euler angles assuming XYZ order
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
        
        return roll, pitch, yaw
    else:
        raise ValueError(f"Unsupported rotation order: {order}")


def validate_calibration(
    T_flange_cam: np.ndarray,
    robot_poses: List[np.ndarray],
    camera_poses: List[np.ndarray],
    threshold: float = 0.01
) -> Tuple[bool, float]:
    """
    Validate hand-eye calibration result by checking reprojection consistency.
    
    Args:
        T_flange_cam: Calibration result (4x4 transformation matrix)
        robot_poses: List of robot flange poses (T_base_flange)
        camera_poses: List of camera poses relative to calibration target (T_cam_target)
        threshold: Maximum acceptable error in meters
        
    Returns:
        valid: Whether calibration is valid
        mean_error: Mean reprojection error
    """
    if len(robot_poses) < 2 or len(camera_poses) < 2:
        return False, float('inf')
    
    errors = []
    
    for i in range(len(robot_poses) - 1):
        for j in range(i + 1, len(robot_poses)):
            # Compute relative motion from robot
            T_robot_rel = np.linalg.inv(robot_poses[i]) @ robot_poses[j]
            
            # Compute relative motion from camera (through hand-eye transform)
            T_cam_rel = np.linalg.inv(camera_poses[i]) @ camera_poses[j]
            
            # These should be related by: T_robot_rel ≈ T_flange_cam @ T_cam_rel @ inv(T_flange_cam)
            T_predicted = T_flange_cam @ T_cam_rel @ np.linalg.inv(T_flange_cam)
            
            # Compute error
            T_error = np.linalg.inv(T_robot_rel) @ T_predicted
            translation_error = np.linalg.norm(T_error[:3, 3])
            errors.append(translation_error)
    
    mean_error = np.mean(errors)
    valid = mean_error < threshold
    
    return valid, mean_error


def draw_chessboard_corners(
    image: np.ndarray,
    pattern_size: Tuple[int, int],
    corners: np.ndarray,
    found: bool
) -> np.ndarray:
    """
    Draw detected chessboard corners on image.
    
    Args:
        image: Input image
        pattern_size: Chessboard pattern size
        corners: Detected corners
        found: Whether corners were successfully detected
        
    Returns:
        image: Image with drawn corners
    """
    result = image.copy()
    cv2.drawChessboardCorners(result, pattern_size, corners, found)
    return result
