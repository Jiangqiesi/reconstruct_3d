"""
Realman RM75 Robot Interface

Provides an interface for the 3D reconstruction pipeline to communicate 
with Realman RM75 robot arm. Based on the RM_controller class from realman/RealMan.py.

The interface provides:
- Robot connection/disconnection
- Current pose retrieval (as 4x4 transformation matrix)
- Movement commands (Cartesian and joint space)
- Hand-eye calibration integration

Usage:
    robot = RealmanRobot("192.168.1.18")
    robot.connect()
    pose = robot.get_current_pose()  # Returns 4x4 matrix
    robot.move_to_pose(target_pose)
    robot.disconnect()
"""

import numpy as np
from typing import Optional, Tuple, List
from scipy.spatial.transform import Rotation
import time
import sys
from pathlib import Path

# Add realman directory to path for importing RM_controller
_realman_path = Path(__file__).parent.parent.parent / 'realman'
if str(_realman_path) not in sys.path:
    sys.path.insert(0, str(_realman_path))


class RealmanRobot:
    """
    Interface for Realman RM75 robot arm for 3D reconstruction.
    
    Wraps RM_controller and provides pose transformations needed
    for hand-eye calibration and trajectory execution.
    """
    
    def __init__(
        self,
        ip: str = "192.168.1.18",
        port: int = 8080,
        thread_mode: str = "triple"
    ):
        """
        Initialize the robot interface.
        
        Args:
            ip: Robot IP address
            port: Robot control port (default 8080 for Realman)
            thread_mode: Thread mode ("single" or "triple")
        """
        self.ip = ip
        self.port = port
        self.thread_mode = thread_mode
        
        self._controller = None
        self._is_connected = False
    
    def connect(self) -> bool:
        """
        Connect to the robot.
        
        Returns:
            success: Whether connection was successful
        """
        try:
            from RealMan import RM_controller
            from Robotic_Arm.rm_robot_interface import rm_thread_mode_e
            
            # Select thread mode
            if self.thread_mode == "triple":
                mode = rm_thread_mode_e.RM_TRIPLE_MODE_E
            else:
                mode = None
            
            # Initialize controller
            self._controller = RM_controller(self.ip, mode)
            
            # Test connection by getting state
            state = self._controller.get_state()
            if state is not None:
                self._is_connected = True
                print(f"Connected to Realman RM75 at {self.ip}")
                print(f"Current joint angles: {state}")
                return True
            else:
                raise ConnectionError("Failed to get robot state")
            
        except ImportError as e:
            print(f"Failed to import Realman SDK: {e}")
            print("Make sure RM_controller and Robotic_Arm are in the path")
            self._is_connected = False
            return False
            
        except Exception as e:
            print(f"Failed to connect to robot: {e}")
            self._is_connected = False
            return False
    
    def disconnect(self):
        """Disconnect from the robot."""
        if self._controller is not None:
            try:
                del self._controller
                self._controller = None
            except:
                pass
        
        self._is_connected = False
        print("Disconnected from robot")
    
    def get_current_pose(self) -> np.ndarray:
        """
        Get the current end-effector (flange) pose as 4x4 transformation matrix.
        
        Returns:
            T_base_flange: 4x4 homogeneous transformation matrix
        """
        if not self._is_connected or self._controller is None:
            raise RuntimeError("Robot not connected")
        
        try:
            # Get full arm state from controller
            result = self._controller.arm_controller.rm_get_current_arm_state()
            
            if result[0] != 0:
                raise RuntimeError(f"Failed to get arm state, error code: {result[0]}")
            
            arm_state = result[1]
            
            # Extract pose from arm state
            # Realman returns pose as dict with 'pose' key containing [x, y, z, rx, ry, rz]
            # Position in mm, rotation in radians (or degrees depending on SDK version)
            if 'pose' in arm_state:
                pose_data = arm_state['pose']
            else:
                # Fallback: calculate from joint angles using FK (if pose not available)
                raise RuntimeError("Pose data not available in arm state")
            
            # Position: convert mm to meters
            position = np.array([
                pose_data[0] / 1000.0,  # x
                pose_data[1] / 1000.0,  # y
                pose_data[2] / 1000.0   # z
            ])
            
            # Rotation: Realman uses rx, ry, rz (Euler angles in radians)
            # Using XYZ Euler convention
            euler_angles = np.array([
                pose_data[3],  # rx (rad)
                pose_data[4],  # ry (rad)
                pose_data[5]   # rz (rad)
            ])
            
            # Convert to transformation matrix
            T = self._pose_to_matrix(position, euler_angles)
            return T
            
        except Exception as e:
            print(f"Error getting robot pose: {e}")
            raise
    
    def get_joint_angles(self) -> np.ndarray:
        """
        Get current joint angles.
        
        Returns:
            angles: Joint angles in radians (7 joints for RM75)
        """
        if not self._is_connected or self._controller is None:
            raise RuntimeError("Robot not connected")
        
        try:
            # Get state returns angles in degrees
            joint_degrees = self._controller.get_state()
            return np.radians(joint_degrees)
            
        except Exception as e:
            print(f"Error getting joint angles: {e}")
            raise
    
    def move_to_pose(
        self,
        T_base_flange: np.ndarray,
        speed: float = 0.2,
        wait: bool = True
    ) -> bool:
        """
        Move the robot to a target pose using Cartesian linear motion.
        
        Args:
            T_base_flange: Target pose as 4x4 transformation matrix
            speed: Movement speed (0-1), maps to 0-100% of max speed
            wait: Whether to wait for motion to complete (blocking)
            
        Returns:
            success: Whether motion was successful
        """
        if not self._is_connected or self._controller is None:
            raise RuntimeError("Robot not connected")
        
        try:
            # Convert matrix to position + Euler angles
            position, euler_angles = self._matrix_to_pose(T_base_flange)
            
            # Convert to robot units: position in mm, angles in radians
            pose = [
                position[0] * 1000.0,  # x (mm)
                position[1] * 1000.0,  # y (mm)
                position[2] * 1000.0,  # z (mm)
                euler_angles[0],       # rx (rad)
                euler_angles[1],       # ry (rad)
                euler_angles[2]        # rz (rad)
            ]
            
            # Use movel for Cartesian linear motion
            # rm_movej_p: pose, speed(%), acceleration(%), 0, block(1=wait)
            speed_percent = int(speed * 100)
            speed_percent = max(1, min(100, speed_percent))
            
            result = self._controller.arm_controller.rm_movej_p(
                pose, 
                speed_percent, 
                0,  # acceleration
                0,  # blend radius
                1 if wait else 0  # blocking
            )
            
            return result == 0
            
        except Exception as e:
            print(f"Error moving robot: {e}")
            return False
    
    def move_to_joint_angles(
        self,
        angles: np.ndarray,
        speed: float = 0.2,
        wait: bool = True
    ) -> bool:
        """
        Move robot using joint angles.
        
        Args:
            angles: Target joint angles in radians (7 joints)
            speed: Movement speed (0-1)
            wait: Whether to wait for motion to complete
            
        Returns:
            success: Whether motion was successful
        """
        if not self._is_connected or self._controller is None:
            raise RuntimeError("Robot not connected")
        
        try:
            # Convert radians to degrees
            angles_deg = np.degrees(angles).tolist()
            
            speed_percent = int(speed * 100)
            speed_percent = max(1, min(100, speed_percent))
            
            result = self._controller.arm_controller.rm_movej(
                angles_deg,
                speed_percent,
                0,  # acceleration
                0,  # blend radius
                1 if wait else 0  # blocking
            )
            
            return result == 0
            
        except Exception as e:
            print(f"Error moving robot: {e}")
            return False
    
    def move_to_pose_list(
        self,
        pose_list: List[float],
        speed: float = 0.2,
        wait: bool = True
    ) -> bool:
        """
        Move robot to pose specified as [x, y, z, rx, ry, rz].
        
        This is a convenience method for direct interface with the robot.
        
        Args:
            pose_list: [x, y, z, rx, ry, rz] - position in meters, rotation in radians
            speed: Movement speed (0-1)
            wait: Whether to wait for motion to complete
            
        Returns:
            success: Whether motion was successful
        """
        if len(pose_list) != 6:
            raise ValueError("pose_list must have 6 elements: [x, y, z, rx, ry, rz]")
        
        # Convert to mm for position
        pose_mm = [
            pose_list[0] * 1000.0,  # x
            pose_list[1] * 1000.0,  # y
            pose_list[2] * 1000.0,  # z
            pose_list[3],          # rx (rad)
            pose_list[4],          # ry (rad)
            pose_list[5]           # rz (rad)
        ]
        
        speed_percent = int(speed * 100)
        speed_percent = max(1, min(100, speed_percent))
        
        result = self._controller.arm_controller.rm_movej_p(
            pose_mm,
            speed_percent,
            0, 0,
            1 if wait else 0
        )
        
        return result == 0
    
    def _pose_to_matrix(
        self,
        position: np.ndarray,
        euler_angles: np.ndarray
    ) -> np.ndarray:
        """
        Convert position and Euler angles to 4x4 transformation matrix.
        
        Args:
            position: [x, y, z] in meters
            euler_angles: [rx, ry, rz] in radians (XYZ Euler)
            
        Returns:
            T: 4x4 homogeneous transformation matrix
        """
        # Create rotation matrix from Euler angles (XYZ convention)
        R = Rotation.from_euler('xyz', euler_angles).as_matrix()
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = position
        
        return T
    
    def _matrix_to_pose(
        self,
        T: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert 4x4 transformation matrix to position and Euler angles.
        
        Args:
            T: 4x4 homogeneous transformation matrix
            
        Returns:
            position: [x, y, z] in meters
            euler_angles: [rx, ry, rz] in radians
        """
        position = T[:3, 3].copy()
        R = Rotation.from_matrix(T[:3, :3])
        euler_angles = R.as_euler('xyz')
        
        return position, euler_angles
    
    @property
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self._is_connected
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False


class MockRealmanRobot:
    """
    Mock robot for testing without hardware.
    
    Simulates robot behavior for development and testing.
    """
    
    def __init__(self, ip: str = "192.168.1.18", port: int = 8080, **kwargs):
        self.ip = ip
        self.port = port
        self._is_connected = False
        
        # Default pose: about 0.5m in front of robot base
        self._current_pose = np.eye(4)
        self._current_pose[:3, 3] = [0.5, 0, 0.4]
        
        self._joint_angles = np.zeros(7)
    
    def connect(self) -> bool:
        self._is_connected = True
        print(f"Mock robot connected at {self.ip}:{self.port}")
        return True
    
    def disconnect(self):
        self._is_connected = False
        print("Mock robot disconnected")
    
    def get_current_pose(self) -> np.ndarray:
        if not self._is_connected:
            raise RuntimeError("Robot not connected")
        return self._current_pose.copy()
    
    def get_joint_angles(self) -> np.ndarray:
        if not self._is_connected:
            raise RuntimeError("Robot not connected")
        return self._joint_angles.copy()
    
    def move_to_pose(
        self,
        T_base_flange: np.ndarray,
        speed: float = 0.2,
        wait: bool = True
    ) -> bool:
        if not self._is_connected:
            raise RuntimeError("Robot not connected")
        
        self._current_pose = T_base_flange.copy()
        
        if wait:
            # Simulate movement time based on distance
            time.sleep(0.3)
        
        print(f"Mock robot moved to position: {T_base_flange[:3, 3]}")
        return True
    
    def move_to_joint_angles(
        self,
        angles: np.ndarray,
        speed: float = 0.2,
        wait: bool = True
    ) -> bool:
        if not self._is_connected:
            raise RuntimeError("Robot not connected")
        
        self._joint_angles = angles.copy()
        
        if wait:
            time.sleep(0.3)
        
        print(f"Mock robot moved to joint angles: {np.degrees(angles)}")
        return True
    
    def move_to_pose_list(
        self,
        pose_list: List[float],
        speed: float = 0.2,
        wait: bool = True
    ) -> bool:
        if not self._is_connected:
            raise RuntimeError("Robot not connected")
        
        print(f"Mock robot moved to pose: {pose_list}")
        return True
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False
