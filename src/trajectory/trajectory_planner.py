"""
Trajectory Planning Module

Generates spherical trajectories for scanning objects with a robot arm.
The camera is positioned on a sphere around the object and always points
toward the object center (LookAt pose).

Usage:
    planner = TrajectoryPlanner(center=[0.5, 0, 0.1], radius=0.4)
    poses = planner.generate_spherical_trajectory(num_views=30)
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation


class SphericalTrajectory:
    """
    Generate viewpoints on a sphere surface for object scanning.
    
    The camera is positioned on a sphere centered at the object and
    always looks toward the sphere center.
    
    Coordinate System:
        - X: Right
        - Y: Forward (toward object)
        - Z: Up
    """
    
    def __init__(
        self,
        center: np.ndarray,
        radius: float,
        up_vector: np.ndarray = None
    ):
        """
        Initialize the spherical trajectory generator.
        
        Args:
            center: Object center position [x, y, z] in robot base frame
            radius: Distance from object center to camera
            up_vector: World up direction (default: [0, 0, 1])
        """
        self.center = np.array(center, dtype=float)
        self.radius = radius
        self.up_vector = np.array(up_vector if up_vector is not None else [0, 0, 1], dtype=float)
        self.up_vector = self.up_vector / np.linalg.norm(self.up_vector)
    
    def spherical_to_cartesian(
        self,
        azimuth: float,
        elevation: float
    ) -> np.ndarray:
        """
        Convert spherical coordinates to Cartesian coordinates.
        
        Args:
            azimuth: Horizontal angle in radians (0 = front, positive = counterclockwise)
            elevation: Vertical angle in radians (0 = horizontal, positive = up)
            
        Returns:
            position: 3D position [x, y, z]
        """
        x = self.radius * np.cos(elevation) * np.cos(azimuth)
        y = self.radius * np.cos(elevation) * np.sin(azimuth)
        z = self.radius * np.sin(elevation)
        
        return self.center + np.array([x, y, z])
    
    def compute_lookat_rotation(
        self,
        camera_position: np.ndarray,
        target_position: np.ndarray
    ) -> np.ndarray:
        """
        Compute rotation matrix that makes the camera look at a target.
        
        The camera Z-axis (optical axis) points toward the target.
        The camera Y-axis points downward (image Y points down).
        
        Args:
            camera_position: Camera position in world frame
            target_position: Target position to look at
            
        Returns:
            R: 3x3 rotation matrix
        """
        # Z-axis: from camera toward target (optical axis)
        z_axis = target_position - camera_position
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        # X-axis: perpendicular to Z and world up (right direction)
        x_axis = np.cross(z_axis, self.up_vector)
        x_norm = np.linalg.norm(x_axis)
        
        # Handle gimbal lock when looking straight up/down
        if x_norm < 1e-6:
            # Use a different reference vector
            ref = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
            x_axis = np.cross(z_axis, ref)
        
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Y-axis: perpendicular to Z and X (down direction in image)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # Construct rotation matrix (columns are camera axes in world frame)
        R = np.column_stack([x_axis, y_axis, z_axis])
        
        return R
    
    def compute_camera_pose(
        self,
        azimuth: float,
        elevation: float
    ) -> np.ndarray:
        """
        Compute the full camera pose (4x4 transformation matrix) for given angles.
        
        Args:
            azimuth: Horizontal angle in radians
            elevation: Vertical angle in radians
            
        Returns:
            T_base_cam: 4x4 transformation matrix (camera pose in robot base frame)
        """
        # Get camera position
        position = self.spherical_to_cartesian(azimuth, elevation)
        
        # Compute rotation matrix (look at center)
        rotation = self.compute_lookat_rotation(position, self.center)
        
        # Construct homogeneous transformation
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = position
        
        return T
    
    def generate_trajectory(
        self,
        num_views: int,
        elevation_range: Tuple[float, float] = (20, 70),
        azimuth_range: Tuple[float, float] = (0, 360),
        pattern: str = 'spiral'
    ) -> List[np.ndarray]:
        """
        Generate a trajectory of camera poses.
        
        Args:
            num_views: Number of viewpoints
            elevation_range: (min, max) elevation in degrees
            azimuth_range: (min, max) azimuth in degrees
            pattern: 'spiral', 'layers', or 'uniform'
            
        Returns:
            poses: List of 4x4 transformation matrices
        """
        elev_min, elev_max = np.radians(elevation_range)
        az_min, az_max = np.radians(azimuth_range)
        
        poses = []
        
        if pattern == 'spiral':
            # Spiral pattern: smoothly vary both azimuth and elevation
            t = np.linspace(0, 1, num_views)
            for i in range(num_views):
                azimuth = az_min + (az_max - az_min) * t[i]
                # Oscillate elevation along the spiral
                elevation = elev_min + (elev_max - elev_min) * (0.5 + 0.5 * np.sin(t[i] * 2 * np.pi))
                poses.append(self.compute_camera_pose(azimuth, elevation))
        
        elif pattern == 'layers':
            # Layer pattern: multiple elevation rings
            num_layers = max(3, int(np.sqrt(num_views)))
            views_per_layer = num_views // num_layers
            
            elevations = np.linspace(elev_min, elev_max, num_layers)
            
            for elev in elevations:
                azimuths = np.linspace(az_min, az_max, views_per_layer, endpoint=False)
                for az in azimuths:
                    poses.append(self.compute_camera_pose(az, elev))
        
        elif pattern == 'uniform':
            # Uniform distribution on sphere segment using Fibonacci lattice
            golden_ratio = (1 + np.sqrt(5)) / 2
            
            for i in range(num_views):
                # Fibonacci lattice for uniform distribution
                theta = 2 * np.pi * i / golden_ratio
                z = elev_min + (elev_max - elev_min) * (i + 0.5) / num_views
                elevation = np.arcsin(np.clip(z / self.radius, -1, 1)) if self.radius > 0 else z
                
                # Scale to desired elevation range
                elevation = elev_min + (elev_max - elev_min) * i / (num_views - 1) if num_views > 1 else (elev_min + elev_max) / 2
                azimuth = az_min + (az_max - az_min) * ((i * golden_ratio) % 1)
                
                poses.append(self.compute_camera_pose(azimuth, elevation))
        
        else:
            raise ValueError(f"Unknown pattern: {pattern}. Use 'spiral', 'layers', or 'uniform'")
        
        return poses


class TrajectoryPlanner:
    """
    High-level trajectory planner for robot arm scanning applications.
    
    This class wraps SphericalTrajectory and provides additional utilities
    for integration with robot arm control.
    """
    
    def __init__(
        self,
        center: List[float],
        radius: float,
        T_flange_cam: Optional[np.ndarray] = None
    ):
        """
        Initialize the trajectory planner.
        
        Args:
            center: Object center [x, y, z] in robot base frame
            radius: Distance from object center
            T_flange_cam: Hand-eye calibration result (optional)
        """
        self.center = np.array(center)
        self.radius = radius
        self.T_flange_cam = T_flange_cam if T_flange_cam is not None else np.eye(4)
        
        self.trajectory_generator = SphericalTrajectory(center, radius)
    
    def set_hand_eye_calibration(self, T_flange_cam: np.ndarray):
        """Set the hand-eye calibration transformation."""
        self.T_flange_cam = T_flange_cam
    
    def camera_pose_to_flange_pose(self, T_base_cam: np.ndarray) -> np.ndarray:
        """
        Convert camera pose to robot flange pose.
        
        Formula: T_base_flange = T_base_cam @ inv(T_flange_cam)
        
        Args:
            T_base_cam: Camera pose in robot base frame
            
        Returns:
            T_base_flange: Flange pose for robot control
        """
        T_cam_flange = np.linalg.inv(self.T_flange_cam)
        return T_base_cam @ T_cam_flange
    
    def generate_spherical_trajectory(
        self,
        num_views: int = 30,
        elevation_range: Tuple[float, float] = (20, 70),
        azimuth_range: Tuple[float, float] = (0, 360),
        pattern: str = 'spiral',
        return_flange_poses: bool = True
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generate a spherical scanning trajectory.
        
        Args:
            num_views: Number of viewpoints
            elevation_range: (min, max) elevation in degrees
            azimuth_range: (min, max) azimuth in degrees  
            pattern: 'spiral', 'layers', or 'uniform'
            return_flange_poses: If True, also return flange poses for robot control
            
        Returns:
            camera_poses: List of camera poses (T_base_cam)
            flange_poses: List of flange poses (T_base_flange) if return_flange_poses=True
        """
        camera_poses = self.trajectory_generator.generate_trajectory(
            num_views=num_views,
            elevation_range=elevation_range,
            azimuth_range=azimuth_range,
            pattern=pattern
        )
        
        if return_flange_poses:
            flange_poses = [self.camera_pose_to_flange_pose(T) for T in camera_poses]
            return camera_poses, flange_poses
        
        return camera_poses, []
    
    def generate_orbit_trajectory(
        self,
        num_views: int = 36,
        elevation: float = 45,
        start_azimuth: float = 0
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generate a simple orbit trajectory at fixed elevation.
        
        Args:
            num_views: Number of viewpoints
            elevation: Fixed elevation angle in degrees
            start_azimuth: Starting azimuth angle in degrees
            
        Returns:
            camera_poses: List of camera poses (T_base_cam)
            flange_poses: List of flange poses (T_base_flange)
        """
        return self.generate_spherical_trajectory(
            num_views=num_views,
            elevation_range=(elevation, elevation),
            azimuth_range=(start_azimuth, start_azimuth + 360),
            pattern='layers'
        )
    
    def poses_to_joint_angles(
        self,
        flange_poses: List[np.ndarray],
        ik_solver
    ) -> List[Optional[np.ndarray]]:
        """
        Convert flange poses to joint angles using inverse kinematics.
        
        Args:
            flange_poses: List of flange poses
            ik_solver: Object with solve(pose) -> joint_angles method
            
        Returns:
            joint_angles: List of joint angle arrays (None if IK fails)
        """
        joint_angles = []
        for pose in flange_poses:
            try:
                angles = ik_solver.solve(pose)
                joint_angles.append(angles)
            except Exception as e:
                print(f"IK failed for pose: {e}")
                joint_angles.append(None)
        
        return joint_angles
    
    def save_trajectory(self, filepath: str, camera_poses: List[np.ndarray]):
        """
        Save trajectory to file.
        
        Args:
            filepath: Output file path (.npy)
            camera_poses: List of camera poses
        """
        poses_array = np.stack(camera_poses, axis=0)
        np.save(filepath, poses_array)
        print(f"Trajectory saved to {filepath}: {len(camera_poses)} poses")
    
    def load_trajectory(self, filepath: str) -> List[np.ndarray]:
        """
        Load trajectory from file.
        
        Args:
            filepath: Input file path (.npy)
            
        Returns:
            camera_poses: List of camera poses
        """
        poses_array = np.load(filepath)
        camera_poses = [poses_array[i] for i in range(poses_array.shape[0])]
        print(f"Trajectory loaded from {filepath}: {len(camera_poses)} poses")
        return camera_poses
    
    def visualize_trajectory(
        self,
        camera_poses: List[np.ndarray],
        show: bool = True
    ):
        """
        Visualize the trajectory using Open3D.
        
        Args:
            camera_poses: List of camera poses
            show: Whether to display the visualization
        """
        try:
            import open3d as o3d
        except ImportError:
            print("Open3D not installed. Cannot visualize trajectory.")
            return
        
        # Create coordinate frames for each camera pose
        geometries = []
        
        # Object center sphere
        center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        center_sphere.translate(self.center)
        center_sphere.paint_uniform_color([1, 0, 0])  # Red
        geometries.append(center_sphere)
        
        # Camera poses as coordinate frames
        for i, pose in enumerate(camera_poses):
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
            frame.transform(pose)
            geometries.append(frame)
        
        # Connect camera positions with lines
        positions = [pose[:3, 3] for pose in camera_poses]
        lines = [[i, i + 1] for i in range(len(positions) - 1)]
        lines.append([len(positions) - 1, 0])  # Close the loop
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(positions)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0]] * len(lines))  # Green
        geometries.append(line_set)
        
        # World coordinate frame
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        geometries.append(world_frame)
        
        if show:
            o3d.visualization.draw_geometries(geometries)
        
        return geometries
