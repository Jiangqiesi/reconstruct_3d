"""
Visualization Module

Provides utilities for visualizing 3D reconstruction results,
camera trajectories, and point clouds.

Usage:
    vis = Visualizer()
    vis.show_point_cloud(pcd)
    vis.show_reconstruction_with_trajectory(pcd, poses)
"""

import numpy as np
import open3d as o3d
from typing import List, Optional, Union


class Visualizer:
    """
    Visualization utilities for 3D reconstruction results.
    
    Features:
        - Point cloud visualization
        - Mesh visualization
        - Camera trajectory visualization
        - Interactive viewer
    """
    
    def __init__(self, window_name: str = "3D Reconstruction"):
        """
        Initialize the visualizer.
        
        Args:
            window_name: Name of the visualization window
        """
        self.window_name = window_name
    
    def show_point_cloud(
        self,
        pcd: o3d.geometry.PointCloud,
        coordinate_frame: bool = True,
        point_size: float = 1.0
    ):
        """
        Display a point cloud.
        
        Args:
            pcd: Point cloud to display
            coordinate_frame: Whether to show world coordinate frame
            point_size: Point rendering size
        """
        geometries = [pcd]
        
        if coordinate_frame:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            geometries.append(frame)
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name=self.window_name,
            point_show_normal=False
        )
    
    def show_mesh(
        self,
        mesh: o3d.geometry.TriangleMesh,
        coordinate_frame: bool = True,
        show_wireframe: bool = False
    ):
        """
        Display a triangle mesh.
        
        Args:
            mesh: Mesh to display
            coordinate_frame: Whether to show world coordinate frame
            show_wireframe: Whether to show wireframe overlay
        """
        geometries = [mesh]
        
        if coordinate_frame:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            geometries.append(frame)
        
        if show_wireframe:
            wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
            wireframe.paint_uniform_color([0, 0, 0])
            geometries.append(wireframe)
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name=self.window_name,
            mesh_show_back_face=True
        )
    
    def create_camera_frustum(
        self,
        pose: np.ndarray,
        intrinsic: Optional[o3d.camera.PinholeCameraIntrinsic] = None,
        scale: float = 0.05,
        color: List[float] = [0, 0, 1]
    ) -> o3d.geometry.LineSet:
        """
        Create a camera frustum visualization.
        
        Args:
            pose: Camera pose 4x4 matrix
            intrinsic: Camera intrinsic (for aspect ratio)
            scale: Size of the frustum
            color: RGB color [0-1]
            
        Returns:
            frustum: Line set representing the frustum
        """
        # Default aspect ratio
        if intrinsic:
            aspect = intrinsic.width / intrinsic.height
        else:
            aspect = 4 / 3
        
        # Frustum corners in camera frame
        h = scale
        w = h * aspect
        d = scale * 2
        
        points_cam = np.array([
            [0, 0, 0],          # Camera center
            [-w, -h, d],        # Bottom-left
            [w, -h, d],         # Bottom-right
            [w, h, d],          # Top-right
            [-w, h, d],         # Top-left
        ])
        
        # Transform to world frame
        points_world = (pose[:3, :3] @ points_cam.T).T + pose[:3, 3]
        
        # Define lines
        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],  # From center to corners
            [1, 2], [2, 3], [3, 4], [4, 1],  # Rectangle
        ]
        
        frustum = o3d.geometry.LineSet()
        frustum.points = o3d.utility.Vector3dVector(points_world)
        frustum.lines = o3d.utility.Vector2iVector(lines)
        frustum.colors = o3d.utility.Vector3dVector([color] * len(lines))
        
        return frustum
    
    def create_trajectory_visualization(
        self,
        poses: List[np.ndarray],
        show_frustums: bool = True,
        frustum_scale: float = 0.03,
        path_color: List[float] = [0, 1, 0]
    ) -> List[o3d.geometry.Geometry]:
        """
        Create visualization for camera trajectory.
        
        Args:
            poses: List of camera poses
            show_frustums: Whether to show camera frustums
            frustum_scale: Size of frustums
            path_color: RGB color for path
            
        Returns:
            geometries: List of geometries (path + frustums)
        """
        geometries = []
        
        # Camera positions
        positions = [pose[:3, 3] for pose in poses]
        
        # Path as line set
        if len(positions) > 1:
            lines = [[i, i + 1] for i in range(len(positions) - 1)]
            path = o3d.geometry.LineSet()
            path.points = o3d.utility.Vector3dVector(positions)
            path.lines = o3d.utility.Vector2iVector(lines)
            path.colors = o3d.utility.Vector3dVector([path_color] * len(lines))
            geometries.append(path)
        
        # Camera frustums
        if show_frustums:
            for i, pose in enumerate(poses):
                # Color gradient from blue to red
                t = i / max(len(poses) - 1, 1)
                color = [t, 0, 1 - t]
                
                frustum = self.create_camera_frustum(pose, scale=frustum_scale, color=color)
                geometries.append(frustum)
        
        return geometries
    
    def show_reconstruction_with_trajectory(
        self,
        geometry: Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh],
        poses: List[np.ndarray],
        show_frustums: bool = True
    ):
        """
        Display reconstruction result with camera trajectory.
        
        Args:
            geometry: Point cloud or mesh
            poses: List of camera poses
            show_frustums: Whether to show camera frustums
        """
        geometries = [geometry]
        
        # Add world frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        geometries.append(frame)
        
        # Add trajectory
        trajectory_geoms = self.create_trajectory_visualization(
            poses, show_frustums=show_frustums
        )
        geometries.extend(trajectory_geoms)
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name=self.window_name
        )
    
    def show_rgbd_frame(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        intrinsic: o3d.camera.PinholeCameraIntrinsic,
        pose: Optional[np.ndarray] = None,
        depth_scale: float = 1.0
    ):
        """
        Display a single RGB-D frame as a point cloud.
        
        Args:
            color: RGB image
            depth: Depth image in meters
            intrinsic: Camera intrinsic
            pose: Optional camera pose for transformation
            depth_scale: Depth scaling factor
        """
        # Create RGB-D image
        color_o3d = o3d.geometry.Image(color[:, :, ::-1].copy())  # BGR to RGB
        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=depth_scale,
            convert_rgb_to_intensity=False
        )
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        
        # Transform if pose provided
        if pose is not None:
            pcd.transform(pose)
        
        self.show_point_cloud(pcd)
    
    def compare_point_clouds(
        self,
        pcd1: o3d.geometry.PointCloud,
        pcd2: o3d.geometry.PointCloud,
        label1: str = "Cloud 1",
        label2: str = "Cloud 2"
    ):
        """
        Display two point clouds side by side.
        
        Args:
            pcd1, pcd2: Point clouds to compare
            label1, label2: Labels for the clouds
        """
        # Offset pcd2 for side-by-side view
        bbox1 = pcd1.get_axis_aligned_bounding_box()
        offset = bbox1.get_extent()[0] * 1.5
        
        pcd2_offset = o3d.geometry.PointCloud(pcd2)
        pcd2_offset.translate([offset, 0, 0])
        
        geometries = [pcd1, pcd2_offset]
        
        # Add coordinate frames
        frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        frame2.translate([offset, 0, 0])
        geometries.extend([frame1, frame2])
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"{label1} vs {label2}"
        )


def visualize_reconstruction(
    filepath: str,
    poses_file: Optional[str] = None
):
    """
    Convenience function to visualize a saved reconstruction.
    
    Args:
        filepath: Path to point cloud or mesh file
        poses_file: Optional path to poses file (.npy)
    """
    vis = Visualizer()
    
    # Load geometry
    if filepath.endswith(('.ply', '.pcd', '.xyz')):
        geometry = o3d.io.read_point_cloud(filepath)
        print(f"Loaded point cloud: {len(geometry.points)} points")
    elif filepath.endswith(('.obj', '.stl')):
        geometry = o3d.io.read_triangle_mesh(filepath)
        geometry.compute_vertex_normals()
        print(f"Loaded mesh: {len(geometry.vertices)} vertices")
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    # Load poses if provided
    if poses_file and poses_file != "":
        poses_array = np.load(poses_file)
        poses = [poses_array[i] for i in range(poses_array.shape[0])]
        vis.show_reconstruction_with_trajectory(geometry, poses)
    else:
        if isinstance(geometry, o3d.geometry.PointCloud):
            vis.show_point_cloud(geometry)
        else:
            vis.show_mesh(geometry)
