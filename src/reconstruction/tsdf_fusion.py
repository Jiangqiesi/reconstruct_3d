"""
TSDF Fusion Module (Route A)

Implements TSDF (Truncated Signed Distance Function) volume integration
for 3D reconstruction using Open3D. This is the standard approach when
using accurate robot arm poses for camera localization.

Usage:
    fusion = TSDFFusion(intrinsic)
    for color, depth, pose in data:
        fusion.integrate(color, depth, pose)
    pcd = fusion.extract_point_cloud()
"""

import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Optional, Tuple, Union, List


class TSDFFusion:
    """
    TSDF Volume Fusion for 3D Reconstruction.
    
    Uses Open3D's ScalableTSDFVolume for efficient large-scale reconstruction.
    Camera poses from robot arm (via hand-eye calibration) are used directly
    without SLAM-based pose estimation.
    
    Attributes:
        intrinsic: Camera intrinsic parameters
        voxel_length: Size of each voxel in meters
        sdf_trunc: Truncation distance for SDF
    """
    
    def __init__(
        self,
        intrinsic: o3d.camera.PinholeCameraIntrinsic,
        voxel_length: float = 0.002,
        sdf_trunc: float = 0.01,
        depth_trunc: float = 3.0,
        color_type: str = 'rgb'
    ):
        """
        Initialize the TSDF fusion pipeline.
        
        Args:
            intrinsic: Open3D camera intrinsic object
            voxel_length: Voxel size in meters (smaller = more detail, more memory)
            sdf_trunc: SDF truncation distance (typically 3-5x voxel_length)
            depth_trunc: Maximum depth to consider (in meters)
            color_type: Color integration type ('rgb' or 'gray')
        """
        self.intrinsic = intrinsic
        self.voxel_length = voxel_length
        self.sdf_trunc = sdf_trunc
        self.depth_trunc = depth_trunc
        
        # Select color type
        if color_type.lower() == 'rgb':
            self.color_type = o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        else:
            self.color_type = o3d.pipelines.integration.TSDFVolumeColorType.Gray32
        
        # Create TSDF volume
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_length,
            sdf_trunc=self.sdf_trunc,
            color_type=self.color_type
        )
        
        self.frame_count = 0
        self.poses = []
        
        print(f"TSDF Volume initialized:")
        print(f"  Voxel size: {voxel_length * 1000:.1f} mm")
        print(f"  Truncation: {sdf_trunc * 1000:.1f} mm")
        print(f"  Max depth: {depth_trunc:.1f} m")
    
    @classmethod
    def from_intrinsics(
        cls,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        **kwargs
    ) -> 'TSDFFusion':
        """
        Create TSDFFusion from intrinsic parameters.
        
        Args:
            fx, fy: Focal lengths
            cx, cy: Principal point
            width, height: Image dimensions
            **kwargs: Additional arguments for TSDFFusion
            
        Returns:
            fusion: TSDFFusion instance
        """
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx, fy, cx, cy
        )
        return cls(intrinsic, **kwargs)
    
    def integrate(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        pose: np.ndarray,
        depth_in_meters: bool = True
    ):
        """
        Integrate a single RGB-D frame into the TSDF volume.
        
        Args:
            color: RGB or BGR color image (H, W, 3)
            depth: Depth image (H, W) - float in meters or uint16 in mm
            pose: Camera pose 4x4 matrix (T_base_cam or T_world_cam)
            depth_in_meters: Whether depth is in meters (True) or mm (False)
        """
        # Convert color to Open3D image
        if color.dtype != np.uint8:
            color = (color * 255).astype(np.uint8)
        
        # Handle BGR to RGB conversion (OpenCV loads as BGR)
        if len(color.shape) == 3 and color.shape[2] == 3:
            color_rgb = color[:, :, ::-1].copy()  # BGR to RGB
        else:
            color_rgb = color
        
        color_o3d = o3d.geometry.Image(color_rgb)
        
        # Convert depth if needed
        if not depth_in_meters:
            depth = depth.astype(np.float32) * 0.001  # mm to meters
        elif depth.dtype != np.float32:
            depth = depth.astype(np.float32)
        
        depth_o3d = o3d.geometry.Image(depth)
        
        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=1.0,  # Already in meters
            depth_trunc=self.depth_trunc,
            convert_rgb_to_intensity=False
        )
        
        # Integrate into volume
        # Note: Open3D expects the INVERSE of the camera pose (extrinsic)
        # pose is T_world_cam, we need T_cam_world
        extrinsic = np.linalg.inv(pose)
        
        self.volume.integrate(rgbd, self.intrinsic, extrinsic)
        
        self.poses.append(pose.copy())
        self.frame_count += 1
        
        if self.frame_count % 10 == 0:
            print(f"Integrated frame {self.frame_count}")
    
    def integrate_from_files(
        self,
        color_path: str,
        depth_path: str,
        pose: np.ndarray,
        depth_scale: float = 0.001
    ):
        """
        Integrate from image files.
        
        Args:
            color_path: Path to color image
            depth_path: Path to depth image (16-bit PNG in mm)
            pose: Camera pose 4x4 matrix
            depth_scale: Depth conversion factor (default: mm to m)
        """
        import cv2
        
        color = cv2.imread(color_path, cv2.IMREAD_COLOR)
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        
        if color is None:
            raise FileNotFoundError(f"Color image not found: {color_path}")
        if depth_raw is None:
            raise FileNotFoundError(f"Depth image not found: {depth_path}")
        
        depth = depth_raw.astype(np.float32) * depth_scale
        
        self.integrate(color, depth, pose, depth_in_meters=True)
    
    def extract_point_cloud(self) -> o3d.geometry.PointCloud:
        """
        Extract point cloud from the TSDF volume.
        
        Returns:
            pcd: Open3D point cloud
        """
        print("Extracting point cloud...")
        pcd = self.volume.extract_point_cloud()
        print(f"Extracted {len(pcd.points)} points")
        return pcd
    
    def extract_mesh(self) -> o3d.geometry.TriangleMesh:
        """
        Extract triangle mesh from the TSDF volume.
        
        Uses marching cubes algorithm.
        
        Returns:
            mesh: Open3D triangle mesh
        """
        print("Extracting mesh...")
        mesh = self.volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        print(f"Extracted mesh with {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        return mesh
    
    def save_point_cloud(
        self,
        filepath: str,
        compute_normals: bool = True
    ):
        """
        Save extracted point cloud to file.
        
        Args:
            filepath: Output file path (.ply, .pcd, .xyz, etc.)
            compute_normals: Whether to estimate normals
        """
        pcd = self.extract_point_cloud()
        
        if compute_normals:
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.voxel_length * 5, max_nn=30
                )
            )
        
        o3d.io.write_point_cloud(filepath, pcd)
        print(f"Point cloud saved to {filepath}")
    
    def save_mesh(
        self,
        filepath: str,
        simplify: bool = False,
        target_triangles: int = 100000
    ):
        """
        Save extracted mesh to file.
        
        Args:
            filepath: Output file path (.ply, .obj, .stl, etc.)
            simplify: Whether to simplify the mesh
            target_triangles: Target number of triangles if simplifying
        """
        mesh = self.extract_mesh()
        
        if simplify and len(mesh.triangles) > target_triangles:
            print(f"Simplifying mesh from {len(mesh.triangles)} to {target_triangles} triangles...")
            mesh = mesh.simplify_quadric_decimation(target_triangles)
            mesh.compute_vertex_normals()
        
        o3d.io.write_triangle_mesh(filepath, mesh)
        print(f"Mesh saved to {filepath}")
    
    def reset(self):
        """Reset the TSDF volume."""
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_length,
            sdf_trunc=self.sdf_trunc,
            color_type=self.color_type
        )
        self.frame_count = 0
        self.poses = []
        print("TSDF volume reset")
    
    def get_trajectory(self) -> List:
        """
        Get the camera trajectory as Open3D geometry.
        
        Returns:
            frames: List of coordinate frame geometries
        """
        frames = []
        for pose in self.poses:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=self.voxel_length * 20
            )
            frame.transform(pose)
            frames.append(frame)
        return frames


def run_reconstruction(
    data_dir: str,
    output_path: str,
    voxel_length: float = 0.002,
    sdf_trunc: float = 0.01,
    depth_trunc: float = 3.0,
    output_mesh: bool = False
) -> Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]:
    """
    Run TSDF reconstruction on a captured dataset.
    
    This is a convenience function that handles the full pipeline.
    
    Args:
        data_dir: Path to captured data directory
        output_path: Path to save result (.ply)
        voxel_length: Voxel size in meters
        sdf_trunc: SDF truncation distance
        depth_trunc: Maximum depth
        output_mesh: If True, output mesh instead of point cloud
        
    Returns:
        result: Reconstructed point cloud or mesh
    """
    # Import here to avoid circular imports
    from ..capture.data_collector import DataLoader
    
    # Load dataset
    loader = DataLoader(data_dir)
    
    # Create fusion pipeline
    fusion = TSDFFusion(
        intrinsic=loader.get_open3d_intrinsic(),
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        depth_trunc=depth_trunc
    )
    
    # Process all frames
    print(f"Processing {len(loader)} frames...")
    
    for i in range(len(loader)):
        color, depth, pose = loader.get_frame(i)
        fusion.integrate(color, depth, pose, depth_in_meters=True)
    
    # Extract and save result
    if output_mesh:
        result = fusion.extract_mesh()
        o3d.io.write_triangle_mesh(output_path, result)
    else:
        result = fusion.extract_point_cloud()
        result.estimate_normals()
        o3d.io.write_point_cloud(output_path, result)
    
    print(f"Reconstruction saved to {output_path}")
    return result
