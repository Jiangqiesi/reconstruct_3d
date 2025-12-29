"""
Data Collection Module

Provides utilities for synchronized capture of RGB-D images and robot poses.
Saves data in a structured format for offline reconstruction.

Data Format:
    capture_dir/
    ├── color/
    │   ├── 000000.png
    │   ├── 000001.png
    │   └── ...
    ├── depth/
    │   ├── 000000.png  (16-bit PNG, values in mm)
    │   ├── 000001.png
    │   └── ...
    ├── poses.txt       (4x4 matrices, one per frame)
    └── intrinsics.json (camera intrinsics)
"""

import numpy as np
import cv2
import json
from pathlib import Path
from typing import Optional, Tuple, List
import time


class DataCollector:
    """
    Synchronized data collector for RGB-D + robot pose capture.
    
    Features:
        - Save RGB as PNG
        - Save depth as 16-bit PNG (mm values)
        - Save poses as 4x4 matrices
        - Save camera intrinsics
    """
    
    def __init__(
        self,
        output_dir: str,
        depth_scale: float = 0.001,
        create_dirs: bool = True
    ):
        """
        Initialize the data collector.
        
        Args:
            output_dir: Root directory for saving data
            depth_scale: Conversion factor from raw depth to meters
            create_dirs: Whether to create output directories
        """
        self.output_dir = Path(output_dir)
        self.depth_scale = depth_scale
        
        self.color_dir = self.output_dir / "color"
        self.depth_dir = self.output_dir / "depth"
        self.pose_file = self.output_dir / "poses.txt"
        self.intrinsics_file = self.output_dir / "intrinsics.json"
        self.metadata_file = self.output_dir / "metadata.json"
        
        if create_dirs:
            self.color_dir.mkdir(parents=True, exist_ok=True)
            self.depth_dir.mkdir(parents=True, exist_ok=True)
        
        self.frame_count = 0
        self.poses: List[np.ndarray] = []
        self.timestamps: List[float] = []
    
    def save_frame(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        pose: np.ndarray,
        timestamp: Optional[float] = None
    ) -> int:
        """
        Save a single frame.
        
        Args:
            color: BGR color image (H, W, 3) as uint8
            depth: Depth image (H, W) as uint16 (mm) or float32 (meters)
            pose: Camera pose 4x4 transformation matrix (T_base_cam)
            timestamp: Optional timestamp
            
        Returns:
            frame_index: Index of the saved frame
        """
        frame_idx = self.frame_count
        
        # Save color image
        color_path = self.color_dir / f"{frame_idx:06d}.png"
        cv2.imwrite(str(color_path), color)
        
        # Save depth image (convert to mm if in meters)
        if depth.dtype == np.float32 or depth.dtype == np.float64:
            # Convert meters to millimeters
            depth_mm = (depth / self.depth_scale).astype(np.uint16)
        else:
            depth_mm = depth.astype(np.uint16)
        
        depth_path = self.depth_dir / f"{frame_idx:06d}.png"
        cv2.imwrite(str(depth_path), depth_mm)
        
        # Store pose and timestamp
        self.poses.append(pose.copy())
        self.timestamps.append(timestamp if timestamp else time.time())
        
        self.frame_count += 1
        
        return frame_idx
    
    def save_intrinsics(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        depth_scale: Optional[float] = None
    ):
        """
        Save camera intrinsic parameters.
        
        Args:
            fx, fy: Focal lengths
            cx, cy: Principal point
            width, height: Image dimensions
            depth_scale: Depth conversion factor
        """
        intrinsics = {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "width": width,
            "height": height,
            "depth_scale": depth_scale if depth_scale else self.depth_scale
        }
        
        with open(self.intrinsics_file, 'w') as f:
            json.dump(intrinsics, f, indent=2)
        
        print(f"Saved intrinsics to {self.intrinsics_file}")
    
    def save_poses(self):
        """Save all collected poses to file."""
        with open(self.pose_file, 'w') as f:
            for i, pose in enumerate(self.poses):
                # Write frame index
                f.write(f"# Frame {i}, timestamp {self.timestamps[i]}\n")
                # Write 4x4 matrix
                for row in pose:
                    f.write(" ".join(f"{v:.8f}" for v in row) + "\n")
                f.write("\n")
        
        print(f"Saved {len(self.poses)} poses to {self.pose_file}")
    
    def save_metadata(self, **kwargs):
        """
        Save metadata about the capture session.
        
        Args:
            **kwargs: Additional metadata to save
        """
        metadata = {
            "num_frames": self.frame_count,
            "depth_scale": self.depth_scale,
            "capture_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            **kwargs
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def finalize(self):
        """Finalize the capture session and save all data."""
        self.save_poses()
        self.save_metadata()
        print(f"Capture session finalized: {self.frame_count} frames saved to {self.output_dir}")


class DataLoader:
    """
    Data loader for reconstruction.
    
    Loads RGB-D images and poses from a captured dataset.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to the capture directory
        """
        self.data_dir = Path(data_dir)
        self.color_dir = self.data_dir / "color"
        self.depth_dir = self.data_dir / "depth"
        self.pose_file = self.data_dir / "poses.txt"
        self.intrinsics_file = self.data_dir / "intrinsics.json"
        
        # Load intrinsics
        self.intrinsics = self._load_intrinsics()
        
        # Load poses
        self.poses = self._load_poses()
        
        # Get frame count
        self.num_frames = len(list(self.color_dir.glob("*.png")))
        
        print(f"Loaded dataset: {self.num_frames} frames")
    
    def _load_intrinsics(self) -> dict:
        """Load camera intrinsics from file."""
        if not self.intrinsics_file.exists():
            print("Warning: intrinsics.json not found, using defaults")
            return {
                "fx": 615.0, "fy": 615.0,
                "cx": 320.0, "cy": 240.0,
                "width": 640, "height": 480,
                "depth_scale": 0.001
            }
        
        with open(self.intrinsics_file, 'r') as f:
            return json.load(f)
    
    def _load_poses(self) -> List[np.ndarray]:
        """Load poses from file."""
        if not self.pose_file.exists():
            print("Warning: poses.txt not found")
            return []
        
        poses = []
        current_matrix = []
        
        with open(self.pose_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    if len(current_matrix) == 4:
                        poses.append(np.array(current_matrix))
                        current_matrix = []
                    continue
                
                values = [float(v) for v in line.split()]
                if len(values) == 4:
                    current_matrix.append(values)
        
        # Handle last matrix
        if len(current_matrix) == 4:
            poses.append(np.array(current_matrix))
        
        return poses
    
    def get_frame(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get a single frame.
        
        Args:
            index: Frame index
            
        Returns:
            color: BGR color image
            depth: Depth image in meters (float32)
            pose: 4x4 camera pose matrix
        """
        # Load color image
        color_path = self.color_dir / f"{index:06d}.png"
        color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        
        if color is None:
            raise FileNotFoundError(f"Color image not found: {color_path}")
        
        # Load depth image
        depth_path = self.depth_dir / f"{index:06d}.png"
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        
        if depth_raw is None:
            raise FileNotFoundError(f"Depth image not found: {depth_path}")
        
        # Convert to meters
        depth = depth_raw.astype(np.float32) * self.intrinsics['depth_scale']
        
        # Get pose
        if index < len(self.poses):
            pose = self.poses[index]
        else:
            print(f"Warning: No pose for frame {index}, using identity")
            pose = np.eye(4)
        
        return color, depth, pose
    
    def get_open3d_intrinsic(self):
        """Get Open3D camera intrinsic object."""
        import open3d as o3d
        
        return o3d.camera.PinholeCameraIntrinsic(
            self.intrinsics['width'],
            self.intrinsics['height'],
            self.intrinsics['fx'],
            self.intrinsics['fy'],
            self.intrinsics['cx'],
            self.intrinsics['cy']
        )
    
    def get_depth_scale(self) -> float:
        """Get depth scale."""
        return self.intrinsics.get('depth_scale', 0.001)
    
    def __len__(self) -> int:
        """Return number of frames."""
        return self.num_frames
    
    def __iter__(self):
        """Iterate over frames."""
        for i in range(self.num_frames):
            yield self.get_frame(i)
