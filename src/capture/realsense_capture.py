"""
Intel RealSense Camera Module

Provides an interface for capturing RGB-D data from Intel RealSense cameras.
Supports depth-color alignment and intrinsic retrieval.

Usage:
    camera = RealSenseCamera()
    camera.start()
    color, depth = camera.capture()
    camera.stop()
"""

import numpy as np
from typing import Tuple, Optional
import time


class RealSenseCamera:
    """
    Intel RealSense RGB-D camera interface.
    
    Features:
        - RGB and depth capture
        - Depth-to-color alignment
        - Camera intrinsics retrieval
        - Configurable resolution and frame rate
    """
    
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        align_to_color: bool = True
    ):
        """
        Initialize the RealSense camera.
        
        Args:
            width: Image width
            height: Image height
            fps: Frame rate
            align_to_color: Whether to align depth to color frame
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.align_to_color = align_to_color
        
        self.pipeline = None
        self.config = None
        self.align = None
        self.depth_scale = None
        self.intrinsics = None
        
        self._is_started = False
    
    def start(self):
        """Start the camera stream."""
        try:
            import pyrealsense2 as rs
        except ImportError:
            raise RuntimeError(
                "pyrealsense2 not installed. "
                "Install with: pip install pyrealsense2"
            )
        
        # Create pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure streams
        self.config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
        )
        self.config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
        )
        
        # Start pipeline
        profile = self.pipeline.start(self.config)
        
        # Get depth scale
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
        # Get intrinsics
        color_stream = profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        self.intrinsics = {
            'width': intrinsics.width,
            'height': intrinsics.height,
            'fx': intrinsics.fx,
            'fy': intrinsics.fy,
            'cx': intrinsics.ppx,
            'cy': intrinsics.ppy
        }
        
        # Create alignment object
        if self.align_to_color:
            self.align = rs.align(rs.stream.color)
        
        # Allow camera to warm up
        for _ in range(30):
            self.pipeline.wait_for_frames()
        
        self._is_started = True
        print(f"RealSense camera started: {self.width}x{self.height}@{self.fps}fps")
        print(f"Depth scale: {self.depth_scale}")
    
    def stop(self):
        """Stop the camera stream."""
        if self.pipeline is not None:
            self.pipeline.stop()
            self._is_started = False
            print("RealSense camera stopped")
    
    def capture(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Capture a frame.
        
        Returns:
            color: RGB image (H, W, 3) as uint8
            depth: Depth image (H, W) as uint16 (in mm)
        """
        if not self._is_started:
            raise RuntimeError("Camera not started. Call start() first.")
        
        import pyrealsense2 as rs
        
        # Wait for frames
        frames = self.pipeline.wait_for_frames()
        
        # Align if needed
        if self.align_to_color and self.align is not None:
            frames = self.align.process(frames)
        
        # Get color and depth frames
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to capture frames")
        
        # Convert to numpy arrays
        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        
        return color, depth
    
    def capture_aligned_rgbd(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Capture aligned RGB-D data.
        
        Returns:
            color: RGB image (H, W, 3) as uint8
            depth: Depth image (H, W) as float32 (in meters)
            timestamp: Capture timestamp
        """
        color, depth_raw = self.capture()
        timestamp = time.time()
        
        # Convert depth to meters
        depth_meters = depth_raw.astype(np.float32) * self.depth_scale
        
        return color, depth_meters, timestamp
    
    def get_intrinsic_matrix(self) -> np.ndarray:
        """
        Get the camera intrinsic matrix (3x3).
        
        Returns:
            K: Camera intrinsic matrix
        """
        if self.intrinsics is None:
            raise RuntimeError("Camera not started. Call start() first.")
        
        K = np.array([
            [self.intrinsics['fx'], 0, self.intrinsics['cx']],
            [0, self.intrinsics['fy'], self.intrinsics['cy']],
            [0, 0, 1]
        ], dtype=np.float64)
        
        return K
    
    def get_open3d_intrinsic(self):
        """
        Get Open3D camera intrinsic object.
        
        Returns:
            intrinsic: open3d.camera.PinholeCameraIntrinsic
        """
        try:
            import open3d as o3d
        except ImportError:
            raise RuntimeError("Open3D not installed")
        
        if self.intrinsics is None:
            raise RuntimeError("Camera not started. Call start() first.")
        
        return o3d.camera.PinholeCameraIntrinsic(
            self.intrinsics['width'],
            self.intrinsics['height'],
            self.intrinsics['fx'],
            self.intrinsics['fy'],
            self.intrinsics['cx'],
            self.intrinsics['cy']
        )
    
    def get_depth_scale(self) -> float:
        """Get the depth scale (conversion factor to meters)."""
        return self.depth_scale if self.depth_scale else 0.001
    
    @property
    def is_started(self) -> bool:
        """Check if camera is running."""
        return self._is_started
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


class MockRealSenseCamera:
    """
    Mock camera for testing without hardware.
    
    Generates synthetic RGB-D data for development and testing.
    """
    
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.depth_scale = 0.001
        self.intrinsics = {
            'width': width,
            'height': height,
            'fx': 615.0,
            'fy': 615.0,
            'cx': width / 2,
            'cy': height / 2
        }
        self._is_started = False
        self._frame_count = 0
    
    def start(self):
        """Start mock camera."""
        self._is_started = True
        print("Mock RealSense camera started")
    
    def stop(self):
        """Stop mock camera."""
        self._is_started = False
        print("Mock RealSense camera stopped")
    
    def capture(self) -> Tuple[np.ndarray, np.ndarray]:
        """Capture synthetic frame."""
        if not self._is_started:
            raise RuntimeError("Camera not started")
        
        self._frame_count += 1
        
        # Generate synthetic color image (gradient)
        x = np.linspace(0, 255, self.width, dtype=np.uint8)
        y = np.linspace(0, 255, self.height, dtype=np.uint8)
        xx, yy = np.meshgrid(x, y)
        
        color = np.stack([
            xx, yy, 
            np.full_like(xx, (self._frame_count * 10) % 256)
        ], axis=-1).astype(np.uint8)
        
        # Generate synthetic depth (sphere)
        cx, cy = self.width // 2, self.height // 2
        xx, yy = np.meshgrid(
            np.arange(self.width) - cx,
            np.arange(self.height) - cy
        )
        r = np.sqrt(xx**2 + yy**2)
        depth = (1000 + 500 * np.clip(1 - r / 200, 0, 1)).astype(np.uint16)
        
        return color, depth
    
    def capture_aligned_rgbd(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Capture with timestamp."""
        color, depth = self.capture()
        depth_meters = depth.astype(np.float32) * self.depth_scale
        return color, depth_meters, time.time()
    
    def get_intrinsic_matrix(self) -> np.ndarray:
        """Get mock intrinsic matrix."""
        return np.array([
            [self.intrinsics['fx'], 0, self.intrinsics['cx']],
            [0, self.intrinsics['fy'], self.intrinsics['cy']],
            [0, 0, 1]
        ], dtype=np.float64)
    
    def get_open3d_intrinsic(self):
        """Get Open3D camera intrinsic."""
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
        return self.depth_scale
    
    @property
    def is_started(self) -> bool:
        return self._is_started
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
