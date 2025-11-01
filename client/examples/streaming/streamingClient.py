#!/usr/bin/env python3
"""
senseSpace Streaming Client Example

Demonstrates receiving multiplexed RGB and depth video streams from senseSpace server.
Displays received frames in a visualization window.

Usage:
    python streamingClient.py --server 192.168.1.100
    python streamingClient.py --server 192.168.1.100 --stream-port 5000
"""

import sys
import argparse
import time
import numpy as np
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'libs'))

from senseSpaceLib.senseSpace.video_streaming import VideoReceiver, MultiCameraVideoReceiver
from senseSpaceLib.senseSpace.client import SenseSpaceClient
from senseSpaceLib.senseSpace.vizWidget import SkeletonGLWidget

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from OpenGL.GL import *
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class StreamingVisualizationWidget(SkeletonGLWidget):
    """
    Extends SkeletonGLWidget to display RGB and depth video streams.
    Supports both single camera and multi-camera (multiplexed) modes.
    """
    
    def __init__(self, num_cameras=1, parent=None):
        super().__init__(parent)
        
        self.num_cameras = num_cameras
        
        # Video stream data - lists for multi-camera support
        self.rgb_frames = [None] * num_cameras
        self.depth_frames = [None] * num_cameras
        
        # Stream receiver (can be VideoReceiver or MultiCameraVideoReceiver)
        self.video_receiver = None
        
        # Texture caching for efficient rendering
        self.rgb_textures = [None] * num_cameras
        self.depth_textures = [None] * num_cameras
        self.texture_sizes = {}  # Track texture dimensions for resize detection
        
        logger.info(f"StreamingVisualizationWidget initialized for {num_cameras} camera(s)")
    
    def set_video_receiver(self, receiver):
        """Set the video receiver instance (VideoReceiver or MultiCameraVideoReceiver)"""
        self.video_receiver = receiver
    
    def on_rgb_frame(self, frame: np.ndarray, camera_idx: int = 0):
        """
        Callback for RGB frame reception.
        
        Args:
            frame: RGB frame data
            camera_idx: Camera index (0 for single camera)
        """
        if camera_idx < self.num_cameras:
            self.rgb_frames[camera_idx] = frame
            # Trigger re-draw
            self.update()
    
    def on_depth_frame(self, frame: np.ndarray, camera_idx: int = 0):
        """
        Callback for depth frame reception.
        
        Args:
            frame: Depth frame data
            camera_idx: Camera index (0 for single camera)
        """
        if camera_idx < self.num_cameras:
            self.depth_frames[camera_idx] = frame
            self.update()
    
    def draw_custom(self, frame):
        """
        Override to draw video streams below status text.
        Called after skeleton/floor drawing but before 2D overlay.
        Handles both single and multi-camera layouts.
        """
        # Calculate grid layout for multiple cameras
        cameras_per_row = 2  # 2 columns for camera grid
        overlay_width = int(self.width() * 0.2)
        x_offset = 10
        y_offset = 80  # Position below status text (was 10, now 80 to clear text at y=58)
        spacing = 10
        
        for cam_idx in range(self.num_cameras):
            # Calculate grid position
            col = cam_idx % cameras_per_row
            row = cam_idx // cameras_per_row
            
            # RGB stream position
            rgb_x = x_offset + col * (overlay_width + spacing) * 2  # *2 for RGB+depth side-by-side
            rgb_y = y_offset + row * (int(overlay_width * 0.75) + spacing)  # Estimate height
            
            # Draw RGB if available
            if self.rgb_frames[cam_idx] is not None:
                self._draw_video_overlay(
                    self.rgb_frames[cam_idx], 
                    'rgb', 
                    camera_idx=cam_idx,
                    position=(rgb_x, rgb_y)
                )
                
                # Calculate depth position (next to RGB)
                overlay_height = int(self.rgb_frames[cam_idx].shape[0] * overlay_width / self.rgb_frames[cam_idx].shape[1])
                depth_x = rgb_x + overlay_width + spacing
                depth_y = rgb_y
            else:
                # Use estimated position if RGB not available yet
                depth_x = rgb_x + overlay_width + spacing
                depth_y = rgb_y
            
            # Draw depth if available
            if self.depth_frames[cam_idx] is not None:
                self._draw_video_overlay(
                    self.depth_frames[cam_idx], 
                    'depth',
                    camera_idx=cam_idx, 
                    position=(depth_x, depth_y)
                )
    
    def _draw_video_overlay(self, frame: np.ndarray, stream_type: str, camera_idx: int = 0, position=(10, 10)):
        """
        Draw video frame as 2D overlay in top-left corner.
        Optimized with texture caching and minimal memory copies.
        
        Args:
            frame: Video frame (RGB or depth)
            stream_type: 'rgb' or 'depth'
            camera_idx: Camera index for multi-camera display
            position: (x, y) position in pixels from top-left
        """
        # Calculate overlay size (20% of window size)
        overlay_width = int(self.width() * 0.2)
        overlay_height = int(frame.shape[0] * overlay_width / frame.shape[1])
        
        x, y = position
        
        # Switch to 2D mode
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width(), self.height(), 0, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Get or create texture for this camera/stream
        texture_key = (camera_idx, stream_type)
        if stream_type == 'rgb':
            texture_id = self.rgb_textures[camera_idx]
        else:
            texture_id = self.depth_textures[camera_idx]
        
        # Check if we need to create or recreate texture
        frame_size = (frame.shape[1], frame.shape[0])
        needs_recreate = (
            texture_id is None or 
            self.texture_sizes.get(texture_key) != frame_size
        )
        
        if needs_recreate:
            # Delete old texture if exists
            if texture_id is not None:
                glDeleteTextures([texture_id])
            
            # Create new texture
            texture_id = glGenTextures(1)
            
            if stream_type == 'rgb':
                self.rgb_textures[camera_idx] = texture_id
            else:
                self.depth_textures[camera_idx] = texture_id
            
            self.texture_sizes[texture_key] = frame_size
        
        # Bind texture
        glBindTexture(GL_TEXTURE_2D, texture_id)
        
        # Set texture parameters (only on creation)
        if needs_recreate:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            try:
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            except:
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        
        # Prepare texture data with minimal copies
        if stream_type == 'rgb':
            # BGR to RGB flip - use view when possible, copy only if needed
            if frame.flags['C_CONTIGUOUS']:
                # Frame is contiguous, we can use it directly with BGR format
                gl_data_format = GL_BGR  # Data format (how pixels are arranged in memory)
                gl_internal_format = GL_RGB8  # Internal format (how OpenGL stores it)
                # Don't flip - frame is already right-side up
                texture_data = np.ascontiguousarray(frame)
            else:
                texture_data = np.ascontiguousarray(frame[:, :, ::-1])
                gl_data_format = GL_RGB
                gl_internal_format = GL_RGB8
        else:  # depth
            # Depth is now float32 in millimeters - apply colormap visualization
            # Red = close, Blue = far
            depth_nonzero = frame[frame > 0]  # Ignore zero/invalid depth
            
            if depth_nonzero.size > 0:
                # Use percentiles to get a better range (ignore outliers)
                depth_min = np.percentile(depth_nonzero, 1)  # 1st percentile (mm)
                depth_max = np.percentile(depth_nonzero, 99)  # 99th percentile (mm)
                
                # Normalize to 0-1 range
                depth_clipped = np.clip(frame, depth_min, depth_max)
                depth_range = depth_max - depth_min
                
                if depth_range > 0:
                    # Normalize to 0-1
                    depth_norm = (depth_clipped - depth_min) / depth_range
                    
                    # Apply colormap: Red (close) -> Yellow -> Green -> Cyan -> Blue (far)
                    # This is similar to a "jet" colormap but simplified
                    r = np.clip(255.0 * (1.0 - depth_norm * 1.5), 0, 255).astype(np.uint8)
                    g = np.clip(255.0 * (1.0 - np.abs(depth_norm - 0.5) * 2.0), 0, 255).astype(np.uint8)
                    b = np.clip(255.0 * (depth_norm * 1.5 - 0.5), 0, 255).astype(np.uint8)
                    
                    # Stack into BGR format
                    texture_data = np.ascontiguousarray(np.stack([b, g, r], axis=-1))
                else:
                    texture_data = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            else:
                texture_data = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            
            gl_data_format = GL_BGR
            gl_internal_format = GL_RGB8
        
        # Upload texture data (this is the main bottleneck, but unavoidable)
        if needs_recreate:
            # Full texture allocation - internal format must be GL_RGB8, not GL_BGR
            glTexImage2D(GL_TEXTURE_2D, 0, gl_internal_format, texture_data.shape[1], texture_data.shape[0],
                        0, gl_data_format, GL_UNSIGNED_BYTE, texture_data)
        else:
            # Update existing texture (slightly faster than full allocation)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, texture_data.shape[1], texture_data.shape[0],
                           gl_data_format, GL_UNSIGNED_BYTE, texture_data)
        
        # Draw textured quad
        glEnable(GL_TEXTURE_2D)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(x, y)
        glTexCoord2f(1, 0); glVertex2f(x + overlay_width, y)
        glTexCoord2f(1, 1); glVertex2f(x + overlay_width, y + overlay_height)
        glTexCoord2f(0, 1); glVertex2f(x, y + overlay_height)
        glEnd()
        
        glDisable(GL_TEXTURE_2D)
        # DON'T delete texture - reuse it next frame!
        
        # Draw border
        glColor4f(1.0, 1.0, 1.0, 0.8)
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(x, y)
        glVertex2f(x + overlay_width, y)
        glVertex2f(x + overlay_width, y + overlay_height)
        glVertex2f(x, y + overlay_height)
        glEnd()
        
        # Draw label
        glColor4f(1.0, 1.0, 1.0, 1.0)
        try:
            label = f"CAM{camera_idx} {'RGB' if stream_type == 'rgb' else 'DEPTH'}"
            self.renderText(x + 5, y + 15, label)
        except Exception:
            pass
        
        # Restore 3D mode
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
    
    def __del__(self):
        """Cleanup textures on widget destruction"""
        # Clean up cached textures
        for texture_id in self.rgb_textures:
            if texture_id is not None:
                try:
                    glDeleteTextures([texture_id])
                except:
                    pass
        
        for texture_id in self.depth_textures:
            if texture_id is not None:
                try:
                    glDeleteTextures([texture_id])
                except:
                    pass


def main():
    parser = argparse.ArgumentParser(description='senseSpace Streaming Client')
    parser.add_argument('--server', type=str, required=True,
                       help='Server IP address (e.g., 192.168.1.100)')
    parser.add_argument('--stream-port', type=int, default=5000,
                       help='Single multiplexed stream port (default: 5000)')
    parser.add_argument('--num-cameras', type=int, default=1,
                       help='Number of cameras in multiplexed stream (default: 1)')
    parser.add_argument('--skeleton-server', type=str, default=None,
                       help='Optional: Skeleton data server IP (if different from stream server)')
    parser.add_argument('--skeleton-port', type=int, default=12345,
                       help='Skeleton data port (default: 12345)')
    
    args = parser.parse_args()
    
    logger.info(f"Starting streaming client...")
    logger.info(f"Server: {args.server}")
    logger.info(f"Stream Port: {args.stream_port}")
    logger.info(f"Number of cameras: {args.num_cameras}")
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Create visualization widget with multi-camera support
    viz_widget = StreamingVisualizationWidget(num_cameras=args.num_cameras)
    viz_widget.setWindowTitle(f'senseSpace Streaming Client - {args.server} ({args.num_cameras} cam)')
    viz_widget.resize(1280, 720)
    viz_widget.show()
    
    # Create video receiver with single stream port (no heartbeat needed with udpsink)
    video_receiver = VideoReceiver(
        server_ip=args.server,
        stream_port=args.stream_port,
        num_cameras=args.num_cameras,
        rgb_callback=lambda frame: viz_widget.on_rgb_frame(frame, camera_idx=0),
        depth_callback=lambda frame: viz_widget.on_depth_frame(frame, camera_idx=0),
        send_heartbeat=False  # Disable heartbeat - using direct udpsink now
    )
    logger.info(f"Using single multiplexed stream on port {args.stream_port}")
    
    viz_widget.set_video_receiver(video_receiver)
    
    # Start receiving video
    video_receiver.start()
    logger.info("Video receiver started")
    
    # Optionally connect to skeleton data server
    skeleton_client = None
    if args.skeleton_server:
        skeleton_server = args.skeleton_server
    else:
        skeleton_server = args.server  # Use same server for skeleton data
    
    try:
        logger.info(f"Connecting to skeleton server: {skeleton_server}:{args.skeleton_port}")
        skeleton_client = SenseSpaceClient(
            server_ip=skeleton_server,
            server_port=args.skeleton_port
        )
        skeleton_client.connect()
        viz_widget.set_client(skeleton_client)
        
        # Update visualization with skeleton data
        def update_skeleton():
            if skeleton_client and skeleton_client.is_connected():
                frame = skeleton_client.get_latest_frame()
                if frame:
                    viz_widget.update_frame(frame)
        
        # Timer for skeleton updates
        skeleton_timer = QTimer()
        skeleton_timer.timeout.connect(update_skeleton)
        skeleton_timer.start(33)  # ~30 FPS
        
        logger.info("Skeleton client connected")
    
    except Exception as e:
        logger.warning(f"Could not connect to skeleton server: {e}")
        logger.info("Running in video-only mode")
    
    logger.info("=== Streaming Client Running ===")
    logger.info("Press 'O' to toggle joint orientations")
    logger.info("Press 'P' to toggle point cloud")
    logger.info("Press 'Q' or close window to quit")
    logger.info("================================")
    
    # Run Qt event loop
    try:
        exit_code = app.exec_()
    finally:
        # Cleanup
        logger.info("Shutting down...")
        video_receiver.stop()
        if skeleton_client:
            skeleton_client.disconnect()
        logger.info("Stopped")
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
