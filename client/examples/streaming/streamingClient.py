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
import cv2
import json
import socket
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'libs'))

from senseSpaceLib.senseSpace.video_streaming import VideoReceiver, MultiCameraVideoReceiver
from senseSpaceLib.senseSpace.client import SenseSpaceClient
from senseSpaceLib.senseSpace.vizWidget import SkeletonGLWidget

# Import v2 per-camera streaming (optional)
try:
    from senseSpaceLib.senseSpace.video_streaming_v2 import PerCameraVideoReceiver
    V2_AVAILABLE = True
except ImportError:
    V2_AVAILABLE = False
    PerCameraVideoReceiver = None

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
            frame: RGB frame data as numpy array (H, W, 3) in BGR format, dtype=uint8
            camera_idx: Camera index (0 for single camera)
        
        PERFORMANCE NOTE:
        ------------------
        This callback approach is efficient for video streaming because:
        1. GStreamer handles frame arrival asynchronously in background threads
        2. Callback is invoked immediately when frame is decoded (minimal latency)
        3. Only the frame reference is passed (no data copying until .update())
        4. Qt's update() batches redraws efficiently (won't redraw 60x/sec if GPU can't keep up)
        
        Alternative approaches (and why callbacks are better):
        - Polling with get_latest_frame(): Adds latency, may miss frames
        - Queue-based: Adds memory overhead, complexity, same thread switching
        - Shared memory: No benefit since we're already getting zero-copy from GStreamer
        
        The frames DO arrive at different times (cameras aren't perfectly synced),
        but this is expected and handled naturally by independent callbacks.
        
        Student Guide - Working with RGB frames:
        ----------------------------------------
        The 'frame' parameter is a numpy array ready to use with OpenCV or other libraries.
        
        Example 1: Display with OpenCV
            cv2.imshow('RGB Camera', frame)
            cv2.waitKey(1)
        
        Example 2: Save as PNG with OpenCV
            cv2.imwrite('rgb_frame.png', frame)
        
        Example 3: Save as PNG with Qt (convert BGR to RGB first)
            from PyQt5.QtGui import QImage
            height, width, channels = frame.shape
            bytes_per_line = channels * width
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            qimage = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            qimage.save('rgb_frame.png')
        
        Example 4: Convert to RGB for processing
            rgb_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Now rgb_array[:,:,0] is Red, [:,:,1] is Green, [:,:,2] is Blue
        
        Example 5: Access pixel values
            height, width, channels = frame.shape
            blue_channel = frame[:, :, 0]   # Blue channel
            green_channel = frame[:, :, 1]  # Green channel
            red_channel = frame[:, :, 2]    # Red channel
            pixel_bgr = frame[y, x]  # Get BGR value at position (x, y)
        """
        if camera_idx < self.num_cameras:
            # Debug: Calculate hash of frame to detect if frames are identical
            if not hasattr(self, '_rgb_frame_counts'):
                self._rgb_frame_counts = {}
                self._rgb_frame_hashes = {}
            if camera_idx not in self._rgb_frame_counts:
                self._rgb_frame_counts[camera_idx] = 0
                self._rgb_frame_hashes[camera_idx] = []
            
            self._rgb_frame_counts[camera_idx] += 1
            
            # Compute simple hash of first 100 pixels to detect identical frames
            frame_hash = hash(frame[:10, :10, :].tobytes())
            self._rgb_frame_hashes[camera_idx].append(frame_hash)
            
            if self._rgb_frame_counts[camera_idx] <= 5:
                # Check if this frame matches any other camera's recent frames
                identical_to = []
                for other_idx in range(self.num_cameras):
                    if other_idx != camera_idx and other_idx in self._rgb_frame_hashes:
                        if frame_hash in self._rgb_frame_hashes[other_idx][-5:]:  # Check last 5 frames
                            identical_to.append(other_idx)
                
                if identical_to:
                    logger.warning(f"⚠ RGB camera {camera_idx} frame #{self._rgb_frame_counts[camera_idx]} IDENTICAL to camera(s) {identical_to}! (hash={frame_hash})")
                else:
                    logger.info(f"✓ RGB frame received for camera {camera_idx} (frame #{self._rgb_frame_counts[camera_idx]}, shape={frame.shape}, hash={frame_hash})")
            
            self.rgb_frames[camera_idx] = frame
            # Trigger re-draw
            self.update()
    
    def on_depth_frame(self, frame: np.ndarray, camera_idx: int = 0):
        """
        Callback for depth frame reception.
        
        Args:
            frame: Depth frame data as numpy array (H, W) in MILLIMETERS, dtype=float32
            camera_idx: Camera index (0 for single camera)
        
        Student Guide - Working with Depth frames:
        ------------------------------------------
        The 'frame' parameter is a 2D numpy array containing depth values in MILLIMETERS.
        - Valid depth values: > 0 (typically 300mm to 20000mm depending on camera)
        - Invalid/unknown depth: 0 or NaN
        
        Example 1: Get depth at specific pixel
            depth_mm = frame[y, x]  # Depth in millimeters at position (x, y)
            depth_m = depth_mm / 1000.0  # Convert to meters
            if depth_mm > 0:
                print(f"Object at ({x},{y}) is {depth_m:.2f} meters away")
        
        Example 2: Find closest object
            valid_depths = frame[frame > 0]  # Get all valid depth values
            if valid_depths.size > 0:
                closest_mm = valid_depths.min()
                closest_m = closest_mm / 1000.0
                print(f"Closest object: {closest_m:.2f}m")
        
        Example 3: Create colored depth visualization for OpenCV/saving
            # Normalize depth to 0-255 range for visualization
            depth_valid = frame.copy()
            depth_valid[depth_valid == 0] = np.nan  # Mark invalid as NaN
            depth_min = np.nanpercentile(depth_valid, 2)
            depth_max = np.nanpercentile(depth_valid, 98)
            depth_norm = np.clip((frame - depth_min) / (depth_max - depth_min), 0, 1)
            depth_uint8 = (depth_norm * 255).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
            
            # Save colored depth map
            cv2.imwrite('depth_colored.png', depth_colored)
        
        Example 4: Save raw depth data (preserves actual distance values)
            # Save as 32-bit float TIFF (preserves millimeter precision)
            cv2.imwrite('depth_raw.tiff', frame)
            
            # Or save as 16-bit PNG (convert mm to 0.1mm units, max ~6.5m range)
            depth_uint16 = (frame * 10).astype(np.uint16)  # 0.1mm precision
            cv2.imwrite('depth_raw.png', depth_uint16)
        
        Example 5: Mask objects by distance
            # Find pixels between 500mm and 2000mm (0.5m to 2m)
            mask = (frame > 500) & (frame < 2000)
            close_objects = frame[mask]
        
        Example 6: Save with Qt (colored visualization)
            from PyQt5.QtGui import QImage
            # First create colored version (see Example 3)
            depth_uint8 = (depth_norm * 255).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
            
            height, width = depth_colored.shape[:2]
            bytes_per_line = 3 * width
            rgb_depth = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
            qimage = QImage(rgb_depth.data, width, height, bytes_per_line, QImage.Format_RGB888)
            qimage.save('depth_visualization.png')
        """
        if camera_idx < self.num_cameras:
            # Debug: Calculate hash of frame to detect if frames are identical
            if not hasattr(self, '_depth_frame_counts'):
                self._depth_frame_counts = {}
                self._depth_frame_hashes = {}
            if camera_idx not in self._depth_frame_counts:
                self._depth_frame_counts[camera_idx] = 0
                self._depth_frame_hashes[camera_idx] = []
            
            self._depth_frame_counts[camera_idx] += 1
            
            # Compute simple hash of first 100 pixels to detect identical frames
            frame_hash = hash(frame[:10, :10].tobytes())
            self._depth_frame_hashes[camera_idx].append(frame_hash)
            
            if self._depth_frame_counts[camera_idx] <= 5:
                # Check if this frame matches any other camera's recent frames
                identical_to = []
                for other_idx in range(self.num_cameras):
                    if other_idx != camera_idx and other_idx in self._depth_frame_hashes:
                        if frame_hash in self._depth_frame_hashes[other_idx][-5:]:  # Check last 5 frames
                            identical_to.append(other_idx)
                
                if identical_to:
                    logger.warning(f"⚠ Depth camera {camera_idx} frame #{self._depth_frame_counts[camera_idx]} IDENTICAL to camera(s) {identical_to}! (hash={frame_hash})")
                else:
                    logger.info(f"✓ Depth frame received for camera {camera_idx} (frame #{self._depth_frame_counts[camera_idx]}, shape={frame.shape}, hash={frame_hash})")
            
            self.depth_frames[camera_idx] = frame
            self.update()
    
    def draw_custom(self, frame):
        """
        Override to draw video streams below status text.
        Called after skeleton/floor drawing but before 2D overlay.
        Handles both single and multi-camera layouts.
        Layout: Side-by-side pairs stacked vertically
        Camera 0: RGB0 | DEPTH0
        Camera 1: RGB1 | DEPTH1 (below camera 0)
        Camera N: RGBN | DEPTHN (below camera N-1)
        """
        # Calculate layout - horizontal RGB+Depth pairs, stacked vertically for multiple cameras
        overlay_width = int(self.width() * 0.2)
        x_offset = 10
        y_offset = 80  # Position below status text
        spacing = 10
        
        current_y = y_offset
        
        for cam_idx in range(self.num_cameras):
            # RGB on left
            rgb_x = x_offset
            rgb_y = current_y
            
            # Draw RGB if available
            rgb_height = 0
            if self.rgb_frames[cam_idx] is not None:
                self._draw_video_overlay(
                    self.rgb_frames[cam_idx], 
                    'rgb', 
                    camera_idx=cam_idx,
                    position=(rgb_x, rgb_y)
                )
                
                # Calculate actual height for proper spacing
                frame_height, frame_width = self.rgb_frames[cam_idx].shape[:2]
                rgb_height = int(overlay_width * frame_height / frame_width)
            
            # Depth on right (next to RGB)
            depth_x = x_offset + overlay_width + spacing
            depth_y = current_y  # Same Y as RGB
            
            # Draw depth if available
            depth_height = 0
            if self.depth_frames[cam_idx] is not None:
                self._draw_video_overlay(
                    self.depth_frames[cam_idx], 
                    'depth',
                    camera_idx=cam_idx, 
                    position=(depth_x, depth_y)
                )
                
                # Calculate depth height
                frame_height, frame_width = self.depth_frames[cam_idx].shape[:2]
                depth_height = int(overlay_width * frame_height / frame_width)
            
            # Update Y position for next camera (below current pair + spacing)
            max_height = max(rgb_height, depth_height) if rgb_height > 0 or depth_height > 0 else 150
            current_y += max_height + spacing * 2  # Extra spacing between camera pairs
    
    def _draw_video_overlay(self, frame: np.ndarray, stream_type: str, camera_idx: int = 0, position=(10, 10)):
        """
        Draw video frame as 2D overlay in top-left corner.
        Optimized with texture caching and minimal memory copies.
        IMPORTANT: Respects aspect ratio of source frame.
        
        Args:
            frame: Video frame (RGB or depth)
            stream_type: 'rgb' or 'depth'
            camera_idx: Camera index for multi-camera display
            position: (x, y) position in pixels from top-left
        """
        # Calculate overlay size (20% of window width, height preserves aspect ratio)
        overlay_width = int(self.width() * 0.2)
        # IMPORTANT: Preserve aspect ratio by calculating height from frame dimensions
        frame_height, frame_width = frame.shape[:2]
        frame_aspect_ratio = frame_width / frame_height  # width / height
        overlay_height = int(overlay_width / frame_aspect_ratio)
        
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
            # Depth visualization with proper colormap
            # Data is float32 in MILLIMETERS (from ZED with UNIT.MILLIMETER), create RED->BLUE colormap
            depth_nonzero = frame[frame > 0]  # Ignore zero/invalid depth
            
            if depth_nonzero.size > 0:
                # Use percentiles for robust range estimation (ignore outliers)
                depth_min = np.percentile(depth_nonzero, 2)   # 2nd percentile
                depth_max = np.percentile(depth_nonzero, 98)  # 98th percentile
                
                # Ensure we have a valid range
                if depth_max - depth_min < 100:  # Less than 10cm range
                    depth_min = depth_nonzero.min()
                    depth_max = depth_nonzero.max()
                
                # Normalize depth to 0-1 range
                depth_clipped = np.clip(frame, depth_min, depth_max)
                depth_range = depth_max - depth_min
                
                if depth_range > 10:  # At least 1cm range
                    depth_norm = (depth_clipped - depth_min) / depth_range
                    
                    # Apply OpenCV TURBO colormap (like ZED SDK) - vibrant rainbow colors
                    # Convert normalized depth (0-1) to 0-255 range for colormap
                    depth_uint8 = (depth_norm * 255).astype(np.uint8)
                    
                    # Apply TURBO colormap (modern, perceptually uniform rainbow)
                    # This gives: Blue (far) → Cyan → Green → Yellow → Red (close)
                    colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
                    
                    # colored is already BGR format, perfect for OpenGL
                    texture_data = np.ascontiguousarray(colored)
                else:
                    # Constant depth - map the single value to a color using TURBO colormap
                    avg_depth = depth_nonzero.mean()
                    norm_val = np.clip(avg_depth / 10000.0, 0.0, 1.0)  # Normalize to 0-10m
                    
                    # Create single-color image using TURBO colormap
                    depth_uint8 = np.full_like(frame, int(norm_val * 255), dtype=np.uint8)
                    colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
                    texture_data = np.ascontiguousarray(colored)
            else:
                # No valid depth - show as black
                black = np.zeros_like(frame, dtype=np.uint8)
                texture_data = np.ascontiguousarray(np.stack([black, black, black], axis=-1))
            
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


def get_server_info(server_ip, tcp_port=12345, timeout=5.0):
    """
    Connect to server via TCP and retrieve streaming configuration.
    
    Args:
        server_ip: Server IP address
        tcp_port: TCP port for skeleton/control data
        timeout: Connection timeout in seconds
    
    Returns:
        dict with server_info data, or None if failed/not available
    """
    try:
        logger.info(f"Querying server info from {server_ip}:{tcp_port}...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((server_ip, tcp_port))
        
        # Read first message (should be server_info)
        buffer = b""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buffer += chunk
                
                # Check if we have a complete JSON message (ends with \n)
                if b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    try:
                        message = json.loads(line.decode('utf-8'))
                        if message.get('type') == 'server_info':
                            sock.close()
                            logger.info(f"✓ Received server info: {message['data']['streaming']['num_cameras']} cameras, "
                                      f"{message['data']['streaming']['camera_width']}x{message['data']['streaming']['camera_height']}@{message['data']['streaming']['framerate']}fps")
                            return message['data']
                    except (json.JSONDecodeError, KeyError):
                        # Not a server_info message, keep reading
                        pass
            except socket.timeout:
                break
        
        sock.close()
        logger.warning("Server did not send server_info message (old server version?)")
        return None
        
    except Exception as e:
        logger.warning(f"Could not retrieve server info: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='senseSpace Streaming Client')
    parser.add_argument('--server', type=str, required=True,
                       help='Server IP address (e.g., 192.168.1.100)')
    parser.add_argument('--stream-port', type=int, default=None,
                       help='Single multiplexed stream port (default: auto-detect from server)')
    parser.add_argument('--num-cameras', type=int, default=None,
                       help='Number of cameras (default: auto-detect from server)')
    parser.add_argument('--skeleton-server', type=str, default=None,
                       help='Optional: Skeleton data server IP (if different from stream server)')
    parser.add_argument('--skeleton-port', type=int, default=12345,
                       help='Skeleton data port (default: 12345)')
    
    # Per-camera streaming options
    parser.add_argument('--camStream', action='store_true',
                       help='Enable per-camera video streaming (unicast, works over WiFi)')
    parser.add_argument('--cameras', nargs='+', type=int, default=None,
                       help='Select specific camera indices (e.g., --cameras 0 2). Default: stream all cameras')
    parser.add_argument('--fps-factor', type=int, default=2,
                       help='FPS decimation factor: 1=full 60fps, 2=30fps, 3=20fps (default: 2)')
    
    args = parser.parse_args()
    
    logger.info(f"Starting streaming client...")
    logger.info(f"Server: {args.server}")
    
    # Check if using per-camera streaming
    if args.camStream:
        if not V2_AVAILABLE:
            logger.error("Per-camera streaming requested but not available (missing video_streaming_v2 module)")
            sys.exit(1)
        
        logger.info("Using per-camera video streaming (unicast)")
        
        # Query server for number of cameras if not selecting specific cameras
        if args.cameras is None:
            # Auto-detect all cameras from server
            server_info = get_server_info(args.server, args.skeleton_port)
            if server_info and server_info.get('streaming', {}).get('num_cameras'):
                total_cameras = server_info['streaming']['num_cameras']
                cameras = list(range(total_cameras))
                logger.info(f"Auto-detected {total_cameras} cameras, streaming all")
            else:
                # Fallback: assume 1 camera
                cameras = [0]
                logger.warning("Could not detect camera count, defaulting to camera 0")
        else:
            # User specified specific cameras
            cameras = args.cameras
            logger.info(f"Streaming selected cameras: {cameras}")
        
        num_cameras = len(cameras)
        logger.info(f"Requesting cameras: {cameras}")
        logger.info(f"FPS factor: {args.fps_factor}")
        
        # Create Qt application
        app = QApplication(sys.argv)
        
        # Create visualization widget
        viz_widget = StreamingVisualizationWidget(num_cameras=num_cameras)
        viz_widget.setWindowTitle(f'senseSpace Camera Stream - {args.server} (cameras: {cameras})')
        viz_widget.resize(1280, 720)
        viz_widget.show()
        
        # Request stream from server
        control_port = args.skeleton_port + 1
        control_socket = None
        try:
            control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            control_socket.connect((args.server, control_port))
            
            request = {
                'cameras': cameras,
                'fps_factor': args.fps_factor
            }
            control_socket.sendall(json.dumps(request).encode('utf-8'))
            
            data = control_socket.recv(4096)
            response = json.loads(data.decode('utf-8'))
            
            # Don't close socket - keep it open for session
            
            if response.get('status') != 'ok':
                logger.error(f"Server rejected stream request: {response.get('message', 'Unknown')}")
                control_socket.close()
                sys.exit(1)
            
            base_port = response['base_port']
            logger.info(f"Stream request accepted, base port: {base_port}")
            logger.info("Control connection kept open - server will cleanup when this closes")
            
        except Exception as e:
            logger.error(f"Failed to request stream: {e}")
            if control_socket:
                control_socket.close()
            sys.exit(1)
        
        # Create v2 receiver
        video_receiver = PerCameraVideoReceiver(
            server_ip=args.server,
            base_port=base_port,
            cameras=cameras,
            rgb_callback=lambda frame, cam_idx: viz_widget.on_rgb_frame(frame, camera_idx=cam_idx),
            depth_callback=lambda frame, cam_idx: viz_widget.on_depth_frame(frame, camera_idx=cam_idx)
        )
        
    else:
        # Original v1 streaming (existing code)
        logger.info("Using v1 multiplexed streaming (multicast - localhost only)")
        logger.warning("For WiFi streaming, use --camStream flag instead")
        
        # Auto-detect server configuration via TCP
        server_info = get_server_info(args.server, args.skeleton_port)
    
        # Original v1 streaming (existing code)
        logger.info("Using v1 multiplexed streaming")
        
        # Auto-detect server configuration via TCP
        server_info = get_server_info(args.server, args.skeleton_port)
    
        # Determine streaming parameters (command-line args override auto-detection)
        if args.stream_port is not None:
            stream_port = args.stream_port
        elif server_info and server_info.get('streaming', {}).get('enabled'):
            stream_port = server_info['streaming']['port']
        else:
            stream_port = 5000  # Default fallback
        
        # Get multicast host from server_info (or use default)
        if server_info and server_info.get('streaming', {}).get('host'):
            stream_host = server_info['streaming']['host']
        else:
            stream_host = "239.255.0.1"  # Default multicast
        
        if args.num_cameras is not None:
            num_cameras = args.num_cameras
        elif server_info and server_info.get('streaming', {}).get('num_cameras'):
            num_cameras = server_info['streaming']['num_cameras']
        else:
            num_cameras = 1  # Default fallback
        
        logger.info(f"Stream Host: {stream_host}")
        logger.info(f"Stream Port: {stream_port}")
        logger.info(f"Number of cameras: {num_cameras}")
        
        if server_info:
            logger.info(f"Depth mode: {server_info['streaming'].get('depth_mode', 'UNKNOWN')}")
            logger.info(f"Resolution: {server_info['streaming'].get('camera_width')}x{server_info['streaming'].get('camera_height')}@{server_info['streaming'].get('framerate')}fps")
        
        # Create Qt application
        app = QApplication(sys.argv)
        
        # Create visualization widget with multi-camera support
        viz_widget = StreamingVisualizationWidget(num_cameras=num_cameras)
        viz_widget.setWindowTitle(f'senseSpace Streaming Client - {args.server} ({num_cameras} cam)')
        viz_widget.resize(1280, 720)
        viz_widget.show()
        
        # Create video receiver with single stream port (no heartbeat needed with udpsink)
        # IMPORTANT: Callbacks receive (frame, camera_idx) - must handle multi-camera!
        # NOTE: Lambda captures cam_idx at call time, not definition time
        # Use stream_host (multicast address) instead of server IP for udpsrc binding
        video_receiver = VideoReceiver(
            server_ip=stream_host,  # Use multicast address from server_info
            stream_port=stream_port,
            num_cameras=num_cameras,
            rgb_callback=lambda frame, cam_idx: viz_widget.on_rgb_frame(frame, camera_idx=cam_idx),
            depth_callback=lambda frame, cam_idx: viz_widget.on_depth_frame(frame, camera_idx=cam_idx),
            send_heartbeat=False  # Disable heartbeat - using direct udpsink now
        )
        logger.info(f"Using single multiplexed stream on port {stream_port}")
    
    # Common code for both v1 and v2    viz_widget.set_video_receiver(video_receiver)
    
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
        
        # Close control connection (triggers server cleanup)
        if args.camStream and control_socket:
            try:
                control_socket.close()
                logger.info("Closed control connection")
            except:
                pass
        
        video_receiver.stop()
        if skeleton_client:
            skeleton_client.disconnect()
        logger.info("Stopped")
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
