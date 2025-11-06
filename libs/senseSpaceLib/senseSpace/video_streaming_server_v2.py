"""
Server integration for per-camera video streaming (Version 2).

This module provides a mixin class and helper functions to add v2 streaming 
to SenseSpaceServer without modifying the core server code.
"""

import threading
import socket
import json
import logging
from typing import Optional, Dict, Any

from .video_streaming_v2 import (
    PerCameraVideoStreamer, 
    StreamRequest
)

logger = logging.getLogger(__name__)


class PerCameraStreamingMixin:
    """
    Mixin class to add per-camera streaming capability to SenseSpaceServer.
    
    Usage:
        class MyServer(PerCameraStreamingMixin, SenseSpaceServer):
            pass
    
    Or use the standalone manager with existing SenseSpaceServer instance.
    """
    
    def init_per_camera_streaming(self, num_cameras: int, camera_width: int, 
                                  camera_height: int, camera_fps: int = 60,
                                  base_port: int = 5000):
        """
        Initialize per-camera streaming.
        
        Args:
            num_cameras: Number of cameras
            camera_width: Frame width
            camera_height: Frame height
            camera_fps: Camera framerate
            base_port: Base UDP port for RTP streams
        """
        self.per_camera_streamer = PerCameraVideoStreamer(
            num_cameras=num_cameras,
            camera_width=camera_width,
            camera_height=camera_height,
            camera_fps=camera_fps,
            base_port=base_port
        )
        
        # Start listening for client stream requests on TCP control port
        self.stream_request_thread = threading.Thread(
            target=self._stream_request_listener,
            daemon=True
        )
        self.stream_request_thread.start()
        
        logger.info("Per-camera streaming initialized")
    
    def _stream_request_listener(self):
        """
        Listen for client stream requests on TCP.
        Clients send JSON: {"cameras": [0, 1], "fps_factor": 2}
        Server responds with: {"status": "ok", "base_port": 5000}
        """
        # Listen on TCP port+1 (e.g., 12346 if main TCP is 12345)
        control_port = self.port + 1
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, control_port))
        sock.listen(5)
        
        logger.info(f"Stream request listener on port {control_port}")
        
        while self.running:
            try:
                sock.settimeout(1.0)
                client_sock, client_addr = sock.accept()
                
                # Handle request in separate thread
                threading.Thread(
                    target=self._handle_stream_request,
                    args=(client_sock, client_addr),
                    daemon=True
                ).start()
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"Stream request listener error: {e}")
        
        sock.close()
    
    def _handle_stream_request(self, client_sock: socket.socket, client_addr: tuple):
        """Handle a single client stream request"""
        client_ip, client_port = client_addr
        client_id = f"{client_ip}:{client_port}"
        
        try:
            # Receive request (max 4KB JSON)
            data = client_sock.recv(4096)
            if not data:
                return
            
            request_data = json.loads(data.decode('utf-8'))
            logger.info(f"Stream request from {client_id}: {request_data}")
            
            # Parse request
            stream_request = StreamRequest.from_dict(request_data)
            
            # Validate cameras
            if not stream_request.cameras:
                response = {"status": "error", "message": "No cameras requested"}
            elif max(stream_request.cameras) >= self.per_camera_streamer.num_cameras:
                response = {"status": "error", 
                           "message": f"Invalid camera index (max: {self.per_camera_streamer.num_cameras-1})"}
            else:
                # Add client
                success = self.per_camera_streamer.add_client(
                    client_id=client_id,
                    client_ip=client_ip,
                    stream_request=stream_request
                )
                
                if success:
                    response = {
                        "status": "ok",
                        "base_port": self.per_camera_streamer.base_port,
                        "cameras": stream_request.cameras,
                        "fps_factor": stream_request.fps_factor
                    }
                else:
                    response = {"status": "error", "message": "Failed to add client"}
            
            # Send response
            client_sock.sendall(json.dumps(response).encode('utf-8'))
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from {client_id}: {e}")
            response = {"status": "error", "message": "Invalid JSON"}
            client_sock.sendall(json.dumps(response).encode('utf-8'))
        except Exception as e:
            logger.error(f"Error handling stream request from {client_id}: {e}")
        finally:
            client_sock.close()
    
    def push_camera_frames_v2(self, rgb_frames, depth_frames):
        """
        Push frames to per-camera streamer.
        
        Args:
            rgb_frames: List of RGB frames [cam0, cam1, ...]
            depth_frames: List of depth frames [cam0, cam1, ...]
        """
        if hasattr(self, 'per_camera_streamer'):
            self.per_camera_streamer.push_frames(rgb_frames, depth_frames)
    
    def shutdown_per_camera_streaming(self):
        """Shutdown per-camera streaming"""
        if hasattr(self, 'per_camera_streamer'):
            self.per_camera_streamer.shutdown()


class PerCameraStreamingManager:
    """
    Standalone manager for adding per-camera streaming to an existing server.
    
    Use this if you don't want to use the mixin approach.
    
    Example:
        server = SenseSpaceServer(...)
        streaming_mgr = PerCameraStreamingManager(
            server=server,
            num_cameras=3,
            camera_width=672,
            camera_height=376,
            camera_fps=60
        )
        streaming_mgr.start()
        
        # In your capture loop:
        streaming_mgr.push_frames(rgb_frames, depth_frames)
    """
    
    def __init__(self, server, num_cameras: int, camera_width: int,
                 camera_height: int, camera_fps: int = 60, base_port: int = 5000):
        """
        Args:
            server: SenseSpaceServer instance
            num_cameras: Number of cameras
            camera_width: Frame width
            camera_height: Frame height
            camera_fps: Camera framerate
            base_port: Base UDP port
        """
        self.server = server
        self.streamer = PerCameraVideoStreamer(
            num_cameras=num_cameras,
            camera_width=camera_width,
            camera_height=camera_height,
            camera_fps=camera_fps,
            base_port=base_port
        )
        
        # Control port = main TCP port + 1
        self.control_port = server.port + 1
        self.control_socket = None
        self.control_thread = None
        
        logger.info(f"PerCameraStreamingManager initialized on port {self.control_port}")
    
    def start(self):
        """Start the control listener"""
        self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.control_socket.bind((self.server.host, self.control_port))
        self.control_socket.listen(5)
        
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        
        logger.info(f"Stream control listener started on port {self.control_port}")
    
    def stop(self):
        """Stop streaming and control listener"""
        if self.control_socket:
            self.control_socket.close()
        
        self.streamer.shutdown()
        logger.info("PerCameraStreamingManager stopped")
    
    def push_frames(self, rgb_frames, depth_frames):
        """Push frames to all active clients"""
        self.streamer.push_frames(rgb_frames, depth_frames)
    
    def _control_loop(self):
        """Control listener loop"""
        while self.server.running:
            try:
                self.control_socket.settimeout(1.0)
                client_sock, client_addr = self.control_socket.accept()
                
                # Handle in separate thread
                threading.Thread(
                    target=self._handle_request,
                    args=(client_sock, client_addr),
                    daemon=True
                ).start()
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.server.running:
                    logger.error(f"Control loop error: {e}")
    
    def _handle_request(self, client_sock: socket.socket, client_addr: tuple):
        """Handle stream request from client"""
        client_ip, client_port = client_addr
        client_id = f"{client_ip}:{client_port}"
        
        try:
            data = client_sock.recv(4096)
            if not data:
                return
            
            request_data = json.loads(data.decode('utf-8'))
            logger.info(f"Stream request from {client_id}: {request_data}")
            
            stream_request = StreamRequest.from_dict(request_data)
            
            # Validate
            if not stream_request.cameras:
                response = {"status": "error", "message": "No cameras requested"}
            elif max(stream_request.cameras) >= self.streamer.num_cameras:
                response = {"status": "error", 
                           "message": f"Invalid camera (max: {self.streamer.num_cameras-1})"}
            else:
                # Add client
                success = self.streamer.add_client(
                    client_id=client_id,
                    client_ip=client_ip,
                    stream_request=stream_request
                )
                
                if success:
                    response = {
                        "status": "ok",
                        "base_port": self.streamer.base_port,
                        "cameras": stream_request.cameras,
                        "fps_factor": stream_request.fps_factor,
                        "num_cameras": self.streamer.num_cameras
                    }
                else:
                    response = {"status": "error", "message": "Failed to add client"}
            
            client_sock.sendall(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            logger.error(f"Error handling request from {client_id}: {e}")
            try:
                response = {"status": "error", "message": str(e)}
                client_sock.sendall(json.dumps(response).encode('utf-8'))
            except:
                pass
        finally:
            client_sock.close()
