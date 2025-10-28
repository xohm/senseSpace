#!/usr/bin/env python3
"""
Point Cloud Client for senseSpace

Receives and visualizes point cloud data from PointCloudServer.
Can run standalone or be integrated into existing visualization clients.
"""

import socket
import struct
import threading
import time
import numpy as np
from typing import Optional, Callable
import sys
import os

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
    _zstd_decompressor = zstd.ZstdDecompressor()
except ImportError:
    ZSTD_AVAILABLE = False
    print("[WARNING] zstandard not available - cannot decompress point clouds")


class PointCloudClient:
    """
    Client for receiving point cloud data from PointCloudServer.
    
    Handles connection, decompression, and provides callbacks for point cloud frames.
    """
    
    def __init__(self, server_ip: str, server_port: int = 12346):
        """
        Initialize point cloud client.
        
        Args:
            server_ip: Server IP address
            server_port: Point cloud server port (default: 12346)
        """
        self.server_ip = server_ip
        self.server_port = server_port
        
        self.socket = None
        self.connected = False
        self.running = False
        
        self.receive_thread = None
        
        # Callbacks
        self.on_pointcloud_received = None  # Callback(points, colors, timestamp)
        self.on_connection_changed = None   # Callback(connected: bool)
        
        # Latest point cloud (thread-safe)
        self._latest_points = None
        self._latest_colors = None
        self._latest_timestamp = 0.0
        self._pc_lock = threading.Lock()
        
        # Statistics
        self.frames_received = 0
        self.bytes_received = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
    
    def connect(self) -> bool:
        """
        Connect to point cloud server.
        
        Returns:
            True if connection successful
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            print(f"[INFO] Connecting to point cloud server at {self.server_ip}:{self.server_port}...")
            self.socket.connect((self.server_ip, self.server_port))
            
            self.connected = True
            self.running = True
            
            print(f"[INFO] Connected to point cloud server")
            
            if self.on_connection_changed:
                self.on_connection_changed(True)
            
            # Start receive thread
            self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.receive_thread.start()
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to connect: {e}")
            self.connected = False
            if self.on_connection_changed:
                self.on_connection_changed(False)
            return False
    
    def disconnect(self):
        """Disconnect from server"""
        self.running = False
        
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        
        self.connected = False
        
        if self.on_connection_changed:
            self.on_connection_changed(False)
        
        print("[INFO] Disconnected from point cloud server")
    
    def _receive_loop(self):
        """Main receive loop (runs in background thread)"""
        buffer = bytearray()
        
        while self.running and self.connected:
            try:
                # Receive data
                data = self.socket.recv(65536)  # Large buffer for point cloud data
                if not data:
                    print("[WARNING] Connection closed by server")
                    break
                
                buffer.extend(data)
                self.bytes_received += len(data)
                
                # Process complete messages in buffer
                while len(buffer) >= 18:  # Minimum header size: 2+8+4+4
                    # Check magic bytes
                    magic = bytes(buffer[:2])
                    
                    if magic == b'\x9f\xd2':  # Compressed point cloud
                        if not ZSTD_AVAILABLE:
                            print("[ERROR] Received compressed data but zstd not available")
                            buffer = bytearray()
                            break
                        
                        # Parse header: [magic:2][timestamp:8][count:4][length:4]
                        if len(buffer) < 18:
                            break
                        
                        timestamp, num_points, payload_length = struct.unpack('>dII', buffer[2:18])
                        
                        # Check if we have complete payload
                        if len(buffer) < 18 + payload_length:
                            break  # Wait for more data
                        
                        # Extract and decompress payload
                        compressed = bytes(buffer[18:18+payload_length])
                        try:
                            payload = _zstd_decompressor.decompress(compressed)
                        except Exception as e:
                            print(f"[ERROR] Decompression failed: {e}")
                            buffer = buffer[18+payload_length:]
                            continue
                        
                        # Remove processed data from buffer
                        buffer = buffer[18+payload_length:]
                        
                    elif magic == b'\x9f\xd3':  # Uncompressed point cloud
                        # Parse header
                        if len(buffer) < 18:
                            break
                        
                        timestamp, num_points, payload_length = struct.unpack('>dII', buffer[2:18])
                        
                        # Check if we have complete payload
                        if len(buffer) < 18 + payload_length:
                            break
                        
                        # Extract payload
                        payload = bytes(buffer[18:18+payload_length])
                        
                        # Remove processed data from buffer
                        buffer = buffer[18+payload_length:]
                        
                    elif magic == b'\x9f\xd4':  # Per-person point clouds
                        # Parse header: [magic:2][timestamp:8][count:4][length:4]
                        if len(buffer) < 18:
                            break
                        
                        timestamp, person_count, payload_length = struct.unpack('>dII', buffer[2:18])
                        
                        # Check if we have complete payload
                        if len(buffer) < 18 + payload_length:
                            break
                        
                        # Extract payload (might be compressed)
                        compressed_payload = bytes(buffer[18:18+payload_length])
                        
                        # Try to decompress
                        try:
                            if ZSTD_AVAILABLE:
                                payload = _zstd_decompressor.decompress(compressed_payload)
                            else:
                                payload = compressed_payload
                        except:
                            # Not compressed
                            payload = compressed_payload
                        
                        # Parse per-person format and merge all people into one point cloud
                        try:
                            all_points = []
                            all_colors = []
                            offset = 0
                            
                            for _ in range(person_count):
                                # Read person ID and point count (big-endian: signed int + unsigned int)
                                person_id, point_count = struct.unpack('>iI', payload[offset:offset+8])
                                offset += 8
                                
                                # Read points (x,y,z floats)
                                points_bytes = point_count * 12
                                points_data = payload[offset:offset+points_bytes]
                                offset += points_bytes
                                
                                # Read colors (r,g,b bytes)
                                colors_bytes = point_count * 3
                                colors_data = payload[offset:offset+colors_bytes]
                                offset += colors_bytes
                                
                                # Convert to arrays
                                points = np.frombuffer(points_data, dtype=np.float32).reshape(-1, 3)
                                colors = np.frombuffer(colors_data, dtype=np.uint8).reshape(-1, 3)
                                
                                all_points.append(points)
                                all_colors.append(colors)
                            
                            # Merge all people
                            if all_points:
                                merged_points = np.vstack(all_points)
                                merged_colors = np.vstack(all_colors)
                                num_points = len(merged_points)
                                
                                # Create payload in expected format for _parse_point_cloud
                                # Format: points (float32 x,y,z) + colors (uint8 r,g,b)
                                payload = merged_points.tobytes() + merged_colors.tobytes()
                            else:
                                num_points = 0
                                payload = b''
                        except Exception as e:
                            print(f"[ERROR] Failed to parse per-person data: {e}")
                            buffer = buffer[18+payload_length:]
                            continue
                        
                        # Remove processed data from buffer
                        buffer = buffer[18+payload_length:]
                        
                    else:
                        # Unknown format - skip byte
                        print(f"[WARNING] Unknown magic bytes: {magic.hex()}")
                        buffer = buffer[1:]
                        continue
                    
                    # Parse point cloud data
                    try:
                        self._parse_point_cloud(payload, num_points, timestamp)
                    except Exception as e:
                        print(f"[ERROR] Failed to parse point cloud: {e}")
                        continue
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[ERROR] Receive error: {e}")
                break
        
        # Connection lost
        self.disconnect()
    
    def _parse_point_cloud(self, payload: bytes, num_points: int, timestamp: float):
        """
        Parse binary point cloud data.
        
        Args:
            payload: Binary payload (points + colors)
            num_points: Number of points
            timestamp: Frame timestamp
        """
        try:
            # Calculate expected sizes
            points_size = num_points * 12  # 3 floats * 4 bytes
            colors_size = num_points * 3   # 3 uint8
            
            if len(payload) != points_size + colors_size:
                print(f"[WARNING] Payload size mismatch: expected {points_size + colors_size}, got {len(payload)}")
                return
            
            # Parse points (Nx3 float32)
            points_bin = payload[:points_size]
            points = np.frombuffer(points_bin, dtype=np.float32).reshape(-1, 3)
            
            # Parse colors (Nx3 uint8)
            colors_bin = payload[points_size:]
            colors = np.frombuffer(colors_bin, dtype=np.uint8).reshape(-1, 3)
            
            # Store latest point cloud
            with self._pc_lock:
                self._latest_points = points
                self._latest_colors = colors
                self._latest_timestamp = timestamp
            
            # Update statistics
            self.frames_received += 1
            now = time.time()
            if now - self.last_fps_time >= 1.0:
                self.current_fps = self.frames_received / (now - self.last_fps_time)
                self.frames_received = 0
                self.last_fps_time = now
            
            # Call callback
            if self.on_pointcloud_received:
                self.on_pointcloud_received(points, colors, timestamp)
                
        except Exception as e:
            print(f"[ERROR] Point cloud parsing failed: {e}")
    
    def get_latest_pointcloud(self) -> Optional[tuple]:
        """
        Get the latest received point cloud (thread-safe).
        
        Returns:
            (points, colors, timestamp) or None if no data received yet
        """
        with self._pc_lock:
            if self._latest_points is None:
                return None
            return self._latest_points.copy(), self._latest_colors.copy(), self._latest_timestamp
    
    def get_statistics(self) -> dict:
        """
        Get client statistics.
        
        Returns:
            Dictionary with 'fps', 'bytes_received', 'connected'
        """
        return {
            'fps': self.current_fps,
            'bytes_received': self.bytes_received,
            'connected': self.connected
        }


def main():
    """Standalone point cloud client with simple visualization"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SenseSpace Point Cloud Client")
    parser.add_argument("--server", type=str, required=True, help="Server IP address")
    parser.add_argument("--port", type=int, default=12346, help="Point cloud server port")
    parser.add_argument("--stats", action="store_true", help="Print statistics instead of point cloud data")
    
    args = parser.parse_args()
    
    # Create client
    client = PointCloudClient(args.server, args.port)
    
    # Setup callbacks
    def on_pointcloud(points, colors, timestamp):
        if not args.stats:
            print(f"[POINTCLOUD] {len(points)} points at {timestamp:.3f}")
            if len(points) > 0:
                print(f"  Point range: X[{points[:,0].min():.1f}, {points[:,0].max():.1f}] "
                      f"Y[{points[:,1].min():.1f}, {points[:,1].max():.1f}] "
                      f"Z[{points[:,2].min():.1f}, {points[:,2].max():.1f}]")
    
    def on_connection(connected):
        if connected:
            print("[STATUS] Connected to server")
        else:
            print("[STATUS] Disconnected from server")
    
    client.on_pointcloud_received = on_pointcloud
    client.on_connection_changed = on_connection
    
    # Connect
    if not client.connect():
        print("[ERROR] Failed to connect to server")
        return 1
    
    try:
        # Run until interrupted
        print("[INFO] Receiving point clouds... Press Ctrl+C to stop")
        
        if args.stats:
            # Print statistics periodically
            while client.connected:
                time.sleep(1.0)
                stats = client.get_statistics()
                print(f"[STATS] FPS: {stats['fps']:.1f}, "
                      f"Received: {stats['bytes_received']/1024/1024:.1f} MB, "
                      f"Connected: {stats['connected']}")
        else:
            # Just keep running
            while client.connected:
                time.sleep(0.1)
        
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    finally:
        client.disconnect()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
