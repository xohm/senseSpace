#!/usr/bin/env python3
"""
Example: Recording Skeleton + Point Cloud Data Together

This example shows how to record both skeleton frames and per-person point cloud data
simultaneously. The recording uses streaming compression for memory efficiency.

Requirements:
- senseSpace server running with skeleton streaming
- Point cloud server running (if recording point clouds)
- zstandard package installed for compression

Usage:
    # Record skeleton + point clouds:
    python pointcloud_recording_example.py --server 192.168.1.100 --record-pc
    
    # Record skeleton only:
    python pointcloud_recording_example.py --server 192.168.1.100
    
    # Press 'R' to start/stop recording during session
"""

import sys
import os
import argparse
import time
from datetime import datetime

# Add parent directory to path to import senseSpace library
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'libs'))

from senseSpaceLib.senseSpace import SenseSpaceClient, FrameRecorder
from senseSpaceLib.senseSpace.protocol import Frame

# Import point cloud client from client directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from pointCloudClient import PointCloudClient


class PointCloudRecordingClient:
    """Client that records both skeleton and point cloud data"""
    
    def __init__(self, server_ip: str, server_port: int = 12345, 
                 pc_port: int = 12346, record_pointcloud: bool = False):
        """
        Initialize recording client.
        
        Args:
            server_ip: Server IP address
            server_port: Skeleton server port (default: 12345)
            pc_port: Point cloud server port (default: 12346)
            record_pointcloud: Whether to record point cloud data
        """
        self.server_ip = server_ip
        self.record_pointcloud = record_pointcloud
        
        # Skeleton client
        self.client = SenseSpaceClient(server_ip, server_port)
        self.client.set_frame_callback(self._on_frame)
        
        # Point cloud client (optional)
        self.pc_client = None
        if record_pointcloud:
            self.pc_client = PointCloudClient(server_ip, pc_port)
            self.pc_client.on_pointcloud_perperson_received = self._on_pointcloud_perperson
        
        # Recorder
        self.recorder = None
        self.recording = False
        
        # Store latest point cloud for synchronization
        self._latest_pointcloud_data = None
        self._latest_pc_timestamp = 0.0
        
    def _on_frame(self, frame: Frame):
        """Handle skeleton frame"""
        # Record frame with point cloud data if recording
        if self.recorder and self.recorder.is_recording():
            # Use point cloud data if available and recent (within 100ms)
            pointcloud_data = None
            if self._latest_pointcloud_data is not None:
                time_diff = abs(frame.timestamp - self._latest_pc_timestamp)
                if time_diff < 0.1:  # 100ms threshold
                    pointcloud_data = self._latest_pointcloud_data
            
            self.recorder.record_frame(frame, pointcloud_data)
        
        # Print frame info
        print(f"\r[FRAME] Persons: {len(frame.persons)} | Recording: {self.recording}", end='', flush=True)
    
    def _on_pointcloud_perperson(self, pointcloud_data: list, timestamp: float):
        """Handle per-person point cloud data"""
        # Store for synchronization with skeleton frames
        self._latest_pointcloud_data = pointcloud_data
        self._latest_pc_timestamp = timestamp
    
    def start_recording(self):
        """Start recording"""
        if self.recording:
            print("\n[WARNING] Already recording")
            return
        
        # Create filename with timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp_str}.ssrec"
        
        # Create recorder
        self.recorder = FrameRecorder(filename, record_pointcloud=self.record_pointcloud)
        self.recorder.start()
        self.recording = True
        
        print(f"\n[RECORDING] Started recording to: {filename}")
        if self.record_pointcloud:
            print("[RECORDING] Recording point clouds enabled")
    
    def stop_recording(self):
        """Stop recording"""
        if not self.recording:
            print("\n[WARNING] Not recording")
            return
        
        if self.recorder:
            self.recorder.stop()
            self.recorder = None
        
        self.recording = False
        print("\n[RECORDING] Stopped recording")
    
    def run(self):
        """Run the client"""
        print(f"[INFO] Connecting to senseSpace server at {self.server_ip}...")
        
        # Connect skeleton client
        if not self.client.connect():
            print("[ERROR] Failed to connect to skeleton server")
            return
        
        print("[INFO] Connected to skeleton server")
        
        # Connect point cloud client if enabled
        if self.pc_client:
            print(f"[INFO] Connecting to point cloud server...")
            if not self.pc_client.connect():
                print("[WARNING] Failed to connect to point cloud server")
                print("[INFO] Continuing with skeleton-only recording")
                self.pc_client = None
                self.record_pointcloud = False
            else:
                print("[INFO] Connected to point cloud server")
        
        print("\n" + "="*60)
        print("RECORDING CLIENT")
        print("="*60)
        print("Commands:")
        print("  R - Toggle recording on/off")
        print("  Q - Quit")
        print("="*60 + "\n")
        
        # Main loop
        try:
            while True:
                # Check for keyboard input (simple version - requires Enter key)
                import select
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.readline().strip().upper()
                    
                    if key == 'R':
                        if self.recording:
                            self.stop_recording()
                        else:
                            self.start_recording()
                    elif key == 'Q':
                        break
                
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        
        finally:
            # Stop recording if active
            if self.recording:
                self.stop_recording()
            
            # Disconnect clients
            print("[INFO] Disconnecting...")
            self.client.disconnect()
            if self.pc_client:
                self.pc_client.disconnect()
            
            print("[INFO] Shutdown complete")


def main():
    parser = argparse.ArgumentParser(description='Record skeleton and point cloud data')
    parser.add_argument('--server', type=str, required=True,
                       help='Server IP address')
    parser.add_argument('--port', type=int, default=12345,
                       help='Skeleton server port (default: 12345)')
    parser.add_argument('--pc-port', type=int, default=12346,
                       help='Point cloud server port (default: 12346)')
    parser.add_argument('--record-pc', action='store_true',
                       help='Record point cloud data (requires point cloud server)')
    
    args = parser.parse_args()
    
    client = PointCloudRecordingClient(
        args.server, 
        args.port, 
        args.pc_port,
        args.record_pc
    )
    
    client.run()


if __name__ == "__main__":
    main()
