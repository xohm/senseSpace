#!/usr/bin/env python3
"""
Video Recording Example for senseSpace

This example demonstrates how to record skeleton data, point clouds, AND video streams
into a single .ssrec file. The video is captured directly from the H.265-encoded
GStreamer pipeline with zero re-encoding overhead.

Requirements:
- senseSpace server running with video streaming enabled
- GStreamer with H.265 support

Usage:
    python video_recording_example.py --server-ip 127.0.0.1 --duration 60

Controls:
    Press 'r' to start recording
    Press 's' to stop recording
    Press 'q' to quit
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'libs'))

from senseSpaceLib.senseSpace import SenseSpaceClient, FrameRecorder
from senseSpaceLib.senseSpace.video_streaming import MultiCameraVideoStreamer


class VideoRecordingClient:
    """Client that records skeleton, point cloud, and video data"""
    
    def __init__(self, server_ip='127.0.0.1', 
                 skeleton_port=9999,
                 pointcloud_port=9998,
                 video_port=5000):
        """
        Initialize video recording client.
        
        Args:
            server_ip: Server IP address
            skeleton_port: Skeleton data port
            pointcloud_port: Point cloud data port
            video_port: Video streaming port
        """
        self.server_ip = server_ip
        self.skeleton_port = skeleton_port
        self.pointcloud_port = pointcloud_port
        self.video_port = video_port
        
        # Initialize clients
        self.skeleton_client = SenseSpaceClient(
            server_ip=server_ip,
            server_port=skeleton_port
        )
        
        # Point cloud client (optional)
        try:
            from senseSpaceLib.senseSpace import PointCloudClient
            self.pc_client = PointCloudClient(
                server_ip=server_ip,
                server_port=pointcloud_port
            )
        except ImportError:
            print("[WARNING] PointCloudClient not available, skipping point clouds")
            self.pc_client = None
        
        # Video streamer (we'll receive video from this)
        self.video_streamer = None
        
        # Recording state
        self.recorder = None
        self.recording = False
        self.record_pointclouds = True
        self.record_video = True
        
    def start_recording(self):
        """Start recording with video"""
        if self.recording:
            print("\n[WARNING] Already recording")
            return
        
        # Create filename with timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp_str}.ssrec"
        
        # Create recorder with video enabled
        self.recorder = FrameRecorder(
            filename=filename,
            record_pointclouds=self.record_pointclouds and self.pc_client is not None,
            record_video=self.record_video
        )
        
        # Start recording
        self.recorder.start_recording()
        self.recording = True
        
        # Enable video recording on streamer if available
        if self.video_streamer and self.record_video:
            self.video_streamer.enable_recording(self.recorder)
            print(f"[RECORDING] Video recording enabled")
        
        print(f"\n[RECORDING] Started recording to: {filename}")
        if self.record_pointclouds:
            print("[RECORDING] Point cloud recording enabled")
        if self.record_video:
            print("[RECORDING] Video recording enabled (H.265)")
        
    def stop_recording(self):
        """Stop recording"""
        if not self.recording:
            print("\n[WARNING] Not recording")
            return
        
        # Disable video recording on streamer
        if self.video_streamer and self.record_video:
            self.video_streamer.disable_recording()
        
        # Stop and close recorder
        if self.recorder:
            self.recorder.stop_recording()
            
            # Print statistics
            print(f"\n[RECORDING] Statistics:")
            print(f"  Frames recorded: {self.recorder.frame_count}")
            print(f"  Total size: {self.recorder.total_bytes_written / 1024 / 1024:.2f} MB")
            if self.recorder.total_pointcloud_points > 0:
                print(f"  Point cloud points: {self.recorder.total_pointcloud_points:,}")
            if self.recorder.total_video_bytes > 0:
                print(f"  Video data: {self.recorder.total_video_bytes / 1024 / 1024:.2f} MB")
                num_cameras = len(self.recorder.video_cameras)
                print(f"  Cameras recorded: {num_cameras}")
            
            self.recorder.close()
            self.recorder = None
        
        self.recording = False
        print("\n[RECORDING] Stopped recording")
    
    def setup_video_receiver(self, camera_serials, rgb_width=1280, rgb_height=720,
                            depth_width=640, depth_height=480, framerate=30):
        """
        Setup video receiver (if recording from server's video stream).
        
        Note: This is needed if you want to capture the video stream that the
        server is already streaming. If you're running this on the server side,
        you can directly hook into the MultiCameraVideoStreamer instance.
        
        Args:
            camera_serials: List of camera serial numbers
            rgb_width: RGB frame width
            rgb_height: RGB frame height
            depth_width: Depth frame width
            depth_height: Depth frame height
            framerate: Streaming framerate
        """
        try:
            self.video_streamer = MultiCameraVideoStreamer(
                camera_serials=camera_serials,
                server_ip=self.server_ip,
                server_port=self.video_port,
                rgb_width=rgb_width,
                rgb_height=rgb_height,
                depth_width=depth_width,
                depth_height=depth_height,
                framerate=framerate
            )
            print(f"[INFO] Video receiver initialized for {len(camera_serials)} cameras")
        except Exception as e:
            print(f"[WARNING] Failed to setup video receiver: {e}")
            self.video_streamer = None
            self.record_video = False
    
    def run(self, duration=None):
        """
        Run the recording client.
        
        Args:
            duration: Recording duration in seconds (None for manual control)
        """
        print(f"[INFO] Connecting to senseSpace server at {self.server_ip}...")
        
        # Connect skeleton client
        if not self.skeleton_client.connect():
            print("[ERROR] Failed to connect to skeleton server")
            return
        
        print("[INFO] Connected to skeleton server")
        
        # Connect point cloud client if available
        if self.pc_client:
            print(f"[INFO] Connecting to point cloud server...")
            if not self.pc_client.connect():
                print("[WARNING] Failed to connect to point cloud server")
                self.pc_client = None
                self.record_pointclouds = False
            else:
                print("[INFO] Connected to point cloud server")
        
        # Start video streamer if configured
        if self.video_streamer:
            print("[INFO] Starting video receiver...")
            self.video_streamer.start()
        
        print("\n[INFO] Ready to record!")
        print("Controls:")
        print("  Press 'r' to start recording")
        print("  Press 's' to stop recording")
        print("  Press 'q' to quit")
        
        # Auto-start if duration specified
        if duration:
            print(f"\n[INFO] Auto-recording for {duration} seconds...")
            self.start_recording()
            start_time = time.time()
        
        frame_count = 0
        try:
            while True:
                # Get skeleton data
                skeleton_data = self.skeleton_client.get_skeleton_data()
                if skeleton_data is None:
                    time.sleep(0.01)
                    continue
                
                # Get point cloud data if enabled
                pointcloud_data = None
                if self.pc_client and self.record_pointclouds:
                    pointcloud_data = self.pc_client.get_pointcloud_data()
                
                # Record frame if recording
                if self.recording and self.recorder:
                    self.recorder.record_frame(
                        skeleton_data=skeleton_data,
                        pointcloud_data=pointcloud_data
                    )
                    frame_count += 1
                    
                    # Print progress every 30 frames (~1 second at 30fps)
                    if frame_count % 30 == 0:
                        print(f"\r[RECORDING] Frames: {frame_count}", end='', flush=True)
                
                # Check duration
                if duration and self.recording:
                    if time.time() - start_time >= duration:
                        print(f"\n[INFO] Recording duration reached ({duration}s)")
                        self.stop_recording()
                        break
                
                # Small delay to avoid CPU overload
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        finally:
            # Cleanup
            if self.recording:
                self.stop_recording()
            
            if self.video_streamer:
                self.video_streamer.stop()
                self.video_streamer.shutdown()
            
            self.skeleton_client.close()
            if self.pc_client:
                self.pc_client.close()
            
            print("[INFO] Cleanup complete")


def main():
    parser = argparse.ArgumentParser(description='Record skeleton, point cloud, and video data')
    parser.add_argument('--server-ip', default='127.0.0.1', help='Server IP address')
    parser.add_argument('--skeleton-port', type=int, default=9999, help='Skeleton data port')
    parser.add_argument('--pointcloud-port', type=int, default=9998, help='Point cloud data port')
    parser.add_argument('--video-port', type=int, default=5000, help='Video streaming port')
    parser.add_argument('--duration', type=int, help='Auto-record for N seconds (optional)')
    parser.add_argument('--no-pointcloud', action='store_true', help='Disable point cloud recording')
    parser.add_argument('--no-video', action='store_true', help='Disable video recording')
    
    # Video configuration
    parser.add_argument('--cameras', nargs='+', help='Camera serial numbers (for video)')
    parser.add_argument('--rgb-width', type=int, default=1280, help='RGB frame width')
    parser.add_argument('--rgb-height', type=int, default=720, help='RGB frame height')
    parser.add_argument('--depth-width', type=int, default=640, help='Depth frame width')
    parser.add_argument('--depth-height', type=int, default=480, help='Depth frame height')
    parser.add_argument('--framerate', type=int, default=30, help='Video framerate')
    
    args = parser.parse_args()
    
    # Create client
    client = VideoRecordingClient(
        server_ip=args.server_ip,
        skeleton_port=args.skeleton_port,
        pointcloud_port=args.pointcloud_port,
        video_port=args.video_port
    )
    
    # Configure recording options
    client.record_pointclouds = not args.no_pointcloud
    client.record_video = not args.no_video
    
    # Setup video receiver if cameras specified
    if args.cameras and not args.no_video:
        client.setup_video_receiver(
            camera_serials=args.cameras,
            rgb_width=args.rgb_width,
            rgb_height=args.rgb_height,
            depth_width=args.depth_width,
            depth_height=args.depth_height,
            framerate=args.framerate
        )
    elif not args.no_video:
        print("\n[WARNING] No cameras specified, video recording disabled")
        print("Use --cameras to specify camera serial numbers")
        client.record_video = False
    
    # Run client
    client.run(duration=args.duration)


if __name__ == '__main__':
    main()
