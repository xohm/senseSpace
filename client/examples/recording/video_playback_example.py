#!/usr/bin/env python3
"""
Video Playback Example for senseSpace

This example demonstrates how to play back .ssrec files with video data,
displaying the decoded video frames alongside skeleton and point cloud data.

Requirements:
- .ssrec file with video data (version 2.0)
- GStreamer with H.265 decoder support
- OpenCV for video display (optional)

Usage:
    python video_playback_example.py recordings/my_session.ssrec
    python video_playback_example.py recordings/my_session.ssrec --show-video
    python video_playback_example.py recordings/my_session.ssrec --export-video output/
"""

import argparse
import sys
import time
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'libs'))

from senseSpaceLib.senseSpace import FramePlayer


class VideoPlaybackClient:
    """Client for playing back recordings with video"""
    
    def __init__(self, filename, show_video=False, export_video=None):
        """
        Initialize video playback client.
        
        Args:
            filename: Path to .ssrec file
            show_video: If True, display video using OpenCV
            export_video: If set, export video frames to this directory
        """
        self.filename = filename
        self.show_video = show_video
        self.export_video = export_video
        
        # Create player
        self.player = FramePlayer(filename=filename)
        
        # Video state
        self.video_frames = {}  # Store latest frames per camera
        self.frame_count = 0
        
        # Export state
        if export_video:
            self.export_dir = Path(export_video)
            self.export_dir.mkdir(parents=True, exist_ok=True)
            self.video_writers = {}  # OpenCV VideoWriter per camera
        
        # Set video callback
        self.player.set_video_callback(self._on_video_frame)
        
    def _on_video_frame(self, camera_idx, rgb_frame, depth_frame):
        """
        Callback for decoded video frames.
        
        Args:
            camera_idx: Camera index
            rgb_frame: numpy array (H, W, 3) BGR uint8
            depth_frame: numpy array (H, W) float32 millimeters
        """
        # Store frames
        self.video_frames[camera_idx] = {
            'rgb': rgb_frame,
            'depth': depth_frame
        }
        
        # Display if enabled
        if self.show_video and rgb_frame is not None:
            # Show RGB
            cv2.imshow(f"Camera {camera_idx} - RGB", rgb_frame)
            
            # Show depth (normalize for display)
            if depth_frame is not None:
                # Normalize depth to 0-255 range for visualization
                depth_display = depth_frame.copy()
                if depth_display.max() > 0:
                    depth_display = (depth_display / depth_display.max() * 255).astype(np.uint8)
                    depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                    cv2.imshow(f"Camera {camera_idx} - Depth", depth_display)
        
        # Export if enabled
        if self.export_video and rgb_frame is not None:
            self._export_frame(camera_idx, rgb_frame, depth_frame)
    
    def _export_frame(self, camera_idx, rgb_frame, depth_frame):
        """Export video frame to file"""
        # Create video writer if needed
        if camera_idx not in self.video_writers:
            h, w = rgb_frame.shape[:2]
            fps = self.player.header.get('framerate', 30)
            
            # RGB video
            rgb_path = self.export_dir / f"camera_{camera_idx}_rgb.mp4"
            self.video_writers[f'{camera_idx}_rgb'] = cv2.VideoWriter(
                str(rgb_path),
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (w, h)
            )
            print(f"[EXPORT] Created RGB video: {rgb_path}")
            
            # Depth video (if available)
            if depth_frame is not None:
                depth_path = self.export_dir / f"camera_{camera_idx}_depth.mp4"
                self.video_writers[f'{camera_idx}_depth'] = cv2.VideoWriter(
                    str(depth_path),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (w, h)
                )
                print(f"[EXPORT] Created depth video: {depth_path}")
        
        # Write RGB frame
        rgb_writer = self.video_writers.get(f'{camera_idx}_rgb')
        if rgb_writer:
            rgb_writer.write(rgb_frame)
        
        # Write depth frame (convert to color for video)
        if depth_frame is not None:
            depth_writer = self.video_writers.get(f'{camera_idx}_depth')
            if depth_writer:
                # Normalize and colorize depth
                depth_display = depth_frame.copy()
                if depth_display.max() > 0:
                    depth_display = (depth_display / depth_display.max() * 255).astype(np.uint8)
                    depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                    depth_writer.write(depth_display)
    
    def print_info(self):
        """Print recording information"""
        print(f"\n{'='*60}")
        print(f"Recording Info: {self.filename}")
        print(f"{'='*60}")
        
        header = self.player.header
        print(f"Version: {header.get('version', 'Unknown')}")
        print(f"Timestamp: {header.get('timestamp', 'Unknown')}")
        print(f"Framerate: {header.get('framerate', 'Unknown')} fps")
        print(f"Has point clouds: {header.get('has_pointcloud', False)}")
        print(f"Has video: {header.get('has_video', False)}")
        
        if header.get('has_video'):
            print(f"\nVideo Cameras:")
            for cam in header.get('video_cameras', []):
                print(f"  Camera {cam['camera_idx']}:")
                print(f"    Resolution: {cam['width']}x{cam['height']}")
                print(f"    FPS: {cam['fps']}")
                print(f"    Codec: {cam['codec']}")
        
        print(f"{'='*60}\n")
    
    def play(self, loop=False, speed=1.0):
        """
        Play back the recording.
        
        Args:
            loop: If True, loop playback
            speed: Playback speed multiplier (1.0 = normal speed)
        """
        self.print_info()
        
        # Calculate frame delay based on framerate and speed
        framerate = self.player.header.get('framerate', 30)
        frame_delay = (1.0 / framerate) / speed
        
        print("[INFO] Starting playback...")
        print("Controls:")
        print("  Press SPACE to pause/resume")
        print("  Press 'q' to quit")
        print()
        
        paused = False
        
        try:
            while True:
                # Start playback
                self.player.play()
                
                # Read frames
                for skeleton_data, pointcloud_data in self.player.get_next_frame():
                    self.frame_count += 1
                    
                    # Print progress
                    if self.frame_count % 30 == 0:
                        print(f"\r[PLAYBACK] Frame: {self.frame_count}", end='', flush=True)
                    
                    # Handle video display
                    if self.show_video:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            print("\n[INFO] Quit requested")
                            return
                        elif key == ord(' '):
                            paused = not paused
                            if paused:
                                print("\n[PLAYBACK] Paused")
                            else:
                                print("\n[PLAYBACK] Resumed")
                    
                    # Pause handling
                    while paused:
                        if self.show_video:
                            key = cv2.waitKey(100) & 0xFF
                            if key == ord(' '):
                                paused = False
                                print("\n[PLAYBACK] Resumed")
                            elif key == ord('q'):
                                print("\n[INFO] Quit requested")
                                return
                        else:
                            time.sleep(0.1)
                    
                    # Frame rate control
                    time.sleep(frame_delay)
                
                print(f"\n[PLAYBACK] Playback complete ({self.frame_count} frames)")
                
                # Loop if enabled
                if loop:
                    print("[PLAYBACK] Looping...")
                    self.frame_count = 0
                    self.player.close()
                    self.player = FramePlayer(filename=self.filename)
                    self.player.set_video_callback(self._on_video_frame)
                else:
                    break
                    
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        # Close video writers
        if hasattr(self, 'video_writers'):
            for writer in self.video_writers.values():
                writer.release()
        
        # Close OpenCV windows
        if self.show_video:
            cv2.destroyAllWindows()
        
        # Close player
        self.player.close()
        
        print("[INFO] Cleanup complete")


def main():
    parser = argparse.ArgumentParser(description='Play back .ssrec recordings with video')
    parser.add_argument('filename', help='Path to .ssrec file')
    parser.add_argument('--show-video', action='store_true', help='Display video using OpenCV')
    parser.add_argument('--export-video', help='Export video to directory (MP4 files)')
    parser.add_argument('--loop', action='store_true', help='Loop playback')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed (1.0 = normal)')
    
    args = parser.parse_args()
    
    # Check file exists
    if not Path(args.filename).exists():
        print(f"[ERROR] File not found: {args.filename}")
        return 1
    
    # Create playback client
    client = VideoPlaybackClient(
        filename=args.filename,
        show_video=args.show_video,
        export_video=args.export_video
    )
    
    # Play
    client.play(loop=args.loop, speed=args.speed)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
