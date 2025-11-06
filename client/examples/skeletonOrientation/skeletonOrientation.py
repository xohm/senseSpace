#!/usr/bin/env python3
"""
Example: Print bone-aligned joint orientations
Press SPACE to print bone-aligned orientations for all skeletons
"""

import argparse
import sys
import os

# Ensure local 'libs' folder is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
libs_path = os.path.join(repo_root, 'libs')
if os.path.isdir(libs_path) and libs_path not in sys.path:
    sys.path.insert(0, libs_path)

# Import from shared library
from senseSpaceLib.senseSpace.vizClient import VisualizationClient
from senseSpaceLib.senseSpace.protocol import Frame
from senseSpaceLib.senseSpace.vizWidget import SkeletonGLWidget
from senseSpaceLib.senseSpace.visualization import (
    compute_bone_aligned_local_orientations,
    quaternion_to_euler
)

from PyQt5.QtCore import Qt


class OrientationWidget(SkeletonGLWidget):
    """Minimal widget that prints orientations on SPACE key"""
    
    def __init__(self, parent=None):
        """Initialize with orientation axes visible by default"""
        super().__init__(parent)
        # Enable orientation visualization by default
        self.show_joint_orientation = True
    
    def keyPressEvent(self, event):
        """Handle keyboard input"""
        if event.key() == Qt.Key_Space:
            # Print orientations for all skeletons
            if hasattr(self, 'latest_frame') and self.latest_frame and hasattr(self.latest_frame, 'people'):
                print("\n" + "="*80)
                print("BONE-ALIGNED JOINT ORIENTATIONS")
                print("="*80)
                
                for person_idx, person in enumerate(self.latest_frame.people):
                    print(f"\nPerson {person_idx}:")
                    
                    # Get skeleton data
                    skeleton = person.skeleton if hasattr(person, 'skeleton') else person.get('skeleton')
                    if not skeleton:
                        print("  No skeleton data")
                        continue
                    
                    # Compute bone-aligned local orientations
                    local_orientations = compute_bone_aligned_local_orientations(skeleton)
                    
                    # Print each joint's orientation
                    for joint_idx, quat in sorted(local_orientations.items()):
                        # Convert to Euler angles
                        euler = quaternion_to_euler(quat, order='XYZ')
                        
                        print(f"  Joint {joint_idx:2d}: "
                              f"Quat=[{quat[0]:7.4f}, {quat[1]:7.4f}, {quat[2]:7.4f}, {quat[3]:7.4f}]  "
                              f"Euler=[{euler[0]:7.2f}°, {euler[1]:7.2f}°, {euler[2]:7.2f}°]")
                
                print("="*80 + "\n")
            else:
                print("[WARNING] No skeleton data available")
        
        else:
            # Pass other keys to parent
            super().keyPressEvent(event)
    
    def draw_custom(self, frame: Frame):
        """Store latest frame for orientation computation"""
        self.latest_frame = frame


def main():
    parser = argparse.ArgumentParser(description="SenseSpace Orientation Example")
    parser.add_argument("--server", "-s", default="localhost", help="Server IP address")
    parser.add_argument("--port", "-p", type=int, default=12345, help="Server port")
    parser.add_argument("--rec", "-r", type=str, default=None, help="Path to recording file")
    
    args = parser.parse_args()
    
    # Create and run visualization client
    client = VisualizationClient(
        viewer_class=OrientationWidget,
        server_ip=args.server,
        server_port=args.port,
        playback_file=args.rec,
        window_title="Skeleton Orientations - Press SPACE to print orientations"
    )
    
    success = client.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
