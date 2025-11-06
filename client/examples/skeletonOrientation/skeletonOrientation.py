#!/usr/bin/env python3
"""
Example: Show TouchDesigner-ready orientation data

Press SPACE: Show SDK local orientations as Euler angles (ready for TouchDesigner)
Press '1': Show raw SDK quaternions
Press '2': Show bone-aligned orientations (alternative approach)
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
        # Store T-pose reference (captured with 'T' key)
        self.tpose_reference = None
    
    def _quat_multiply(self, q1, q2):
        """Multiply two quaternions: q1 * q2"""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return [
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ]
    
    def _quat_inverse(self, q):
        """Compute quaternion inverse (conjugate for unit quaternions)"""
        x, y, z, w = q
        return [-x, -y, -z, w]
    
    def keyPressEvent(self, event):
        """Handle keyboard input"""
        if event.key() == Qt.Key_Space:
            # TOUCHDESIGNER OUTPUT: SDK local orientations as Euler angles
            if hasattr(self, 'latest_frame') and self.latest_frame and hasattr(self.latest_frame, 'people'):
                print("\n" + "="*80)
                print("TOUCHDESIGNER OUTPUT - SDK Local Orientations (Euler XYZ)")
                print("="*80)
                print("\nThis is what you should send to TouchDesigner:")
                print("- SDK provides local orientations (relative to parent)")
                print("- Already close to zero in T-pose")
                print("- Just normalize quaternions and convert to Euler")
                print()
                
                for person_idx, person in enumerate(self.latest_frame.people):
                    print(f"Person {person_idx}:")
                    skeleton = person.skeleton if hasattr(person, 'skeleton') else person.get('skeleton')
                    if not skeleton:
                        print("  No skeleton data")
                        continue
                    
                    for i, joint in enumerate(skeleton):
                        ori = joint.ori if hasattr(joint, 'ori') else joint.get('ori')
                        if not ori:
                            continue
                        
                        # Get SDK local orientation
                        if hasattr(ori, 'x'):
                            quat = [ori.x, ori.y, ori.z, ori.w]
                        else:
                            quat = [ori["x"], ori["y"], ori["z"], ori["w"]]
                        
                        # Normalize (Unity does this)
                        import math
                        length = math.sqrt(quat[0]**2 + quat[1]**2 + quat[2]**2 + quat[3]**2)
                        if length > 0.0001:
                            quat = [quat[0]/length, quat[1]/length, quat[2]/length, quat[3]/length]
                        
                        # Convert to Euler
                        euler = quaternion_to_euler(quat, order='XYZ')
                        
                        # Format for TouchDesigner (joint index, euler angles)
                        print(f"  Joint {i:2d}: euler_x={euler[0]:7.2f}°, euler_y={euler[1]:7.2f}°, euler_z={euler[2]:7.2f}°")
                
                print("\n" + "="*80)
                print("SEND THIS DATA TO TOUCHDESIGNER:")
                print("For each joint: joint_index, euler_x, euler_y, euler_z")
                print("="*80 + "\n")
            else:
                print("[WARNING] No skeleton data available")
        
        elif event.key() == Qt.Key_1:
            # Option 1: Show raw SDK quaternions
            if hasattr(self, 'latest_frame') and self.latest_frame and hasattr(self.latest_frame, 'people'):
                print("\n" + "="*80)
                print("RAW SDK LOCAL ORIENTATIONS (Quaternions)")
                print("="*80)
                
                for person_idx, person in enumerate(self.latest_frame.people):
                    print(f"\nPerson {person_idx}:")
                    skeleton = person.skeleton if hasattr(person, 'skeleton') else person.get('skeleton')
                    if not skeleton:
                        print("  No skeleton data")
                        continue
                    
                    for i, joint in enumerate(skeleton):
                        ori = joint.ori if hasattr(joint, 'ori') else joint.get('ori')
                        if not ori:
                            continue
                        
                        if hasattr(ori, 'x'):
                            quat = [ori.x, ori.y, ori.z, ori.w]
                        else:
                            quat = [ori["x"], ori["y"], ori["z"], ori["w"]]
                        
                        print(f"  Joint {i:2d}: quat=[{quat[0]:7.4f}, {quat[1]:7.4f}, {quat[2]:7.4f}, {quat[3]:7.4f}]")
                
                print("="*80 + "\n")
            else:
                print("[WARNING] No skeleton data available")
        
        elif event.key() == Qt.Key_2:
            # Option 2: Show bone-aligned local orientations
            if hasattr(self, 'latest_frame') and self.latest_frame and hasattr(self.latest_frame, 'people'):
                print("\n" + "="*80)
                print("OPTION 2: BONE-ALIGNED LOCAL ORIENTATIONS (Y along bone)")
                print("="*80)
                
                for person_idx, person in enumerate(self.latest_frame.people):
                    print(f"\nPerson {person_idx}:")
                    skeleton = person.skeleton if hasattr(person, 'skeleton') else person.get('skeleton')
                    if not skeleton:
                        print("  No skeleton data")
                        continue
                    
                    # Compute bone-aligned orientations
                    local_orientations = compute_bone_aligned_local_orientations(skeleton)
                    
                    for joint_idx, quat in sorted(local_orientations.items()):
                        euler = quaternion_to_euler(quat, order='XYZ')
                        print(f"  Joint {joint_idx:2d}: Euler = [{euler[0]:7.2f}°, {euler[1]:7.2f}°, {euler[2]:7.2f}°]")
                
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
