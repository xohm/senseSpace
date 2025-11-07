#!/usr/bin/env python3
"""
Example: TouchDesigner-ready orientation data with GEOMETRIC FIX for shoulders

This version ACTUALLY FIXES shoulder flipping by reconstructing orientations
from bone directions + stable reference vectors (torso up).

KEYBOARD CONTROLS:
Press SPACE: Show orientations as Euler angles (TouchDesigner-ready)
Press 'O': Toggle orientation axes visualization
Press 'B': Cycle blend factor (0.0=pure geometric, 0.3=default, 1.0=pure SDK)
Press '+/-': Adjust temporal smoothing

WHAT'S DIFFERENT:
- Shoulders (joints 7, 11) are reconstructed from shoulder→elbow direction + torso up
- Elbows (joints 8, 12) are reconstructed from elbow→wrist direction + torso up
- Other joints use SDK orientations
- Result: Shoulders NEVER flip because orientation is geometrically constrained
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
from senseSpaceLib.senseSpace.visualization import quaternion_to_euler
from senseSpaceLib.senseSpace.geometric_orientation import GeometricOrientationReconstructor
import numpy as np

from PyQt5.QtCore import Qt


class FixedOrientationWidget(SkeletonGLWidget):
    """Widget that shows geometrically-reconstructed orientations"""
    
    def __init__(self, parent=None):
        """Initialize with geometric reconstructor"""
        super().__init__(parent)
        self.show_joint_orientation = True
        self.show_orientation = True
        
        # Geometric reconstructor - this is the key!
        self.reconstructor = GeometricOrientationReconstructor(
            blend_factor=0.3,  # 30% SDK, 70% geometric
            use_filtering=True,
            filter_smoothing=0.2
        )
        
        # Cache for orientations
        self.latest_frame = None
        self.fixed_orientations = {}
    
    def keyPressEvent(self, event):
        """Handle keyboard input"""
        if event.key() == Qt.Key_Space:
            # Print orientations as Euler angles for TouchDesigner
            if hasattr(self, 'latest_frame') and self.latest_frame and hasattr(self.latest_frame, 'people'):
                print("\n" + "="*80)
                print("TOUCHDESIGNER OUTPUT - Geometrically Fixed Orientations")
                print("="*80)
                print(f"Blend factor: {self.reconstructor.blend_factor:.1f}")
                print(f"  0.0 = pure geometric (most stable)")
                print(f"  0.3 = default (stable + motion nuance)")
                print(f"  1.0 = pure SDK (original flipping)")
                print()
                
                for person_idx, person in enumerate(self.latest_frame.people):
                    print(f"Person {person_idx}:")
                    skeleton = person.skeleton if hasattr(person, 'skeleton') else person.get('skeleton')
                    if not skeleton:
                        print("  No skeleton data")
                        continue
                    
                    if person_idx in self.fixed_orientations:
                        orientations = self.fixed_orientations[person_idx]
                        
                        for joint_idx in sorted(orientations.keys()):
                            quat = orientations[joint_idx]
                            
                            # Convert numpy array to list if needed
                            if hasattr(quat, 'tolist'):
                                quat = quat.tolist()
                            
                            euler = quaternion_to_euler(quat, order='XYZ')
                            
                            # Mark which joints are geometrically fixed
                            if joint_idx in [7, 8, 11, 12]:  # Shoulders & elbows
                                marker = " ← GEOMETRIC FIX"
                            else:
                                marker = ""
                            
                            print(f"  Joint {joint_idx:2d}: rx={euler[0]:7.2f}°, ry={euler[1]:7.2f}°, rz={euler[2]:7.2f}°{marker}")
                
                print("\n" + "="*80)
                print("SEND THIS TO TOUCHDESIGNER (will NOT flip!)")
                print("="*80 + "\n")
            else:
                print("[WARNING] No skeleton data available")
        
        elif event.key() == Qt.Key_B:
            # Cycle blend factor
            current = self.reconstructor.blend_factor
            if current < 0.2:
                new_blend = 0.3
            elif current < 0.4:
                new_blend = 0.5
            elif current < 0.6:
                new_blend = 0.7
            elif current < 0.9:
                new_blend = 1.0
            else:
                new_blend = 0.0
            
            self.reconstructor.set_blend_factor(new_blend)
            print(f"\n[INFO] Blend factor set to {new_blend:.1f}")
            print(f"       (0.0=pure geometric, 1.0=pure SDK)")
            print(f"       Reset reconstructor...")
            self.reconstructor.reset()
        
        elif event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            # Increase smoothing
            new_smoothing = min(1.0, self.reconstructor.filter_smoothing + 0.1)
            self.reconstructor.set_filter_smoothing(new_smoothing)
            print(f"\n[INFO] Smoothing increased to {new_smoothing:.2f}")
        
        elif event.key() == Qt.Key_Minus or event.key() == Qt.Key_Underscore:
            # Decrease smoothing
            new_smoothing = max(0.0, self.reconstructor.filter_smoothing - 0.1)
            self.reconstructor.set_filter_smoothing(new_smoothing)
            print(f"\n[INFO] Smoothing decreased to {new_smoothing:.2f}")
        
        else:
            # Pass other keys to parent
            super().keyPressEvent(event)
    
    def draw_custom(self, frame: Frame):
        """Compute geometrically-fixed orientations each frame"""
        self.latest_frame = frame
        
        if hasattr(frame, 'people') and frame.people:
            self.fixed_orientations = {}
            
            for person_idx, person in enumerate(frame.people):
                skeleton = person.skeleton if hasattr(person, 'skeleton') else person.get('skeleton')
                if not skeleton:
                    continue
                
                # Get SDK orientations
                sdk_orientations = {}
                for joint_idx, joint in enumerate(skeleton):
                    ori = joint.ori if hasattr(joint, 'ori') else joint.get('ori')
                    if ori:
                        if hasattr(ori, 'x'):
                            sdk_orientations[joint_idx] = [ori.x, ori.y, ori.z, ori.w]
                        else:
                            sdk_orientations[joint_idx] = [ori["x"], ori["y"], ori["z"], ori["w"]]
                
                # Reconstruct shoulders & elbows geometrically
                fixed = self.reconstructor.reconstruct_skeleton_orientations(skeleton)
                
                # Merge: use geometric for shoulders/elbows, SDK for others
                all_orientations = sdk_orientations.copy()
                for joint_idx, quat in fixed.items():
                    all_orientations[joint_idx] = quat
                
                self.fixed_orientations[person_idx] = all_orientations
    
    def draw_skeletons(self, frame: Frame):
        """Draw skeletons with geometrically-fixed orientation axes"""
        if not hasattr(frame, 'people') or not frame.people:
            return
        
        from senseSpaceLib.senseSpace.visualization import draw_skeletons_with_bones
        from OpenGL.GL import glBegin, glEnd, glVertex3f, glColor3f, GL_LINES, glLineWidth
        
        # Draw skeleton (joints and bones)
        draw_skeletons_with_bones(
            frame.people, 
            joint_color=(0.2, 0.8, 1.0), 
            bone_color=(0.8, 0.2, 0.2),
            show_orientation=False
        )
        
        # Draw orientation axes
        if self.show_orientation and hasattr(frame, 'people'):
            for person_idx, person in enumerate(frame.people):
                if person_idx not in self.fixed_orientations:
                    continue
                
                skeleton = person.skeleton if hasattr(person, 'skeleton') else person.get('skeleton')
                if not skeleton:
                    continue
                
                self._draw_fixed_orientations(skeleton, self.fixed_orientations[person_idx])
    
    def _draw_fixed_orientations(self, skeleton, orientations, axis_length=150.0):
        """Draw orientation axes (geometrically-fixed joints highlighted)"""
        from OpenGL.GL import glBegin, glEnd, glVertex3f, glColor3f, GL_LINES, glLineWidth
        
        glLineWidth(2.0)
        
        for joint_idx, quat in orientations.items():
            if joint_idx >= len(skeleton):
                continue
            
            joint = skeleton[joint_idx]
            pos = joint.pos if hasattr(joint, 'pos') else joint.get('pos')
            if not pos:
                continue
            
            # Get position
            if hasattr(pos, 'x'):
                px, py, pz = pos.x, pos.y, pos.z
            else:
                px, py, pz = pos['x'], pos['y'], pos['z']
            
            # Convert quaternion to rotation matrix
            if hasattr(quat, 'tolist'):
                quat = quat.tolist()
            
            x, y, z, w = quat
            
            # Rotation matrix columns = axis directions
            x_axis = [1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y)]
            y_axis = [2*(x*y - w*z), 1 - 2*(x*x + z*z), 2*(y*z + w*x)]
            z_axis = [2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y)]
            
            # Highlight geometrically-fixed joints (shoulders & elbows)
            is_fixed = joint_idx in [7, 8, 11, 12]
            
            if is_fixed:
                # Bright colors + longer axes for fixed joints
                x_color = (1.0, 0.3, 1.0)  # Bright magenta
                y_color = (0.3, 1.0, 1.0)  # Bright cyan
                z_color = (1.0, 1.0, 0.3)  # Bright yellow
                length = axis_length * 1.5
            else:
                # Standard RGB
                x_color = (1.0, 0.0, 0.0)  # Red
                y_color = (0.0, 1.0, 0.0)  # Green
                z_color = (0.0, 0.0, 1.0)  # Blue
                length = axis_length
            
            # Draw X axis
            glBegin(GL_LINES)
            glColor3f(*x_color)
            glVertex3f(px, py, pz)
            glVertex3f(px + x_axis[0]*length, py + x_axis[1]*length, pz + x_axis[2]*length)
            glEnd()
            
            # Draw Y axis
            glBegin(GL_LINES)
            glColor3f(*y_color)
            glVertex3f(px, py, pz)
            glVertex3f(px + y_axis[0]*length, py + y_axis[1]*length, pz + y_axis[2]*length)
            glEnd()
            
            # Draw Z axis
            glBegin(GL_LINES)
            glColor3f(*z_color)
            glVertex3f(px, py, pz)
            glVertex3f(px + z_axis[0]*length, py + z_axis[1]*length, pz + z_axis[2]*length)
            glEnd()
        
        glLineWidth(1.0)


def main():
    parser = argparse.ArgumentParser(description="SenseSpace Fixed Orientation Example")
    parser.add_argument("--server", "-s", default="localhost", help="Server IP address")
    parser.add_argument("--port", "-p", type=int, default=12345, help="Server port")
    parser.add_argument("--rec", "-r", type=str, default=None, help="Path to recording file")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("GEOMETRIC ORIENTATION FIX - Prevents Shoulder Flipping")
    print("="*80)
    print("\nThis example reconstructs shoulder/elbow orientations from:")
    print("  - Bone direction (shoulder → elbow)")
    print("  - Stable reference (torso up vector)")
    print("  - Blended with SDK orientation for nuance")
    print("\nResult: Shoulders NEVER flip 180° during arm movement!")
    print("\nPress 'O' to see highlighted orientation axes")
    print("Press SPACE to print orientations for TouchDesigner")
    print("="*80 + "\n")
    
    # Create and run visualization client
    client = VisualizationClient(
        viewer_class=FixedOrientationWidget,
        server_ip=args.server,
        server_port=args.port,
        playback_file=args.rec,
        window_title="Fixed Skeleton Orientations - No Shoulder Flipping!"
    )
    
    success = client.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
