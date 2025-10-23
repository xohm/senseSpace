#!/usr/bin/env python3
"""
Example: Raised arm detection with universal joint enum
"""

import argparse
import sys
import os

# Ensure local 'libs' folder is on sys.path when running from repo
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
libs_path = os.path.join(repo_root, 'libs')
if os.path.isdir(libs_path) and libs_path not in sys.path:
    sys.path.insert(0, libs_path)

# Import from shared library
from senseSpaceLib.senseSpace.vizClient import VisualizationClient
from senseSpaceLib.senseSpace.protocol import Frame, Person
from senseSpaceLib.senseSpace.vizWidget import SkeletonGLWidget
from senseSpaceLib.senseSpace.enums import UniversalJoint

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QVector3D
from OpenGL.GL import *
from OpenGL.GLU import *


def check_raised_arm(person: Person, body_model: str = "BODY_34") -> tuple[bool, bool]:
    """Check if person's arms are raised above shoulders
    
    Args:
        person: Person object with skeleton data
        body_model: "BODY_18" or "BODY_34"
        
    Returns:
        (left_raised, right_raised) tuple of booleans
    """
    # Get joint positions using universal joint enum
    left_shoulder_data = person.get_joint(UniversalJoint.LEFT_SHOULDER, body_model)
    left_wrist_data = person.get_joint(UniversalJoint.LEFT_WRIST, body_model)
    right_shoulder_data = person.get_joint(UniversalJoint.RIGHT_SHOULDER, body_model)
    right_wrist_data = person.get_joint(UniversalJoint.RIGHT_WRIST, body_model)
    
    left_raised = False
    right_raised = False
    
    # Check left arm
    if left_shoulder_data and left_wrist_data:
        shoulder_pos, _ = left_shoulder_data
        wrist_pos, _ = left_wrist_data
        # Arm is raised if wrist is 100mm or more above shoulder
        if wrist_pos.y > shoulder_pos.y + 100:
            left_raised = True
    
    # Check right arm
    if right_shoulder_data and right_wrist_data:
        shoulder_pos, _ = right_shoulder_data
        wrist_pos, _ = right_wrist_data
        # Arm is raised if wrist is 100mm or more above shoulder
        if wrist_pos.y > shoulder_pos.y + 100:
            right_raised = True
    
    return left_raised, right_raised


class CustomSkeletonWidget(SkeletonGLWidget):
    """Custom visualization with raised arm detection"""
    
    def onInit(self):
        """Initialize custom state"""
        self.sphere_radius = 80.0  # Normal size
        self.sphere_size_large = False  # Track size state
        self.quadric = gluNewQuadric()  # Create quadric once
    
    def onClose(self):
        """Cleanup resources"""
        if hasattr(self, 'quadric') and self.quadric:
            gluDeleteQuadric(self.quadric)
    
    def keyPressEvent(self, event):
        """Handle keyboard input"""
        if event.key() == Qt.Key_Space:
            # Toggle sphere size
            self.sphere_size_large = not self.sphere_size_large
            if self.sphere_size_large:
                self.sphere_radius = 160.0  # 2x bigger
                print("[INFO] Sphere size: LARGE (160mm)")
            else:
                self.sphere_radius = 80.0  # Normal
                print("[INFO] Sphere size: NORMAL (80mm)")
        else:
            # Pass other keys to parent
            super().keyPressEvent(event)
       
    def draw_custom(self, frame: Frame):
        """Draw red spheres at raised hands"""
        if not hasattr(frame, 'people') or not frame.people:
            return
        
        # Get body model from frame
        body_model = frame.body_model if frame.body_model else "BODY_34"
        
        # Check each person
        for person in frame.people:
            left_raised, right_raised = check_raised_arm(person, body_model)
            
            # Draw sphere at left hand if raised
            if left_raised:
                wrist_data = person.get_joint(UniversalJoint.LEFT_WRIST, body_model)
                if wrist_data:
                    wrist_pos, _ = wrist_data
                    glPushMatrix()
                    glTranslatef(wrist_pos.x, wrist_pos.y, wrist_pos.z)
                    glColor4f(1.0, 0.0, 0.0, 0.6)  # Semi-transparent red
                    gluSphere(self.quadric, self.sphere_radius, 32, 32)
                    glPopMatrix()
            
            # Draw sphere at right hand if raised
            if right_raised:
                wrist_data = person.get_joint(UniversalJoint.RIGHT_WRIST, body_model)
                if wrist_data:
                    wrist_pos, _ = wrist_data
                    glPushMatrix()
                    glTranslatef(wrist_pos.x, wrist_pos.y, wrist_pos.z)
                    glColor4f(1.0, 0.0, 0.0, 0.6)  # Semi-transparent red
                    gluSphere(self.quadric, self.sphere_radius, 32, 32)
                    glPopMatrix()


def main():
    parser = argparse.ArgumentParser(description="SenseSpace Raised Arm Detection")
    parser.add_argument("--server", "-s", default="localhost", help="Server IP address")
    parser.add_argument("--port", "-p", type=int, default=12345, help="Server port")
    
    args = parser.parse_args()
    
    # Create and run visualization client
    client = VisualizationClient(
        viewer_class=CustomSkeletonWidget,
        server_ip=args.server,
        server_port=args.port,
        window_title="Raised Arm Detection - Press SPACE to toggle sphere size"
    )
    
    success = client.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
