#!/usr/bin/env python3
"""
Example: Raised arm detection - draws arms in red when raised
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
from senseSpaceLib.senseSpace.protocol import Frame
from senseSpaceLib.senseSpace.vizWidget import SkeletonGLWidget
from senseSpaceLib.senseSpace.enums import Body34Joint, Body18Joint

from PyQt5.QtCore import Qt
from OpenGL.GL import *
from OpenGL.GLU import *

from PyQt5.QtCore import Qt

def check_raised_arm(person, body_format="BODY_34"):
    """Check which arms are raised for a person
    
    Args:
        person: Person object with skeleton data
        body_format: "BODY_34" or "BODY_18" skeleton format
        
    Returns:
        tuple: (left_raised, right_raised) - boolean flags for each arm
    """
    if not person.skeleton:
        return False, False
    
    # Select the appropriate joint enum based on body format
    if body_format == "BODY_18":
        J = Body18Joint
        min_joints = 18
    else:  # BODY_34
        J = Body34Joint
        min_joints = 34
    
    skeleton = person.skeleton
    
    # Check if we have enough joints
    if len(skeleton) < min_joints:
        return False, False
    
    left_arm_raised = False
    right_arm_raised = False
    
    # Check left arm - wrist higher than shoulder (higher Y value = higher position)
    left_shoulder_idx = J.LEFT_SHOULDER.value
    left_wrist_idx = J.LEFT_WRIST.value
    if left_shoulder_idx < len(skeleton) and left_wrist_idx < len(skeleton):
        shoulder_y = skeleton[left_shoulder_idx].pos.y
        wrist_y = skeleton[left_wrist_idx].pos.y
        if wrist_y > shoulder_y + 100:  # 100mm threshold - wrist ABOVE shoulder
            left_arm_raised = True
    
    # Check right arm
    right_shoulder_idx = J.RIGHT_SHOULDER.value
    right_wrist_idx = J.RIGHT_WRIST.value
    if right_shoulder_idx < len(skeleton) and right_wrist_idx < len(skeleton):
        shoulder_y = skeleton[right_shoulder_idx].pos.y
        wrist_y = skeleton[right_wrist_idx].pos.y
        if wrist_y > shoulder_y + 100:  # 100mm threshold - wrist ABOVE shoulder
            right_arm_raised = True
    
    return left_arm_raised, right_arm_raised


class CustomSkeletonWidget(SkeletonGLWidget):
    """Custom visualization that shows red spheres at raised hands"""

    def onInit(self):
        print("[INFO] Initializing CustomSkeletonWidget")
        self.sphere_radius = 40.0  # Normal size
        self.sphere_size_large = False
        self.custom_data = []
        # Create quadric once for efficiency
        self.quadric = gluNewQuadric()

    def onClose(self):
        print("[INFO] Closing CustomSkeletonWidget")
        # Clean up quadric
        if hasattr(self, 'quadric') and self.quadric:
            gluDeleteQuadric(self.quadric)
            self.quadric = None
    
    def keyPressEvent(self, event):
        """Handle key press events"""
        if event.key() == Qt.Key_Space:
            # Toggle sphere size
            self.sphere_size_large = not self.sphere_size_large
            if self.sphere_size_large:
                self.sphere_radius = 80.0  # 2x bigger
                print("[INFO] Sphere size: LARGE (160mm)")
            else:
                self.sphere_radius = 40.0  # Normal size
                print("[INFO] Sphere size: NORMAL (80mm)")
        else:
            # Pass other keys to parent
            super().keyPressEvent(event)

    def draw_custom(self, frame: Frame):
        """Draw red spheres at hands when arms are raised"""
        if not hasattr(frame, 'people') or not frame.people:
            return
        
        # Get body model from frame (defaults to BODY_34)
        body_model = frame.body_model if frame.body_model else "BODY_34"
        
        # Select the appropriate joint enum based on body model
        if body_model == "BODY_18":
            J = Body18Joint
        else:  # BODY_34
            J = Body34Joint
        
        for person in frame.people:
            # Check which arms are raised (pass body model)
            left_raised, right_raised = check_raised_arm(person, body_model)
            
            if not person.skeleton:
                continue
            
            skeleton = person.skeleton
            
            # Draw red sphere at left hand if raised
            if left_raised:
                left_hand_idx = J.LEFT_HAND.value
                if left_hand_idx < len(skeleton):
                    hand_pos = skeleton[left_hand_idx].pos
                    
                    glPushMatrix()
                    glTranslatef(hand_pos.x, hand_pos.y, hand_pos.z)
                    glColor4f(1.0, 0.0, 0.0, 0.7)  # Red semi-transparent
                    gluSphere(self.quadric, self.sphere_radius, 16, 16)
                    glPopMatrix()
            
            # Draw red sphere at right hand if raised
            if right_raised:
                right_hand_idx = J.RIGHT_HAND.value
                if right_hand_idx < len(skeleton):
                    hand_pos = skeleton[right_hand_idx].pos
                    
                    glPushMatrix()
                    glTranslatef(hand_pos.x, hand_pos.y, hand_pos.z)
                    glColor4f(1.0, 0.0, 0.0, 0.7)  # Red semi-transparent
                    gluSphere(self.quadric, self.sphere_radius, 16, 16)
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
        window_title="Raised Arm Detection Example"
    )
    
    success = client.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
