#!/usr/bin/env python3
"""
Example: Detection sphere at camera center using QVector3D
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

from PyQt5.QtGui import QVector3D
from OpenGL.GL import *
from OpenGL.GLU import *


class CustomSkeletonWidget(SkeletonGLWidget):
    """Custom visualization with detection sphere at camera center"""
       
    def onInit(self):
        print("[INFO] Initializing CustomSkeletonWidget")
        self.sphere_radius = 80.0
        self.custom_data = []

    def onClose(self):
        print("[INFO] Closing CustomSkeletonWidget")

    def draw_custom(self, frame: Frame):
        """Draw a detection sphere at the center of all cameras"""
        # Parse camera positions as QVector3D
        camera_positions = []
        camera_list = self._parse_camera_list(frame)
        
        for cam_dict in camera_list:
            pos = self._parse_position(cam_dict.get('position'))
            if pos:
                camera_positions.append(QVector3D(pos[0], pos[1], pos[2]))
        
        if not camera_positions:
            return
        
        # Calculate center of all cameras using QVector3D
        center = QVector3D(0, 0, 0)
        for pos in camera_positions:
            center += pos
        center /= len(camera_positions)
        
        # Check if any skeleton joint is inside the 1m sphere
        sphere_radius = 1000.0  # 1 meter = 1000mm
        person_detected = False
        
        if hasattr(frame, 'people') and frame.people:
            for person in frame.people:
                if not person.skeleton:
                    continue
                
                for joint in person.skeleton:
                    # Calculate distance from joint to sphere center
                    joint_pos = QVector3D(joint.pos.x, joint.pos.y, joint.pos.z)
                    distance = center.distanceToPoint(joint_pos)
                    
                    if distance <= sphere_radius:
                        person_detected = True
                        break
                
                if person_detected:
                    break
        
        # Draw sphere at center
        glPushMatrix()
        glTranslatef(center.x(), center.y(), center.z())
        
        # Set color: red if person detected, green otherwise
        if person_detected:
            glColor4f(1.0, 0.0, 0.0, 0.5)  # Semi-transparent red
        else:
            glColor4f(0.0, 1.0, 0.0, 0.5)  # Semi-transparent green
        
        # Draw sphere
        quadric = gluNewQuadric()
        gluSphere(quadric, sphere_radius, 32, 32)
        gluDeleteQuadric(quadric)
        
        glPopMatrix()


def main():
    parser = argparse.ArgumentParser(description="SenseSpace Detection Sphere")
    parser.add_argument("--server", "-s", default="localhost", help="Server IP address")
    parser.add_argument("--port", "-p", type=int, default=12345, help="Server port")
    
    args = parser.parse_args()
    
    # Create and run visualization client - that's it!
    client = VisualizationClient(
        viewer_class=CustomSkeletonWidget,
        server_ip=args.server,
        server_port=args.port,
        window_title="Detection Sphere Example"
    )
    
    success = client.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()