#!/usr/bin/env python3
"""
Example: Custom skeleton visualization with thicker lines
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

# Import visualization helper
from senseSpaceLib.senseSpace.visualization import draw_skeletons_with_bones
        
from OpenGL.GL import *


class CustomSkeletonWidget(SkeletonGLWidget):
    """Custom visualization with thicker lines and custom colors"""
    
    def onInit(self):
        print("[INFO] Initializing CustomSkeletonWidget")
        self.sphere_radius = 80.0
        self.custom_data = []

    def onClose(self):
        print("[INFO] Closing CustomSkeletonWidget")

    def draw_skeletons(self, frame: Frame):
        """Override to draw skeletons with thicker lines"""
        if not hasattr(frame, 'people') or not frame.people:
            return
        
        # Set thicker lines
        glLineWidth(5.0)  # Much thicker than default (2.0)
        glPointSize(12.0)  # Larger joint markers
        
        # Draw with custom colors - bright cyan joints, red bones
        draw_skeletons_with_bones(
            frame.people,
            joint_color=(0.0, 1.0, 1.0),  # Cyan
            bone_color=(1.0, 1.0, 1.0)     # white
        )
        
        # Reset to defaults for other rendering
        glLineWidth(2.0)
        glPointSize(8.0)

def main():
    parser = argparse.ArgumentParser(description="Custom SenseSpace Skeleton Viewer")
    parser.add_argument("--server", "-s", default="localhost", help="Server IP address")
    parser.add_argument("--port", "-p", type=int, default=12345, help="Server port")
    
    args = parser.parse_args()
    
    # Create and run visualization client - that's it!
    client = VisualizationClient(
        viewer_class=CustomSkeletonWidget,
        server_ip=args.server,
        server_port=args.port,
        window_title="Thick Lines Example"
    )
    
    success = client.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
