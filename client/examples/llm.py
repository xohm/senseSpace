# -----------------------------------------------------------------------------
# Sense Space
# -----------------------------------------------------------------------------
# Open AI LLM Client Example
# -----------------------------------------------------------------------------
# IAD, Zurich University of the Arts / zhdk.ch
# Max Rheiner
# -----------------------------------------------------------------------------

import argparse
import sys
import time
from typing import Optional

import os, sys
# Ensure local 'libs' folder is on sys.path when running from repo
# repo structure: <repo_root>/client/senseSpaceClient.py and <repo_root>/libs/senseSpaceLib
# so the repo root is one level up from the client folder.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
libs_path = os.path.join(repo_root, 'libs')
if os.path.isdir(libs_path) and libs_path not in sys.path:
    sys.path.insert(0, libs_path)
elif repo_root not in sys.path:
    # Fallback: add repo root so an installed package or alternative layout can still import
    sys.path.insert(0, repo_root)

# Import from our shared library
from senseSpaceLib.senseSpace.client import SenseSpaceClient, CommandLineClient
from senseSpaceLib.senseSpace.protocol import Frame


class VisualizationClient(SenseSpaceClient):
    """Client with Qt OpenGL visualization"""
    
    def __init__(self, server_ip="localhost", server_port=12345):
        super().__init__(server_ip, server_port)
        
        # Qt components
        self.qt_app = None
        self.qt_viewer = None
        
        # Set up callbacks
        self.set_frame_callback(self._on_frame_received)
        self.set_connection_callback(self._on_connection_changed)
    
    def _on_frame_received(self, frame: Frame):
        """Update Qt viewer with new frame"""
        if self.qt_viewer:
            self.qt_viewer.update_frame(frame)
    
    def _on_connection_changed(self, connected: bool):
        """Handle connection status changes"""
        if connected:
            print(f"[INFO] Connected to {self.server_ip}:{self.server_port}")
        else:
            print(f"[INFO] Disconnected from {self.server_ip}:{self.server_port}")
    
    def run(self) -> bool:
        """Run client in visualization mode"""
        try:
            from PyQt5 import QtWidgets, QtCore
            from qt_client_viewer import ClientSkeletonGLWidget
        except ImportError as e:
            print(f"[ERROR] PyQt5 not available for visualization mode: {e}")
            print("[INFO] Install PyQt5 and PyOpenGL: pip install PyQt5 PyOpenGL")
            return False
        
        if not self.connect():
            return False
        
        # Create Qt application
        self.qt_app = QtWidgets.QApplication(sys.argv)
        
        # Create main window
        main_window = QtWidgets.QMainWindow()
        main_window.setWindowTitle(f"SenseSpace Client - {self.server_ip}:{self.server_port}")
        main_window.resize(800, 600)
        
        # Create OpenGL viewer widget
        self.qt_viewer = ClientSkeletonGLWidget()
        main_window.setCentralWidget(self.qt_viewer)
        
        # Show window
        main_window.show()
        
        print("[INFO] Running in visualization mode. Close window to exit...")
        
        try:
            exit_code = self.qt_app.exec_()
            return exit_code == 0
        finally:
            self.disconnect()


def main():
    parser = argparse.ArgumentParser(description="SenseSpace Client - Connect to server and display body tracking data")
    parser.add_argument("--server", "-s", default="localhost", help="Server IP address (default: localhost)")
    parser.add_argument("--port", "-p", type=int, default=12345, help="Server port (default: 12345)")
    parser.add_argument("--viz", "--visualization", action="store_true", help="Enable visualization mode (Qt OpenGL)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output in command line mode")
    
    args = parser.parse_args()
    
    # Create appropriate client type
    if args.viz:
        client = VisualizationClient(
            server_ip=args.server,
            server_port=args.port
        )
    else:
        client = CommandLineClient(
            server_ip=args.server,
            server_port=args.port,
            verbose=args.verbose
        )
    
    # Run client
    success = client.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
