#!/usr/bin/env python3
"""
Visualization client for SenseSpace - combines client connection with Qt OpenGL viewer
"""

import sys
from typing import Optional

from .client import SenseSpaceClient
from .protocol import Frame
from .vizWidget import SkeletonGLWidget


class VisualizationClient(SenseSpaceClient):
    """Client with Qt OpenGL visualization
    
    Usage:
        client = VisualizationClient(
            viewer_class=CustomSkeletonWidget,  # Your custom widget class
            server_ip="192.168.1.2",
            server_port=12345
        )
        client.run()
    """
    
    def __init__(self, viewer_class=None, server_ip="localhost", server_port=12345, window_title=None, playback_file=None):
        """
        Args:
            viewer_class: Custom SkeletonGLWidget subclass (default: SkeletonGLWidget)
            server_ip: Server IP address
            server_port: Server port
            window_title: Custom window title (default: auto-generated)
            playback_file: Path to recording file for playback mode
        """
        super().__init__(server_ip, server_port, playback_file=playback_file)
        
        self.viewer_class = viewer_class or SkeletonGLWidget
        
        # Update window title for playback mode
        if playback_file:
            self.window_title = window_title or f"SenseSpace Viewer - Playback: {playback_file}"
        else:
            self.window_title = window_title or f"SenseSpace Viewer - {server_ip}:{server_port}"
        
        # Qt components
        self.qt_app = None
        self.qt_viewer = None
        self.main_window = None
        
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
    
    def run(self, window_width=800, window_height=600) -> bool:
        """Run client with visualization window
        
        Args:
            window_width: Initial window width (default: 800)
            window_height: Initial window height (default: 600)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from PyQt5 import QtWidgets, QtCore
        except ImportError as e:
            print(f"[ERROR] PyQt5 not available: {e}")
            print("[INFO] Install: pip install PyQt5 PyOpenGL")
            return False
        
        if not self.connect():
            return False
        
        # Create Qt application
        self.qt_app = QtWidgets.QApplication(sys.argv)
        
        # Create main window
        self.main_window = QtWidgets.QMainWindow()
        self.main_window.setWindowTitle(self.window_title)
        self.main_window.resize(window_width, window_height)
        
        # Create viewer widget (custom or default)
        self.qt_viewer = self.viewer_class()
        
        # Pass client reference to viewer for recording control
        if hasattr(self.qt_viewer, 'set_client'):
            self.qt_viewer.set_client(self)
        
        self.main_window.setCentralWidget(self.qt_viewer)
        
        # Show window
        self.main_window.show()
        
        print(f"[INFO] Running visualization. Close window to exit...")
        
        try:
            exit_code = self.qt_app.exec_()
            return exit_code == 0
        finally:
            self.disconnect()