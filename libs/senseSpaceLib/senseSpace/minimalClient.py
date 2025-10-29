# -----------------------------------------------------------------------------
# Sense Space
# -----------------------------------------------------------------------------
# Minimal Client - Simple wrapper around SenseSpace client
# -----------------------------------------------------------------------------
# IAD, Zurich University of the Arts / zhdk.ch
# Max Rheiner
# -----------------------------------------------------------------------------

import sys
import time
from typing import Callable, Optional

from senseSpaceLib.senseSpace.client import CommandLineClient
from senseSpaceLib.senseSpace.vizClient import VisualizationClient
from senseSpaceLib.senseSpace.protocol import Frame


class MinimalClient:
    """Minimal client wrapper with three callbacks"""
    
    def __init__(
        self, 
        server_ip="localhost", 
        server_port=12345, 
        viz=False,
        playback_file=None,
        on_init: Optional[Callable[[], None]] = None,
        on_frame: Optional[Callable[[Frame], None]] = None,
        on_connection_changed: Optional[Callable[[bool], None]] = None
    ):
        self.server_ip = server_ip
        self.server_port = server_port
        self.viz = viz
        self.playback_file = playback_file
        self.client = None
        self.qt_app = None
        self.qt_viewer = None
        
        # Store callbacks (use no-op defaults if not provided)
        self._on_init = on_init or (lambda: None)
        self._on_frame = on_frame or (lambda f: None)
        self._on_connection_changed = on_connection_changed or (lambda c: None)
        
        # Optional LLM callback (set from example)
        self.llm_callback = None
    
    def run(self) -> bool:
        """Run the client (blocking)"""
        if self.viz:
            return self._run_viz()
        else:
            return self._run_cli()
    
    def _run_cli(self) -> bool:
        """Run in command line mode"""
        self.client = CommandLineClient(
            server_ip=self.server_ip,
            server_port=self.server_port
        )
        
        # Set up callbacks
        self.client.set_frame_callback(self._on_frame)
        self.client.set_connection_callback(self._on_connection_changed)
        
        # Connect
        if not self.client.connect():
            print("[ERROR] Failed to connect")
            return False
        
        # Call init callback
        self._on_init()
        
        # Run (blocking)
        print("[INFO] Running in CLI mode. Press Ctrl+C to exit...")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[INFO] Shutting down...")
            self.client.disconnect()
            return True
    
    def _run_viz(self) -> bool:
        """Run in visualization mode"""
        try:
            from PyQt5 import QtWidgets, QtCore
            from senseSpaceLib.senseSpace.vizWidget import SkeletonGLWidget
        except ImportError as e:
            print(f"[ERROR] Visualization requires PyQt5: {e}")
            return False
        
        # Create visualization client
        self.client = VisualizationClient(
            viewer_class=SkeletonGLWidget,
            server_ip=self.server_ip,
            server_port=self.server_port,
            window_title=f"SenseSpace - {self.server_ip}:{self.server_port}",
            playback_file=self.playback_file
        )
        
        # Override frame callback to call both viewer update and user callback
        original_on_frame = self.client._on_frame_received
        def combined_callback(frame):
            original_on_frame(frame)  # Update viewer
            self._on_frame(frame)      # Call user callback
        self.client.set_frame_callback(combined_callback)
        
        # Set connection callback
        self.client.set_connection_callback(self._on_connection_changed)
        
        # Call init callback before running
        self._on_init()
        
        # Run (blocking) - this handles Qt app creation and event loop
        return self.client.run()
