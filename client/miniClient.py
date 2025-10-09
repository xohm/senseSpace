import sys
import time
from typing import Callable, Optional

from senseSpaceLib.senseSpace.client import SenseSpaceClient, CommandLineClient
from senseSpaceLib.senseSpace.protocol import Frame


class MinimalClient:
    """Minimal client wrapper with three callbacks"""
    
    def __init__(
        self, 
        server_ip="localhost", 
        server_port=12345, 
        viz=False,
        on_init: Optional[Callable[[], None]] = None,
        on_frame: Optional[Callable[[Frame], None]] = None,
        on_connection_changed: Optional[Callable[[bool], None]] = None
    ):
        self.server_ip = server_ip
        self.server_port = server_port
        self.viz = viz
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
            from qt_client_viewer import ClientSkeletonGLWidget
        except ImportError as e:
            print(f"[ERROR] Visualization requires PyQt5: {e}")
            return False
        
        # Create Qt app
        self.qt_app = QtWidgets.QApplication(sys.argv)
        
        # Create client
        self.client = SenseSpaceClient(
            server_ip=self.server_ip,
            server_port=self.server_port
        )
        
        # Set up callbacks
        self.client.set_frame_callback(self._on_frame_wrapper)
        self.client.set_connection_callback(self._on_connection_changed)
        
        # Connect
        if not self.client.connect():
            print("[ERROR] Failed to connect")
            return False
        
        # Create viewer
        main_window = QtWidgets.QMainWindow()
        main_window.setWindowTitle(f"SenseSpace - {self.server_ip}:{self.server_port}")
        main_window.resize(800, 600)
        
        self.qt_viewer = ClientSkeletonGLWidget()
        main_window.setCentralWidget(self.qt_viewer)
        
        # Install event filter for keyboard input
        if self.llm_callback:
            main_window.keyPressEvent = self._handle_key_press
        
        main_window.show()
        
        # Call init callback
        self._on_init()
        
        print("[INFO] Running in visualization mode. Close window to exit...")
        
        # Run Qt event loop
        try:
            exit_code = self.qt_app.exec_()
            return exit_code == 0
        finally:
            self.client.disconnect()
    
    def _handle_key_press(self, event):
        """Handle keyboard events"""
        if self.llm_callback:
            # Convert Qt key to string character
            key_text = event.text()  # Get the character as string
            if not key_text:
                # For special keys (Space, Enter, etc), use key name
                from PyQt5 import QtCore
                key_map = {
                    QtCore.Qt.Key_Space: ' ',
                    QtCore.Qt.Key_Return: '\n',
                    QtCore.Qt.Key_Enter: '\n',
                    QtCore.Qt.Key_Escape: 'ESC',
                }
                key_text = key_map.get(event.key(), '')
            
            if key_text:
                self.llm_callback(key_text)
        
        event.accept()
    
    def _on_frame_wrapper(self, frame: Frame):
        """Internal wrapper for viz mode - updates viewer and calls user callback"""
        # Update viewer
        if self.qt_viewer:
            self.qt_viewer.update_frame(frame)
        # Call user callback
        self._on_frame(frame)