#!/usr/bin/env python3
"""
SenseSpace Client Library

Core client functionality for connecting to SenseSpace servers and handling data.
"""

import json
import socket
import threading
import time
import struct
import os
from typing import Optional, Callable

from .protocol import Frame
from .communication import deserialize_message
from .recorder import FrameRecorder, FramePlayer


class SenseSpaceClient:
    """Client for connecting to SenseSpace server and receiving body tracking data"""
    
    def __init__(self, server_ip="localhost", server_port=12345, playback_file: Optional[str] = None):
        self.server_ip = server_ip
        self.server_port = server_port
        
        self.socket = None
        self.connected = False
        self.latest_frame: Optional[Frame] = None
        self.latest_timestamp: float = 0.0  # Track latest frame timestamp for dropping old frames
        self.running = True
        
        # Callbacks
        self.frame_callback: Optional[Callable[[Frame], None]] = None
        self.connection_callback: Optional[Callable[[bool], None]] = None
        
        # Threading
        self.receive_thread = None
        
        # Recording/Playback
        self.recorder: Optional[FrameRecorder] = None
        self.player: Optional[FramePlayer] = None
        self.playback_mode = playback_file is not None
        self.playback_file = playback_file
        
    def set_frame_callback(self, callback: Callable[[Frame], None]):
        """Set callback function to be called when a new frame is received"""
        self.frame_callback = callback
    
    def set_connection_callback(self, callback: Callable[[bool], None]):
        """Set callback function to be called when connection status changes"""
        self.connection_callback = callback
    
    def connect(self) -> bool:
        """Connect to the senseSpace server via TCP or start playback"""
        # Playback mode - no server connection needed
        if self.playback_mode:
            return self._start_playback()
        
        # Normal server connection
        try:
            # Use TCP (stream) for reliable connection
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10.0)  # 10 second timeout for connection
            
            # Connect to server
            self.socket.connect((self.server_ip, self.server_port))
            self.socket.settimeout(None)  # No timeout for receiving (blocking)
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # Disable Nagle's algorithm
            
            self.connected = True
            self.running = True
            
            # Start receiving frames in background thread
            self.receive_thread = threading.Thread(target=self._receive_frames_tcp, daemon=True)
            self.receive_thread.start()
            
            if self.connection_callback:
                self.connection_callback(True)
                
            print(f"[INFO] Connected to server at {self.server_ip}:{self.server_port} (TCP)")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to connect to server {self.server_ip}:{self.server_port}: {e}")
            self.connected = False
            if self.connection_callback:
                self.connection_callback(False)
            return False
    
    def disconnect(self):
        """Disconnect from server"""
        self.running = False
        self.connected = False
        
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        
        if self.connection_callback:
            self.connection_callback(False)
            
        print("[INFO] Disconnected from server")
    
    def is_connected(self) -> bool:
        """Check if client is currently connected to server"""
        return self.connected
    
    def get_latest_frame(self) -> Optional[Frame]:
        """Get the most recently received frame"""
        return self.latest_frame
    
    def _receive_frames_tcp(self):
        """Background thread to receive frames from server"""
        json_buffer = ""
        
        while self.running and self.connected:
            try:
                # Receive data
                data = self.socket.recv(65536)
                if not data:
                    print("[WARNING] Server closed connection")
                    break
                
                # Try to detect protocol: check first byte
                # JSON starts with '{' (0x7b), MessagePack magic is 0x9f
                if data[0:1] == b'{':
                    # JSON protocol (legacy)
                    try:
                        json_buffer += data.decode('utf-8')
                        
                        # Process complete messages (newline-delimited JSON)
                        while '\n' in json_buffer:
                            line, json_buffer = json_buffer.split('\n', 1)
                            if line.strip():
                                try:
                                    message = json.loads(line)
                                    self._handle_message(message)
                                except json.JSONDecodeError as e:
                                    print(f"[WARNING] Failed to parse JSON: {e}")
                    except UnicodeDecodeError:
                        print(f"[WARNING] Received binary data but failed to decode as UTF-8")
                        continue
                        
                elif data[0:2] == b'\x9f\xd0' or data[0:2] == b'\x9f\xd1':
                    # MessagePack protocol (binary)
                    # Format: [magic:2][length:4][payload]
                    if len(data) < 6:
                        print(f"[WARNING] Incomplete MessagePack header")
                        continue
                    
                    try:
                        message = deserialize_message(data)
                        if message:
                            self._handle_message(message)
                    except Exception as e:
                        print(f"[WARNING] Failed to deserialize MessagePack: {e}")
                else:
                    print(f"[WARNING] Unknown protocol magic bytes: {data[0:2].hex()}")
                    
            except Exception as e:
                if self.running:
                    print(f"[ERROR] Error receiving data: {e}")
                break
        
        # Connection lost
        self.connected = False
        if self.connection_callback:
            self.connection_callback(False)
    
    def _handle_message(self, message: dict):
        """Handle received message from server"""
        if message.get("type") == "frame":
            frame_data = message.get("data")
            if frame_data:
                try:
                    frame = Frame.from_dict(frame_data)
                    self.latest_frame = frame
                    
                    # Record frame if recording is active
                    if self.recorder and self.recorder.is_recording():
                        self.recorder.record_frame(frame)
                    
                    # Call frame callback if set
                    if self.frame_callback:
                        self.frame_callback(frame)
                        
                except Exception as e:
                    print(f"[WARNING] Failed to parse frame data: {e}")
    
    def _start_playback(self) -> bool:
        """Start playback from recorded file"""
        try:
            self.player = FramePlayer(self.playback_file, loop=True, speed=1.0)
            
            if not self.player.load_header():
                print(f"[ERROR] Failed to load recording: {self.playback_file}")
                return False
            
            # Set up playback callback to handle frames
            def on_playback_frame(frame: Frame):
                self.latest_frame = frame
                if self.frame_callback:
                    self.frame_callback(frame)
            
            self.player.set_frame_callback(on_playback_frame)
            
            # Start playback
            if not self.player.start():
                return False
            
            self.connected = True
            self.running = True
            
            if self.connection_callback:
                self.connection_callback(True)
            
            print(f"[INFO] Playing back recording: {self.playback_file}")
            info = self.player.get_info()
            if 'frame_count' in info:
                print(f"[INFO] {info['frame_count']} frames, {info['duration']:.1f}s duration")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to start playback: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def start_recording(self, filepath: Optional[str] = None):
        """
        Start recording frames to file.
        
        Args:
            filepath: Path to save recording (default: recordings/recording_TIMESTAMP.ssrec)
        """
        if self.playback_mode:
            print("[WARNING] Cannot record while in playback mode")
            return False
        
        if self.recorder and self.recorder.is_recording():
            print("[WARNING] Already recording")
            return False
        
        # Generate default filename if not provided
        if filepath is None:
            recordings_dir = os.path.join(os.getcwd(), 'recordings')
            os.makedirs(recordings_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(recordings_dir, f"recording_{timestamp}.ssrec")
        
        self.recorder = FrameRecorder(filepath)
        return self.recorder.start()
    
    def stop_recording(self):
        """Stop recording frames"""
        if self.recorder:
            self.recorder.stop()
            self.recorder = None
    
    def toggle_recording(self, filepath: Optional[str] = None):
        """Toggle recording on/off"""
        if self.recorder and self.recorder.is_recording():
            self.stop_recording()
        else:
            self.start_recording(filepath)
    
    def is_recording(self) -> bool:
        """Check if currently recording"""
        return self.recorder is not None and self.recorder.is_recording()
    
    def toggle_playback_pause(self):
        """Toggle pause/play for playback mode"""
        if not self.playback_mode or not self.player:
            print("[WARNING] Not in playback mode")
            return False
        
        if self.player.paused:
            self.player.resume()
            return True  # Now playing
        else:
            self.player.pause()
            return False  # Now paused
    
    def is_playback_paused(self) -> bool:
        """Check if playback is currently paused"""
        if not self.playback_mode or not self.player:
            return False
        return self.player.paused
    
    def wait_for_connection(self, timeout: float = 10.0) -> bool:
        """Wait for connection to be established with timeout"""
        start_time = time.time()
        while not self.connected and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        return self.connected
    
    def get_connection_info(self) -> dict:
        """Get information about the current connection"""
        return {
            "server_ip": self.server_ip,
            "server_port": self.server_port,
            "connected": self.connected,
            "latest_frame_time": self.latest_frame.timestamp if self.latest_frame else None,
            "people_count": len(self.latest_frame.people) if self.latest_frame else 0
        }


class CommandLineClient(SenseSpaceClient):
    """Client that prints frame information to console"""
    
    def __init__(self, server_ip="localhost", server_port=12345, verbose=False, playback_file=None):
        super().__init__(server_ip, server_port, playback_file=playback_file)
        self.verbose = verbose
        self.set_frame_callback(self._on_frame_received)
        self.set_connection_callback(self._on_connection_changed)
    
    def _on_frame_received(self, frame: Frame):
        """Print frame information to console"""
        people_count = len(frame.people)
        # Only show floor info when the server provided a ZED-detected value (may be None)
        floor_info = f", floor: {frame.floor_height:.0f}mm" if frame.floor_height is not None else ""
        print(f"[FRAME] ts={frame.timestamp:.3f}, people={people_count}{floor_info}")

        if self.verbose:
            # Print detailed person information
            for person in frame.people:
                joint_count = len(person.skeleton)
                print(f"  Person {person.id}: {person.tracking_state}, conf={person.confidence:.2f}, joints={joint_count}")

                if joint_count > 0:
                    # Print a few key joint positions
                    for i, joint in enumerate(person.skeleton[:5]):  # First 5 joints only
                        pos = joint.pos
                        # Support both Position object and dict
                        if hasattr(pos, 'x'):
                            print(f"    Joint {i}: ({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f})")
                        else:
                            print(f"    Joint {i}: ({pos['x']:.1f}, {pos['y']:.1f}, {pos['z']:.1f})")
    
    def _on_connection_changed(self, connected: bool):
        """Handle connection status changes"""
        if connected:
            print(f"[INFO] Connected to {self.server_ip}:{self.server_port}")
        else:
            print(f"[INFO] Disconnected from {self.server_ip}:{self.server_port}")
    
    def run(self):
        """Run the command line client"""
        if not self.connect():
            return False
        
        print("[INFO] Running in command line mode. Press Ctrl+C to exit...")
        
        try:
            while self.running and self.connected:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[INFO] Shutting down...")
        finally:
            self.disconnect()
        
        return True
