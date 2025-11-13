#!/usr/bin/env python3
"""
SenseSpace Server Library

Core server functionality for ZED SDK body tracking and TCP broadcasting.
"""

import socket
import json
import time
import threading
import queue
from typing import List, Optional, Callable
import pyzed.sl as sl
import os
import numpy as np
from PyQt5.QtGui import QVector3D, QQuaternion

# Import protocol classes
try:
    from .protocol import Frame, Person, Joint, Position, Quaternion
except ImportError:
    try:
        from senseSpaceLib.senseSpace.protocol import Frame, Person, Joint, Position, Quaternion
    except ImportError:
        # Fallback for development
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__)))
        from protocol import Frame, Person, Joint, Position, Quaternion

# Import video streaming classes (optional)
try:
    from .video_streaming import MultiCameraVideoStreamer
    STREAMING_AVAILABLE = True
except ImportError:
    try:
        from senseSpaceLib.senseSpace.video_streaming import MultiCameraVideoStreamer
        STREAMING_AVAILABLE = True
    except ImportError:
        STREAMING_AVAILABLE = False
        MultiCameraVideoStreamer = None

# Import v2 per-camera streaming (optional)
try:
    from .video_streaming_server_v2 import PerCameraStreamingManager
    STREAMING_V2_AVAILABLE = True
except ImportError:
    try:
        from senseSpaceLib.senseSpace.video_streaming_server_v2 import PerCameraStreamingManager
        STREAMING_V2_AVAILABLE = True
    except ImportError:
        STREAMING_V2_AVAILABLE = False
        PerCameraStreamingManager = None

# Import body tracking filter
try:
    from .body_tracking_filter import BodyTrackingFilter
except ImportError:
    try:
        from senseSpaceLib.senseSpace.body_tracking_filter import BodyTrackingFilter
    except ImportError:
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__)))
        from body_tracking_filter import BodyTrackingFilter


def get_local_ip():
    """Get the local IP address of this machine"""
    try:
        # Connect to a remote address to determine which local interface to use
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        try:
            # Fallback: try to get hostname IP
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"


class SenseSpaceServer:
    """Main server class for ZED SDK body tracking and TCP broadcasting"""
    
    # Camera configuration - used for both single and fusion modes
    # VGA @ 60fps provides best performance for fusion tracking
    CAMERA_RESOLUTION = sl.RESOLUTION.VGA  # 672x376 @ 60fps
    CAMERA_FPS = 60  # 60fps for smooth tracking

    def __init__(self, host: str = "0.0.0.0", port: int = 12345, use_udp: bool = False,
                 enable_streaming: bool = False, stream_host: str = None,
                 stream_rgb_port: int = 5000, stream_depth_port: int = 5001,
                 enable_body_filter: bool = False,  # Changed to False - filter can cause tracking instability
                 camera_resolution: int = 2,  # 0=HD720, 1=HD1080, 2=VGA (default)
                 camera_fps: int = 60,  # 60fps default for smooth tracking
                 tracking_accuracy: int = 1,  # 0=FAST, 1=MEDIUM (default), 2=ACCURATE
                 max_detection_range: float = 10.0,  # Maximum detection range in meters (increased for better tracking)
                 enable_body_fitting: bool = True,  # Body mesh fitting (default: enabled for BODY_34)
                 prediction_timeout: float = 2.0,  # Tracking prediction timeout (2.0s - balance between stability and ghost reduction)
                 enable_skeleton_filter: bool = True):  # Skeleton ID continuity filter (prevents ghost duplicates)
        self.host = host
        self.port = port
        self.local_ip = get_local_ip()
        self.use_udp = use_udp
        
        # Apply camera and tracking parameters (override class defaults)
        resolution_map = {
            0: sl.RESOLUTION.HD720,   # 1280x720
            1: sl.RESOLUTION.HD1080,  # 1920x1080
            2: sl.RESOLUTION.VGA      # 672x376
        }
        self.CAMERA_RESOLUTION = resolution_map.get(camera_resolution, sl.RESOLUTION.VGA)
        self.CAMERA_FPS = camera_fps
        
        accuracy_map = {
            0: sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST,
            1: sl.BODY_TRACKING_MODEL.HUMAN_BODY_MEDIUM,
            2: sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
        }
        self.body_tracking_model = accuracy_map.get(tracking_accuracy, sl.BODY_TRACKING_MODEL.HUMAN_BODY_MEDIUM)
        self.max_detection_range = max_detection_range
        self.enable_body_fitting = enable_body_fitting
        self.prediction_timeout = prediction_timeout
        
        # Print configuration
        resolution_names = {0: "HD720 (1280x720)", 1: "HD1080 (1920x1080)", 2: "VGA (672x376)"}
        accuracy_names = {0: "FAST", 1: "MEDIUM", 2: "ACCURATE"}
        print(f"[INFO] Camera configuration:")
        print(f"[INFO]   Resolution: {resolution_names.get(camera_resolution, 'VGA')}")
        print(f"[INFO]   FPS: {camera_fps}")
        print(f"[INFO]   Depth mode: NEURAL")
        print(f"[INFO]   Tracking accuracy: {accuracy_names.get(tracking_accuracy, 'MEDIUM')}")
        print(f"[INFO]   Max detection range: {max_detection_range}m")
        print(f"[INFO]   Prediction timeout: {prediction_timeout}s")
        print(f"[INFO]   Body fitting (mesh): {'enabled' if enable_body_fitting else 'disabled (--no-body-fitting)'}")
        
        # Skeleton ID continuity filter - prevents duplicate skeletons
        # Track each unique person with a stable ID
        # Tracker structure: {
        #   'current_sdk_id': int or None,
        #   'created': timestamp,
        #   'last_seen': timestamp,
        #   'sdk_id_start': timestamp,
        #   'sdk_ids': [history of previous SDK IDs],
        #   'last_position': [x, y, z] or None - pelvis position when went inactive
        # }
        self.tracked_skeletons = {}
        self.next_stable_id = 0
        # Persistent set of SDK IDs that were identified as ghosts
        # Once an SDK ID is a ghost, it stays a ghost (prevents reactivation with ghost IDs)
        self.ghost_sdk_ids = set()
        self.enable_skeleton_filter = enable_skeleton_filter
        if enable_skeleton_filter:
            print(f"[INFO] Skeleton ID continuity filter: enabled (prevents ghost duplicates)")
        else:
            print(f"[INFO] Skeleton ID continuity filter: DISABLED (raw ZED SDK output, may show duplicates)")
        
        # Video streaming configuration
        self.enable_streaming = enable_streaming
        # V2 per-camera streaming is always available (client-activated)
        self.per_camera_streaming_manager = None
        
        # Auto-detect: Use unicast for localhost-only, multicast if on LAN
        if stream_host is None:
            # Check if we have a real LAN IP (not localhost)
            if self.local_ip and self.local_ip not in ["127.0.0.1", "localhost", "::1"]:
                # We're on a real network - use multicast
                self.stream_host = "239.255.0.1"
            else:
                # Localhost only - use unicast
                self.stream_host = "127.0.0.1"
                print(f"[INFO] No LAN connection detected - using unicast streaming")
        elif stream_host in ["localhost", "127.0.0.1", "::1"]:
            # Force unicast for localhost testing
            self.stream_host = "127.0.0.1"
            print(f"[INFO] Using unicast streaming for localhost")
        else:
            self.stream_host = stream_host
            
        self.stream_rgb_port = stream_rgb_port
        self.stream_depth_port = stream_depth_port
        self.video_streamer = None
        
        if enable_streaming:
            stream_mode = "unicast" if self.stream_host == "127.0.0.1" else "multicast"
            print(f"[INFO] Video streaming enabled ({stream_mode})")
            print(f"[INFO]   Stream host: {self.stream_host}")
            print(f"[INFO]   Stream port: {self.stream_rgb_port}")
        
        # Server state
        self.running = True
        
        # V2 per-camera streaming control listener (starts immediately, lazy-init streamer)
        self.v2_control_thread = None
        self._start_v2_control_listener()
        
        # TCP server attributes
        self.server_socket = None
        self.server_thread = None
        self.clients = []
        self._client_queues = {}
        self._client_sender_threads = {}
        
        # UDP broadcast attributes
        self.udp_socket = None
        self.udp_clients = set()  # Set of (ip, port) tuples
        
        # Camera instances
        self.camera = None
        self.fusion = None
        self.is_fusion_mode = False
        
        # Body tracking filter - DISABLED (causes more problems than it solves)
        # The ZED SDK's own tracking is more reliable
        self.enable_body_filter = False
        self.body_filter = None
        print("[INFO] Body tracking filter: disabled (using ZED SDK raw output)")
        
        # Floor detection
        self.detected_floor_height = None
        
        # UI callback
        self.update_callback = None
        
        # Camera pose cache
        self._camera_pose_cache = None
        self._camera_pose_cache_time = 0.0
        # Make camera poses computed once at startup (no periodic recompute).
        # Set to None to indicate permanent cache.
        self._camera_pose_cache_ttl = None

        # No background floor detection here; follow SDK best-practice: detect once at init
        # Per-client send queues and sender threads (avoid per-frame thread creation)
        self._client_queues = {}  # socket -> Queue
        self._client_sender_threads = {}  # socket -> Thread
        
        # Initialize video streaming if enabled
        if self.enable_streaming:
            if not STREAMING_AVAILABLE:
                print("[WARNING] Video streaming requested but GStreamer not available")
                print("[WARNING] Install system packages: sudo apt-get install python3-gi gstreamer1.0-*")
                self.enable_streaming = False
            else:
                print(f"[INFO] Video streaming will be initialized after cameras (port: {self.stream_rgb_port})")

    def set_update_callback(self, callback: Callable):
        """Set callback function for UI updates (e.g., Qt viewer updates)"""
        self.update_callback = callback

    def start_tcp_server(self):
        """Start the server (TCP or UDP based on configuration)"""
        if self.use_udp:
            self._start_udp_server()
        else:
            self._start_tcp_server()
    
    def _start_udp_server(self):
        """Start UDP broadcast server"""
        try:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.udp_socket.bind((self.host, self.port))
            print(f"[INFO] UDP server listening on {self.host}:{self.port}")
            if self.host == "0.0.0.0":
                print(f"[INFO] Clients connect to: {self.local_ip}:{self.port}")
            
            # Start thread to listen for client handshakes
            def udp_listener():
                while self.running:
                    try:
                        data, addr = self.udp_socket.recvfrom(1024)
                        if data == b"HELLO":
                            # Client handshake - add to broadcast list
                            if addr not in self.udp_clients:
                                self.udp_clients.add(addr)
                                print(f"[INFO] UDP client registered from {addr}")
                            # Send acknowledgment
                            self.udp_socket.sendto(b"HELLO", addr)
                    except socket.timeout:
                        continue
                    except Exception as e:
                        if self.running:
                            print(f"[WARNING] UDP listener error: {e}")
                        break
            
            self.server_thread = threading.Thread(target=udp_listener, daemon=True)
            self.server_thread.start()
            
        except Exception as e:
            print(f"[ERROR] Failed to start UDP server: {e}")
    
    def _start_tcp_server(self):
        """Start the TCP server"""
        # Pre-populate camera pose cache when TCP server starts
        try:
            self.get_camera_poses()
        except Exception as e:
            print(f"[WARNING] Failed to pre-populate camera poses: {e}")
        
        def run_server():
            try:
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.server_socket.bind((self.host, self.port))
                self.server_socket.listen(5)
                print(f"[INFO] TCP server listening on {self.host}:{self.port}")
                if self.host == "0.0.0.0":
                    print(f"[INFO] Connect clients to: {self.local_ip}:{self.port}")
                
                while self.running:
                    try:
                        client_socket, addr = self.server_socket.accept()
                        client_socket.settimeout(5.0)
                        client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        print(f"[INFO] Client connected from {addr}")
                        
                        # Add client to the list
                        self.clients.append(client_socket)
                        
                        # Create queue and sender thread for this client
                        q = queue.Queue(maxsize=8)
                        self._client_queues[client_socket] = q
                        sender_thread = threading.Thread(target=self._client_sender_worker, args=(client_socket, q), daemon=True)
                        self._client_sender_threads[client_socket] = sender_thread
                        sender_thread.start()
                        
                        threading.Thread(target=self._client_handler, args=(client_socket, addr), daemon=True).start()
                    except socket.error:
                        if self.running:
                            print("[ERROR] Socket error in TCP server")
                        break
            except Exception as e:
                print(f"[ERROR] Failed to start TCP server: {e}")
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()

    def stop_tcp_server(self):
        """Stop the TCP server and close all connections"""
        self.running = False

        # Close all client connections
        for client in list(self.clients):
            try:
                client.close()
            except:
                pass
        self.clients.clear()

        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
            self.server_socket = None

        print("[SERVER] TCP server stopped")

    def _client_handler(self, conn, addr):
        """Handle individual client connection"""
        print(f"[CLIENT CONNECTED] {addr}")
        
        # Send server info to client (backward-compatible - old clients will ignore)
        self._send_server_info(conn)
        
        # If video streaming is enabled, notify the streamer about the new client
        if hasattr(self, 'video_streamer') and self.video_streamer is not None:
            client_ip = addr[0]  # Extract IP from (IP, port) tuple
            # Send heartbeat on behalf of the TCP client to trigger streaming
            try:
                import socket as sock_module
                heartbeat_socket = sock_module.socket(sock_module.AF_INET, sock_module.SOCK_DGRAM)
                heartbeat_port = self.video_streamer.stream_port + 100  # Heartbeat port
                # Send from the client's IP perspective (server sends to itself)
                heartbeat_socket.sendto(b"STREAMING_CLIENT", ('127.0.0.1', heartbeat_port))
                heartbeat_socket.close()
                print(f"[INFO] Auto-triggered video streaming for TCP client {client_ip}")
            except Exception as e:
                print(f"[WARNING] Failed to auto-trigger video streaming: {e}")
        
        try:
            while self.running:
                try:
                    data = conn.recv(1024)  # Read larger chunks
                    if not data:
                        break
                except socket.timeout:
                    continue
                except Exception:
                    break
        except:
            pass
        finally:
            print(f"[CLIENT DISCONNECTED] {addr}")
            if conn in self.clients:
                self.clients.remove(conn)
            # Clean up sender queue and thread mapping
            try:
                if conn in self._client_queues:
                    try:
                        # signal sender thread to exit
                        self._client_queues[conn].put_nowait(None)
                    except Exception:
                        pass
                    del self._client_queues[conn]
            except Exception:
                pass
            try:
                if conn in self._client_sender_threads:
                    del self._client_sender_threads[conn]
            except Exception:
                pass
            try:
                conn.close()
            except:
                pass

    def _client_sender_worker(self, conn, q: "queue.Queue"):
        """Worker thread that sends queued messages for a single client socket.
        Accepts either already-encoded bytes or a payload dict/object to serialize
        and send. Exits when the queue yields None or on socket failure.
        """
        def _serialize_message(msg):
            # msg is expected to be a dict/serializable object
            try:
                return (json.dumps(msg) + "\n").encode("utf-8")
            except TypeError:
                try:
                    def _fallback(o):
                        if hasattr(o, 'to_dict'):
                            return o.to_dict()
                        if hasattr(o, 'tolist'):
                            return list(o.tolist())
                        try:
                            return list(o)
                        except Exception:
                            return str(o)
                    return (json.dumps(msg, default=_fallback) + "\n").encode("utf-8")
                except Exception:
                    return None

        try:
            while self.running:
                try:
                    item = q.get(timeout=0.5)
                except Exception:
                    # timeout, check running flag
                    continue
                if item is None:
                    break

                # If item is raw bytes, send directly; otherwise serialize here.
                # Accept Frame objects, dicts, or pre-serialized bytes.
                if isinstance(item, (bytes, bytearray)):
                    msg_bytes = item
                else:
                    # If it's a Frame-like object, try to call to_dict() before serializing.
                    try:
                        # avoid importing Frame type here; duck-type
                        if hasattr(item, 'to_dict') and not isinstance(item, dict):
                            payload = item.to_dict()
                        else:
                            payload = item
                    except Exception:
                        # fallback: send string representation
                        payload = str(item)

                    msg_bytes = _serialize_message(payload)
                    if msg_bytes is None:
                        # serialization failed; drop this message for this client
                        continue

                try:
                    conn.sendall(msg_bytes)
                except Exception:
                    # On error, remove client and break
                    try:
                        if conn in self.clients:
                            self.clients.remove(conn)
                    except Exception:
                        pass
                    try:
                        conn.close()
                    except Exception:
                        pass
                    break
        except Exception:
            pass
        finally:
            # ensure queue entry and thread mapping removed
            try:
                if conn in self._client_queues:
                    del self._client_queues[conn]
            except Exception:
                pass
            try:
                if conn in self._client_sender_threads:
                    del self._client_sender_threads[conn]
            except Exception:
                pass
    
    def _send_server_info(self, conn):
        """Send server configuration info to newly connected client.
        
        Backward-compatible: Old clients will receive but can ignore this message.
        New clients use it to auto-configure streaming parameters.
        """
        try:
            # Use the actual stream host (multicast works on localhost with loop=true)
            effective_stream_host = self.stream_host
            
            info = {
                "type": "server_info",
                "data": {
                    "version": "2.0",  # Protocol version
                    "streaming": {
                        "enabled": hasattr(self, 'video_streamer') and self.video_streamer is not None,
                        "host": effective_stream_host if hasattr(self, 'stream_host') else "239.255.0.1",
                        "port": self.stream_rgb_port if hasattr(self, 'stream_rgb_port') else 5000,
                        "num_cameras": 0,
                        "camera_width": 1280,
                        "camera_height": 720,
                        "framerate": 60,
                        "depth_mode": "NEURAL"
                    },
                    "tcp_port": self.port
                }
            }
            
            # Populate actual camera info if available
            if hasattr(self, 'video_streamer') and self.video_streamer is not None:
                info["data"]["streaming"]["num_cameras"] = self.video_streamer.num_cameras
                info["data"]["streaming"]["camera_width"] = self.video_streamer.camera_width
                info["data"]["streaming"]["camera_height"] = self.video_streamer.camera_height
                info["data"]["streaming"]["framerate"] = self.video_streamer.framerate
            elif self.is_fusion_mode and hasattr(self, 'fusion') and self.fusion is not None:
                # Get camera info from fusion mode
                if hasattr(self, 'num_fusion_cameras'):
                    info["data"]["streaming"]["num_cameras"] = self.num_fusion_cameras
                # Get resolution from first sender
                try:
                    if hasattr(self, '_fusion_senders'):
                        for sender in self._fusion_senders.values():
                            camera_info = sender.get_camera_information()
                            res = camera_info.camera_configuration.resolution
                            fps = camera_info.camera_configuration.fps
                            info["data"]["streaming"]["camera_width"] = res.width
                            info["data"]["streaming"]["camera_height"] = res.height
                            info["data"]["streaming"]["framerate"] = fps
                            break
                except:
                    pass
            elif hasattr(self, 'camera') and self.camera is not None:
                # Get info from single camera
                info["data"]["streaming"]["num_cameras"] = 1
                try:
                    camera_info = self.camera.get_camera_information()
                    res = camera_info.camera_configuration.resolution
                    fps = camera_info.camera_configuration.fps
                    info["data"]["streaming"]["camera_width"] = res.width
                    info["data"]["streaming"]["camera_height"] = res.height
                    info["data"]["streaming"]["framerate"] = fps
                except:
                    pass
            
            # Get depth mode from camera if available
            if hasattr(self, 'camera') and self.camera is not None:
                try:
                    init_params = self.camera.get_init_parameters()
                    depth_mode = str(init_params.depth_mode).split('.')[-1]  # Extract enum name
                    info["data"]["streaming"]["depth_mode"] = depth_mode
                except:
                    pass
            elif self.is_fusion_mode and hasattr(self, '_fusion_senders'):
                # Try to get from fusion senders
                try:
                    for sender in self._fusion_senders.values():
                        init_params = sender.get_init_parameters()
                        depth_mode = str(init_params.depth_mode).split('.')[-1]
                        info["data"]["streaming"]["depth_mode"] = depth_mode
                        break
                except:
                    pass
            
            # Send as JSON
            message = json.dumps(info) + "\n"
            conn.sendall(message.encode('utf-8'))
            
            # Small delay to ensure server_info is received before skeleton frames start
            time.sleep(0.1)
            
            print(f"[INFO] Sent server info to client: {info['data']['streaming']['num_cameras']} cameras, "
                  f"{info['data']['streaming']['camera_width']}x{info['data']['streaming']['camera_height']}@{info['data']['streaming']['framerate']}fps, "
                  f"depth mode: {info['data']['streaming']['depth_mode']}")
        except Exception as e:
            print(f"[WARNING] Failed to send server info: {e}")

    def broadcast_frame(self, frame: Frame):
        """Broadcast a frame to all connected clients (TCP or UDP).
        
        Optimized to serialize the frame ONCE, then send the same bytes to all clients.
        This avoids redundant to_dict() and serialization calls per client.
        """
        if self.use_udp:
            if not self.udp_clients:
                return
        else:
            if not self.clients:
                return
        
        # OPTIMIZATION: Serialize frame ONCE for all clients
        try:
            frame_dict = frame.to_dict() if hasattr(frame, 'to_dict') else frame
            message_dict = {"type": "frame", "data": frame_dict}
            
            # Use the communication module's serialize_message for MessagePack+zstd support
            from .communication import serialize_message
            serialized_bytes = serialize_message(message_dict, use_msgpack=True, use_compression=True)
            
            # Check size limit for UDP (max ~65KB)
            if self.use_udp and len(serialized_bytes) > 65000:
                print(f"[WARNING] Frame too large for UDP ({len(serialized_bytes)} bytes), dropping")
                return
                
        except Exception as e:
            print(f"[Server] Frame serialization failed: {e}")
            return
        
        # Broadcast based on protocol
        if self.use_udp:
            # UDP: Send to all registered clients
            for client_addr in list(self.udp_clients):
                try:
                    self.udp_socket.sendto(serialized_bytes, client_addr)
                except Exception as e:
                    # Remove dead client
                    print(f"[WARNING] Failed to send to UDP client {client_addr}: {e}")
                    self.udp_clients.discard(client_addr)
        else:
            # TCP: Send via queues (existing code)
            for client in list(self.clients):
                q = self._client_queues.get(client)
                if q is None:
                    # No queue: send directly (legacy path for clients without queues)
                    try:
                        client.sendall(serialized_bytes)
                    except Exception as e:
                        try:
                            if client in self.clients:
                                self.clients.remove(client)
                        except Exception:
                            pass
                        try:
                            client.close()
                        except Exception:
                            pass
                    continue

                try:
                    # Put the pre-serialized bytes in the queue (not the Frame object!)
                    q.put_nowait(serialized_bytes)
                except queue.Full:
                    # drop message for this slow client
                    pass

    def find_floor_plane(self, cam=None):
        """Detect floor plane using ZED SDK - works for both fusion and single camera mode"""
        # Try multiple times with short backoff to improve reliability
        attempts = 3
        wait_secs = 0.2
        self.detected_floor_height = None
        
        for attempt in range(attempts):
            try:
                plane = sl.Plane()
                reset_tracking_floor_frame = sl.Transform()
                
                # Choose detection method based on mode
                if self.is_fusion_mode and hasattr(self, '_fusion_senders') and self._fusion_senders:
                    # Use first available fusion sender camera for floor detection
                    first_serial = next(iter(self._fusion_senders.keys()))
                    first_cam = self._fusion_senders[first_serial]
                    status = first_cam.find_floor_plane(plane, reset_tracking_floor_frame)
                elif cam is not None:
                    # Use provided camera for floor detection
                    status = cam.find_floor_plane(plane, reset_tracking_floor_frame)
                elif not self.is_fusion_mode and self.camera is not None:
                    # Use single camera mode
                    status = self.camera.find_floor_plane(plane, reset_tracking_floor_frame)
                else:
                    print("[WARNING] No camera available for floor detection")
                    return
                
                if status == sl.ERROR_CODE.SUCCESS:
                    
                    # Extract the floor height from the plane
                    try:
                        # Try to get the plane center point (this gives us the floor level)
                        plane_center = plane.get_center()
                        if plane_center is not None:
                            # Y coordinate of the plane center gives us the floor height
                            self.detected_floor_height = float(plane_center[1])
                        else:
                            # Fallback: try to get bounds and calculate center
                            bounds = plane.get_bounds()
                            if bounds is not None and len(bounds) > 0:
                                # Calculate average Y coordinate from bounds
                                y_coords = [point[1] for point in bounds]
                                self.detected_floor_height = float(sum(y_coords) / len(y_coords))
                            else:
                                # Last resort: use plane equation
                                plane_eq = plane.get_plane_equation()
                                if plane_eq is not None and len(plane_eq) >= 4:
                                    a, b, c, d = plane_eq[0], plane_eq[1], plane_eq[2], plane_eq[3]
                                    # For plane equation ax + by + cz + d = 0, 
                                    # Y intercept at x=0, z=0 is y = -d/b
                                    if abs(b) > 1e-6:
                                        self.detected_floor_height = float(-d / b)
                                    else:
                                        print("[WARNING] Invalid plane equation (b coefficient too small)")
                                        self.detected_floor_height = 0.0
                                else:
                                    print("[WARNING] Could not get plane equation")
                                    self.detected_floor_height = 0.0
                    except Exception as e:
                        print(f"[WARNING] Error extracting floor height: {e}")
                        # Fallback to 0.0
                        self.detected_floor_height = 0.0
                    
                    if self.detected_floor_height is not None:
                        if self.is_fusion_mode and hasattr(self, '_fusion_senders'):
                            detection_source = f"fusion sender camera {first_serial}"
                        else:
                            detection_source = "camera"
                        
                        # Convert to millimeters if we're using UNIT.MILLIMETER coordinate system
                        # Check if the value seems to be in meters (typically floor height is < 5 meters)
                        if abs(self.detected_floor_height) < 10.0:
                            # Likely in meters, convert to mm
                            self.detected_floor_height *= 1000.0
                            print(f"[INFO] Floor detected at height: {self.detected_floor_height} mm (via {detection_source}, converted from meters)")
                        else:
                            print(f"[INFO] Floor detected at height: {self.detected_floor_height} mm (via {detection_source})")
                        
                        self._last_floor_detect_time = time.time()
                    break
                else:
                    # Keep trying
                    time.sleep(wait_secs)
                    wait_secs *= 2
            except Exception as e:
                print(f"[WARNING] Floor plane detection error on attempt {attempt + 1}: {e}")
                time.sleep(wait_secs)
                wait_secs *= 2
                
        if self.detected_floor_height is None:
            if self.is_fusion_mode and hasattr(self, '_fusion_senders'):
                detection_source = f"fusion sender cameras"
            else:
                detection_source = "camera"
            print(f"[WARNING] Floor plane detection failed after {attempts} retries (via {detection_source})")
        else:
            self._last_floor_detect_time = time.time()

    def get_detected_floor_height(self) -> Optional[float]:
        """Return the detected floor height from ZED SDK, or None if not detected."""
        return self.detected_floor_height


    def get_camera_poses(self):
        """
        Return a dict with camera poses and floor info, converted into OpenGL coords.
        - Detect floor first
        - In fusion mode: read from calibration file and adjust for floor
        - In single camera mode: use SDK defaults
        """
        if self._camera_pose_cache is not None:
            return self._camera_pose_cache

        poses = []
        floor_info = None

        # --- Detect floor height first ---
        detected_floor_height = None
        try:
            self.find_floor_plane()
            if self.detected_floor_height is not None:
                detected_floor_height = self.detected_floor_height
                detection_source = "fusion" if self.is_fusion_mode else "camera"
                print(f"[INFO] Floor detected at height {detected_floor_height} mm (via {detection_source})")
                floor_info = {"height": detected_floor_height}
        except Exception as e:
            print(f"[WARNING] Floor detection error: {e}")

        floor_height_mm = detected_floor_height or self.detected_floor_height or 0.0

        if self.is_fusion_mode:
            try:
                calib_file = self._find_calibration_file()
                if not calib_file:
                    print("[WARNING] No calibration file found for camera poses")
                    return {"cameras": poses, "floor": floor_info}

                with open(calib_file, 'r') as f:
                    calib_data = json.load(f)

                for key, entry in calib_data.items():
                    try:
                        fusion_config = entry.get('FusionConfiguration', {})
                        serial = fusion_config.get('serial_number', key)
                        pose_data = fusion_config.get('pose')
                        if not pose_data:
                            continue

                        # Parse pose string into flat list of 16 floats
                        if isinstance(pose_data, str):
                            values = pose_data.strip().split()
                            if len(values) != 16:
                                print(f"[WARNING] Expected 16 values for {serial}, got {len(values)}")
                                continue
                            flat_matrix = [float(v) for v in values]
                            matrix = [
                                flat_matrix[0:4],
                                flat_matrix[4:8],
                                flat_matrix[8:12],
                                flat_matrix[12:16],
                            ]
                        else:
                            continue

                        # --- Convert to OpenGL coords (F * M * F) ---
                        F = [
                            [1, 0,  0, 0],
                            [0, 1,  0, 0],
                            [0, 0, -1, 0],
                            [0, 0,  0, 1]
                        ]

                        def matmul(A, B):
                            return [[sum(A[i][k]*B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]

                        M_zed = matrix
                        M_gl = matmul(matmul(F, M_zed), F)

                        # --- Extract position (mm) ---
                        position = {
                            'x': M_gl[0][3] * 1000.0,
#                            'y': M_gl[1][3] * 1000.0 - (2 * floor_height_mm),
                            'y': -(M_gl[1][3] * 1000.0) - floor_height_mm,
                            'z': M_gl[2][3] * 1000.0
                        }

                        # --- Extract rotation matrix ---
                        R = [
                            [M_gl[0][0], M_gl[0][1], M_gl[0][2]],
                            [M_gl[1][0], M_gl[1][1], M_gl[1][2]],
                            [M_gl[2][0], M_gl[2][1], M_gl[2][2]]
                        ]

                        # --- Convert to quaternion ---
                        quat = self._matrix_to_quaternion(R)  # returns (x,y,z,w)

                        orientation = {
                            'x': quat[0],
                            'y': quat[1],
                            'z': quat[2],
                            'w': quat[3]
                        }

                        poses.append({
                            "serial": str(serial),
                            "position": position,
                            "orientation": orientation
                        })

                        print(f"[DEBUG] Camera {serial}: pos={position}, quat={orientation}")

                    except Exception as e:
                        print(f"[WARNING] Failed to parse pose for {key}: {e}")

            except Exception as e:
                print(f"[WARNING] Failed to read calibration file: {e}")

        else:
            # Single camera mode
            try:
                serial = str(self.camera.get_camera_information().serial_number)
            except Exception:
                serial = "unknown"

            poses.append({
                "serial": serial,
                "position": {'x': 0.0, 'y': 0.0, 'z': 0.0},
                "orientation": {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
            })

        result = {"cameras": poses, "floor": floor_info}
        self._camera_pose_cache = result
        print(f"[INFO] Cached {len(poses)} camera poses with floor height {floor_height_mm} mm")
        return result

    def _try_get_fusion_camera_pose(self, serial):
        """Return sl.Pose or None. Prints status for debugging."""
        if not getattr(self, "fusion", None):
            print("[WARNING] Fusion not initialized")
            return None

        try:
            # reuse stored CameraIdentifier if available
            uid = None
            if hasattr(self, "_fusion_camera_identifiers"):
                for cid in self._fusion_camera_identifiers:
                    try:
                        if str(getattr(cid, "serial_number", "")) == str(serial):
                            uid = cid
                            break
                    except Exception:
                        continue

            # build identifier if we don't have one saved
            if uid is None:
                uid = sl.CameraIdentifier()
                try:
                    uid.serial_number = int(serial)
                except Exception:
                    uid.serial_number = serial

            # NOTE: fusion.get_position expects a sl.Pose, not sl.Transform
            pose = sl.Pose()
            status = self.fusion.get_position(pose, sl.REFERENCE_FRAME.WORLD)
            print(f"[DEBUG] fusion.get_position status: {status}")

            if status == sl.FUSION_ERROR_CODE.SUCCESS:
                print(f"[DEBUG] pose: {pose}")
                return pose
            else:
                print(f"[WARNING] fusion.get_position failed for {serial}: {status}")
                return None

        except Exception as e:
            print(f"[ERROR] exception calling fusion.get_position: {e}")
            return None

    def _ensure_frame_has_camera(self, frame):
        """Ensure frame has camera information for single-camera setups"""
        if frame.cameras:
            return  # already has cameras

        # For single camera setups, add a default camera if none present
        if not self.is_fusion_mode:
            try:
                poses = self.get_camera_poses()
                if poses:
                    frame.cameras = poses
                else:
                    # Fallback: create a simple default camera
                    try:
                        serial = "default"
                        if self.camera is not None:
                            serial = str(self.camera.get_camera_information().serial_number)
                    except Exception:
                        pass

                    frame.cameras = [{
                        'serial': serial,
                        'position': (0.0, 0.0, 0.0),
                        'target': (0.0, 0.0, -200.0)
                    }]
            except Exception:
                pass

    def initialize_cameras(self, serial_numbers: Optional[List[str]] = None,
                         enable_body_tracking: bool = True,
                         enable_floor_detection: bool = True) -> bool:
        """Initialize cameras for body tracking"""
        if serial_numbers is None or len(serial_numbers) == 0:
            # Auto-detect cameras
            try:
                device_list = sl.Camera.get_device_list()
                if len(device_list) == 0:
                    print("[ERROR] No ZED cameras detected")
                    return False

                if len(device_list) == 1:
                    # Single camera mode
                    return self._initialize_single_camera(device_list[0], enable_body_tracking, enable_floor_detection)
                else:
                    # Multi-camera fusion mode - require calibration file
                    return self._initialize_fusion_cameras(device_list, enable_body_tracking, enable_floor_detection)
            except Exception as e:
                print(f"[ERROR] Failed to detect cameras: {e}")
                return False
        else:
            # Use specified serial numbers
            if len(serial_numbers) == 1:
                return self._initialize_single_camera_by_serial(serial_numbers[0], enable_body_tracking, enable_floor_detection)
            else:
                # Multi-camera fusion by serials - require calibration file
                calib_file = self._find_calibration_file()
                if not calib_file:
                    print("[ERROR] Fusion mode requires calibration file in server/calib/ directory")
                    return False
                return self._initialize_fusion_cameras_by_serial(serial_numbers, enable_body_tracking, enable_floor_detection)

    def _initialize_single_camera(self, device_info, enable_body_tracking: bool = True, enable_floor_detection: bool = True) -> bool:
        """Initialize single camera for body tracking"""
        try:
            self.camera = sl.Camera()

            # Set initialization parameters
            init_params = sl.InitParameters()
            init_params.camera_resolution = self.CAMERA_RESOLUTION
            init_params.camera_fps = self.CAMERA_FPS
            init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # NEURAL mode (recommended by ZED SDK)
            init_params.coordinate_units = sl.UNIT.MILLIMETER
            init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
            init_params.depth_stabilization = 1  # Enable depth stabilization (int: 0=off, 1=on)

            # Open camera
            status = self.camera.open(init_params)
            if status != sl.ERROR_CODE.SUCCESS:
                print(f"[ERROR] Failed to open camera: {status}")
                return False

            # Enable positional tracking (required for body tracking)
            tracking_params = sl.PositionalTrackingParameters()
            tracking_params.set_as_static = True  # Camera is static, improves tracking stability
            status = self.camera.enable_positional_tracking(tracking_params)
            if status != sl.ERROR_CODE.SUCCESS:
                print(f"[ERROR] Failed to enable positional tracking: {status}")
                self.camera.close()
                return False

            # Enable body tracking if requested
            if enable_body_tracking:
                body_params = sl.BodyTrackingParameters()
                body_params.detection_model = self.body_tracking_model  # Use runtime-configured model
                body_params.body_format = sl.BODY_FORMAT.BODY_34
                body_params.enable_body_fitting = self.enable_body_fitting  # Enabled by default for BODY_34 mesh data
                body_params.enable_tracking = True
                body_params.prediction_timeout_s = self.prediction_timeout  # Use runtime-configured timeout
                body_params.max_range = self.max_detection_range  # Use runtime-configured range

                status = self.camera.enable_body_tracking(body_params)
                if status != sl.ERROR_CODE.SUCCESS:
                    print(f"[ERROR] Failed to enable body tracking: {status}")
                    self.camera.close()
                    return False

            # Detect floor if requested
            if enable_floor_detection:
                self.find_floor_plane(self.camera)

            self.is_fusion_mode = False
            print(f"[INFO] Single camera initialized successfully")
            
            # Initialize video streaming if enabled
            if self.enable_streaming:
                self._initialize_video_streamer(num_cameras=1)
            
            return True

        except Exception as e:
            print(f"[ERROR] Failed to initialize single camera: {e}")
            return False

    def _initialize_single_camera_by_serial(self, serial: str, enable_body_tracking: bool = True, enable_floor_detection: bool = True) -> bool:
        """Initialize single camera by serial number"""
        try:
            device_list = sl.Camera.get_device_list()
            for device in device_list:
                if str(device.serial_number) == str(serial):
                    return self._initialize_single_camera(device, enable_body_tracking, enable_floor_detection)

            print(f"[ERROR] Camera with serial {serial} not found")
            return False
        except Exception as e:
            print(f"[ERROR] Failed to find camera by serial: {e}")
            return False

    @staticmethod
    def pose_to_qt(pose_values):
        """
        Convert a 16-element list or space-separated string into
        OpenGL position (QVector3D) + orientation (QQuaternion),
        using only Qt classes.

        Args:
            pose_values: list of 16 floats OR space-separated string

        Returns:
            (QVector3D, QQuaternion)
        """
        import math

        # Parse input
        if isinstance(pose_values, str):
            pose_values = list(map(float, pose_values.strip().split()))
        if len(pose_values) != 16:
            raise ValueError("Pose must have 16 values")

        # Convert to 4x4 row-major matrix
        m = [pose_values[i*4:(i+1)*4] for i in range(4)]

        # Flip Z axis for OpenGL (multiply by diag(1,1,-1,1) on both sides)
        F = [
            [1, 0,  0, 0],
            [0, 1,  0, 0],
            [0, 0, -1, 0],
            [0, 0,  0, 1]
        ]
        # M_gl = F * m * F
        def mat_mult(a, b):
            return [
                [sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)]
                for i in range(4)
            ]
        M_gl = mat_mult(F, mat_mult(m, F))

        # Extract translation
        tx = M_gl[0][3]
        ty = M_gl[1][3]
        tz = M_gl[2][3]
        pos = QVector3D(tx, ty, tz)

        # Extract rotation (top-left 3x3)
        r00, r01, r02 = M_gl[0][0], M_gl[0][1], M_gl[0][2]
        r10, r11, r12 = M_gl[1][0], M_gl[1][1], M_gl[1][2]
        r20, r21, r22 = M_gl[2][0], M_gl[2][1], M_gl[2][2]

        # Quaternion from rotation matrix (robust, normalized)
        trace = r00 + r11 + r22
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (r21 - r12) * s
            y = (r02 - r20) * s
            z = (r10 - r01) * s
        elif r00 > r11 and r00 > r22:
            s = 2.0 * math.sqrt(1.0 + r00 - r11 - r22)
            w = (r21 - r12) / s
            x = 0.25 * s
            y = (r01 + r10) / s
            z = (r02 + r20) / s
        elif r11 > r22:
            s = 2.0 * math.sqrt(1.0 + r11 - r00 - r22)
            w = (r02 - r20) / s
            x = (r01 + r10) / s
            y = 0.25 * s
            z = (r12 + r21) / s
        else:
            s = 2.0 * math.sqrt(1.0 + r22 - r00 - r11)
            w = (r10 - r01) / s
            x = (r02 + r20) / s
            y = (r12 + r21) / s
            z = 0.25 * s

        quat = QQuaternion(w, x, y, z)  # Qt expects (scalar=w, x, y, z)

        return pos, quat

    def _initialize_video_streamer(self, num_cameras: int):
        """Initialize video streamer after cameras are ready"""
        if not STREAMING_AVAILABLE:
            print("[WARNING] Video streaming not available - install required dependencies")
            return
        
        try:
            print(f"[INFO] Initializing video streamer for {num_cameras} camera(s)")
            print(f"[INFO] All streams multiplexed on single port: {self.stream_rgb_port}")
            
            # Get camera resolution from actual camera
            camera_width = 1280  # Default fallback
            camera_height = 720  # Default fallback
            framerate = 60  # Default fallback
            
            # Try to get actual resolution from camera
            if self.is_fusion_mode and hasattr(self, '_fusion_senders') and self._fusion_senders:
                # Get resolution from first fusion sender camera
                first_cam = next(iter(self._fusion_senders.values()))
                camera_info = first_cam.get_camera_information()
                res = camera_info.camera_configuration.resolution
                camera_width = res.width
                camera_height = res.height
                fps = camera_info.camera_configuration.fps
                framerate = fps
                print(f"[INFO] Detected fusion camera resolution: {camera_width}x{camera_height}@{framerate}fps")
            elif hasattr(self, 'camera') and self.camera is not None:
                # Get resolution from single camera
                camera_info = self.camera.get_camera_information()
                res = camera_info.camera_configuration.resolution
                camera_width = res.width
                camera_height = res.height
                fps = camera_info.camera_configuration.fps
                framerate = fps
                print(f"[INFO] Detected camera resolution: {camera_width}x{camera_height}@{framerate}fps")
            else:
                print(f"[WARNING] Could not detect camera resolution, using default: {camera_width}x{camera_height}@{framerate}fps")
            
            # Cap streaming framerate at 30 FPS for bandwidth efficiency
            streaming_framerate = min(framerate, 30)
            if streaming_framerate != framerate:
                print(f"[INFO] Streaming framerate capped at {streaming_framerate}fps (camera runs at {framerate}fps)")
            
            self.video_streamer = MultiCameraVideoStreamer(
                stream_port=self.stream_rgb_port,  # Use RGB port as the single stream port
                num_cameras=num_cameras,
                host=self.stream_host,
                camera_width=camera_width,
                camera_height=camera_height,
                framerate=streaming_framerate,
                enable_client_detection=False  # Using udpsink - no client detection needed
            )
            
            print(f"[INFO] Video streamer created with host={self.stream_host}, port={self.stream_rgb_port}")
            
            # Start streaming immediately (using udpsink, no heartbeat needed)
            self.video_streamer.start()
            
            print("[INFO] Video streamer started - pipelines should be PLAYING")
            
            # Test: Push a test frame to verify pipeline works
            try:
                import numpy as np
                test_rgb = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
                test_depth = np.zeros((camera_height, camera_width), dtype=np.float32)
                test_frames_rgb = [test_rgb] * num_cameras
                test_frames_depth = [test_depth] * num_cameras
                print(f"[DEBUG] Pushing test frame to verify pipeline...")
                self.video_streamer.push_camera_frames(test_frames_rgb, test_frames_depth)
                print(f"[DEBUG] Test frame pushed successfully!")
            except Exception as e:
                print(f"[ERROR] Failed to push test frame: {e}")
                import traceback
                traceback.print_exc()
            
            print("[INFO] Video streamer initialized successfully")
        except Exception as e:
            print(f"[ERROR] Failed to initialize video streamer: {e}")
            self.video_streamer = None
    
    def _start_v2_control_listener(self):
        """Start TCP control listener for v2 per-camera streaming requests"""
        # Track active client handler sockets for cleanup
        self.v2_client_sockets = []
        self.v2_client_sockets_lock = threading.Lock()
        
        self.v2_control_thread = threading.Thread(
            target=self._v2_control_loop,
            daemon=True
        )
        self.v2_control_thread.start()
    
    def _v2_control_loop(self):
        """Control listener for v2 streaming requests (port = main_port + 1)"""
        control_port = self.port + 1
        
        try:
            self.v2_control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.v2_control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.v2_control_socket.bind((self.host, control_port))
            self.v2_control_socket.listen(5)
            
            print(f"[INFO] V2 streaming control listener on port {control_port}")
            
            while self.running:
                try:
                    self.v2_control_socket.settimeout(1.0)
                    client_sock, client_addr = self.v2_control_socket.accept()
                    
                    # Handle in separate thread
                    threading.Thread(
                        target=self._handle_v2_stream_request,
                        args=(client_sock, client_addr),
                        daemon=True
                    ).start()
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"[WARNING] V2 control listener error: {e}")
            
            self.v2_control_socket.close()
        except Exception as e:
            print(f"[ERROR] Failed to start V2 control listener: {e}")
    
    def _handle_v2_stream_request(self, client_sock: socket.socket, client_addr: tuple):
        """Handle v2 stream request from client - keeps connection open until client disconnects"""
        client_ip, client_port = client_addr
        client_id = f"{client_ip}:{client_port}"
        
        # Register this socket for cleanup
        with self.v2_client_sockets_lock:
            self.v2_client_sockets.append(client_sock)
        
        try:
            # Receive request
            data = client_sock.recv(4096)
            if not data:
                return
            
            request_data = json.loads(data.decode('utf-8'))
            print(f"[INFO] V2 stream request from {client_id}: {request_data}")
            
            # Parse cameras (if not specified, use all cameras)
            requested_cameras = request_data.get('cameras')
            fps_factor = request_data.get('fps_factor', 2)
            
            # Determine total cameras available
            if self.is_fusion_mode and hasattr(self, 'num_fusion_cameras'):
                total_cameras = self.num_fusion_cameras
            elif hasattr(self, 'camera') and self.camera is not None:
                total_cameras = 1
            else:
                response = {"status": "error", "message": "No cameras initialized"}
                client_sock.sendall(json.dumps(response).encode('utf-8'))
                return
            
            # If no cameras specified, stream all
            if requested_cameras is None or len(requested_cameras) == 0:
                cameras = list(range(total_cameras))
                print(f"[INFO] No cameras specified, streaming all {total_cameras} cameras")
            else:
                cameras = requested_cameras
                # Validate camera indices
                if max(cameras) >= total_cameras:
                    response = {"status": "error", 
                               "message": f"Invalid camera index (max: {total_cameras-1})"}
                    client_sock.sendall(json.dumps(response).encode('utf-8'))
                    return
            
            # Lazy-initialize v2 streaming if not already done
            if self.per_camera_streaming_manager is None:
                print("[INFO] Lazy-initializing V2 per-camera streaming...")
                
                # Get camera parameters
                if self.is_fusion_mode and hasattr(self, '_fusion_senders'):
                    # Get from fusion sender
                    for sender in self._fusion_senders.values():
                        camera_info = sender.get_camera_information()
                        res = camera_info.camera_configuration.resolution
                        fps = camera_info.camera_configuration.fps
                        camera_width = res.width
                        camera_height = res.height
                        camera_fps = fps
                        break
                elif hasattr(self, 'camera') and self.camera is not None:
                    camera_info = self.camera.get_camera_information()
                    res = camera_info.camera_configuration.resolution
                    fps = camera_info.camera_configuration.fps
                    camera_width = res.width
                    camera_height = res.height
                    camera_fps = fps
                else:
                    response = {"status": "error", "message": "Camera info not available"}
                    client_sock.sendall(json.dumps(response).encode('utf-8'))
                    return
                
                # Initialize streaming manager
                self._initialize_per_camera_streaming(
                    num_cameras=total_cameras,
                    camera_width=camera_width,
                    camera_height=camera_height,
                    camera_fps=camera_fps
                )
                
                if self.per_camera_streaming_manager is None:
                    response = {"status": "error", "message": "Failed to initialize streaming"}
                    client_sock.sendall(json.dumps(response).encode('utf-8'))
                    return
            
            # Create stream request
            from .video_streaming_v2 import StreamRequest
            stream_request = StreamRequest(cameras=cameras, fps_factor=fps_factor)
            
            # Add client to streaming manager
            success = self.per_camera_streaming_manager.streamer.add_client(
                client_id=client_id,
                client_ip=client_ip,
                stream_request=stream_request
            )
            
            if success:
                response = {
                    "status": "ok",
                    "base_port": 5000,
                    "cameras": cameras,
                    "fps_factor": fps_factor,
                    "num_cameras": total_cameras
                }
                print(f"[INFO] V2 streaming started for {client_id}: cameras {cameras} @ fps/{fps_factor}")
            else:
                response = {"status": "error", "message": "Failed to add client"}
            
            client_sock.sendall(json.dumps(response).encode('utf-8'))
            
            if success:
                # Keep connection open - when it closes, cleanup the stream
                print(f"[INFO] Keeping connection open for {client_id}, will cleanup on disconnect")
                try:
                    # Set a keepalive timeout
                    client_sock.settimeout(1.0)  # Short timeout for responsive shutdown
                    # Wait for disconnect or keepalive
                    while self.running:
                        try:
                            data = client_sock.recv(1024)
                            if not data:
                                # Client disconnected
                                print(f"[INFO] Client {client_id} disconnected, cleaning up stream")
                                break
                        except socket.timeout:
                            # Timeout - check if server is still running
                            continue
                except Exception as e:
                    print(f"[INFO] Client {client_id} connection lost: {e}")
                
                # Cleanup: remove client from streaming
                if self.per_camera_streaming_manager:
                    self.per_camera_streaming_manager.streamer.remove_client(client_id)
                    print(f"[INFO] Removed streaming client {client_id}")
        
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON from {client_id}: {e}")
            try:
                response = {"status": "error", "message": "Invalid JSON"}
                client_sock.sendall(json.dumps(response).encode('utf-8'))
            except:
                pass
        except Exception as e:
            print(f"[ERROR] Error handling V2 stream request from {client_id}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Unregister socket
            with self.v2_client_sockets_lock:
                if client_sock in self.v2_client_sockets:
                    self.v2_client_sockets.remove(client_sock)
            
            try:
                client_sock.close()
            except:
                pass
    
    def _initialize_per_camera_streaming(self, num_cameras: int, camera_width: int,
                                         camera_height: int, camera_fps: int):
        """Initialize v2 per-camera streaming manager (lazy - called on first client request)"""
        if not STREAMING_V2_AVAILABLE:
            print("[ERROR] Per-camera streaming (v2) not available - missing dependencies")
            return
        
        try:
            from .video_streaming_v2 import PerCameraVideoStreamer
            
            # Create a simple wrapper to hold the streamer
            class StreamingWrapper:
                def __init__(self, streamer):
                    self.streamer = streamer
                
                def push_frames(self, rgb_frames, depth_frames):
                    self.streamer.push_frames(rgb_frames, depth_frames)
                
                def stop(self):
                    self.streamer.shutdown()
            
            streamer = PerCameraVideoStreamer(
                num_cameras=num_cameras,
                camera_width=camera_width,
                camera_height=camera_height,
                camera_fps=camera_fps,
                base_port=5000
            )
            
            self.per_camera_streaming_manager = StreamingWrapper(streamer)
            
            print(f"[INFO] Per-camera streaming (v2) initialized:")
            print(f"[INFO]   Cameras: {num_cameras}, Resolution: {camera_width}x{camera_height}@{camera_fps}fps")
            print(f"[INFO]   Base RTP port: 5000")
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize per-camera streaming: {e}")
            import traceback
            traceback.print_exc()
            self.per_camera_streaming_manager = None
    
    def _initialize_fusion_cameras(self, device_list, enable_body_tracking: bool = True, enable_floor_detection: bool = True) -> bool:
        """Initialize multiple cameras for fusion mode using a fusion configuration (calib) file."""
        try:
            # Find newest calibration file
            calib_file = self._find_calibration_file()
            if not calib_file:
                print("[ERROR] Fusion mode requires calibration file in server/calib/ directory")
                return False

            # Read fusion configurations using ZED SDK helper
            fusion_configurations = sl.read_fusion_configuration_file(
                calib_file,
                sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP,
                sl.UNIT.MILLIMETER
            )
            if not fusion_configurations:
                print("[ERROR] No valid fusion configurations found in calib file")
                return False

            print(f"[INFO] Found {len(fusion_configurations)} cameras in configuration")

            # Common parameters
            init_params = sl.InitParameters()
            init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
            init_params.coordinate_units = sl.UNIT.MILLIMETER
            init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # NEURAL mode (recommended by ZED SDK)
            init_params.camera_resolution = self.CAMERA_RESOLUTION
            init_params.camera_fps = self.CAMERA_FPS
            init_params.depth_stabilization = 1  # Enable depth stabilization (int: 0=off, 1=on)

            communication_parameters = sl.CommunicationParameters()
            communication_parameters.set_for_shared_memory()

            positional_tracking_parameters = sl.PositionalTrackingParameters()
            positional_tracking_parameters.set_as_static = True

            body_tracking_parameters = sl.BodyTrackingParameters()
            body_tracking_parameters.detection_model = self.body_tracking_model  # Use runtime-configured model
            body_tracking_parameters.body_format = sl.BODY_FORMAT.BODY_34
            body_tracking_parameters.enable_body_fitting = self.enable_body_fitting  # Enabled by default for BODY_34 mesh data
            body_tracking_parameters.enable_tracking = True
            body_tracking_parameters.prediction_timeout_s = self.prediction_timeout  # Use runtime-configured timeout
            body_tracking_parameters.max_range = self.max_detection_range  # Use runtime-configured range
            # Removed allow_reduced_precision_inference - let SDK use defaults for stability

            # Start local senders
            senders = {}
            for conf in fusion_configurations:
                serial = conf.serial_number
                print(f"[INFO] Try to open ZED {serial}")
                
                # Skip network cameras (already running)
                if conf.communication_parameters.comm_type == sl.COMM_TYPE.LOCAL_NETWORK:
                    print(f"[INFO] Camera {serial} is network sender")
                    continue

                # Open local camera
                init_params.input = conf.input_type
                init_params.set_from_serial_number(serial)

                cam = sl.Camera()
                status = cam.open(init_params)
                if status != sl.ERROR_CODE.SUCCESS:
                    print(f"[WARNING] Error opening camera {serial}: {status}")
                    continue

                # Enable positional tracking
                status = cam.enable_positional_tracking(positional_tracking_parameters)
                if status != sl.ERROR_CODE.SUCCESS:
                    print(f"[WARNING] Error enabling positional tracking for {serial}")
                    cam.close()
                    continue

                # Enable body tracking on each camera (required for fusion)
                # Fusion merges body detections from all cameras
                status = cam.enable_body_tracking(body_tracking_parameters)
                if status != sl.ERROR_CODE.SUCCESS:
                    print(f"[WARNING] Error enabling body tracking for {serial}")
                    cam.close()
                    continue

                # Start publishing
                cam.start_publishing(communication_parameters)
                senders[serial] = cam
                print(f"[INFO] Camera {serial} is open and publishing")

            if not senders:
                print("[ERROR] No cameras could be started")
                return False

            # Warmup - grab a few frames from each sender
            print("[INFO] Senders started")
            bodies = sl.Bodies()
            for serial, cam in senders.items():
                for _ in range(3):
                    if cam.grab() == sl.ERROR_CODE.SUCCESS:
                        try:
                            cam.retrieve_bodies(bodies)
                        except:
                            pass
                        break
                    time.sleep(0.05)

            # Initialize Fusion
            print("[INFO] Running the fusion...")
            init_fusion_parameters = sl.InitFusionParameters()
            init_fusion_parameters.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
            init_fusion_parameters.coordinate_units = sl.UNIT.MILLIMETER
            init_fusion_parameters.output_performance_metrics = False
            init_fusion_parameters.verbose = True

            self.fusion = sl.Fusion()
            status = self.fusion.init(init_fusion_parameters)
            if status != sl.FUSION_ERROR_CODE.SUCCESS:
                print(f"[ERROR] Failed to initialize fusion: {status}")
                for cam in senders.values():
                    cam.close()
                return False

            # Subscribe to cameras
            camera_identifiers = []
            for conf in fusion_configurations:
                uuid = sl.CameraIdentifier()
                uuid.serial_number = conf.serial_number
                
                print(f"[INFO] Subscribing to {conf.serial_number}")
                status = self.fusion.subscribe(uuid, conf.communication_parameters, conf.pose)
                
                if status == sl.FUSION_ERROR_CODE.SUCCESS:
                    camera_identifiers.append(uuid)
                    print(f"[INFO] Subscribed to {conf.serial_number}")
                else:
                    print(f"[WARNING] Unable to subscribe to {conf.serial_number}: {status}")
                    
            # store identifiers so later lookups use the exact SDK object/type
            self._fusion_camera_identifiers = camera_identifiers

            if not camera_identifiers:
                print("[ERROR] No cameras subscribed to fusion")
                for cam in senders.values():
                    cam.close()
                self.fusion.close()
                return False

            # Enable fusion body tracking
            if enable_body_tracking:
                body_tracking_fusion_params = sl.BodyTrackingFusionParameters()
                body_tracking_fusion_params.enable_tracking = True
                body_tracking_fusion_params.enable_body_fitting = self.enable_body_fitting  # Enabled by default for BODY_34 mesh data
                
                status = self.fusion.enable_body_tracking(body_tracking_fusion_params)
                if status != sl.FUSION_ERROR_CODE.SUCCESS:
                    print(f"[ERROR] Failed to enable fusion body tracking: {status}")
                    for cam in senders.values():
                        cam.close()
                    self.fusion.close()
                    return False

                print("[INFO] Fusion body tracking enabled")

            # Store references for cleanup
            self._fusion_senders = senders
            # IMPORTANT: Store serial order for video streaming (must match camera index order)
            self._fusion_serials_ordered = [conf.serial_number for conf in fusion_configurations]
            self.num_fusion_cameras = len(camera_identifiers)
            self.is_fusion_mode = True
            print(f"[INFO] Fusion mode initialized with {self.num_fusion_cameras} cameras")
            print(f"[INFO] Camera order: {self._fusion_serials_ordered}")
            
            # Initialize video streaming if enabled
            if self.enable_streaming:
                self._initialize_video_streamer(num_cameras=len(camera_identifiers))
            
            return True

        except Exception as e:
            print(f"[ERROR] Failed to initialize fusion cameras: {e}")
            return False

    def _initialize_fusion_cameras_by_serial(self, serials: List[str], enable_body_tracking: bool = True, enable_floor_detection: bool = True) -> bool:
        """Initialize fusion cameras by serial numbers"""
        try:
            device_list = sl.Camera.get_device_list()
            selected_devices = []

            for serial in serials:
                found = False
                for device in device_list:
                    if str(device.serial_number) == str(serial):
                        selected_devices.append(device)
                        found = True
                        break
                if not found:
                    print(f"[WARNING] Camera with serial {serial} not found")

            if len(selected_devices) == 0:
                print("[ERROR] No specified cameras found")
                return False

            return self._initialize_fusion_cameras(selected_devices, enable_body_tracking, enable_floor_detection)

        except Exception as e:
            print(f"[ERROR] Failed to initialize cameras by serial: {e}")
            return False

    def _find_calibration_file(self, serial: str = None) -> Optional[str]:
        """Find the newest calibration file in the calib directory"""
        try:
            calib_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "server", "calib")
            
            if not os.path.isdir(calib_dir):
                return None
            
            # Find all .json files in calib directory
            calib_files = []
            for filename in os.listdir(calib_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(calib_dir, filename)
                    if os.path.isfile(filepath):
                        # Get modification time
                        mtime = os.path.getmtime(filepath)
                        calib_files.append((filepath, mtime))
            
            if not calib_files:
                return None
            
            # Sort by modification time (newest first)
            calib_files.sort(key=lambda x: x[1], reverse=True)
            newest_file = calib_files[0][0]
            
            print(f"[INFO] Using calibration file: {os.path.basename(newest_file)}")
            return newest_file
            
        except Exception as e:
            print(f"[WARNING] Failed to find calibration file: {e}")
            return None

    def run_body_tracking_loop(self):
        """Main body tracking loop - calls appropriate mode-specific loop"""
        if not self.camera and not self.fusion:
            print("[ERROR] No cameras initialized")
            return

        if self.is_fusion_mode:
            self._run_fusion_loop()
        else:
            self._run_single_camera_loop()

    def _run_single_camera_loop(self):
        """Body tracking loop for single camera"""
        runtime_params = sl.RuntimeParameters()
        bodies = sl.Bodies()

        MAX_FPS = 30
        FRAME_INTERVAL = 1.0 / MAX_FPS
        last_time = time.time()

        print("[INFO] Starting single camera body tracking loop")

        while self.running:
            now = time.time()
            if now - last_time < FRAME_INTERVAL:
                time.sleep(FRAME_INTERVAL - (now - last_time))
            last_time = time.time()

            if self.camera.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                # Capture and stream video frames if enabled
                rgb_frame = None
                depth_frame = None
                
                if self.video_streamer is not None or self.per_camera_streaming_manager is not None:
                    try:
                        # Create image containers
                        rgb_mat = sl.Mat()
                        depth_mat = sl.Mat()
                        
                        # Retrieve RGB and depth images
                        err = self.camera.retrieve_image(rgb_mat, sl.VIEW.LEFT, sl.MEM.CPU)
                        if err != sl.ERROR_CODE.SUCCESS:
                            print(f"[WARNING] retrieve_image failed: {err}")
                            raise RuntimeError(f"retrieve_image failed: {err}")
                        
                        err = self.camera.retrieve_measure(depth_mat, sl.MEASURE.DEPTH, sl.MEM.CPU)
                        if err != sl.ERROR_CODE.SUCCESS:
                            print(f"[WARNING] retrieve_measure failed: {err}")
                            raise RuntimeError(f"retrieve_measure failed: {err}")
                        
                        # Convert to numpy arrays
                        rgb_frame = rgb_mat.get_data()
                        depth_frame = depth_mat.get_data()
                        
                        # ZED returns BGRA, convert to BGR (remove alpha channel)
                        if rgb_frame.shape[2] == 4:
                            rgb_frame = rgb_frame[:, :, :3]
                        
                        # Push to v1 if enabled
                        if self.video_streamer is not None:
                            self.video_streamer.push_camera_frames([rgb_frame], [depth_frame])
                        
                        # Push to v2 if enabled
                        if self.per_camera_streaming_manager is not None:
                            self.per_camera_streaming_manager.push_frames([rgb_frame], [depth_frame])
                            
                    except Exception as e:
                        print(f"[WARNING] Failed to stream video frame: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Retrieve bodies (simplified - match fusion approach)
                try:
                    self.camera.retrieve_bodies(bodies)
                except Exception as e:
                    print(f'[WARNING] retrieve_bodies failed: {e}')
                    continue
                
                # Use raw ZED SDK output (filter disabled - causes tracking issues)
                # Process bodies - use helper function only
                frame = self._process_bodies_single_camera(bodies)
                if frame:
                    frame.floor_height = self.detected_floor_height

                    self.broadcast_frame(frame)

                    # Update UI if callback is set (provide detected floor height or None)
                    if self.update_callback:
                        try:
                            # Call directly without creating new threads - callback should be fast
                            self.update_callback(frame.people, self.detected_floor_height)
                        except Exception:
                            pass

    def _run_fusion_loop(self):
        """Body tracking loop for fusion mode"""
        bodies = sl.Bodies()

        MAX_FPS = 30  # Fusion at 30fps is more stable than 60fps
        FRAME_INTERVAL = 1.0 / MAX_FPS
        last_time = time.time()

        print("[INFO] Starting fusion body tracking loop")

        while self.running:
            now = time.time()
            if now - last_time < FRAME_INTERVAL:
                time.sleep(FRAME_INTERVAL - (now - last_time))
            last_time = time.time()

            # Grab frames from local senders sequentially (essential for fusion to work)
            # IMPORTANT: Must iterate in same order as camera indices (0, 1, 2...)
            # Note: Sequential is required to avoid USB bandwidth conflicts
            rgb_frames = []
            depth_frames = []
            
            try:
                # Use ordered serial list to maintain camera index consistency
                if hasattr(self, '_fusion_serials_ordered') and hasattr(self, '_fusion_senders'):
                    for cam_idx, serial in enumerate(self._fusion_serials_ordered):
                        zed = self._fusion_senders.get(serial)
                        if not zed:
                            # Camera not available, add None placeholders
                            rgb_frames.append(None)
                            depth_frames.append(None)
                            continue
                        
                        try:
                            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                                try:
                                    zed.retrieve_bodies(bodies)
                                except Exception:
                                    pass
                                
                                # Capture video frames if streaming enabled (v1 or v2)
                                if self.video_streamer is not None or self.per_camera_streaming_manager is not None:
                                    try:
                                        # Create image containers
                                        rgb_mat = sl.Mat()
                                        depth_mat = sl.Mat()
                                        
                                        # Retrieve RGB and depth images
                                        zed.retrieve_image(rgb_mat, sl.VIEW.LEFT, sl.MEM.CPU)
                                        zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH, sl.MEM.CPU)
                                        
                                        # Convert to numpy arrays
                                        # CRITICAL: get_data() returns a VIEW, must copy IMMEDIATELY
                                        rgb_frame = rgb_mat.get_data().copy()
                                        depth_frame = depth_mat.get_data().copy()
                                        
                                        # ZED returns BGRA, convert to BGR (remove alpha channel)
                                        if rgb_frame.shape[2] == 4:
                                            rgb_frame = rgb_frame[:, :, :3].copy()
                                        
                                        # Frames already copied above, these are independent buffers
                                        
                                        # DEBUG: Log frame hashes for first few frames to verify different data
                                        if not hasattr(self, '_server_frame_counts'):
                                            self._server_frame_counts = {}
                                            self._server_frame_hashes = {}
                                        if serial not in self._server_frame_counts:
                                            self._server_frame_counts[serial] = 0
                                            self._server_frame_hashes[serial] = []
                                        
                                        self._server_frame_counts[serial] += 1
                                        
                                        # Collect frames
                                        rgb_frames.append(rgb_frame)
                                        depth_frames.append(depth_frame)
                                    except Exception as e:
                                        print(f"[WARNING] Failed to capture video frame from camera {serial}: {e}")
                                        # Add None placeholders to maintain camera index alignment
                                        rgb_frames.append(None)
                                        depth_frames.append(None)
                                else:
                                    # Streaming not enabled, don't collect frames
                                    pass
                            else:
                                # Grab failed for this camera
                                if self.video_streamer is not None or self.per_camera_streaming_manager is not None:
                                    rgb_frames.append(None)
                                    depth_frames.append(None)
                        except Exception as e:
                            print(f"[WARNING] Exception grabbing from camera {serial}: {e}")
                            if self.video_streamer is not None or self.per_camera_streaming_manager is not None:
                                rgb_frames.append(None)
                                depth_frames.append(None)
            except Exception:
                pass
            
            # Push v2 per-camera streaming frames if enabled (independent of v1)
            if self.per_camera_streaming_manager and len(rgb_frames) > 0:
                try:
                    # Check if we have at least some valid frames
                    valid_rgb_count = sum(1 for f in rgb_frames if f is not None)
                    valid_depth_count = sum(1 for f in depth_frames if f is not None)
                    
                    if valid_rgb_count > 0 or valid_depth_count > 0:
                        # Replace None frames with black/zero frames to maintain camera index alignment
                        final_rgb_frames = []
                        final_depth_frames = []
                        
                        num_cameras = len(rgb_frames)
                        for i in range(num_cameras):
                            # RGB: Use existing frame or create black frame
                            if rgb_frames[i] is not None:
                                final_rgb_frames.append(rgb_frames[i])
                            else:
                                # Create black placeholder (need dimensions from somewhere)
                                if hasattr(self, '_fusion_senders') and self._fusion_senders:
                                    sender = next(iter(self._fusion_senders.values()))
                                    cam_info = sender.get_camera_information()
                                    w, h = cam_info.camera_configuration.resolution.width, cam_info.camera_configuration.resolution.height
                                    black_frame = np.zeros((h, w, 3), dtype=np.uint8)
                                    final_rgb_frames.append(black_frame)
                                else:
                                    final_rgb_frames.append(None)
                            
                            # Depth: Use existing frame or create zero frame
                            if depth_frames[i] is not None:
                                final_depth_frames.append(depth_frames[i])
                            else:
                                # Create zero placeholder
                                if hasattr(self, '_fusion_senders') and self._fusion_senders:
                                    sender = next(iter(self._fusion_senders.values()))
                                    cam_info = sender.get_camera_information()
                                    w, h = cam_info.camera_configuration.resolution.width, cam_info.camera_configuration.resolution.height
                                    zero_depth = np.zeros((h, w), dtype=np.float32)
                                    final_depth_frames.append(zero_depth)
                                else:
                                    final_depth_frames.append(None)
                        
                        # Push to v2 streaming
                        self.per_camera_streaming_manager.push_frames(final_rgb_frames, final_depth_frames)
                except Exception as e:
                    print(f"[WARNING] Failed to push v2 video frames: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Push v1 multicast frames if enabled
            if self.video_streamer is not None and len(rgb_frames) > 0:
                try:
                    # Check if we have at least some valid frames
                    valid_rgb_count = sum(1 for f in rgb_frames if f is not None)
                    valid_depth_count = sum(1 for f in depth_frames if f is not None)
                    
                    if valid_rgb_count > 0 or valid_depth_count > 0:
                        # Replace None frames with black/zero frames to maintain camera index alignment
                        final_rgb_frames = []
                        final_depth_frames = []
                        
                        num_cameras = len(rgb_frames)  # Use actual number of frames collected
                        for i in range(num_cameras):
                            # RGB: Use existing frame or create black frame
                            if i < len(rgb_frames) and rgb_frames[i] is not None:
                                final_rgb_frames.append(rgb_frames[i])
                            else:
                                # Create black BGR frame matching expected resolution
                                black_frame = np.zeros((self.video_streamer.camera_height, 
                                                       self.video_streamer.camera_width, 3), dtype=np.uint8)
                                final_rgb_frames.append(black_frame)
                            
                            # Depth: Use existing frame or create zero depth frame
                            if i < len(depth_frames) and depth_frames[i] is not None:
                                final_depth_frames.append(depth_frames[i])
                            else:
                                # Create zero depth frame matching expected resolution
                                zero_depth = np.zeros((self.video_streamer.camera_height, 
                                                      self.video_streamer.camera_width), dtype=np.float32)
                                final_depth_frames.append(zero_depth)
                        
                        # Push frames (with placeholders for failed cameras) to v1
                        self.video_streamer.push_camera_frames(final_rgb_frames, final_depth_frames)
                except Exception as e:
                    print(f"[WARNING] Failed to stream v1 video frames: {e}")

            fusion_status = self.fusion.process()
            if fusion_status == sl.FUSION_ERROR_CODE.SUCCESS:
                # Clear bodies list before retrieving (SDK appends, doesn't replace!)
                bodies.body_list.clear()
                
                # Retrieve bodies
                try:
                    if hasattr(sl, 'BodyTrackingFusionRuntimeParameters'):
                        self.fusion.retrieve_bodies(bodies, sl.BodyTrackingFusionRuntimeParameters())
                    else:
                        self.fusion.retrieve_bodies(bodies)
                    
                    # Debug: Log number of bodies detected
                    if not hasattr(self, '_fusion_body_log_counter'):
                        self._fusion_body_log_counter = 0
                    self._fusion_body_log_counter += 1
                    if self._fusion_body_log_counter % 30 == 0:  # Log every 30 frames (~1 second)
                        print(f"[DEBUG] Fusion detected {len(bodies.body_list)} bodies")
                        
                except Exception as e:
                    print(f'[WARNING] Fusion retrieve_bodies failed: {e}')
                    continue

                # Apply skeleton continuity filter to prevent ghost duplicates (if enabled)
                if self.enable_skeleton_filter:
                    bodies = self._apply_skeleton_continuity_filter(bodies)
                
                # Process bodies
                frame = self._process_bodies_fusion(bodies)
                if frame:
                    frame.floor_height = self.detected_floor_height

                    self.broadcast_frame(frame)

                    # Update UI if callback is set
                    if self.update_callback:
                        try:
                            # Call directly without creating new threads - callback should be fast
                            self.update_callback(frame.people, self.detected_floor_height)
                        except Exception:
                            pass

                #test
                #ret = self._try_get_fusion_camera_pose(serial)
                #print("-----------", ret)

            else:
                # Fusion failed - could be timeout, no data, etc.
                # Check for common non-critical fusion errors
                non_critical_errors = []
                try:
                    # Try to get common non-critical error codes if they exist
                    if hasattr(sl.FUSION_ERROR_CODE, 'CAMERA_FPS_TOO_LOW'):
                        non_critical_errors.append(sl.FUSION_ERROR_CODE.CAMERA_FPS_TOO_LOW)
                    if hasattr(sl.FUSION_ERROR_CODE, 'NO_NEW_DATA'):
                        non_critical_errors.append(sl.FUSION_ERROR_CODE.NO_NEW_DATA)
                    if hasattr(sl.FUSION_ERROR_CODE, 'TIMEOUT'):
                        non_critical_errors.append(sl.FUSION_ERROR_CODE.TIMEOUT)
                except Exception:
                    pass
                
                # Only print warning for critical errors
                """
                if fusion_status not in non_critical_errors:
                    print(f"[WARNING] Fusion process failed: {fusion_status}")
                """
                
                # Still send empty frame to keep clients connected
                empty_frame = Frame(
                    timestamp=time.time(),
                    people=[],
                    body_model="BODY_34",
                    floor_height=self.detected_floor_height,
                    cameras=self.get_camera_poses()
                )
                self.broadcast_frame(empty_frame)

    def _process_bodies_single_camera(self, bodies) -> Optional[Frame]:
        """Process bodies from single camera and create Frame object"""
        people = []

        for person in bodies.body_list:
            # Handle different SDK versions for keypoints
            kp_attr = getattr(person, 'keypoint', None)
            if kp_attr is None:
                kp_attr = getattr(person, 'keypoints', None)
            keypoints = kp_attr if kp_attr is not None else []

            # Handle different SDK versions for confidences
            conf_attr = getattr(person, 'keypoint_confidence', None)
            if conf_attr is None:
                conf_attr = getattr(person, 'keypoint_confidences', None)
            confidences = conf_attr if conf_attr is not None else []

            # Get orientations from ZED SDK
            # Note: ZED SDK only provides local_orientation_per_joint and global_root_orientation
            # There is NO global_orientation_per_joint in the SDK!
            local_orientations = getattr(person, 'local_orientation_per_joint', None)
            global_root_ori = getattr(person, 'global_root_orientation', None)
            
            # DEBUG: Check what orientation data we're getting
            if not hasattr(self, '_orientation_data_logged'):
                self._orientation_data_logged = True
                print("[ORIENTATION DATA DEBUG]")
                print(f"  local_orientation_per_joint: {type(local_orientations)}, shape={getattr(local_orientations, 'shape', 'N/A')}")
                print(f"  global_root_orientation: {type(global_root_ori)}, shape={getattr(global_root_ori, 'shape', 'N/A')}")
                
                # Show global_root_orientation value
                if global_root_ori is not None:
                    print(f"  global_root_orientation = [{global_root_ori[0]:.3f}, {global_root_ori[1]:.3f}, {global_root_ori[2]:.3f}, {global_root_ori[3]:.3f}]")
                
                # Check which one we'll actually use for joint.ori
                if local_orientations is not None:
                    print("  --> Storing in joint.ori: local_orientation_per_joint (LOCAL/relative to parent)")
                    if hasattr(local_orientations, 'shape') and len(local_orientations.shape) == 2:
                        print(f"      {local_orientations.shape[0]} joints")
                        print(f"      Pelvis (joint 0) local: [{local_orientations[0][0]:.3f}, {local_orientations[0][1]:.3f}, {local_orientations[0][2]:.3f}, {local_orientations[0][3]:.3f}] (should be ~identity)")
                        for idx in [5, 12, 18]:
                            if idx < len(local_orientations):
                                ori = local_orientations[idx]
                                print(f"      Joint {idx} local: [{ori[0]:.3f}, {ori[1]:.3f}, {ori[2]:.3f}, {ori[3]:.3f}]")
                else:
                    print("  --> Using global_root_orientation (FALLBACK)")
            
            # Use local orientations (these are what the SDK actually provides)
            orientations = local_orientations
            if orientations is None:
                # Fallback to global_root if local orientations not available
                orientations = global_root_ori

            joints = []
            for i, kp in enumerate(keypoints):
                pos_obj = Position(x=float(kp[0]), y=float(kp[1]), z=float(kp[2]))

                # Use orientations from ZED SDK (local_orientation_per_joint)
                if orientations is not None:
                    try:
                        if len(orientations.shape) == 2:
                            # Array of quaternions, one per joint - these are LOCAL (relative to parent)
                            ori = orientations[i] if i < len(orientations) else orientations[0]
                        else:
                            # Single quaternion - use for all joints
                            ori = orientations
                        ori_obj = Quaternion(x=float(ori[0]), y=float(ori[1]), z=float(ori[2]), w=float(ori[3]))
                    except (IndexError, AttributeError):
                        ori_obj = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                else:
                    ori_obj = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

                conf = float(confidences[i]) if i < len(confidences) else 0.0
                joints.append(Joint(i=i, pos=pos_obj, ori=ori_obj, conf=conf))

            # Extract global_root_orientation if available
            global_root_quat = None
            if global_root_ori is not None:
                try:
                    global_root_quat = Quaternion(
                        x=float(global_root_ori[0]),
                        y=float(global_root_ori[1]),
                        z=float(global_root_ori[2]),
                        w=float(global_root_ori[3])
                    )
                except (IndexError, AttributeError, TypeError):
                    pass

            people.append(Person(
                id=person.id,
                tracking_state=str(person.tracking_state),
                confidence=float(person.confidence),
                skeleton=joints,
                global_root_orientation=global_root_quat
            ))

        if people or True:  # Always send frames for debugging
            # Get camera poses data
            camera_poses_data = self.get_camera_poses()

            # Normalize into separate camera list and floor height
            camera_list = []
            floor_height = None

            if camera_poses_data:
                if isinstance(camera_poses_data, dict):
                    camera_list = camera_poses_data.get('cameras', []) or []
                    floor_info = camera_poses_data.get('floor')
                    if isinstance(floor_info, dict):
                        floor_height = floor_info.get('height')
                elif isinstance(camera_poses_data, list):
                    camera_list = camera_poses_data
                else:
                    # unknown format - ignore
                    camera_list = []

            # fallback to detected value if available
            if floor_height is None:
                floor_height = getattr(self, 'detected_floor_height', None)

            return Frame(
                timestamp=bodies.timestamp.get_seconds(),
                people=people,
                body_model="BODY_34",
                floor_height=floor_height,
                cameras=camera_list
            )

        return None

    def _apply_skeleton_continuity_filter(self, bodies):
        """
        Maintains stable skeleton IDs across tracking interruptions.
        
        The ZED SDK may create new IDs when tracking is briefly lost, resulting in
        duplicate skeletons for the same person. This filter:
        1. Detects when a "new" skeleton is actually a duplicate of an existing one
        2. Maintains a stable ID for each unique person
        3. Tracks which SDK ID is currently active for each stable ID
        4. Keeps history of all SDK IDs that represented the same person
        
        Strategy:
        - Each person gets one stable_id that clients always see
        - The stable_id tracks which current_sdk_id to use for skeleton data
        - When SDK creates duplicate, we update current_sdk_id but keep stable_id
        - Only output skeletons that have ACTIVE data in current frame
        """
        current_time = time.time()
        current_sdk_ids = set()
        
        # Build lookup dict for quick access to skeleton data by SDK ID
        persons_by_sdk_id = {}
        for person in bodies.body_list:
            sdk_id = person.id
            current_sdk_ids.add(sdk_id)
            persons_by_sdk_id[sdk_id] = person
        
        # Build reverse lookup: which stable_id is currently using each sdk_id
        sdk_to_stable = {}
        for stable_id, tracker in self.tracked_skeletons.items():
            if tracker['current_sdk_id'] is not None:
                sdk_to_stable[tracker['current_sdk_id']] = stable_id
        
        # Process each SDK ID in current frame
        # Keep track of which SDK IDs we've already assigned to trackers in THIS frame
        processed_sdk_ids = set()
        skipped_ghost_ids = set()  # Track SDK IDs that were identified as ghosts and skipped
        
        for sdk_id in current_sdk_ids:
            person = persons_by_sdk_id[sdk_id]
            
            # Check if this SDK ID is already being tracked
            if sdk_id in sdk_to_stable:
                # Update timestamp for this tracked skeleton
                stable_id = sdk_to_stable[sdk_id]
                self.tracked_skeletons[stable_id]['last_seen'] = current_time
                processed_sdk_ids.add(sdk_id)
                continue
            
            # Skip if this SDK ID was already identified as a ghost (either in this frame or previously)
            if sdk_id in skipped_ghost_ids or sdk_id in self.ghost_sdk_ids:
                continue
            
            # New SDK ID - check if it's a ghost/duplicate of ANY tracker (active OR recently inactive)
            is_duplicate = False
            best_match_stable_id = None
            REACTIVATION_WINDOW = 2.0  # Check inactive trackers for 2 seconds
            
            # FIRST: Check against OTHER SDK IDs we've already processed in THIS frame
            # (This detects when SDK creates 2 IDs for same person in same frame)
            for other_sdk_id in processed_sdk_ids:
                if other_sdk_id == sdk_id:
                    continue
                
                other_person = persons_by_sdk_id[other_sdk_id]
                
                # Check if this new SDK ID is a ghost of the already-processed one
                if self._are_skeletons_same_person(person, other_person):
                    # This is a ghost! Find which stable ID is tracking the other SDK ID
                    if other_sdk_id in sdk_to_stable:
                        best_match_stable_id = sdk_to_stable[other_sdk_id]
                        is_duplicate = True
                        print(f"[FILTER] Ghost in same frame: SDK ID {sdk_id} is duplicate of SDK ID {other_sdk_id} (stable ID {best_match_stable_id})")
                        break
            
            # SECOND: If not a ghost in current frame, check existing trackers
            if not is_duplicate:
                for stable_id, tracker in self.tracked_skeletons.items():
                    current_tracked_sdk_id = tracker['current_sdk_id']
                    
                    # For ACTIVE trackers: compare directly
                    if current_tracked_sdk_id is not None and current_tracked_sdk_id in persons_by_sdk_id:
                        tracked_person = persons_by_sdk_id[current_tracked_sdk_id]
                        
                        # Check if this new SDK ID is a ghost of the active tracker
                        if self._are_skeletons_same_person(person, tracked_person):
                            best_match_stable_id = stable_id
                            is_duplicate = True
                            break
                    
                    # For INACTIVE trackers: check if recently inactive (might be same person reappearing)
                    elif current_tracked_sdk_id is None:
                        time_inactive = current_time - tracker['last_seen']
                        
                        # Only consider very recently inactive trackers
                        # BUT: don't reactivate with an SDK ID that was identified as a ghost
                        if time_inactive < REACTIVATION_WINDOW and sdk_id not in skipped_ghost_ids and sdk_id not in self.ghost_sdk_ids:
                            # This is likely the same person - reactivate the tracker
                            # (We can't compare skeletons since tracker is inactive, but timing suggests it's the same)
                            best_match_stable_id = stable_id
                            is_duplicate = True
                            break
            
            if is_duplicate and best_match_stable_id is not None:
                tracker = self.tracked_skeletons[best_match_stable_id]
                old_sdk_id = tracker['current_sdk_id']
                
                if old_sdk_id is not None and old_sdk_id in current_sdk_ids:
                    # Both the tracker's current SDK ID and the new SDK ID exist in same frame
                    # This means ZED SDK created a duplicate for the same person
                    # The HIGHER SDK ID is usually the NEWER tracking (ZED SDK increments IDs)
                    # So we should use whichever ID is higher
                    
                    if sdk_id > old_sdk_id:
                        # New SDK ID is higher → it's the new tracking, replace the old one
                        print(f"[FILTER] SDK upgrade: SDK ID {sdk_id} replaces SDK ID {old_sdk_id} for stable ID {best_match_stable_id} (newer tracking)")
                        
                        # Record the old SDK ID in history
                        tracker['sdk_ids'].append({
                            'sdk_id': old_sdk_id,
                            'start': tracker.get('sdk_id_start', tracker['created']),
                            'end': current_time
                        })
                        
                        # Switch to using the new (higher) SDK ID
                        tracker['current_sdk_id'] = sdk_id
                        tracker['sdk_id_start'] = current_time
                        tracker['last_seen'] = current_time
                        processed_sdk_ids.add(sdk_id)
                        
                    else:
                        # New SDK ID is lower → it's an old ghost, ignore it
                        print(f"[FILTER] Ghost ignored: SDK ID {sdk_id} is old ghost of SDK ID {old_sdk_id} (stable ID {best_match_stable_id})")
                        skipped_ghost_ids.add(sdk_id)
                        self.ghost_sdk_ids.add(sdk_id)
                        # Keep the higher (current) SDK ID active
                        tracker['last_seen'] = current_time
                        processed_sdk_ids.add(old_sdk_id)
                    
                    continue
                elif old_sdk_id is not None:
                    # Old SDK ID is gone but new one appeared - this is a replacement
                    tracker['sdk_ids'].append({
                        'sdk_id': old_sdk_id,
                        'start': tracker.get('sdk_id_start', tracker['created']),
                        'end': current_time
                    })
                    tracker['current_sdk_id'] = sdk_id
                    tracker['sdk_id_start'] = current_time
                    tracker['last_seen'] = current_time
                    processed_sdk_ids.add(sdk_id)
                    print(f"[FILTER] Ghost replaced: SDK ID {sdk_id} replaces disappeared ghost {old_sdk_id} for stable ID {best_match_stable_id}")
                else:
                    # Reactivating inactive tracker
                    tracker['current_sdk_id'] = sdk_id
                    tracker['sdk_id_start'] = current_time
                    tracker['last_seen'] = current_time
                    processed_sdk_ids.add(sdk_id)
                    print(f"[FILTER] Reactivated: stable ID {best_match_stable_id} with SDK ID {sdk_id} (was inactive {current_time - tracker['last_seen']:.2f}s)")
            
            if not is_duplicate:
                # Truly new person - create a new tracker
                stable_id = self.next_stable_id
                self.next_stable_id += 1
                
                self.tracked_skeletons[stable_id] = {
                    'current_sdk_id': sdk_id,
                    'created': current_time,
                    'last_seen': current_time,
                    'sdk_id_start': current_time,
                    'sdk_ids': []  # History of previous SDK IDs
                }
                print(f"[FILTER] New person: stable ID {stable_id} (SDK ID {sdk_id})")
                processed_sdk_ids.add(sdk_id)
        
        # Cleanup: Mark trackers as inactive if their current SDK ID is gone
        # But DON'T delete them - they might get a replacement in future frames
        for stable_id, tracker in list(self.tracked_skeletons.items()):
            current_sdk_id = tracker['current_sdk_id']
            
            if current_sdk_id is not None and current_sdk_id not in current_sdk_ids:
                # This SDK ID is gone - record it in history and mark as inactive
                tracker['sdk_ids'].append({
                    'sdk_id': current_sdk_id,
                    'start': tracker.get('sdk_id_start', tracker['created']),
                    'end': current_time
                })
                tracker['current_sdk_id'] = None  # Mark as inactive
                print(f"[FILTER] Stable ID {stable_id} lost SDK ID {current_sdk_id} (marked inactive)")
        
        # Optional: Clean up old inactive trackers after timeout (e.g., 10 seconds)
        timeout = 10.0
        for stable_id in list(self.tracked_skeletons.keys()):
            tracker = self.tracked_skeletons[stable_id]
            if tracker['current_sdk_id'] is None:
                time_inactive = current_time - tracker['last_seen']
                if time_inactive > timeout:
                    print(f"[FILTER] Removing stable ID {stable_id} after {time_inactive:.1f}s inactive")
                    del self.tracked_skeletons[stable_id]
        
        # Don't modify bodies.body_list here - ZED SDK doesn't support it properly
        # Filtering will happen in _process_bodies_fusion
        return bodies

    def _are_skeletons_same_person(self, person_a, person_b):
        """Check if two skeletons represent the same person"""
        MAX_POSITION_DIFF = 1500  # 1.5m
        POSE_SIMILARITY_THRESHOLD = 0.3  # 30% joint similarity
        
        # Get keypoints
        kp_a = getattr(person_a, 'keypoint', None)
        if kp_a is None:
            kp_a = getattr(person_a, 'keypoints', None)
        kp_b = getattr(person_b, 'keypoint', None)
        if kp_b is None:
            kp_b = getattr(person_b, 'keypoints', None)
        
        if kp_a is None or kp_b is None or len(kp_a) == 0 or len(kp_b) == 0:
            return False
        
        # Check distance between pelvis positions
        pelvis_a, pelvis_b = kp_a[0], kp_b[0]
        dx = pelvis_a[0] - pelvis_b[0]
        dy = pelvis_a[1] - pelvis_b[1]
        dz = pelvis_a[2] - pelvis_b[2]
        distance = (dx*dx + dy*dy + dz*dz) ** 0.5
        
        if distance > MAX_POSITION_DIFF:
            return False
        
        # Check pose similarity
        num_joints = min(len(kp_a), len(kp_b))
        similar_joints = 0
        for k in range(num_joints):
            jdx = kp_a[k][0] - kp_b[k][0]
            jdy = kp_a[k][1] - kp_b[k][1]
            jdz = kp_a[k][2] - kp_b[k][2]
            joint_dist = (jdx*jdx + jdy*jdy + jdz*jdz) ** 0.5
            if joint_dist < 300:  # Within 30cm
                similar_joints += 1
        
        pose_similarity = similar_joints / num_joints if num_joints > 0 else 0
        return pose_similarity >= POSE_SIMILARITY_THRESHOLD

    def _process_bodies_fusion(self, bodies) -> Optional[Frame]:
        """
        Process bodies from fusion and create Frame object.
        
        Uses tracked_skeletons to maintain stable IDs while using newest skeleton data:
        - Only output skeletons that have ACTIVE skeleton data in current frame
        - Use stable_id for client-facing ID
        - Use current_sdk_id to fetch skeleton data from bodies.body_list
        """
        people = []
        
        # Build dict for quick person lookup by SDK ID
        persons_by_sdk_id = {person.id: person for person in bodies.body_list}
        
        # Build set of SDK IDs that are currently being used by trackers
        active_sdk_ids = set()
        for tracker in self.tracked_skeletons.values():
            if tracker['current_sdk_id'] is not None:
                active_sdk_ids.add(tracker['current_sdk_id'])
        
        # Process each tracked skeleton
        for stable_id, tracker in self.tracked_skeletons.items():
            current_sdk_id = tracker['current_sdk_id']
            
            # Skip if this tracker has no active SDK ID
            if current_sdk_id is None:
                continue
            
            # Skip if the SDK ID doesn't have skeleton data in current frame
            if current_sdk_id not in persons_by_sdk_id:
                continue
            
            # Get skeleton data from current SDK ID
            person = persons_by_sdk_id[current_sdk_id]
            output_id = stable_id  # Use stable ID for clients
            
            # Handle different SDK versions for keypoints
            kp_attr = getattr(person, 'keypoint', None)
            if kp_attr is None:
                kp_attr = getattr(person, 'keypoints', None)
            keypoints = kp_attr if kp_attr is not None else []

            # Handle different SDK versions for confidences
            conf_attr = getattr(person, 'keypoint_confidence', None)
            if conf_attr is None:
                conf_attr = getattr(person, 'keypoint_confidences', None)
            confidences = conf_attr if conf_attr is not None else []

            # Get orientations from ZED SDK
            # Note: ZED SDK only provides local_orientation_per_joint and global_root_orientation
            local_orientations = getattr(person, 'local_orientation_per_joint', None)
            global_root_ori = getattr(person, 'global_root_orientation', None)
            
            # Use local orientations (these are what the SDK actually provides)
            orientations = local_orientations
            if orientations is None:
                # Fallback to global_root if local orientations not available
                orientations = global_root_ori

            joints = []
            for i, kp in enumerate(keypoints):
                pos_obj = Position(x=float(kp[0]), y=float(kp[1]), z=float(kp[2]))

                # Use orientations from ZED SDK (local_orientation_per_joint)
                if orientations is not None:
                    try:
                        if len(orientations.shape) == 2:
                            # Array of quaternions, one per joint - these are LOCAL (relative to parent)
                            ori = orientations[i] if i < len(orientations) else orientations[0]
                        else:
                            # Single quaternion - use for all joints
                            ori = orientations
                        ori_obj = Quaternion(x=float(ori[0]), y=float(ori[1]), z=float(ori[2]), w=float(ori[3]))
                    except (IndexError, AttributeError):
                        ori_obj = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                else:
                    ori_obj = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

                conf = float(confidences[i]) if i < len(confidences) else 0.0
                joints.append(Joint(i=i, pos=pos_obj, ori=ori_obj, conf=conf))

            # Extract global_root_orientation if available
            global_root_quat = None
            if global_root_ori is not None:
                try:
                    global_root_quat = Quaternion(
                        x=float(global_root_ori[0]),
                        y=float(global_root_ori[1]),
                        z=float(global_root_ori[2]),
                        w=float(global_root_ori[3])
                    )
                except (IndexError, AttributeError, TypeError):
                    pass

            people.append(Person(
                id=output_id,  # Use stable old ID for client, not SDK's current ID
                tracking_state=str(person.tracking_state),
                confidence=float(person.confidence),
                skeleton=joints,
                global_root_orientation=global_root_quat
            ))

        if people or True:  # Always send frames for debugging
            # Get camera poses data
            camera_poses_data = self.get_camera_poses()

            # Normalize into separate camera list and floor height
            camera_list = []
            floor_height = None

            if camera_poses_data:
                if isinstance(camera_poses_data, dict):
                    camera_list = camera_poses_data.get('cameras', []) or []
                    floor_info = camera_poses_data.get('floor')
                    if isinstance(floor_info, dict):
                        floor_height = floor_info.get('height')
                elif isinstance(camera_poses_data, list):
                    camera_list = camera_poses_data
                else:
                    # unknown format - ignore
                    camera_list = []

            # fallback to detected value if available
            if floor_height is None:
                floor_height = getattr(self, 'detected_floor_height', None)

            return Frame(
                timestamp=bodies.timestamp.get_seconds(),
                people=people,
                body_model="BODY_34",
                floor_height=floor_height,
                cameras=camera_list
            )

        return None

    def _matrix_to_quaternion(self, rotation_matrix):
        """Convert 3x3 rotation matrix to quaternion [x, y, z, w]"""
        try:
            import math
            
            R = rotation_matrix
            trace = R[0][0] + R[1][1] + R[2][2]
            
            if trace > 0:
                s = math.sqrt(trace + 1.0) * 2  # s = 4 * qw
                w = 0.25 * s
                x = (R[2][1] - R[1][2]) / s
                y = (R[0][2] - R[2][0]) / s
                z = (R[1][0] - R[0][1]) / s
            elif R[0][0] > R[1][1] and R[0][0] > R[2][2]:
                s = math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2  # s = 4 * qx
                w = (R[2][1] - R[1][2]) / s
                x = 0.25 * s
                y = (R[0][1] + R[1][0]) / s
                z = (R[0][2] + R[2][0]) / s
            elif R[1][1] > R[2][2]:
                s = math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2  # s = 4 * qy
                w = (R[0][2] - R[2][0]) / s
                x = (R[0][1] + R[1][0]) / s
                y = 0.25 * s
                z = (R[1][2] + R[2][1]) / s
            else:
                s = math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2  # s = 4 * qz
                w = (R[1][0] - R[0][1]) / s
                x = (R[0][2] + R[2][0]) / s
                y = (R[1][2] + R[2][1]) / s
                z = 0.25 * s

            return [x, y, z, w]
        except Exception as e:
            print(f"[WARNING] Failed to convert matrix to quaternion: {e}")
            return [0.0, 0.0, 0.0, 1.0]  # Identity quaternion

    def cleanup(self):
        """Clean up resources"""
        print("[INFO] Starting cleanup...")
        self.running = False
        
        # Close all active v2 client handler sockets to unblock recv() calls
        if hasattr(self, 'v2_client_sockets') and hasattr(self, 'v2_client_sockets_lock'):
            with self.v2_client_sockets_lock:
                for sock in self.v2_client_sockets[:]:  # Copy list to avoid modification during iteration
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                        sock.close()
                    except:
                        pass
                self.v2_client_sockets.clear()
                print(f"[INFO] Closed all v2 client handler sockets")
        
        # Close v2 control socket to unblock accept()
        if hasattr(self, 'v2_control_socket'):
            try:
                self.v2_control_socket.close()
                print("[INFO] Closed v2 control socket")
            except:
                pass
        
        # Shutdown v2 per-camera streaming first (stops GStreamer pipelines)
        if hasattr(self, 'per_camera_streaming_manager') and self.per_camera_streaming_manager:
            try:
                print("[INFO] Stopping per-camera streaming...")
                self.per_camera_streaming_manager.stop()
            except Exception as e:
                print(f"[WARNING] Error stopping per-camera streaming: {e}")
        
        # Give daemon threads a moment to see running=False and exit
        import time
        print("[INFO] Waiting for threads to exit...")
        time.sleep(1.0)
        
        # Force exit if still hanging (threads are daemons, so this is safe)
        import threading
        active_threads = threading.active_count()
        if active_threads > 1:  # More than just main thread
            print(f"[WARNING] {active_threads} threads still active, forcing exit...")
            import os
            os._exit(0)

        if self.camera:
            try:
                self.camera.disable_body_tracking()
                self.camera.close()
            except:
                pass
            self.camera = None

        if self.fusion:
            try:
                self.fusion.close()
            except:
                pass
            self.fusion = None

        self.stop_tcp_server()

