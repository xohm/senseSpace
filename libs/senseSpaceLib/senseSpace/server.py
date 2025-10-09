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
from PyQt5.QtGui import QVector3D, QQuaternion

# Import protocol classes
try:
    from .protocol import Frame, Person, Joint
except ImportError:
    try:
        from senseSpaceLib.senseSpace.protocol import Frame, Person, Joint
    except ImportError:
        # Fallback for development
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__)))
        from protocol import Frame, Person, Joint


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

    def __init__(self, host: str = "0.0.0.0", port: int = 12345):
        self.host = host
        self.port = port
        self.local_ip = get_local_ip()
        
        # Server state
        self.running = True
        
        # TCP server attributes
        self.server_socket = None
        self.server_thread = None
        self.clients = []
        self._client_queues = {}
        self._client_sender_threads = {}
        
        # Camera instances
        self.camera = None
        self.fusion = None
        self.is_fusion_mode = False
        
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

    def set_update_callback(self, callback: Callable):
        """Set callback function for UI updates (e.g., Qt viewer updates)"""
        self.update_callback = callback

    def start_tcp_server(self):
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

    def broadcast_frame(self, frame: Frame):
        """Broadcast a frame to all connected clients"""
        if not self.clients:
            return
        # Enqueue the Frame object itself: sender worker will call to_dict() and json.dumps().
        message = {"type": "frame", "data": frame}
        for client in list(self.clients):
            q = self._client_queues.get(client)
            if q is None:
                # No queue: best-effort short-path; serialize minimally here
                try:
                    payload = frame.to_dict() if hasattr(frame, 'to_dict') else frame
                    text = json.dumps({"type": "frame", "data": payload}, default=lambda o: getattr(o, 'to_dict', lambda: str(o))())
                    client.sendall((text + "\n").encode("utf-8"))
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
                q.put_nowait(message)
            except queue.Full:
                # drop message for this slow client
                continue

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
                            'y': M_gl[1][3] * 1000.0 + (floor_height_mm),
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
            init_params.camera_resolution = sl.RESOLUTION.HD1080
            init_params.camera_fps = 30
            init_params.depth_mode = sl.DEPTH_MODE.NEURAL
            init_params.coordinate_units = sl.UNIT.MILLIMETER
            init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

            # Open camera
            status = self.camera.open(init_params)
            if status != sl.ERROR_CODE.SUCCESS:
                print(f"[ERROR] Failed to open camera: {status}")
                return False

            # Enable positional tracking (required for body tracking)
            tracking_params = sl.PositionalTrackingParameters()
            status = self.camera.enable_positional_tracking(tracking_params)
            if status != sl.ERROR_CODE.SUCCESS:
                print(f"[ERROR] Failed to enable positional tracking: {status}")
                self.camera.close()
                return False

            # Enable body tracking if requested
            if enable_body_tracking:
                body_params = sl.BodyTrackingParameters()
                body_params.enable_tracking = True
                body_params.enable_body_fitting = True
                body_params.body_format = sl.BODY_FORMAT.BODY_34
                body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE

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
            init_params.depth_mode = sl.DEPTH_MODE.NEURAL
            init_params.camera_resolution = sl.RESOLUTION.HD720
            init_params.camera_fps = 30

            communication_parameters = sl.CommunicationParameters()
            communication_parameters.set_for_shared_memory()

            positional_tracking_parameters = sl.PositionalTrackingParameters()
            positional_tracking_parameters.set_as_static = True

            body_tracking_parameters = sl.BodyTrackingParameters()
            body_tracking_parameters.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
            body_tracking_parameters.body_format = sl.BODY_FORMAT.BODY_34
            body_tracking_parameters.enable_body_fitting = False
            body_tracking_parameters.enable_tracking = True

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

                # Enable body tracking
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
            print("[INFO] Senders started, warming up...")
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
                body_tracking_fusion_params.enable_body_fitting = False

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
            self.is_fusion_mode = True
            print(f"[INFO] Fusion mode initialized with {len(camera_identifiers)} cameras")
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
                # Retrieve bodies with proper error handling for different SDK versions
                try:
                    if hasattr(sl, 'BodyTrackingRuntimeParameters'):
                        self.camera.retrieve_bodies(bodies, sl.BodyTrackingRuntimeParameters(), 0)
                    else:
                        self.camera.retrieve_bodies(bodies)
                except TypeError:
                    try:
                        self.camera.retrieve_bodies(bodies, 0)
                    except Exception as e:
                        print(f'[WARNING] retrieve_bodies failed: {e}')
                        continue

                # Process bodies - use helper function only
                frame = self._process_bodies_single_camera(bodies)
                if frame:
                    frame.floor_height = self.detected_floor_height

                    self.broadcast_frame(frame)

                    # Update UI if callback is set (provide detected floor height or None)
                    if self.update_callback:
                        try:
                            # Run UI update in separate thread so Qt rendering doesn't block capture
                            threading.Thread(target=self.update_callback, args=(frame.people, self.detected_floor_height), daemon=True).start()
                        except Exception:
                            pass

    def _run_fusion_loop(self):
        """Body tracking loop for fusion mode"""
        bodies = sl.Bodies()

        MAX_FPS = 30
        FRAME_INTERVAL = 1.0 / MAX_FPS
        last_time = time.time()

        print("[INFO] Starting fusion body tracking loop")

        while self.running:
            now = time.time()
            if now - last_time < FRAME_INTERVAL:
                time.sleep(FRAME_INTERVAL - (now - last_time))
            last_time = time.time()

            # Grab frames from local senders first (essential for fusion to work)
            try:
                if hasattr(self, '_fusion_senders'):
                    for serial, zed in self._fusion_senders.items():
                        try:
                            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                                try:
                                    zed.retrieve_bodies(bodies)
                                except Exception:
                                    pass
                        except Exception:
                            pass
            except Exception:
                pass

            fusion_status = self.fusion.process()
            if fusion_status == sl.FUSION_ERROR_CODE.SUCCESS:
                # Retrieve bodies
                try:
                    if hasattr(sl, 'BodyTrackingFusionRuntimeParameters'):
                        self.fusion.retrieve_bodies(bodies, sl.BodyTrackingFusionRuntimeParameters())
                    else:
                        self.fusion.retrieve_bodies(bodies)
                except Exception as e:
                    print(f'[WARNING] Fusion retrieve_bodies failed: {e}')
                    continue

                # Process bodies
                frame = self._process_bodies_fusion(bodies)
                if frame:
                    frame.floor_height = self.detected_floor_height

                    self.broadcast_frame(frame)

                    # Update UI if callback is set
                    if self.update_callback:
                        try:
                            threading.Thread(target=self.update_callback, args=(frame.people, self.detected_floor_height), daemon=True).start()
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

            # Handle orientations
            orientations = getattr(person, 'global_orientation', None)
            if orientations is None:
                orientations = getattr(person, 'global_root_orientation', None)

            joints = []
            for i, kp in enumerate(keypoints):
                pos = {"x": float(kp[0]), "y": float(kp[1]), "z": float(kp[2])}

                # Handle orientations
                if orientations is not None:
                    try:
                        if len(orientations.shape) == 2:
                            ori = orientations[i] if i < len(orientations) else orientations[0]
                        else:
                            ori = orientations
                        ori_dict = {"x": float(ori[0]), "y": float(ori[1]), "z": float(ori[2]), "w": float(ori[3])}
                    except (IndexError, AttributeError):
                        ori_dict = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
                else:
                    ori_dict = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}

                conf = float(confidences[i]) if i < len(confidences) else 0.0
                joints.append(Joint(i=i, pos=pos, ori=ori_dict, conf=conf))

            people.append(Person(
                id=person.id,
                tracking_state=str(person.tracking_state),
                confidence=float(person.confidence),
                skeleton=joints
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

    def _process_bodies_fusion(self, bodies) -> Optional[Frame]:
        """Process bodies from fusion and create Frame object"""
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

            # Handle orientations
            orientations = getattr(person, 'global_orientation', None)
            if orientations is None:
                orientations = getattr(person, 'global_root_orientation', None)

            joints = []
            for i, kp in enumerate(keypoints):
                pos = {"x": float(kp[0]), "y": float(kp[1]), "z": float(kp[2])}

                # Handle orientations
                if orientations is not None:
                    try:
                        if len(orientations.shape) == 2:
                            ori = orientations[i] if i < len(orientations) else orientations[0]
                        else:
                            ori = orientations
                        ori_dict = {"x": float(ori[0]), "y": float(ori[1]), "z": float(ori[2]), "w": float(ori[3])}
                    except (IndexError, AttributeError):
                        ori_dict = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
                else:
                    ori_dict = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}

                conf = float(confidences[i]) if i < len(confidences) else 0.0
                joints.append(Joint(i=i, pos=pos, ori=ori_dict, conf=conf))

            people.append(Person(
                id=person.id,
                tracking_state=str(person.tracking_state),
                confidence=float(person.confidence),
                skeleton=joints
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
        self.running = False

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

