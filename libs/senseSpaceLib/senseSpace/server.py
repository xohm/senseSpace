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

    def find_floor_plane(self, cam):
        """Detect floor plane using ZED SDK"""
        # Try multiple times with short backoff to improve reliability
        attempts = 3
        wait_secs = 0.2
        self.detected_floor_height = None
        for attempt in range(attempts):
            try:
                plane = sl.Plane()
                reset_tracking_floor_frame = sl.Transform()
                status = cam.find_floor_plane(plane, reset_tracking_floor_frame)
                if status == sl.ERROR_CODE.SUCCESS:
                    # Different ZED SDK versions expose different Plane APIs.
                    # Try common accessors, fall back to computing distance from plane coefficients.
                    distance_to_origin = None
                    try:
                        # Newer API
                        distance_to_origin = plane.get_distance_to_origin()
                    except Exception:
                        pass

                    if distance_to_origin is None:
                        try:
                            # Older API may expose coefficients a,b,c,d or get_coefficients()
                            coeffs = None
                            if hasattr(plane, 'get_coefficients'):
                                coeffs = plane.get_coefficients()
                            elif hasattr(plane, 'coeffs'):
                                coeffs = plane.coeffs
                            elif hasattr(plane, 'a') and hasattr(plane, 'b') and hasattr(plane, 'c') and hasattr(plane, 'd'):
                                coeffs = (plane.a, plane.b, plane.c, plane.d)

                            if coeffs is not None and len(coeffs) >= 4:
                                        a, b, c, d = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
                                        # Compute Y intercept of plane ax + by + cz + d = 0 at X=0,Z=0 => y = -d / b
                                        try:
                                            if abs(b) > 1e-6:
                                                y_at_origin = -d / b
                                                # Use this as the detected floor height in camera/world units
                                                self.detected_floor_height = float(y_at_origin)
                                            else:
                                                # If plane is vertical-ish (b ~= 0), fall back to distance computation
                                                denom = (a * a + b * b + c * c) ** 0.5
                                                if denom != 0:
                                                    distance_to_origin = abs(d) / denom
                                        except Exception:
                                            distance_to_origin = None
                        except Exception:
                            distance_to_origin = None

                    # As a last resort, try to get a normal and a point on plane
                    if self.detected_floor_height is None and distance_to_origin is None:
                        try:
                            normal = None
                            if hasattr(plane, 'get_normal'):
                                normal = plane.get_normal()
                            elif hasattr(plane, 'normal'):
                                normal = plane.normal
                            if normal is not None and hasattr(normal, '__len__'):
                                # If we have a normal and a distance, approximate Y coordinate of the
                                # closest point on the plane to the origin: point = normal * (-distance)
                                # If we only have normal, we can't compute distance; default to None.
                                if distance_to_origin is not None:
                                    try:
                                        # assume normal is normalized
                                        py = float(normal[1])
                                        self.detected_floor_height = py * (-distance_to_origin)
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                    # If we computed distance_to_origin but not direct Y, and detected_floor_height still None,
                    # attempt to infer a Y using normal if available
                    if self.detected_floor_height is None and distance_to_origin is not None:
                        try:
                            normal = None
                            if hasattr(plane, 'get_normal'):
                                normal = plane.get_normal()
                            elif hasattr(plane, 'normal'):
                                normal = plane.normal
                            if normal is not None and hasattr(normal, '__len__'):
                                py = float(normal[1])
                                self.detected_floor_height = py * (-distance_to_origin)
                            else:
                                # unable to compute Y; leave as None
                                self.detected_floor_height = None
                        except Exception:
                            self.detected_floor_height = None
                    break
                else:
                    # Keep trying
                    time.sleep(wait_secs)
                    wait_secs *= 2
            except Exception as e:
                print(f"[WARNING] Floor plane detection error: {e}")
                time.sleep(wait_secs)
                wait_secs *= 2
        if self.detected_floor_height is None:
            print("[WARNING] Floor plane detection failed after retries")
        else:
            self._last_floor_detect_time = time.time()

    def get_detected_floor_height(self) -> Optional[float]:
        """Return the detected floor height from ZED SDK, or None if not detected."""
        return self.detected_floor_height

    def get_camera_poses(self):
        """
        Return a list of camera poses known to the server (from calibration files).
        Each pose is a dict: { 'serial': <serial>, 'position': (x,y,z), 'target': (x,y,z) }
        If no calib file is present, returns an empty list.
        """
        # Return cached result (permanent cache - compute once)
        if self._camera_pose_cache is not None:
            return self._camera_pose_cache

        poses = []

        try:
            calib_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "server", "calib")
            calib_file = os.path.join(calib_dir, "calib.json")
            # For single camera mode, always provide a default camera pose
            if not self.is_fusion_mode and self.camera is not None:
                try:
                    serial = str(self.camera.get_camera_information().serial_number)
                except Exception:
                    try:
                        devs = sl.Camera.get_device_list()
                        if devs and len(devs) >= 1:
                            serial = str(devs[0].serial_number)
                        else:
                            serial = "unknown"
                    except Exception:
                        serial = "unknown"

                poses.append({
                    'serial': serial,
                    'position': (0.0, 0.0, 0.0),
                    'target': (0.0, 0.0, -200.0)
                })
                # Cache and return for single camera mode
                self._camera_pose_cache = poses
                self._camera_pose_cache_time = time.time()
                return poses

            if os.path.isfile(calib_file):
                with open(calib_file, 'r') as fh:
                    data = json.load(fh)

                # Build list of available camera serials for filtering
                available_serials = set()
                try:
                    # Add serials from fusion subscriptions if available
                    if self.fusion is not None:
                        # fusion API doesn't easily expose subscribed serials; skip
                        pass

                    # Add serial from single camera if available
                    if self.camera is not None:
                        try:
                            serial = str(self.camera.get_camera_information().serial_number)
                            available_serials.add(serial)
                        except Exception:
                            pass

                    # Add serials from SDK device list
                    try:
                        devs = sl.Camera.get_device_list()
                        if devs:
                            for dev in devs:
                                try:
                                    available_serials.add(str(dev.serial_number))
                                except Exception:
                                    pass
                    except Exception:
                        pass
                except Exception:
                    pass

                for key, entry in data.items():
                    try:
                        cfg = entry.get('FusionConfiguration', {})
                        pose_str = cfg.get('pose')
                        serial = cfg.get('serial_number', key)
                        # Only include calib entries for serials that are actually available
                        if available_serials and str(serial) not in available_serials:
                            continue

                        if not pose_str:
                            # no pose provided -> default
                            position = (0.0, 0.0, 0.0)
                            target = (0.0, 0.0, -200.0)
                        else:
                            # Default values
                            position = (0.0, 0.0, 0.0)
                            target = (0.0, 0.0, -200.0)

                            # Try JSON-like matrix first (e.g. [[...],[...],...]) or flat 16 numbers
                            parsed = None
                            try:
                                import json as _json
                                txt = pose_str.strip()
                                # Normalize single-quotes -> double-quotes for json.loads if it looks like a list
                                if txt.startswith('['):
                                    try:
                                        parsed = _json.loads(txt.replace("'", '"'))
                                    except Exception:
                                        parsed = None
                                else:
                                    parsed = None
                            except Exception:
                                parsed = None

                            if parsed:
                                # parsed could be nested lists or flat list
                                try:
                                    # nested 4x4 matrix case
                                    if (isinstance(parsed, list) and len(parsed) == 4 and
                                            all(isinstance(r, list) and len(r) == 4 for r in parsed)):
                                        m = parsed  # row-major 4x4
                                        # translation is last column
                                        tx, ty, tz = float(m[0][3]), float(m[1][3]), float(m[2][3])
                                        position = (tx, ty, tz)
                                        # compute forward = -R[:,2] (third column of rotation)
                                        try:
                                            fx = -float(m[0][2]); fy = -float(m[1][2]); fz = -float(m[2][2])
                                            target = (position[0] + fx * 200.0,
                                                      position[1] + fy * 200.0,
                                                      position[2] + fz * 200.0)
                                        except Exception:
                                            target = (position[0], position[1], position[2] - 200.0)
                                    # flat list of 16 numbers
                                    elif isinstance(parsed, list) and len(parsed) == 16:
                                        flat = [float(x) for x in parsed]
                                        # reshape to 4x4 row-major
                                        m = [flat[0:4], flat[4:8], flat[8:12], flat[12:16]]
                                        tx, ty, tz = float(m[0][3]), float(m[1][3]), float(m[2][3])
                                        position = (tx, ty, tz)
                                        try:
                                            fx = -float(m[0][2]); fy = -float(m[1][2]); fz = -float(m[2][2])
                                            target = (position[0] + fx * 200.0,
                                                      position[1] + fy * 200.0,
                                                      position[2] + fz * 200.0)
                                        except Exception:
                                            target = (position[0], position[1], position[2] - 200.0)
                                except Exception:
                                    parsed = None

                            if parsed is None:
                                # Fallback: parse key=value pairs like "tx=...,ty=...,tz=...,rx=...,..."
                                pose_dict = {}
                                for part in pose_str.split(','):
                                    if '=' in part:
                                        k, v = part.strip().split('=', 1)
                                        try:
                                            pose_dict[k] = float(v)
                                        except Exception:
                                            continue
                                position = (
                                    pose_dict.get('tx', 0.0),
                                    pose_dict.get('ty', 0.0),
                                    pose_dict.get('tz', 0.0)
                                )
                                # if rotation present (rx,ry,rz) we could compute forward, but fallback to -Z
                                target = (position[0], position[1], position[2] - 200.0)

                        poses.append({
                            'serial': str(serial),
                            'position': position,
                            'target': target
                        })
                    except Exception as e:
                        print(f"[WARNING] Failed to parse camera pose for {key}: {e}")
                        # continue to next entry

        except Exception as e:
            print(f"[WARNING] Failed to load camera poses: {e}")

        # Cache the result permanently
        self._camera_pose_cache = poses
        self._camera_pose_cache_time = time.time()
        return poses

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
        # resolve expected calibration file path
        try:
            calib_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "server", "calib")
            calib_file = os.path.join(calib_dir, "calib.json")
        except Exception:
            calib_file = None

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
                    if not calib_file or not os.path.isfile(calib_file):
                        print("[ERROR] Fusion mode requires calibration file at server/calib/calib.json")
                        return False
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
                if not calib_file or not os.path.isfile(calib_file):
                    print("[ERROR] Fusion mode requires calibration file at server/calib/calib.json")
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

    def _initialize_fusion_cameras(self, device_list, enable_body_tracking: bool = True, enable_floor_detection: bool = True) -> bool:
        """Initialize multiple cameras for fusion mode using a fusion configuration (calib) file."""
        try:
            # Find and load calib file
            calib_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "server", "calib")
            calib_file = os.path.join(calib_dir, "calib.json")
            if not os.path.isfile(calib_file):
                print("[ERROR] Fusion mode requires calibration file at server/calib/calib.json")
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

    def _find_calibration_file(self, serial: str) -> Optional[str]:
        """Find calibration file for camera serial"""
        try:
            calib_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "server", "calib")
            calib_file = os.path.join(calib_dir, "calib.json")

            if os.path.isfile(calib_file):
                return calib_file
            return None
        except Exception:
            return None

    def _load_pose_for_serial(self, serial: str):
        """
        Try to load/parse the pose entry for a given serial from server/calib/calib.json.
        Returns one of:
          - a nested 4x4 list (row-major) or flat 16-list
          - the raw pose string (fallback)
          - None if no pose found
        """
        try:
            calib_file = self._find_calibration_file(serial)
            if not calib_file:
                return None
            with open(calib_file, "r") as fh:
                data = json.load(fh)

            for key, entry in data.items():
                cfg = entry.get("FusionConfiguration", {})
                serial_cfg = cfg.get("serial_number", key)
                if str(serial_cfg) != str(serial):
                    continue
                pose_str = cfg.get("pose")
                if not pose_str:
                    return None

                # Try JSON list-like first
                try:
                    txt = pose_str.strip()
                    if txt.startswith('['):
                        parsed = json.loads(txt.replace("'", '"'))
                        return parsed
                except Exception:
                    pass

                # fallback: return raw string (key=value style)
                return pose_str
        except Exception:
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
                    frame.cameras = self.get_camera_poses()

                    self.broadcast_frame(frame)

                    # Update UI if callback is set
                    if self.update_callback:
                        try:
                            threading.Thread(target=self.update_callback, args=(frame.people, self.detected_floor_height), daemon=True).start()
                        except Exception:
                            pass
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
                if fusion_status not in non_critical_errors:
                    print(f"[WARNING] Fusion process failed: {fusion_status}")
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
            return Frame(
                timestamp=bodies.timestamp.get_seconds(),
                people=people,
                body_model="BODY_34",
                floor_height=self.detected_floor_height,
                cameras=self.get_camera_poses()  # Pre-populate to avoid per-frame work
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
            return Frame(
                timestamp=bodies.timestamp.get_seconds(),
                people=people,
                body_model="BODY_34",
                floor_height=self.detected_floor_height,
                cameras=self.get_camera_poses()
            )

        return None

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

