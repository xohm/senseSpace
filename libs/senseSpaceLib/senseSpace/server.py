#!/usr/bin/env python3
"""
SenseSpace Server Library

Core server functionality for ZED SDK body tracking and TCP broadcasting.
"""

import socket
import threading
import json
import time
import glob
import os
from typing import Optional, Callable, List

import pyzed.sl as sl
from .protocol import Frame, Person, Joint
from .visualization import get_stable_floor_height


class SenseSpaceServer:
    """Main server class for ZED SDK body tracking and TCP broadcasting"""
    
    def __init__(self, host="0.0.0.0", port=12345):
        self.host = host
        self.port = port
        self.clients = []
        self.server_socket = None
        self.running = False

        # ZED SDK state
        self.detected_floor_height = None
        self.fusion = None
        self.camera = None
        self.is_fusion_mode = False
        self._last_floor_detect_time = 0.0

        # Callback for UI updates
        self.update_callback = None
        # No background floor detection here; follow SDK best-practice: detect once at init
        
    def set_update_callback(self, callback: Callable):
        """Set callback function for UI updates (e.g., Qt viewer updates)"""
        self.update_callback = callback
    
    def start_tcp_server(self):
        """Start the TCP server in a background thread"""
        if self.running:
            return
            
        self.running = True
        server_thread = threading.Thread(target=self._tcp_server_loop, daemon=True)
        server_thread.start()
        print(f"[SERVER] TCP server started on {self.host}:{self.port}")
    
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
    
    def _tcp_server_loop(self):
        """Main TCP server loop"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen()
            
            while self.running:
                try:
                    conn, addr = self.server_socket.accept()
                    self.clients.append(conn)
                    client_thread = threading.Thread(
                        target=self._client_handler, 
                        args=(conn, addr), 
                        daemon=True
                    )
                    client_thread.start()
                except Exception as e:
                    if self.running:
                        print(f"[SERVER] Error accepting connection: {e}")
                    break
        except Exception as e:
            print(f"[SERVER] TCP server error: {e}")
    
    def _client_handler(self, conn, addr):
        """Handle individual client connection"""
        print(f"[CLIENT CONNECTED] {addr}")
        try:
            while self.running:
                data = conn.recv(1)  # keep alive
                if not data:
                    break
        except:
            pass
        finally:
            print(f"[CLIENT DISCONNECTED] {addr}")
            if conn in self.clients:
                self.clients.remove(conn)
            try:
                conn.close()
            except:
                pass
    
    def broadcast_frame(self, frame: Frame):
        """Broadcast a frame to all connected clients"""
        if not self.clients:
            return
            
        message = {
            "type": "frame",
            "data": frame.to_dict()
        }
        
        msg_bytes = (json.dumps(message) + "\n").encode("utf-8")
        
        for client in list(self.clients):
            try:
                client.sendall(msg_bytes)
            except:
                # Remove disconnected clients
                if client in self.clients:
                    self.clients.remove(client)
                try:
                    client.close()
                except:
                    pass
    
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

    # No background floor worker: follow SDK best practice — detect once at init and only re-run on demand
    
    def initialize_cameras(self) -> bool:
        """Initialize ZED cameras (fusion or single camera mode)"""
        # Look for latest calibration file in 'calib' folder
        calib_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "server", "calib")
        latest_calib = None
        if os.path.isdir(calib_dir):
            calib_files = sorted(
                [f for f in glob.glob(os.path.join(calib_dir, "*.json"))],
                key=os.path.getmtime, reverse=True)
            if calib_files:
                latest_calib = calib_files[0]
                print(f"[FUSION] Using calibration file: {latest_calib}")
        
        # Discover available cameras
        cam_list = sl.Camera.get_device_list()
        num_cams = len(cam_list)
        
        if num_cams == 0:
            print("[ERROR] No ZED cameras found. Please connect at least one camera.")
            return False
        
        if num_cams == 1:
            return self._initialize_single_camera(cam_list[0])
        else:
            return self._initialize_fusion_cameras(cam_list, latest_calib)
    
    def _initialize_single_camera(self, cam_info) -> bool:
        """Initialize single camera mode"""
        print("[INFO] Single camera detected — using direct Camera mode.")
        
        self.camera = sl.Camera()
        self.is_fusion_mode = False
        
        # Set init params
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1080
        init_params.camera_fps = 30
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        # Prefer NEURAL depth mode if available (newer ZED SDKs); fall back to ULTRA
        try:
            init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        except Exception:
            init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        
        # Open camera
        status = self.camera.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"[ERROR] Failed to open camera: {status}")
            return False
        
        # Enable positional tracking (required by object/body tracking on many SDK versions)
        try:
            pt_params = sl.PositionalTrackingParameters()
            # some SDKs require passing init params or enabling with default
            self.camera.enable_positional_tracking(pt_params)
        except Exception:
            # Older or different SDKs may expose different signatures; ignore non-fatal failures
            pass

        # Now attempt floor plane detection in the positional/world frame
        try:
            self.find_floor_plane(self.camera)
        except Exception:
            pass

        # Enable body tracking
        bt_params = sl.BodyTrackingParameters()
        bt_params.enable_tracking = True
        bt_params.body_format = sl.BODY_FORMAT.BODY_34
        bt_params.enable_body_fitting = True
        
        status = self.camera.enable_body_tracking(bt_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"[ERROR] Failed to enable body tracking: {status}")
            return False
        
        # Set runtime parameters
        try:
            if hasattr(self.camera, 'set_body_tracking_runtime_parameters'):
                self.camera.set_body_tracking_runtime_parameters(sl.BodyTrackingRuntimeParameters(), 0)
        except Exception:
            pass  # non-fatal

        print("[INFO] Camera opened and body tracking enabled (single-camera mode)")
        return True
    
    def _initialize_fusion_cameras(self, cam_list, latest_calib) -> bool:
        """Initialize fusion mode with multiple cameras"""
        print(f"[INFO] {len(cam_list)} cameras detected — using Fusion mode.")
        
        self.fusion = sl.Fusion()
        self.is_fusion_mode = True
        
        # Initialize fusion
        init_params = sl.InitFusionParameters()
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        
        status = self.fusion.init(init_params)
        if status != sl.FUSION_ERROR_CODE.SUCCESS:
            print(f"[ERROR] Failed to initialize fusion: {status}")
            return False
        
        # Subscribe to cameras
        for i, cam_info in enumerate(cam_list):
            print(f"[FUSION] Subscribing to camera {i}: SN={cam_info.serial_number}")
            
            init_params = sl.InitParameters()
            init_params.camera_resolution = sl.RESOLUTION.HD1080
            init_params.camera_fps = 30
            init_params.coordinate_units = sl.UNIT.MILLIMETER
            init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
            try:
                init_params.depth_mode = sl.DEPTH_MODE.NEURAL
            except Exception:
                init_params.depth_mode = sl.DEPTH_MODE.ULTRA
            
            comm_params = sl.CommunicationParameters()
            comm_params.set_for_shared_memory()
            
            status = self.fusion.subscribe(
                cam_info.serial_number,
                comm_params,
                sl.Transform(),
                init_params
            )
            
            if status != sl.FUSION_ERROR_CODE.SUCCESS:
                print(f"[WARNING] Failed to subscribe to camera {i}: {status}")
        
        # Enable body tracking
        bt_params = sl.BodyTrackingFusionParameters()
        bt_params.enable_tracking = True
        bt_params.body_format = sl.BODY_FORMAT.BODY_34
        bt_params.enable_body_fitting = True
        
        status = self.fusion.enable_body_tracking(bt_params)
        if status != sl.FUSION_ERROR_CODE.SUCCESS:
            print(f"[ERROR] Failed to enable fusion body tracking: {status}")
            return False
        
        print("[INFO] Fusion initialized and body tracking enabled")
        return True
    
    def run_body_tracking_loop(self):
        """Main body tracking loop"""
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
                
                # Process bodies and create frame
                people = []
                # Build people list similar to _process_bodies_single_camera but inline
                for person in bodies.body_list:
                    kp_attr = getattr(person, 'keypoint', None)
                    if kp_attr is None:
                        kp_attr = getattr(person, 'keypoints', None)
                    keypoints = kp_attr if kp_attr is not None else []

                    conf_attr = getattr(person, 'keypoint_confidence', None)
                    if conf_attr is None:
                        conf_attr = getattr(person, 'keypoint_confidences', None)
                    confidences = conf_attr if conf_attr is not None else []

                    orientations = getattr(person, 'global_orientation', None)
                    if orientations is None:
                        orientations = getattr(person, 'global_root_orientation', None)

                    joints = []
                    for i, kp in enumerate(keypoints):
                        pos = {"x": float(kp[0]), "y": float(kp[1]), "z": float(kp[2])}
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

                frame = None
                try:
                    frame = self._process_bodies_single_camera(bodies)
                except Exception:
                    # fallback: build frame manually if helper fails
                    pass
                # Only include floor height when ZED provided a detected value.
                people_for_floor = frame.people if frame else people
                detected_floor = self.detected_floor_height

                # If frame exists, update its floor_height to the detected value (may be None).
                if frame:
                    frame.floor_height = detected_floor
                else:
                    frame = Frame(
                        timestamp=time.time(),
                        people=people_for_floor,
                        body_model="BODY_34",
                        floor_height=detected_floor
                    )

                # Broadcast to clients
                self.broadcast_frame(frame)

                # Update UI if callback is set (provide detected floor height or None)
                if self.update_callback:
                    try:
                        self.update_callback(frame.people, detected_floor)
                    except Exception:
                        pass

                # Do not attempt floor re-detection inside the tight capture loop.
                # Follow SDK best practice: floor is detected once at init and only
                # re-detected on-demand. This avoids periodic costly plane-fitting
                # operations that stall the body-tracking loop.
    
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
            
            if self.fusion.process() == sl.FUSION_ERROR_CODE.SUCCESS:
                status = self.fusion.retrieve_bodies(bodies)
                if status == sl.FUSION_ERROR_CODE.SUCCESS:
                    # Process bodies and create frame
                    frame = self._process_bodies_fusion(bodies)
                    if frame:
                        # Ensure frame.floor_height reflects only the ZED-detected value (may be None)
                        frame.floor_height = self.detected_floor_height

                        # Broadcast to clients
                        self.broadcast_frame(frame)

                        # Update UI if callback is set
                        if self.update_callback:
                            self.update_callback(frame.people, self.detected_floor_height)
    
    def _process_bodies_single_camera(self, bodies) -> Optional[Frame]:
        """Process bodies from single camera and create Frame object"""
        people = []
        
        for person in bodies.body_list:
            # Handle different SDK versions for keypoint attributes
            kp_attr = getattr(person, 'keypoint', None)
            if kp_attr is None:
                kp_attr = getattr(person, 'keypoints', None)
            keypoints = kp_attr if kp_attr is not None else []

            conf_attr = getattr(person, 'keypoint_confidence', None)
            if conf_attr is None:
                conf_attr = getattr(person, 'keypoint_confidences', None)
            confidences = conf_attr if conf_attr is not None else []

            orientations = getattr(person, 'global_orientation', None)
            if orientations is None:
                orientations = getattr(person, 'global_root_orientation', None)
            
            joints = []
            for i, kp in enumerate(keypoints):
                pos = {"x": float(kp[0]), "y": float(kp[1]), "z": float(kp[2])}
                
                if orientations is not None:
                    try:
                        if len(orientations.shape) == 2:  # Nx4 array
                            ori = orientations[i] if i < len(orientations) else orientations[0]
                        else:  # Single quaternion
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
                floor_height=self.detected_floor_height
            )
        
        return None
    
    def _process_bodies_fusion(self, bodies) -> Optional[Frame]:
        """Process bodies from fusion and create Frame object"""
        people = []
        
        for person in bodies.body_list:
            kp_attr = getattr(person, 'keypoint', None)
            if kp_attr is None:
                kp_attr = getattr(person, 'keypoints', None)
            keypoints = kp_attr if kp_attr is not None else []

            conf_attr = getattr(person, 'keypoint_confidence', None)
            if conf_attr is None:
                conf_attr = getattr(person, 'keypoint_confidences', None)
            confidences = conf_attr if conf_attr is not None else []
            
            joints = []
            for i, kp in enumerate(keypoints):
                pos = {"x": float(kp[0]), "y": float(kp[1]), "z": float(kp[2])}
                ori = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}  # Default orientation
                conf = float(confidences[i]) if i < len(confidences) else 0.0
                joints.append(Joint(i=i, pos=pos, ori=ori, conf=conf))
            
            people.append(Person(
                id=person.id,
                tracking_state=str(person.tracking_state),
                confidence=float(person.confidence),
                skeleton=joints
            ))
        
        return Frame(
            timestamp=bodies.timestamp.get_seconds(),
            people=people,
            body_model="BODY_34",
            floor_height=self.detected_floor_height
        )
    
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
                self.fusion.disable_body_tracking()
                self.fusion.close()
            except:
                pass
            self.fusion = None
        
        self.stop_tcp_server()
        print("[INFO] Server cleanup completed")
