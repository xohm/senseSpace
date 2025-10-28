#!/usr/bin/env python3
"""
Segmentation-based Point Cloud Server for senseSpace

Uses ZED SDK for skeleton tracking and MediaPipe for person segmentation.
Combines ZED skeleton IDs with MediaPipe segmentation masks for accurate per-person point clouds.
"""

import socket
import time
import threading
import queue
import struct
import numpy as np
from typing import List, Optional, Callable
import pyzed.sl as sl
import os
import sys

# Add library path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
libs_path = os.path.join(repo_root, 'libs')
if libs_path not in sys.path:
    sys.path.insert(0, libs_path)

from senseSpaceLib.senseSpace.server import SenseSpaceServer, get_local_ip
from senseSpaceLib.senseSpace.protocol import Frame
from senseSpaceLib.senseSpace.communication import serialize_message

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[ERROR] MediaPipe not available. Install with: pip install mediapipe")

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
    _zstd_compressor = zstd.ZstdCompressor(level=3)
except ImportError:
    ZSTD_AVAILABLE = False
    print("[WARNING] zstandard not available - point clouds will not be compressed")


class SegmentationPointCloudServer(SenseSpaceServer):
    """
    Server that uses ZED SDK for skeleton tracking and MediaPipe for segmentation.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 12345, 
                 pointcloud_port: int = 12346, 
                 downsample_factor: int = 4,
                 max_distance: float = 5000.0,
                 voxel_size: float = 20.0,
                 mode: str = "per-person",
                 quality: int = 0):
        
        super().__init__(host=host, port=port)
        
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe is required for this server. Install with: pip install mediapipe")
        
        # Quality presets: (resolution, fps)
        self.quality_presets = {
            0: (sl.RESOLUTION.HD720, 60),   # 1280x720 @ 60fps (default, best)
            1: (sl.RESOLUTION.HD720, 30),   # 1280x720 @ 30fps
            2: (sl.RESOLUTION.VGA, 60),     # 672x376 @ 60fps
            3: (sl.RESOLUTION.VGA, 30),     # 672x376 @ 30fps
            4: (sl.RESOLUTION.HD1080, 30),  # 1920x1080 @ 30fps (highest quality)
        }
        
        if quality not in self.quality_presets:
            print(f"[WARNING] Invalid quality {quality}, using default (0)")
            quality = 0
        
        self.quality = quality
        self.camera_resolution, self.camera_fps = self.quality_presets[quality]
        
        # Point cloud settings
        self.pointcloud_port = pointcloud_port
        self.downsample_factor = max(1, downsample_factor)
        self.max_distance = max_distance
        self.voxel_size = voxel_size
        self.pointcloud_mode = mode  # Only "per-person" supported
        
        if mode != "per-person":
            print("[WARNING] Only 'per-person' mode is supported with MediaPipe segmentation")
            self.pointcloud_mode = "per-person"
        
        # Point cloud TCP server
        self.pc_server_socket = None
        self.pc_server_thread = None
        self.pc_clients = []
        self.pc_client_queues = {}
        self.pc_client_sender_threads = {}
        
        # Point cloud data
        self.point_cloud_mat = None
        self.pc_lock = threading.Lock()
        
        # MediaPipe segmentation
        self.mp_segmenter = None
        self.segmentation_lock = threading.Lock()
        
        # Cached data for per-person mode
        self._cached_bodies = None  # ZED Bodies for skeleton tracking (single camera)
        self._cached_camera_image = None  # Camera image for MediaPipe (single camera)
        self._cached_fusion_data = {}  # {serial: (bodies, camera_image, pc_mat)} for fusion
        
        # Debug image saving
        self._save_debug_image = False  # Flag to trigger debug image saving
        
        # Ensure tmp directory exists
        self.tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp')
        os.makedirs(self.tmp_dir, exist_ok=True)
        
        # Get resolution name for display
        res_names = {
            sl.RESOLUTION.HD720: "HD720 (1280x720)",
            sl.RESOLUTION.VGA: "VGA (672x376)",
            sl.RESOLUTION.HD1080: "HD1080 (1920x1080)"
        }
        res_name = res_names.get(self.camera_resolution, str(self.camera_resolution))
        
        print(f"[INFO] Segmentation Point Cloud Server configured:")
        print(f"       Skeleton port: {port}")
        print(f"       Point cloud port: {pointcloud_port}")
        print(f"       Mode: {mode}")
        print(f"       Quality: {quality} ({res_name} @ {self.camera_fps} FPS)")
        print(f"       Segmentation: MediaPipe")
        print(f"       Downsample factor: {downsample_factor}")
        print(f"       Max distance: {max_distance}mm")
    
    def _initialize_single_camera(self, device_info, enable_body_tracking: bool = True, enable_floor_detection: bool = True) -> bool:
        """Initialize ZED camera with body tracking (no segmentation needed from ZED)."""
        try:
            self.camera = sl.Camera()

            # Set initialization parameters
            init_params = sl.InitParameters()
            init_params.camera_resolution = self.camera_resolution
            init_params.camera_fps = self.camera_fps
            init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
            init_params.coordinate_units = sl.UNIT.MILLIMETER
            init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

            # Open camera
            status = self.camera.open(init_params)
            if status != sl.ERROR_CODE.SUCCESS:
                print(f"[ERROR] Failed to open camera: {status}")
                return False

            # Enable positional tracking
            tracking_params = sl.PositionalTrackingParameters()
            status = self.camera.enable_positional_tracking(tracking_params)
            if status != sl.ERROR_CODE.SUCCESS:
                print(f"[ERROR] Failed to enable positional tracking: {status}")
                self.camera.close()
                return False

            # Enable body tracking (for skeleton only, not segmentation)
            if enable_body_tracking:
                body_params = sl.BodyTrackingParameters()
                body_params.enable_tracking = True
                body_params.enable_body_fitting = True
                body_params.body_format = sl.BODY_FORMAT.BODY_34
                body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
                # NO segmentation from ZED - we use MediaPipe instead
                
                status = self.camera.enable_body_tracking(body_params)
                if status != sl.ERROR_CODE.SUCCESS:
                    print(f"[ERROR] Failed to enable body tracking: {status}")
                    self.camera.close()
                    return False

            # Detect floor if requested
            if enable_floor_detection:
                self.find_floor_plane(self.camera)

            self.is_fusion_mode = False
            print("[INFO] Single camera initialized successfully")
            
            # Initialize MediaPipe segmentation
            self._initialize_mediapipe()
            
            return True

        except Exception as e:
            print(f"[ERROR] Camera initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _initialize_mediapipe(self):
        """Initialize MediaPipe person segmentation."""
        try:
            mp_selfie_segmentation = mp.solutions.selfie_segmentation
            self.mp_segmenter = mp_selfie_segmentation.SelfieSegmentation(
                model_selection=1  # 1 = general model (better for full body)
            )
            print("[INFO] MediaPipe segmentation initialized")
        except Exception as e:
            print(f"[ERROR] Failed to initialize MediaPipe: {e}")
            raise
    
    def _initialize_fusion_cameras(self, device_list, enable_body_tracking: bool = True, enable_floor_detection: bool = True) -> bool:
        """
        Initialize fusion mode with MediaPipe segmentation on each camera.
        Overrides parent class to add MediaPipe support and store camera poses.
        """
        # Call parent class fusion initialization (handles ZED fusion setup)
        success = super()._initialize_fusion_cameras(device_list, enable_body_tracking, enable_floor_detection)
        
        if not success:
            return False
        
        # Store camera poses for point cloud transformation (if not already stored by parent)
        if not hasattr(self, '_fusion_camera_poses'):
            self._fusion_camera_poses = {}
        
        # Initialize MediaPipe segmentation (shared across all cameras)
        self._initialize_mediapipe()
        
        # Initialize point cloud mats for each camera
        self._pc_mats = {}
        for serial, cam in self._fusion_senders.items():
            resolution = cam.get_camera_information().camera_configuration.resolution
            self._pc_mats[serial] = sl.Mat(resolution.width, resolution.height, sl.MAT_TYPE.F32_C4)
        
        print(f"[INFO] Fusion mode with MediaPipe segmentation initialized ({len(self._fusion_senders)} cameras)")
        return True
    
    def start_tcp_server(self):
        """Start both skeleton and point cloud TCP servers."""
        # Start skeleton server (parent class)
        super().start_tcp_server()
        
        # Start point cloud server
        self._start_point_cloud_server()
    
    def _start_point_cloud_server(self):
        """Start point cloud TCP server."""
        try:
            self.pc_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.pc_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.pc_server_socket.bind((self.host, self.pointcloud_port))
            self.pc_server_socket.listen(5)
            
            print(f"[INFO] Point cloud TCP server listening on {self.host}:{self.pointcloud_port}")
            print(f"[INFO] Connect point cloud clients to: {get_local_ip()}:{self.pointcloud_port}")
            
            self.pc_server_thread = threading.Thread(target=self._accept_pc_clients, daemon=True)
            self.pc_server_thread.start()
            
            # Start point cloud streaming loop
            pc_loop_thread = threading.Thread(target=self._run_point_cloud_loop, daemon=True)
            pc_loop_thread.start()
            
        except Exception as e:
            print(f"[ERROR] Failed to start point cloud server: {e}")
    
    def _accept_pc_clients(self):
        """Accept point cloud client connections."""
        while self.running:
            try:
                client_socket, address = self.pc_server_socket.accept()
                print(f"[INFO] Point cloud client connected: {address}")
                
                self.pc_clients.append(client_socket)
                client_queue = queue.Queue(maxsize=10)
                self.pc_client_queues[client_socket] = client_queue
                
                sender_thread = threading.Thread(
                    target=self._send_to_pc_client,
                    args=(client_socket, client_queue),
                    daemon=True
                )
                self.pc_client_sender_threads[client_socket] = sender_thread
                sender_thread.start()
                
            except Exception as e:
                if self.running:
                    print(f"[ERROR] Error accepting point cloud client: {e}")
    
    def _send_to_pc_client(self, client_socket, client_queue):
        """Send point cloud data to a specific client."""
        try:
            while self.running and client_socket in self.pc_clients:
                try:
                    data = client_queue.get(timeout=1.0)
                    client_socket.sendall(data)
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"[ERROR] Failed to send to point cloud client: {e}")
                    break
        finally:
            self._remove_pc_client(client_socket)
    
    def _remove_pc_client(self, client_socket):
        """Remove a point cloud client."""
        if client_socket in self.pc_clients:
            self.pc_clients.remove(client_socket)
        if client_socket in self.pc_client_queues:
            del self.pc_client_queues[client_socket]
        if client_socket in self.pc_client_sender_threads:
            del self.pc_client_sender_threads[client_socket]
        try:
            client_socket.close()
        except:
            pass
    
    def _run_single_camera_loop(self):
        """Override to cache Bodies and camera image for MediaPipe."""
        runtime_params = sl.RuntimeParameters()
        bodies = sl.Bodies()
        camera_image = sl.Mat()

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
                # Retrieve bodies
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

                # Retrieve camera image for MediaPipe
                self.camera.retrieve_image(camera_image, sl.VIEW.LEFT)
                
                # Cache for per-person point cloud extraction
                with self.pc_lock:
                    self._cached_bodies = bodies
                    self._cached_camera_image = camera_image.get_data().copy()

                # Process bodies for skeleton tracking
                frame = self._process_bodies_single_camera(bodies)
                if frame:
                    frame.floor_height = self.detected_floor_height
                    self.broadcast_frame(frame)

                    # Update UI if callback is set
                    if self.update_callback:
                        try:
                            self.update_callback(frame.people, self.detected_floor_height)
                        except Exception:
                            pass
    
    def _run_fusion_loop(self):
        """
        Override fusion loop to capture camera images for MediaPipe segmentation.
        Similar to pointCloudServer but runs MediaPipe on each camera.
        """
        import pyzed.sl as sl
        
        bodies = sl.Bodies()
        MAX_FPS = 60
        FRAME_INTERVAL = 1.0 / MAX_FPS
        last_time = time.time()
        
        print("[INFO] Starting fusion loop with MediaPipe segmentation")
        
        while self.running:
            now = time.time()
            if now - last_time < FRAME_INTERVAL:
                time.sleep(FRAME_INTERVAL - (now - last_time))
            last_time = time.time()
            
            # Grab frames from local senders and capture data for MediaPipe
            try:
                if hasattr(self, '_fusion_senders'):
                    fusion_data = {}
                    
                    for serial, zed in self._fusion_senders.items():
                        try:
                            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                                # Retrieve bodies
                                try:
                                    zed.retrieve_bodies(bodies)
                                except Exception:
                                    pass
                                
                                # Retrieve camera image for MediaPipe
                                camera_image = sl.Mat()
                                zed.retrieve_image(camera_image, sl.VIEW.LEFT)
                                
                                # Retrieve point cloud
                                if self.pc_clients or hasattr(self, '_viz_point_cloud_callback'):
                                    try:
                                        zed.retrieve_measure(self._pc_mats[serial], sl.MEASURE.XYZRGBA)
                                        
                                        # Cache: bodies, camera image, and point cloud mat
                                        fusion_data[serial] = (
                                            bodies,
                                            camera_image.get_data().copy(),
                                            self._pc_mats[serial]
                                        )
                                    except Exception as e:
                                        print(f"[WARNING] Failed to retrieve data from {serial}: {e}")
                        except Exception:
                            pass
                    
                    # Cache fusion data for point cloud extraction
                    if fusion_data:
                        with self.pc_lock:
                            self._cached_fusion_data = fusion_data
            except Exception:
                pass
            
            # Process fusion (skeleton tracking)
            fusion_status = self.fusion.process()
            if fusion_status == sl.FUSION_ERROR_CODE.SUCCESS:
                # Retrieve fused bodies
                try:
                    if hasattr(sl, 'BodyTrackingFusionRuntimeParameters'):
                        self.fusion.retrieve_bodies(bodies, sl.BodyTrackingFusionRuntimeParameters())
                    else:
                        self.fusion.retrieve_bodies(bodies)
                except Exception as e:
                    print(f'[WARNING] Fusion retrieve_bodies failed: {e}')
                    continue
                
                # Process and broadcast skeleton frame
                frame = self._process_bodies_fusion(bodies)
                if frame:
                    frame.floor_height = self.detected_floor_height
                    self.broadcast_frame(frame)
                    
                    if self.update_callback:
                        try:
                            self.update_callback(frame.people, self.detected_floor_height)
                        except Exception:
                            pass
    
    def _run_point_cloud_loop(self):
        """Point cloud streaming loop using MediaPipe segmentation."""
        MAX_FPS = 30
        FRAME_INTERVAL = 1.0 / MAX_FPS
        last_time = time.time()
        
        print("[INFO] Starting point cloud streaming loop with MediaPipe segmentation")
        
        while self.running:
            now = time.time()
            if now - last_time < FRAME_INTERVAL:
                time.sleep(FRAME_INTERVAL - (now - last_time))
            last_time = time.time()
            
            try:
                # Retrieve per-person point clouds using MediaPipe
                result = self._retrieve_per_person_point_clouds()
                if result:
                    per_person_list, timestamp = result
                    self._broadcast_per_person_point_clouds(per_person_list, timestamp)
                    
            except Exception as e:
                print(f"[ERROR] Point cloud loop error: {e}")
                import traceback
                traceback.print_exc()
    
    def _retrieve_per_person_point_clouds(self):
        """Retrieve per-person point clouds using MediaPipe segmentation (single or fusion mode)."""
        # Check if fusion or single camera mode
        if self.is_fusion_mode:
            return self._retrieve_fusion_per_person_point_clouds()
        else:
            return self._retrieve_single_per_person_point_clouds()
    
    def _retrieve_single_per_person_point_clouds(self):
        """Retrieve per-person point clouds for single camera mode."""
        if not self.camera:
            return None
        
        try:
            # Allocate point cloud mat if needed
            if self.point_cloud_mat is None:
                resolution = self.camera.get_camera_information().camera_configuration.resolution
                self.point_cloud_mat = sl.Mat(resolution.width, resolution.height, sl.MAT_TYPE.F32_C4)
            
            # Retrieve XYZRGBA point cloud
            self.camera.retrieve_measure(self.point_cloud_mat, sl.MEASURE.XYZRGBA)
            timestamp = time.time()
            
            # Get cached bodies and camera image
            with self.pc_lock:
                cached_bodies = self._cached_bodies
                cached_image = self._cached_camera_image
            
            if cached_bodies is None or cached_image is None:
                return None
            
            # Extract per-person point clouds using MediaPipe
            per_person_list = self._extract_per_person_with_mediapipe(
                cached_bodies, 
                self.point_cloud_mat,
                cached_image
            )
            
            if per_person_list:
                return (per_person_list, timestamp)
            else:
                return None
                
        except Exception as e:
            print(f"[ERROR] Failed to retrieve per-person point clouds: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _retrieve_fusion_per_person_point_clouds(self):
        """Retrieve and merge per-person point clouds from fusion cameras using MediaPipe."""
        try:
            with self.pc_lock:
                cached_fusion_data = dict(self._cached_fusion_data)  # Copy
            
            if not cached_fusion_data:
                return None
            
            timestamp = time.time()
            all_person_clouds = {}  # {person_id: {"points": [], "colors": []}}
            
            # Process each camera
            for serial, (bodies, camera_image, pc_mat) in cached_fusion_data.items():
                # Extract per-person point clouds using MediaPipe for this camera
                person_clouds = self._extract_per_person_with_mediapipe(bodies, pc_mat, camera_image)
                
                # Transform points to fusion coordinate space
                if hasattr(self, '_fusion_camera_poses') and serial in self._fusion_camera_poses:
                    camera_pose = self._fusion_camera_poses[serial]
                    
                    for pc in person_clouds:
                        # Transform points using camera pose
                        pc["points"] = self._transform_points_to_fusion(pc["points"], camera_pose)
                
                # Merge point clouds from same person across cameras
                for pc in person_clouds:
                    person_id = pc["person_id"]
                    if person_id not in all_person_clouds:
                        all_person_clouds[person_id] = {"points": [], "colors": []}
                    
                    all_person_clouds[person_id]["points"].append(pc["points"])
                    all_person_clouds[person_id]["colors"].append(pc["colors"])
            
            # Concatenate points from all cameras for each person
            per_person_list = []
            for person_id, data in all_person_clouds.items():
                if data["points"]:
                    merged_points = np.vstack(data["points"])
                    merged_colors = np.vstack(data["colors"])
                    per_person_list.append({
                        "person_id": person_id,
                        "points": merged_points,
                        "colors": merged_colors
                    })
            
            return (per_person_list, timestamp) if per_person_list else None
            
        except Exception as e:
            print(f"[ERROR] Failed to retrieve fusion per-person point clouds: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _transform_points_to_fusion(self, points: np.ndarray, pose_matrix: sl.Transform) -> np.ndarray:
        """
        Transform points from camera coordinate space to fusion coordinate space.
        
        Args:
            points: Nx3 array of points in camera space
            pose_matrix: sl.Transform containing 4x4 transformation matrix
        
        Returns:
            Nx3 array of points in fusion space
        """
        if len(points) == 0:
            return points
        
        try:
            # Get 4x4 transformation matrix from ZED Transform object
            transform_matrix = pose_matrix.m  # 4x4 numpy array
            
            # Extract rotation (3x3) and translation (3x1)
            rotation = transform_matrix[:3, :3]
            translation = transform_matrix[:3, 3]
            
            # Transform: P_fusion = R * P_camera + T
            # points is Nx3, we need to transpose for matrix multiplication
            transformed = (rotation @ points.T).T + translation
            
            return transformed.astype(np.float32)
            
        except Exception as e:
            print(f"[WARNING] Failed to transform points: {e}")
            import traceback
            traceback.print_exc()
            return points  # Return untransformed if transformation fails
    
    def _extract_per_person_with_mediapipe(self, bodies: sl.Bodies, pc_mat: sl.Mat, camera_image: np.ndarray):
        """
        Extract per-person point clouds using MediaPipe segmentation.
        
        Args:
            bodies: ZED Bodies for skeleton tracking and person IDs
            pc_mat: Point cloud data
            camera_image: RGB camera image (BGRA format from ZED)
        
        Returns:
            List of dicts with person_id, points, colors
        """
        per_person_clouds = []
        
        # Get point cloud data
        pc_data = pc_mat.get_data()  # Shape: (H, W, 4) - XYZRGBA
        height, width = pc_data.shape[:2]
        
        # Convert BGRA to RGB for MediaPipe
        rgb_image = camera_image[:, :, [2, 1, 0]]  # BGR -> RGB
        
        # Run MediaPipe segmentation
        with self.segmentation_lock:
            results = self.mp_segmenter.process(rgb_image)
        
        if results.segmentation_mask is None:
            return per_person_clouds
        
        # MediaPipe returns a float mask [0.0, 1.0]
        # Convert to binary mask (threshold at 0.5)
        mediapipe_mask = (results.segmentation_mask > 0.5).astype(np.uint8)
        
        # Now match MediaPipe mask to ZED skeleton persons
        # For each tracked person, extract their portion of the MediaPipe mask
        for person in bodies.body_list:
            try:
                person_id = person.id
                tracking_state = str(person.tracking_state)
                
                if tracking_state != "OK":
                    continue
                
                # Get 2D bounding box from ZED skeleton
                bbox_2d = person.bounding_box_2d
                if len(bbox_2d) < 4:
                    continue
                
                top_left = bbox_2d[0]
                bottom_right = bbox_2d[2]
                
                bbox_x_min = max(0, int(top_left[0]))
                bbox_y_min = max(0, int(top_left[1]))
                bbox_x_max = min(width, int(bottom_right[0]))
                bbox_y_max = min(height, int(bottom_right[1]))
                
                # Extract mask region within bounding box
                bbox_mask = mediapipe_mask[bbox_y_min:bbox_y_max, bbox_x_min:bbox_x_max]
                
                # Find person pixels in the bounding box
                local_y, local_x = np.where(bbox_mask > 0)
                
                if len(local_y) == 0:
                    continue
                
                # Convert to global image coordinates
                img_y = bbox_y_min + local_y
                img_x = bbox_x_min + local_x
                
                # Extract person's points
                person_pc = pc_data[img_y, img_x]  # Shape: (N, 4)
                
                # Filter out invalid points and apply distance filter
                points = person_pc[:, :3]
                colors_raw = person_pc[:, 3]
                
                # Check for valid points (not NaN or too far)
                valid_mask = np.isfinite(points).all(axis=1)
                if self.max_distance > 0:
                    distances = np.linalg.norm(points, axis=1)
                    valid_mask &= (distances < self.max_distance)
                
                if not valid_mask.any():
                    continue
                
                points = points[valid_mask]
                colors_raw = colors_raw[valid_mask]
                
                # Depth-based outlier filtering
                # Calculate depth (distance from camera) for each point
                depths = np.linalg.norm(points, axis=1)
                
                if len(depths) > 10:  # Only filter if we have enough points
                    # Use median depth as reference (more robust than mean)
                    median_depth = np.median(depths)
                    
                    # Filter out points that are too far from median depth
                    # Threshold: 500mm (50cm) from median depth
                    # This removes background/foreground outliers
                    depth_threshold = 500.0  # mm
                    depth_mask = np.abs(depths - median_depth) < depth_threshold
                    
                    # Apply depth filter
                    points = points[depth_mask]
                    colors_raw = colors_raw[depth_mask]
                
                if len(points) == 0:
                    continue
                
                # Unpack RGBA
                colors_uint = colors_raw.view(np.uint32)
                r = (colors_uint & 0xFF)
                g = ((colors_uint >> 8) & 0xFF)
                b = ((colors_uint >> 16) & 0xFF)
                colors = np.stack([r, g, b], axis=1).astype(np.uint8)
                
                # Apply downsampling if needed
                if self.downsample_factor > 1:
                    indices = np.arange(0, len(points), self.downsample_factor)
                    points = points[indices]
                    colors = colors[indices]
                
                # Save debug images if requested
                if self._save_debug_image:
                    self._save_mediapipe_overlay_images(
                        camera_image, 
                        mediapipe_mask, 
                        bbox_2d, 
                        person_id
                    )
                
                per_person_clouds.append({
                    "person_id": person_id,
                    "points": points,
                    "colors": colors
                })
                
            except Exception as e:
                print(f"[WARNING] Failed to extract point cloud for person {person.id}: {e}")
                continue
        
        # Reset debug flag after processing all persons
        if self._save_debug_image:
            self._save_debug_image = False
        
        return per_person_clouds
    
    def _save_mediapipe_overlay_images(self, camera_image, mediapipe_mask, bbox_2d, person_id):
        """
        Save debug images: camera image, MediaPipe mask, and overlay.
        
        Args:
            camera_image: Camera image (BGRA format from ZED)
            mediapipe_mask: MediaPipe segmentation mask (0 or 1, full size)
            bbox_2d: 2D bounding box from ZED skeleton
            person_id: Person ID
        """
        try:
            from PyQt5.QtGui import QImage
            
            height, width = camera_image.shape[:2]
            
            # Convert BGRA to RGB
            camera_rgb = camera_image[:, :, [2, 1, 0]].copy()  # BGR -> RGB
            
            # Save full camera image
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            qimg = QImage(camera_rgb.data, width, height, width * 3, QImage.Format_RGB888)
            camera_path = os.path.join(self.tmp_dir, f"debug_camera_{person_id}_{timestamp_str}.png")
            qimg.save(camera_path)
            print(f"[INFO] Saved camera image: {camera_path}")
            
            # Get bounding box
            top_left = bbox_2d[0]
            bottom_right = bbox_2d[2]
            bbox_x_min = int(top_left[0])
            bbox_y_min = int(top_left[1])
            bbox_x_max = int(bottom_right[0])
            bbox_y_max = int(bottom_right[1])
            
            print(f"[DEBUG] BBox: ({bbox_x_min}, {bbox_y_min}) to ({bbox_x_max}, {bbox_y_max})")
            print(f"[DEBUG] MediaPipe mask shape: {mediapipe_mask.shape}, Camera shape: {height}x{width}")
            
            # Create mask visualization (convert binary mask to 0-255)
            mask_uint8 = (mediapipe_mask * 255).astype(np.uint8)
            
            # Save raw mask
            qimg_mask = QImage(mask_uint8.data, width, height, width, QImage.Format_Grayscale8)
            mask_path = os.path.join(self.tmp_dir, f"debug_mask_mediapipe_{person_id}_{timestamp_str}.png")
            qimg_mask.save(mask_path)
            print(f"[INFO] Saved MediaPipe mask: {mask_path}")
            
            # Create overlay image (camera + mask)
            overlay_img = camera_rgb.copy()
            
            # Create semi-transparent green overlay where mask == 1
            alpha = 0.5
            overlay_img[mediapipe_mask > 0, 1] = (overlay_img[mediapipe_mask > 0, 1] * (1 - alpha) + 255 * alpha).astype(np.uint8)
            
            # Draw bounding box in red
            cv2_available = True
            try:
                import cv2
            except ImportError:
                cv2_available = False
            
            if cv2_available:
                import cv2
                cv2.rectangle(overlay_img, (bbox_x_min, bbox_y_min), (bbox_x_max, bbox_y_max), (255, 0, 0), 2)
            
            # Save overlay image
            qimg_overlay = QImage(overlay_img.data, width, height, width * 3, QImage.Format_RGB888)
            overlay_path = os.path.join(self.tmp_dir, f"debug_overlay_{person_id}_{timestamp_str}.png")
            qimg_overlay.save(overlay_path)
            print(f"[INFO] Saved overlay image: {overlay_path}")
            
        except Exception as e:
            print(f"[WARNING] Failed to save debug images: {e}")
            import traceback
            traceback.print_exc()
    
    def _broadcast_per_person_point_clouds(self, per_person_list, timestamp):
        """Broadcast per-person point clouds to clients and visualization."""
        if not per_person_list:
            return
        
        # Update visualization if available
        if hasattr(self, '_viz_point_cloud_callback') and self._viz_point_cloud_callback:
            try:
                # Merge all persons for visualization
                all_points = np.vstack([p["points"] for p in per_person_list])
                all_colors = np.vstack([p["colors"] for p in per_person_list])
                self._viz_point_cloud_callback(all_points, all_colors)
            except Exception as e:
                print(f"[WARNING] Visualization callback failed: {e}")
        
        # Serialize and send to clients
        if self.pc_clients:
            try:
                serialized = self._serialize_per_person_point_clouds(per_person_list, timestamp)
                
                for client_socket in list(self.pc_clients):
                    try:
                        queue_obj = self.pc_client_queues.get(client_socket)
                        if queue_obj:
                            if queue_obj.full():
                                try:
                                    queue_obj.get_nowait()
                                except queue.Empty:
                                    pass
                            queue_obj.put_nowait(serialized)
                    except Exception as e:
                        print(f"[WARNING] Failed to queue for client: {e}")
                        
            except Exception as e:
                print(f"[ERROR] Serialization failed: {e}")
    
    def _serialize_per_person_point_clouds(self, per_person_list, timestamp):
        """Serialize per-person point clouds for network transmission."""
        # Per-person format:
        # Magic: 0x9f 0xd4
        # Timestamp: 8 bytes (double)
        # Person count: 4 bytes
        # Payload length: 4 bytes
        # Compressed payload:
        #   For each person:
        #     Person ID: 4 bytes
        #     Point count: 4 bytes
        #     Points: N * 12 bytes (x, y, z as floats)
        #     Colors: N * 3 bytes (r, g, b)
        
        # Build payload for all persons
        payload_parts = []
        for person_data in per_person_list:
            person_id = person_data["person_id"]
            points = person_data["points"]
            colors = person_data["colors"]
            
            payload_parts.append(struct.pack('>iI', person_id, len(points)))
            payload_parts.append(points.astype(np.float32).tobytes())
            payload_parts.append(colors.tobytes())
        
        payload = b''.join(payload_parts)
        
        # Compress if available
        if ZSTD_AVAILABLE:
            payload = _zstd_compressor.compress(payload)
        
        # Header: [magic:2][timestamp:8][person_count:4][length:4]
        header = struct.pack('BB', 0x9f, 0xd4) + struct.pack('>dII', timestamp, len(per_person_list), len(payload))
        
        return header + payload
    
    def cleanup(self):
        """Cleanup resources."""
        print("[INFO] Cleaning up segmentation point cloud server...")
        
        # Cleanup MediaPipe
        if self.mp_segmenter:
            self.mp_segmenter.close()
        
        # Cleanup point cloud server
        if self.pc_server_socket:
            try:
                self.pc_server_socket.close()
            except:
                pass
        
        for client in list(self.pc_clients):
            self._remove_pc_client(client)
        
        # Cleanup parent (ZED camera, etc.)
        super().cleanup()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SenseSpace Segmentation Point Cloud Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host address")
    parser.add_argument("--port", type=int, default=12345, help="Skeleton data port")
    parser.add_argument("--pc-port", type=int, default=12346, help="Point cloud data port")
    parser.add_argument("--downsample", type=int, default=4, help="Point cloud downsample factor")
    parser.add_argument("--max-distance", type=float, default=5000.0, help="Maximum point distance in mm")
    parser.add_argument("--quality", type=int, default=0, choices=[0, 1, 2, 3, 4],
                       help="Camera quality: 0=720p@60fps (default), 1=720p@30fps, 2=VGA@60fps, 3=VGA@30fps, 4=1080p@30fps")
    parser.add_argument("--viz", action="store_true", help="Enable Qt visualization window")
    
    args = parser.parse_args()
    
    # Optional Qt visualization
    app = None
    main_window = None
    
    if args.viz:
        try:
            from qt_viewer import MainWindow
            from PyQt5.QtWidgets import QApplication
            print("[INFO] Qt visualization enabled")
            app = QApplication(sys.argv)
            main_window = MainWindow()
            main_window.show()
        except ImportError as e:
            print(f"[WARNING] Qt visualization not available: {e}")
            args.viz = False
    
    # Create server
    server = SegmentationPointCloudServer(
        host=args.host,
        port=args.port,
        pointcloud_port=args.pc_port,
        downsample_factor=args.downsample,
        max_distance=args.max_distance,
        mode="per-person",
        quality=args.quality
    )
    
    try:
        # Initialize camera
        print("[INFO] Initializing camera...")
        if not server.initialize_cameras():
            print("[ERROR] Failed to initialize cameras")
            return 1
        
        # Connect visualization if enabled
        if args.viz and main_window:
            # Pass server reference
            main_window.server = server
            if hasattr(main_window, 'glWidget'):
                main_window.glWidget.server = server
            
            # Create callback wrapper
            def qt_update_callback(people_data, floor_height):
                try:
                    if hasattr(main_window, 'people_signal'):
                        main_window.people_signal.emit(people_data)
                        if hasattr(main_window, 'floor_height_signal') and floor_height is not None:
                            main_window.floor_height_signal.emit(floor_height)
                    elif hasattr(main_window, 'update_people'):
                        main_window.update_people(people_data)
                        if hasattr(main_window, 'set_floor_height') and floor_height is not None:
                            main_window.set_floor_height(floor_height)
                except RuntimeError:
                    pass
            
            server.set_update_callback(qt_update_callback)
            
            # Set point cloud visualization callback
            if hasattr(main_window, 'glWidget') and hasattr(main_window.glWidget, 'set_point_cloud'):
                def viz_point_cloud_callback(points, colors):
                    try:
                        main_window.glWidget.set_point_cloud(points, colors)
                    except RuntimeError:
                        pass
                server._viz_point_cloud_callback = viz_point_cloud_callback
                print("[INFO] Point cloud visualization enabled (press 'P' to toggle)")
            
            print("[INFO] Visualization connected to server")
        
        # Start servers
        print("[INFO] Starting servers...")
        server.start_tcp_server()
        
        # Run tracking loop
        print("[INFO] Starting tracking and point cloud streaming...")
        if args.viz and app:
            # Run server in background thread
            server_thread = threading.Thread(target=server.run_body_tracking_loop, daemon=True)
            server_thread.start()
            # Run Qt event loop in main thread
            app.exec_()
        else:
            # Run server in main thread
            server.run_body_tracking_loop()
        
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    except Exception as e:
        print(f"[ERROR] Server error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        server.cleanup()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
