#!/usr/bin/env python3
"""
Point Cloud Server for senseSpace

Streams point cloud data from ZED cameras (single or fusion mode) with efficient compression.
Supports both skeleton tracking and point cloud streaming simultaneously.
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
    import zstandard as zstd
    ZSTD_AVAILABLE = True
    _zstd_compressor = zstd.ZstdCompressor(level=3)
except ImportError:
    ZSTD_AVAILABLE = False
    print("[WARNING] zstandard not available - point clouds will not be compressed")


class PointCloudServer(SenseSpaceServer):
    """
    Extended SenseSpace server that also streams point cloud data.
    
    Inherits body tracking from SenseSpaceServer and adds point cloud streaming.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 12345, 
                 pointcloud_port: int = 12346, 
                 use_udp: bool = False,
                 downsample_factor: int = 4,
                 max_distance: float = 5000.0,
                 voxel_size: float = 20.0,
                 mode: str = "full"):
        """
        Initialize point cloud server.
        
        Args:
            host: Server host address
            port: Port for skeleton data (inherited from SenseSpaceServer)
            pointcloud_port: Port for point cloud data streaming
            use_udp: Use UDP for skeleton data (not used for point cloud)
            downsample_factor: Factor to downsample point cloud (1=no downsampling, 4=1/4 resolution)
            max_distance: Maximum distance in mm to include points (filters far points)
            voxel_size: Voxel size in mm for deduplication (20mm = 2cm cubes, 0 = disable)
            mode: Point cloud mode - "full" (entire scene) or "per-person" (segmented by tracked persons)
        """
        super().__init__(host=host, port=port, use_udp=use_udp)
        
        # Point cloud specific settings
        self.pointcloud_port = pointcloud_port
        self.downsample_factor = max(1, downsample_factor)
        self.max_distance = max_distance
        self.voxel_size = voxel_size
        self.pointcloud_mode = mode  # "full" or "per-person"
        
        # Validate mode
        if mode == "per-person":
            print(f"[INFO] Per-person mode requires fusion mode (multi-camera setup)")
            print(f"[INFO] Single camera will fall back to full mode")
        
        # Point cloud TCP server
        self.pc_server_socket = None
        self.pc_server_thread = None
        self.pc_clients = []
        self.pc_client_queues = {}
        self.pc_client_sender_threads = {}
        
        # Point cloud data
        self.point_cloud_mat = None
        self.pc_lock = threading.Lock()
        
        # Cached point clouds (retrieved during skeleton grab to avoid redundant transfers)
        self._cached_point_clouds = {}  # {serial: (points, colors)} for full mode
        self._cached_bodies = None  # sl.Bodies object for single camera per-person mode
        self._cached_pc_timestamp = 0.0
        
        # Debug image saving
        self._save_debug_image = False  # Flag to trigger debug image saving
        
        # Ensure tmp directory exists
        self.tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp')
        os.makedirs(self.tmp_dir, exist_ok=True)
        
        # Pre-allocated buffers for point cloud processing (will be initialized after cameras start)
        self._pc_buffer_points = None
        self._pc_buffer_colors = None
        self._max_buffer_size = 0
        
        print(f"[INFO] Point cloud server configured:")
        print(f"       Skeleton port: {port}")
        print(f"       Point cloud port: {pointcloud_port}")
        print(f"       Mode: {mode}")
        print(f"       Downsample factor: {downsample_factor}")
        print(f"       Max distance: {max_distance}mm")
        print(f"       Voxel size: {voxel_size}mm {'(deduplication enabled)' if voxel_size > 0 else '(disabled)'}")
    
    def _initialize_single_camera(self, device_info, enable_body_tracking: bool = True, enable_floor_detection: bool = True) -> bool:
        """
        Override parent's single camera initialization to enable segmentation for per-person mode.
        """
        import pyzed.sl as sl
        
        try:
            self.camera = sl.Camera()

            # Set initialization parameters (optimized for performance)
            init_params = sl.InitParameters()
            init_params.camera_resolution = sl.RESOLUTION.VGA  # 672x376 for better performance
            init_params.camera_fps = 60  # Higher FPS for smoother tracking
            init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Faster depth processing
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

            # Enable body tracking if requested (optimized for performance)
            if enable_body_tracking:
                body_params = sl.BodyTrackingParameters()
                body_params.enable_tracking = True
                body_params.enable_body_fitting = True
                body_params.body_format = sl.BODY_FORMAT.BODY_34
                body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE  # Accurate model for full-res masks
                
                # CRITICAL: Enable segmentation for per-person point cloud mode
                if self.pointcloud_mode == "per-person":
                    body_params.enable_segmentation = True
                    print("[INFO] Segmentation enabled for per-person point cloud mode")

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
    
    def _initialize_fusion_cameras(self, device_list, enable_body_tracking: bool = True, enable_floor_detection: bool = True) -> bool:
        """
        Override parent's fusion initialization to enable segmentation for per-person mode.
        """
        import pyzed.sl as sl
        
        # Call parent implementation but we need to modify body_tracking_parameters
        # So we'll duplicate the parent code with segmentation enabled
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

            # Common parameters (optimized for performance)
            init_params = sl.InitParameters()
            init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
            init_params.coordinate_units = sl.UNIT.MILLIMETER
            init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Faster depth processing
            init_params.camera_resolution = sl.RESOLUTION.VGA  # 672x376 for better performance
            init_params.camera_fps = 60  # Higher FPS for smoother tracking

            communication_parameters = sl.CommunicationParameters()
            communication_parameters.set_for_shared_memory()

            positional_tracking_parameters = sl.PositionalTrackingParameters()
            positional_tracking_parameters.set_as_static = True

            body_tracking_parameters = sl.BodyTrackingParameters()
            body_tracking_parameters.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE  # Accurate model for full-res masks
            body_tracking_parameters.body_format = sl.BODY_FORMAT.BODY_34
            body_tracking_parameters.enable_body_fitting = False
            body_tracking_parameters.enable_tracking = True
            
            # CRITICAL: Enable segmentation for per-person point cloud mode
            if self.pointcloud_mode == "per-person":
                body_tracking_parameters.enable_segmentation = True
                print("[INFO] Segmentation enabled for per-person point cloud mode")

            # Continue with rest of parent implementation...
            # (calling parent's implementation from here would override our parameters)
            # So we need the full implementation - let me continue:
            
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
            camera_poses = {}  # Store camera poses for point cloud transformation
            
            for conf in fusion_configurations:
                uuid = sl.CameraIdentifier()
                uuid.serial_number = conf.serial_number
                
                print(f"[INFO] Subscribing to {conf.serial_number}")
                status = self.fusion.subscribe(uuid, conf.communication_parameters, conf.pose)
                
                if status == sl.FUSION_ERROR_CODE.SUCCESS:
                    camera_identifiers.append(uuid)
                    # Store camera pose (4x4 transformation matrix) for point cloud transformation
                    pose_matrix = conf.pose.pose_data(sl.Transform())
                    camera_poses[conf.serial_number] = pose_matrix
                    print(f"[INFO] Subscribed to {conf.serial_number}")
                else:
                    print(f"[WARNING] Unable to subscribe to {conf.serial_number}: {status}")
                    
            # store identifiers so later lookups use the exact SDK object/type
            self._fusion_camera_identifiers = camera_identifiers
            self._fusion_camera_poses = camera_poses  # Store for point cloud transformation

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
    
    def _initialize_point_cloud_buffers(self):
        """
        Initialize worst-case buffers based on actual camera resolution.
        Called after cameras are initialized.
        """
        if self._pc_buffer_points is not None:
            return  # Already initialized
        
        # Detect resolution from cameras
        max_points_per_camera = 0
        num_cameras = 0
        
        if self.is_fusion_mode and hasattr(self, '_fusion_senders'):
            # Fusion mode: get resolution from first camera
            for cam in self._fusion_senders.values():
                try:
                    res = cam.get_camera_information().camera_configuration.resolution
                    max_points_per_camera = res.width * res.height
                    num_cameras = len(self._fusion_senders)
                    break
                except Exception:
                    continue
        elif self.camera is not None:
            # Single camera mode
            try:
                res = self.camera.get_camera_information().camera_configuration.resolution
                max_points_per_camera = res.width * res.height
                num_cameras = 1
            except Exception:
                pass
        
        # Calculate worst-case: all cameras × full resolution (before downsampling/dedup)
        if max_points_per_camera > 0:
            # Add 10% safety margin for edge cases
            worst_case = int(max_points_per_camera * num_cameras * 1.1)
            self._max_buffer_size = worst_case
            
            # Allocate buffers (one-time cost)
            self._pc_buffer_points = np.zeros((worst_case, 3), dtype=np.float32)
            self._pc_buffer_colors = np.zeros((worst_case, 3), dtype=np.uint8)
            
            mem_mb = (worst_case * 3 * 4 + worst_case * 3) / 1024 / 1024
            print(f"[INFO] Point cloud buffers allocated: {worst_case:,} points ({mem_mb:.1f} MB)")
        else:
            # Fallback if detection fails
            print("[WARNING] Could not detect camera resolution, using fallback buffer size")
            self._max_buffer_size = 1000000  # 1M points fallback
            self._pc_buffer_points = np.zeros((1000000, 3), dtype=np.float32)
            self._pc_buffer_colors = np.zeros((1000000, 3), dtype=np.uint8)
    
    def start_tcp_server(self):
        """Start both skeleton and point cloud TCP servers"""
        # Start skeleton server (parent class)
        super().start_tcp_server()
        
        # Start point cloud server
        self._start_pointcloud_server()
    
    def _start_pointcloud_server(self):
        """Start the point cloud TCP server"""
        def run_server():
            try:
                self.pc_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.pc_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.pc_server_socket.bind((self.host, self.pointcloud_port))
                self.pc_server_socket.listen(5)
                print(f"[INFO] Point cloud TCP server listening on {self.host}:{self.pointcloud_port}")
                if self.host == "0.0.0.0":
                    print(f"[INFO] Connect point cloud clients to: {self.local_ip}:{self.pointcloud_port}")
                
                while self.running:
                    try:
                        client_socket, addr = self.pc_server_socket.accept()
                        client_socket.settimeout(5.0)
                        client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        print(f"[INFO] Point cloud client connected from {addr}")
                        
                        # Add client to the list
                        self.pc_clients.append(client_socket)
                        
                        # Create queue and sender thread for this client
                        q = queue.Queue(maxsize=4)  # Smaller queue for large point cloud data
                        self.pc_client_queues[client_socket] = q
                        sender_thread = threading.Thread(
                            target=self._pc_client_sender_worker, 
                            args=(client_socket, q), 
                            daemon=True
                        )
                        self.pc_client_sender_threads[client_socket] = sender_thread
                        sender_thread.start()
                        
                        threading.Thread(
                            target=self._pc_client_handler, 
                            args=(client_socket, addr), 
                            daemon=True
                        ).start()
                    except socket.error:
                        if self.running:
                            print("[ERROR] Socket error in point cloud TCP server")
                        break
            except Exception as e:
                print(f"[ERROR] Failed to start point cloud TCP server: {e}")
        
        self.pc_server_thread = threading.Thread(target=run_server, daemon=True)
        self.pc_server_thread.start()
    
    def _pc_client_handler(self, conn, addr):
        """Handle individual point cloud client connection"""
        print(f"[PC CLIENT CONNECTED] {addr}")
        try:
            while self.running:
                try:
                    # Just keep connection alive, clients don't send data
                    data = conn.recv(1024)
                    if not data:
                        break
                except socket.timeout:
                    continue
                except Exception:
                    break
        except:
            pass
        finally:
            print(f"[PC CLIENT DISCONNECTED] {addr}")
            if conn in self.pc_clients:
                self.pc_clients.remove(conn)
            # Clean up sender queue and thread mapping
            try:
                if conn in self.pc_client_queues:
                    try:
                        self.pc_client_queues[conn].put_nowait(None)
                    except Exception:
                        pass
                    del self.pc_client_queues[conn]
            except Exception:
                pass
            try:
                if conn in self.pc_client_sender_threads:
                    del self.pc_client_sender_threads[conn]
            except Exception:
                pass
            try:
                conn.close()
            except:
                pass
    
    def _pc_client_sender_worker(self, conn, q: "queue.Queue"):
        """Worker thread that sends queued point cloud messages"""
        try:
            while self.running:
                try:
                    item = q.get(timeout=0.5)
                except Exception:
                    continue
                if item is None:
                    break
                
                # Item is pre-serialized bytes
                try:
                    conn.sendall(item)
                except Exception:
                    # On error, remove client and break
                    try:
                        if conn in self.pc_clients:
                            self.pc_clients.remove(conn)
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
            try:
                if conn in self.pc_client_queues:
                    del self.pc_client_queues[conn]
            except Exception:
                pass
            try:
                if conn in self.pc_client_sender_threads:
                    del self.pc_client_sender_threads[conn]
            except Exception:
                pass
    
    def _voxel_deduplicate(self, points: np.ndarray, colors: np.ndarray) -> tuple:
        """
        Deduplicate overlapping points from multiple cameras using voxel grid.
        Also performs averaging of positions and colors within each voxel.
        
        OPTIMIZED: Uses pre-allocated worst-case buffers (no reallocations).
        
        Args:
            points: Nx3 array of xyz positions (float32)
            colors: Nx3 array of rgb colors (uint8)
        
        Returns:
            (deduplicated_points, deduplicated_colors)
        """
        if self.voxel_size <= 0 or len(points) == 0:
            return points, colors
        
        # Quantize points to voxel grid
        voxel_indices = np.floor(points / self.voxel_size).astype(np.int32)
        
        # Build voxel dictionary (unavoidable - structure changes each frame)
        voxel_dict = {}
        
        for i in range(len(points)):
            voxel_key = tuple(voxel_indices[i])
            
            if voxel_key not in voxel_dict:
                voxel_dict[voxel_key] = {
                    'points': [points[i]],
                    'colors': [colors[i]]
                }
            else:
                voxel_dict[voxel_key]['points'].append(points[i])
                voxel_dict[voxel_key]['colors'].append(colors[i])
        
        # Use pre-allocated worst-case buffers (sliced to actual size)
        num_voxels = len(voxel_dict)
        
        # Safety check: ensure buffers are initialized
        if self._pc_buffer_points is None:
            # Fallback: allocate on-demand if called before initialization
            deduped_points = np.zeros((num_voxels, 3), dtype=np.float32)
            deduped_colors = np.zeros((num_voxels, 3), dtype=np.uint8)
        elif num_voxels > self._max_buffer_size:
            # Should never happen with worst-case allocation, but guard anyway
            print(f"[WARNING] Voxel count {num_voxels} exceeds buffer size {self._max_buffer_size}. Using slower allocation.")
            deduped_points = np.zeros((num_voxels, 3), dtype=np.float32)
            deduped_colors = np.zeros((num_voxels, 3), dtype=np.uint8)
        else:
            # Use preallocated buffers (just slice to current size)
            deduped_points = self._pc_buffer_points[:num_voxels]
            deduped_colors = self._pc_buffer_colors[:num_voxels]
        
        # Fill buffers with averaged data
        for idx, voxel_data in enumerate(voxel_dict.values()):
            # Average positions
            pts = np.array(voxel_data['points'], dtype=np.float32)
            deduped_points[idx] = pts.mean(axis=0)
            
            # Average colors
            cols = np.array(voxel_data['colors'], dtype=np.float32)  # Use float for averaging
            deduped_colors[idx] = cols.mean(axis=0).astype(np.uint8)
        
        # Return copies to avoid aliasing (caller may modify)
        return deduped_points.copy(), deduped_colors.copy()
    
    def _downsample_point_cloud(self, points: np.ndarray, colors: np.ndarray) -> tuple:
        """
        Downsample point cloud using simple stride-based approach.
        
        Args:
            points: Nx3 array of xyz positions
            colors: Nx3 array of rgb colors (0-255)
        
        Returns:
            (downsampled_points, downsampled_colors)
        """
        if self.downsample_factor <= 1 or len(points) == 0:
            return points, colors
        
        # Simple stride-based downsampling (fast)
        indices = np.arange(0, len(points), self.downsample_factor)
        return points[indices], colors[indices]
    
    def _filter_point_cloud(self, points: np.ndarray, colors: np.ndarray) -> tuple:
        """
        Filter point cloud by removing invalid and distant points.
        
        Args:
            points: Nx3 array of xyz positions
            colors: Nx3 array of rgb colors
        
        Returns:
            (filtered_points, filtered_colors)
        """
        if len(points) == 0:
            return points, colors
        
        # Remove NaN and infinite values
        valid_mask = np.isfinite(points).all(axis=1)
        
        # Remove points beyond max distance
        if self.max_distance > 0:
            distances = np.linalg.norm(points, axis=1)
            distance_mask = distances < self.max_distance
            valid_mask = valid_mask & distance_mask
        
        return points[valid_mask], colors[valid_mask]
    
    def _serialize_point_cloud(self, points: np.ndarray, colors: np.ndarray, timestamp: float) -> bytes:
        """
        Serialize point cloud data to binary format with compression.
        
        Format:
            - Magic: 2 bytes (0x9f 0xd2 for point cloud with zstd)
            - Timestamp: 8 bytes (double)
            - Point count: 4 bytes (uint32)
            - Compressed payload:
                - Points: N * 12 bytes (3 floats: x, y, z)
                - Colors: N * 3 bytes (3 uint8: r, g, b)
        
        Args:
            points: Nx3 float32 array (x, y, z in mm)
            colors: Nx3 uint8 array (r, g, b 0-255)
            timestamp: Frame timestamp
        
        Returns:
            Serialized and compressed bytes
        """
        MAGIC_PC_ZSTD = b'\x9f\xd2'
        
        num_points = len(points)
        
        # Convert to binary (ensure correct dtypes)
        points_bin = points.astype(np.float32).tobytes()  # 12 bytes per point
        colors_bin = colors.astype(np.uint8).tobytes()    # 3 bytes per point
        
        # Combine payload
        payload = points_bin + colors_bin
        
        # Compress if available
        if ZSTD_AVAILABLE:
            payload = _zstd_compressor.compress(payload)
            header = MAGIC_PC_ZSTD + struct.pack('>dI', timestamp, num_points)
        else:
            # Uncompressed fallback (use different magic)
            header = b'\x9f\xd3' + struct.pack('>dI', timestamp, num_points)
        
        payload_length = len(payload)
        
        # Final format: [magic:2][timestamp:8][count:4][length:4][payload]
        return header + struct.pack('>I', payload_length) + payload
    
    def _serialize_per_person_point_clouds(self, per_person_list: list, timestamp: float) -> bytes:
        """
        Serialize per-person point clouds into binary format.
        
        Format:
            - Magic: 2 bytes (0x9f 0xd4 for per-person compressed)
            - Timestamp: 8 bytes (double)
            - Person count: 4 bytes (uint32)
            - Payload length: 4 bytes (uint32)
            - Compressed payload:
                For each person:
                    - Person ID: 4 bytes (int32)
                    - Point count: 4 bytes (uint32)
                    - Points: N * 12 bytes (3 floats: x, y, z)
                    - Colors: N * 3 bytes (3 uint8: r, g, b)
        
        Args:
            per_person_list: List of dicts with "person_id", "points", "colors"
            timestamp: Frame timestamp
        
        Returns:
            Serialized and compressed bytes
        """
        MAGIC_PC_PER_PERSON_ZSTD = b'\x9f\xd4'
        
        num_persons = len(per_person_list)
        
        # Build payload for all persons
        payload_parts = []
        for person_data in per_person_list:
            person_id = person_data["person_id"]
            points = person_data["points"]
            colors = person_data["colors"]
            
            num_points = len(points)
            
            # Serialize: [person_id:4][count:4][points:N*12][colors:N*3]
            person_header = struct.pack('>iI', person_id, num_points)
            points_bin = points.astype(np.float32).tobytes()
            colors_bin = colors.astype(np.uint8).tobytes()
            
            payload_parts.append(person_header + points_bin + colors_bin)
        
        # Combine all persons
        payload = b''.join(payload_parts)
        
        # Compress if available
        if ZSTD_AVAILABLE:
            payload = _zstd_compressor.compress(payload)
            header = MAGIC_PC_PER_PERSON_ZSTD + struct.pack('>dI', timestamp, num_persons)
        else:
            # Uncompressed fallback (use different magic)
            header = b'\x9f\xd5' + struct.pack('>dI', timestamp, num_persons)
        
        payload_length = len(payload)
        
        # Final format: [magic:2][timestamp:8][person_count:4][length:4][payload]
        return header + struct.pack('>I', payload_length) + payload
    
    def _retrieve_point_cloud_single_camera(self) -> Optional[tuple]:
        """
        Retrieve point cloud from single camera.
        
        For per-person mode: Uses cached Bodies from skeleton tracking loop
        For full mode: Returns entire scene point cloud
        
        Returns:
            For "full" mode: (points, colors, timestamp) or None
            For "per-person" mode: (per_person_list, timestamp) or None
        """
        if not self.camera:
            return None
        
        try:
            # Allocate point cloud mat if needed
            if self.point_cloud_mat is None:
                resolution = self.camera.get_camera_information().camera_configuration.resolution
                self.point_cloud_mat = sl.Mat(resolution.width, resolution.height, sl.MAT_TYPE.F32_C4)
            
            # Retrieve XYZRGBA point cloud
            self.camera.retrieve_measure(self.point_cloud_mat, sl.MEASURE.XYZRGBA)
            
            # Get timestamp
            timestamp = time.time()
            
            # Check if per-person mode is requested
            if self.pointcloud_mode == "per-person":
                # Per-person mode: ONLY show person point clouds, nothing else
                if not hasattr(self, '_cached_bodies'):
                    return None
                    
                with self.pc_lock:
                    cached_bodies = self._cached_bodies
                
                if cached_bodies is None:
                    return None
                
                # Extract per-person point clouds using cached Bodies
                per_person_list = self._extract_per_person_point_clouds(cached_bodies, self.point_cloud_mat)
                if per_person_list:
                    return (per_person_list, timestamp)
                else:
                    return None
            
            # Full mode: extract entire scene
            print(f"[DEBUG] Using FULL MODE - extracting entire scene")
            # Convert to numpy
            pc_data = self.point_cloud_mat.get_data()  # Shape: (H, W, 4) - XYZRGBA
            
            # Reshape to point list
            h, w = pc_data.shape[:2]
            pc_flat = pc_data.reshape(-1, 4)
            
            # Split into points and colors
            points = pc_flat[:, :3]  # XYZ
            colors_raw = pc_flat[:, 3]  # RGBA packed as float
            
            # Unpack RGBA (stored as uint32 in float)
            colors_uint = colors_raw.view(np.uint32)
            r = (colors_uint & 0xFF)
            g = ((colors_uint >> 8) & 0xFF)
            b = ((colors_uint >> 16) & 0xFF)
            colors = np.stack([r, g, b], axis=1).astype(np.uint8)
            
            return points, colors, timestamp
            
        except Exception as e:
            print(f"[WARNING] Failed to retrieve point cloud: {e}")
            return None
    
    def _retrieve_point_cloud_fusion(self) -> Optional[tuple]:
        """
        Retrieve and merge point clouds from all fusion cameras.
        
        OPTIMIZED: Uses cached point clouds retrieved during skeleton grab loop
        to avoid redundant GPU->CPU transfers.
        
        Returns:
            For "full" mode: (points, colors, timestamp) or None
            For "per-person" mode: (per_person_list, timestamp) or None
                where per_person_list = [{"person_id": int, "points": np.ndarray, "colors": np.ndarray}, ...]
        """
        if not hasattr(self, '_cached_point_clouds') or not self._cached_point_clouds:
            return None
        
        try:
            with self.pc_lock:
                timestamp = self._cached_pc_timestamp
                cached_data = dict(self._cached_point_clouds)  # Copy to release lock quickly
            
            if self.pointcloud_mode == "per-person":
                # Per-person mode: extract segmented point clouds for each tracked person
                all_person_clouds = {}  # {person_id: {"points": [], "colors": []}}
                
                for serial, cache_entry in cached_data.items():
                    if cache_entry[0] != "per-person":
                        continue
                    
                    _, pc_mat, bodies = cache_entry
                    
                    # Extract per-person point clouds from this camera
                    person_clouds = self._extract_per_person_point_clouds(bodies, pc_mat)
                    
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
            
            else:
                # Full mode: merge all point clouds into one
                all_points = []
                all_colors = []
                
                for serial, cache_entry in cached_data.items():
                    if isinstance(cache_entry, tuple) and len(cache_entry) == 2:
                        points, colors = cache_entry
                        
                        # Transform points to fusion coordinate space
                        if hasattr(self, '_fusion_camera_poses') and serial in self._fusion_camera_poses:
                            camera_pose = self._fusion_camera_poses[serial]
                            points = self._transform_points_to_fusion(points, camera_pose)
                        
                        all_points.append(points)
                        all_colors.append(colors)
                
                if not all_points:
                    return None
                
                # Merge all point clouds
                merged_points = np.vstack(all_points)
                merged_colors = np.vstack(all_colors)
                
                return merged_points, merged_colors, timestamp
            
        except Exception as e:
            print(f"[WARNING] Failed to merge cached point clouds: {e}")
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
    
    def _extract_point_cloud_from_mat(self, pc_mat: sl.Mat) -> tuple:
        """
        Extract points and colors from ZED Mat.
        
        Args:
            pc_mat: sl.Mat with XYZRGBA data
        
        Returns:
            (points, colors) as numpy arrays
        """
        # Convert to numpy
        pc_data = pc_mat.get_data()  # Shape: (H, W, 4) - XYZRGBA
        pc_flat = pc_data.reshape(-1, 4)
        
        # Split into points and colors
        points = pc_flat[:, :3]  # XYZ
        colors_raw = pc_flat[:, 3]  # RGBA packed as float
        
        # Unpack RGBA (stored as uint32 in float)
        colors_uint = colors_raw.view(np.uint32)
        r = (colors_uint & 0xFF)
        g = ((colors_uint >> 8) & 0xFF)
        b = ((colors_uint >> 16) & 0xFF)
        colors = np.stack([r, g, b], axis=1).astype(np.uint8)
        
        return points, colors
    
    def _extract_per_person_point_clouds(self, bodies: sl.Bodies, pc_mat: sl.Mat) -> list:
        """
        Extract point clouds for each tracked person using ZED SDK segmentation masks.
        
        Args:
            bodies: sl.Bodies object with tracked persons
            pc_mat: sl.Mat with XYZRGBA point cloud data
        
        Returns:
            List of dicts: [{"person_id": int, "points": np.ndarray, "colors": np.ndarray}, ...]
        """
        per_person_clouds = []
        
        # Get full point cloud data
        pc_data = pc_mat.get_data()  # Shape: (H, W, 4) - XYZRGBA
        height, width = pc_data.shape[:2]
        
        for person in bodies.body_list:
            try:
                person_id = person.id
                
                # Check tracking state
                tracking_state = str(person.tracking_state)
                
                # Get segmentation mask
                mask_mat = getattr(person, 'mask', None)
                if mask_mat is None:
                    continue
                
                if not mask_mat.is_init():
                    continue
                
                # Get 2D bounding box - mask is relative to this!
                bbox_2d = person.bounding_box_2d
                if len(bbox_2d) < 4:
                    continue
                
                top_left = bbox_2d[0]
                bottom_right = bbox_2d[2]
                
                bbox_x_min = int(top_left[0])
                bbox_y_min = int(top_left[1])
                bbox_x_max = int(bottom_right[0])
                bbox_y_max = int(bottom_right[1])
                
                bbox_width = bbox_x_max - bbox_x_min
                bbox_height = bbox_y_max - bbox_y_min
                
                # Get mask data - it's at lower resolution and RELATIVE to bounding box
                mask_data = mask_mat.get_data()
                mask_h, mask_w = mask_data.shape[:2]
                
                # Find person pixels in the mask (255 = person)
                mask_y, mask_x = np.where(mask_data == 255)
                
                if len(mask_y) == 0:
                    continue
                
                # Scale mask coordinates to bounding box size, then add bbox offset
                scale_y = bbox_height / mask_h
                scale_x = bbox_width / mask_w
                
                img_y = bbox_y_min + (mask_y * scale_y).astype(np.int32)
                img_x = bbox_x_min + (mask_x * scale_x).astype(np.int32)
                
                # Clamp to valid image range
                img_y = np.clip(img_y, 0, height - 1)
                img_x = np.clip(img_x, 0, width - 1)
                
                # Extract person's points using scaled and offset coordinates
                person_pc = pc_data[img_y, img_x]  # Shape: (N, 4)
                
                if len(person_pc) == 0:
                    continue
                
                # Split into points and colors
                points = person_pc[:, :3]
                colors_raw = person_pc[:, 3]
                
                # Unpack RGBA
                colors_uint = colors_raw.view(np.uint32)
                r = (colors_uint & 0xFF)
                g = ((colors_uint >> 8) & 0xFF)
                b = ((colors_uint >> 16) & 0xFF)
                colors = np.stack([r, g, b], axis=1).astype(np.uint8)
                
                # Save debug images if requested
                if self._save_debug_image:
                    self._save_mask_overlay_images(person, pc_data, mask_data, bbox_2d, person_id)
                
                per_person_clouds.append({
                    "person_id": person_id,
                    "points": points,
                    "colors": colors
                })
                
            except Exception as e:
                print(f"[WARNING] Failed to extract point cloud for person {person.id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return per_person_clouds
    
    def _save_mask_overlay_images(self, person, pc_data, mask_data, bbox_2d, person_id):
        """
        Save debug images: camera image and camera image with mask overlay.
        
        Args:
            person: sl.BodyData object
            pc_data: Point cloud data (H, W, 4) - XYZRGBA
            mask_data: Mask data (mask_h, mask_w) - relative to bounding box
            bbox_2d: 2D bounding box
            person_id: Person ID
        """
        try:
            from PyQt5.QtGui import QImage
            
            # We need the actual camera image, not the point cloud colors
            # Get left camera image instead
            if not self.camera:
                print("[WARNING] No camera available for debug images")
                return
            
            # Retrieve left camera image
            left_image = sl.Mat()
            self.camera.retrieve_image(left_image, sl.VIEW.LEFT)
            left_data = left_image.get_data()  # Shape: (H, W, 4) - BGRA
            
            height, width = left_data.shape[:2]
            
            # Convert BGRA to RGB
            camera_img = left_data[:, :, [2, 1, 0]].copy()  # BGR -> RGB
            
            # Save full camera image using Qt
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            qimg = QImage(camera_img.data, width, height, width * 3, QImage.Format_RGB888)
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
            bbox_width = bbox_x_max - bbox_x_min
            bbox_height = bbox_y_max - bbox_y_min
            
            mask_h, mask_w = mask_data.shape[:2]
            
            print(f"[DEBUG] BBox: ({bbox_x_min}, {bbox_y_min}) to ({bbox_x_max}, {bbox_y_max}), size: {bbox_width}x{bbox_height}")
            print(f"[DEBUG] Mask shape: {mask_h}x{mask_w}, Image shape: {height}x{width}")
            
            # Create full-size mask
            full_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Find person pixels in mask
            mask_y, mask_x = np.where(mask_data == 255)
            
            if len(mask_y) > 0:
                # Scale mask coordinates to bbox size and add offset
                scale_y = bbox_height / mask_h
                scale_x = bbox_width / mask_w
                
                img_y = bbox_y_min + (mask_y * scale_y).astype(np.int32)
                img_x = bbox_x_min + (mask_x * scale_x).astype(np.int32)
                
                # Clamp
                img_y = np.clip(img_y, 0, height - 1)
                img_x = np.clip(img_x, 0, width - 1)
                
                # Place in full mask
                full_mask[img_y, img_x] = 255
            
            # Save raw mask
            qimg_mask = QImage(full_mask.data, width, height, width, QImage.Format_Grayscale8)
            mask_path = os.path.join(self.tmp_dir, f"debug_mask_raw_{person_id}_{timestamp_str}.png")
            qimg_mask.save(mask_path)
            print(f"[INFO] Saved raw mask: {mask_path}")
            
            # Create overlay image (camera + mask)
            overlay_img = camera_img.copy()
            
            # Create semi-transparent green overlay where mask == 255
            alpha = 0.5
            overlay_img[full_mask == 255, 1] = (overlay_img[full_mask == 255, 1] * (1 - alpha) + 255 * alpha).astype(np.uint8)
            
            # Save overlay image using Qt
            qimg_overlay = QImage(overlay_img.data, width, height, width * 3, QImage.Format_RGB888)
            overlay_path = os.path.join(self.tmp_dir, f"debug_overlay_{person_id}_{timestamp_str}.png")
            qimg_overlay.save(overlay_path)
            print(f"[INFO] Saved overlay image: {overlay_path}")
            
            # Reset flag
            self._save_debug_image = False
            
        except Exception as e:
            print(f"[WARNING] Failed to save debug images: {e}")
            import traceback
            traceback.print_exc()
    
    def _broadcast_point_cloud(self, data, timestamp: float):
        """
        Broadcast point cloud to all connected clients and optionally to visualization.
        
        Args:
            data: For full mode: (points, colors) tuple
                  For per-person mode: list of person dicts
            timestamp: Frame timestamp
        """
        if not self.pc_clients and not hasattr(self, '_viz_point_cloud_callback'):
            return
        
        # Detect mode based on data type (more robust than checking config)
        # Per-person mode: list of dicts with person data
        # Full mode: tuple of (points, colors)
        is_per_person = isinstance(data, list)
        
        if is_per_person:
            # Per-person mode
            # Per-person mode
            per_person_list = data
            
            if not per_person_list:
                return
            
            # Process each person's point cloud
            processed_persons = []
            for person_data in per_person_list:
                points = person_data["points"]
                colors = person_data["colors"]
                person_id = person_data["person_id"]
                
                # Filter invalid points
                points, colors = self._filter_point_cloud(points, colors)
                if len(points) == 0:
                    continue
                
                # Downsample if needed (skip voxel dedup for per-person - already segmented)
                points, colors = self._downsample_point_cloud(points, colors)
                
                processed_persons.append({
                    "person_id": person_id,
                    "points": points,
                    "colors": colors
                })
            
            if not processed_persons:
                return
            
            # Update visualization (show all persons merged for viz)
            if hasattr(self, '_viz_point_cloud_callback') and self._viz_point_cloud_callback:
                try:
                    # Merge all persons for visualization
                    all_points = np.vstack([p["points"] for p in processed_persons])
                    all_colors = np.vstack([p["colors"] for p in processed_persons])
                    self._viz_point_cloud_callback(all_points, all_colors)
                except Exception as e:
                    print(f"[WARNING] Visualization callback failed: {e}")
            
            # Serialize and send to clients
            if self.pc_clients:
                try:
                    serialized = self._serialize_per_person_point_clouds(processed_persons, timestamp)
                except Exception as e:
                    print(f"[ERROR] Per-person point cloud serialization failed: {e}")
                    return
                
                for client in list(self.pc_clients):
                    q = self.pc_client_queues.get(client)
                    if q is None:
                        continue
                    try:
                        q.put_nowait(serialized)
                    except queue.Full:
                        pass
        
        else:
            # Full mode (original behavior)
            points, colors = data
            
            # Filter invalid points
            points, colors = self._filter_point_cloud(points, colors)
            if len(points) == 0:
                return
            
            # Deduplicate overlapping points from multiple cameras (voxel grid)
            points, colors = self._voxel_deduplicate(points, colors)
            if len(points) == 0:
                return
            
            # Downsample if needed (after deduplication)
            points, colors = self._downsample_point_cloud(points, colors)
            
            # Update visualization if callback is set (optional)
            if hasattr(self, '_viz_point_cloud_callback') and self._viz_point_cloud_callback:
                try:
                    self._viz_point_cloud_callback(points, colors)
                except Exception as e:
                    print(f"[WARNING] Visualization callback failed: {e}")
            
            # Serialize once for all clients (skip if no clients)
            if not self.pc_clients:
                return
            
            try:
                serialized = self._serialize_point_cloud(points, colors, timestamp)
            except Exception as e:
                print(f"[ERROR] Point cloud serialization failed: {e}")
                return
            
            # Send to all clients via queues
            for client in list(self.pc_clients):
                q = self.pc_client_queues.get(client)
                if q is None:
                    continue
                try:
                    q.put_nowait(serialized)
                except queue.Full:
                    # Drop frame for slow clients
                    pass
    
    def run_body_tracking_loop(self):
        """
        Override to add point cloud streaming.
        
        OPTIMIZED: Override parent's fusion loop to retrieve point clouds
        in the same pass as skeletons (single GPU->CPU transfer).
        """
        if self.is_fusion_mode:
            # Use optimized fusion loop
            self._run_optimized_fusion_loop()
        else:
            # Single camera: override to cache Bodies for per-person mode
            pc_thread = threading.Thread(target=self._run_point_cloud_loop, daemon=True)
            pc_thread.start()
            self._run_single_camera_loop()  # Use our override instead of parent's
    
    def _run_single_camera_loop(self):
        """
        Override parent's single camera loop to cache Bodies object for per-person mode.
        """
        import pyzed.sl as sl
        
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

                # Cache Bodies object for per-person point cloud extraction
                if self.pointcloud_mode == "per-person":
                    with self.pc_lock:
                        self._cached_bodies = bodies

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
    
    def _run_point_cloud_loop(self):
        """Point cloud streaming loop"""
        MAX_FPS = 30  # Point cloud streaming at 30 FPS (easily achievable with ~16-29ms processing)
        FRAME_INTERVAL = 1.0 / MAX_FPS
        last_time = time.time()
        
        print("[INFO] Starting point cloud streaming loop")
        
        # Initialize worst-case buffers based on camera resolution
        self._initialize_point_cloud_buffers()
        
        while self.running:
            now = time.time()
            if now - last_time < FRAME_INTERVAL:
                time.sleep(FRAME_INTERVAL - (now - last_time))
            last_time = time.time()
            
            # Skip if no clients AND no visualization callback
            if not self.pc_clients and not hasattr(self, '_viz_point_cloud_callback'):
                time.sleep(0.1)
                continue
            
            # Retrieve point cloud based on mode
            if self.is_fusion_mode:
                result = self._retrieve_point_cloud_fusion()
            else:
                # Single camera supports both full and per-person modes
                result = self._retrieve_point_cloud_single_camera()
            
            if result is None:
                continue
            
            # Handle result based on actual mode used
            # Check if result is per-person format (list of dicts) or full format (tuple)
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], list):
                # Per-person mode result: (per_person_list, timestamp)
                per_person_list, timestamp = result
                self._broadcast_point_cloud(per_person_list, timestamp)
            else:
                # Full mode result: (points, colors, timestamp)
                points, colors, timestamp = result
                self._broadcast_point_cloud((points, colors), timestamp)
    
    def stop_tcp_server(self):
        """Stop both skeleton and point cloud servers"""
        # Stop skeleton server (parent)
        super().stop_tcp_server()
        
        # Stop point cloud server
        self.running = False
        
        for client in list(self.pc_clients):
            try:
                client.close()
            except:
                pass
        self.pc_clients.clear()
        
        if self.pc_server_socket:
            try:
                self.pc_server_socket.close()
            except:
                pass
            self.pc_server_socket = None
        
        print("[SERVER] Point cloud server stopped")
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_tcp_server()
        super().cleanup()
    
    def _run_optimized_fusion_loop(self):
        """
        OPTIMIZED fusion loop that retrieves skeletons AND point clouds
        in a single pass to avoid redundant GPU->CPU transfers.
        
        This replaces the parent's _run_fusion_loop() for better performance.
        """
        import pyzed.sl as sl
        
        bodies = sl.Bodies()
        MAX_FPS = 60  # Match parent class
        FRAME_INTERVAL = 1.0 / MAX_FPS
        last_time = time.time()
        
        print("[INFO] Starting OPTIMIZED fusion loop (skeletons + point clouds)")
        
        # Initialize worst-case buffers based on camera resolution
        self._initialize_point_cloud_buffers()
        
        # Allocate point cloud mats for each camera
        if not hasattr(self, '_pc_mats'):
            self._pc_mats = {}
        
        for serial, cam in self._fusion_senders.items():
            if serial not in self._pc_mats:
                resolution = cam.get_camera_information().camera_configuration.resolution
                self._pc_mats[serial] = sl.Mat(resolution.width, resolution.height, sl.MAT_TYPE.F32_C4)
        
        while self.running:
            now = time.time()
            if now - last_time < FRAME_INTERVAL:
                time.sleep(FRAME_INTERVAL - (now - last_time))
            last_time = time.time()
            
            # Grab frames from local senders sequentially (USB is serialized anyway)
            try:
                if hasattr(self, '_fusion_senders'):
                    cached_pcs = {}
                    
                    for serial, zed in self._fusion_senders.items():
                        try:
                            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                                # Retrieve bodies (for fusion processing)
                                try:
                                    zed.retrieve_bodies(bodies)
                                except Exception:
                                    pass
                                
                                # OPTIMIZATION: Retrieve point cloud in SAME pass
                                # (both skeleton and PC are from same grab, no extra GPU->CPU transfer)
                                if self.pc_clients or hasattr(self, '_viz_point_cloud_callback'):  # For clients or visualization
                                    try:
                                        zed.retrieve_measure(self._pc_mats[serial], sl.MEASURE.XYZRGBA)
                                        
                                        if self.pointcloud_mode == "per-person":
                                            # Per-person mode: cache bodies AND point cloud mat for later extraction
                                            cached_pcs[serial] = ("per-person", self._pc_mats[serial], bodies)
                                        else:
                                            # Full mode: extract point cloud immediately
                                            points, colors = self._extract_point_cloud_from_mat(self._pc_mats[serial])
                                            cached_pcs[serial] = (points, colors)
                                    except Exception as e:
                                        print(f"[WARNING] Failed to retrieve PC from {serial}: {e}")
                        except Exception:
                            pass
                    
                    # Cache point clouds for broadcasting thread
                    if cached_pcs:
                        with self.pc_lock:
                            self._cached_point_clouds = cached_pcs
                            self._cached_pc_timestamp = time.time()
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
                
                # Broadcast point cloud (uses cached data from grab loop above)
                if self.pc_clients or hasattr(self, '_viz_point_cloud_callback'):
                    result = self._retrieve_point_cloud_fusion()
                    if result:
                        if self.pointcloud_mode == "per-person":
                            # Per-person mode: result is (per_person_list, timestamp)
                            per_person_list, timestamp = result
                            self._broadcast_point_cloud(per_person_list, timestamp)
                        else:
                            # Full mode: result is (points, colors, timestamp)
                            points, colors, timestamp = result
                            self._broadcast_point_cloud((points, colors), timestamp)
            
            else:
                # Fusion failed - send empty frame
                non_critical_errors = []
                try:
                    if hasattr(sl.FUSION_ERROR_CODE, 'CAMERA_FPS_TOO_LOW'):
                        non_critical_errors.append(sl.FUSION_ERROR_CODE.CAMERA_FPS_TOO_LOW)
                    if hasattr(sl.FUSION_ERROR_CODE, 'NO_NEW_DATA'):
                        non_critical_errors.append(sl.FUSION_ERROR_CODE.NO_NEW_DATA)
                    if hasattr(sl.FUSION_ERROR_CODE, 'TIMEOUT'):
                        non_critical_errors.append(sl.FUSION_ERROR_CODE.TIMEOUT)
                except Exception:
                    pass
                
                # Import Frame for empty frame creation
                from senseSpaceLib.senseSpace.protocol import Frame
                empty_frame = Frame(
                    timestamp=time.time(),
                    people=[],
                    body_model="BODY_34",
                    floor_height=self.detected_floor_height,
                    cameras=self.get_camera_poses()
                )
                self.broadcast_frame(empty_frame)


def main():
    """Main entry point for point cloud server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SenseSpace Point Cloud Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host address")
    parser.add_argument("--port", type=int, default=12345, help="Skeleton data port")
    parser.add_argument("--pc-port", type=int, default=12346, help="Point cloud data port")
    parser.add_argument("--mode", type=str, choices=["full", "per-person"], default="full", 
                       help="Point cloud mode: 'full' (entire scene) or 'per-person' (segmented by tracked persons)")
    parser.add_argument("--downsample", type=int, default=4, help="Point cloud downsample factor (1=full resolution)")
    parser.add_argument("--max-distance", type=float, default=5000.0, help="Maximum point distance in mm")
    parser.add_argument("--voxel-size", type=float, default=20.0, help="Voxel size in mm for deduplication (0=disable)")
    parser.add_argument("--serials", type=str, nargs="+", help="Camera serial numbers (optional)")
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
            print("[WARNING] Continuing without visualization")
            args.viz = False
    
    # Create server
    server = PointCloudServer(
        host=args.host,
        port=args.port,
        pointcloud_port=args.pc_port,
        downsample_factor=args.downsample,
        max_distance=args.max_distance,
        voxel_size=args.voxel_size,
        mode=args.mode
    )
    
    try:
        # Initialize cameras
        print("[INFO] Initializing cameras...")
        if not server.initialize_cameras(serial_numbers=args.serials):
            print("[ERROR] Failed to initialize cameras")
            return 1
        
        # Connect visualization if enabled
        if args.viz and main_window:
            # Pass server reference to window for debug image saving
            main_window.server = server
            if hasattr(main_window, 'glWidget'):
                main_window.glWidget.server = server
            
            # Create callback wrapper to handle (people_data, floor_height) signature
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
                    # Window deleted, ignore
                    pass
            
            server.set_update_callback(qt_update_callback)
            
            # Set point cloud visualization callback (optional)
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
        
        # Run tracking loop (in thread if using Qt)
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
