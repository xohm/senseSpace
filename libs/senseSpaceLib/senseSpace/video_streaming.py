#!/usr/bin/env python3
"""
GStreamer-based video streaming for RGB and depth images.

Supports:
- H.265 encoding for RGB (hardware accelerated)
- H.265 lossless encoding for depth (hardware accelerated)
- RTP streaming over network
- Multi-camera support (multiple images in single stream)
- Cross-platform (Linux, Windows, Mac)
"""

# MUST set GST_DEBUG before importing GStreamer
import os
if os.getenv('GST_DEBUG') is None:
    # Set debug level: 2 for warnings and errors only (less verbose)
    os.environ['GST_DEBUG'] = '2'
    # Uncomment for verbose debugging: os.environ['GST_DEBUG'] = 'udpsrc:6,rtph265depay:5'

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import numpy as np
import threading
import time
import platform
import socket
import struct
from typing import Optional, Callable, Tuple, List
import logging

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

# Initialize GStreamer
Gst.init(None)

logger = logging.getLogger(__name__)

if not ZSTD_AVAILABLE:
    logger.warning("zstandard not available - depth streaming will use fallback compression")


class GStreamerPlatform:
    """Platform-specific GStreamer element selection for GPU encoding/decoding"""
    
    # Cache for detected encoders/decoders
    _encoder_cache = {}
    _decoder_cache = None
    
    @staticmethod
    def _check_element_available(element_name):
        """Check if a GStreamer element is available"""
        try:
            factory = Gst.ElementFactory.find(element_name)
            return factory is not None
        except:
            return False
    
    @staticmethod
    def get_encoder(encoder_type='rgb'):
        """
        Get platform-specific H.265 encoder element with fallback.
        Tries hardware encoders first, falls back to software.
        
        Args:
            encoder_type: 'rgb' for standard H.265, 'depth' for lossless H.265
        
        Returns:
            tuple: (encoder_name, properties_dict)
        """
        # Check cache
        cache_key = encoder_type
        if cache_key in GStreamerPlatform._encoder_cache:
            return GStreamerPlatform._encoder_cache[cache_key]
        
        system = platform.system()
        
        # Define encoder preferences per platform
        # Format: [(element_name, properties_dict), ...]
        encoder_options = []
        
        if system == 'Linux':
            # NVIDIA NVENC (best for NVIDIA GPUs)
            encoder_options.append(('nvh265enc', {'preset': 'low-latency-hq', 'bitrate': 4000}))
            # Intel/AMD VAAPI
            encoder_options.append(('vaapih265enc', {'bitrate': 4000, 'rate-control': 1}))
            # Software fallback
            encoder_options.append(('x265enc', {'speed-preset': 'fast', 'bitrate': 4000}))
        
        elif system == 'Windows':
            # NVIDIA NVENC
            encoder_options.append(('nvh265enc', {'preset': 'low-latency-hq', 'bitrate': 4000}))
            # Microsoft Media Foundation (Intel Quick Sync, AMD VCE)
            encoder_options.append(('mfh265enc', {'bitrate': 4000, 'rate-control': 1}))
            # Software fallback
            encoder_options.append(('x265enc', {'speed-preset': 'fast', 'bitrate': 4000}))
        
        elif system == 'Darwin':  # macOS
            # Apple VideoToolbox (M1/M2/Intel with T2)
            encoder_options.append(('vtenc_h265', {'bitrate': 4000, 'allow-frame-reordering': False}))
            # Software fallback
            encoder_options.append(('x265enc', {'speed-preset': 'fast', 'bitrate': 4000}))
        
        else:
            # Unknown platform - software only
            encoder_options.append(('x265enc', {'speed-preset': 'fast', 'bitrate': 4000}))
        
        # Try each encoder in order
        selected_encoder = None
        selected_props = {}
        
        for encoder_name, base_props in encoder_options:
            if GStreamerPlatform._check_element_available(encoder_name):
                selected_encoder = encoder_name
                selected_props = base_props.copy()
                
                # Log what we found
                hw_type = "hardware" if encoder_name not in ['x265enc', 'avenc_h265'] else "software"
                print(f"[INFO] Using {hw_type} H.265 encoder: {encoder_name}")
                logger.info(f"Selected {hw_type} encoder: {encoder_name}")
                break
        
        if not selected_encoder:
            # Ultimate fallback - should always be available
            selected_encoder = 'x265enc'
            selected_props = {'speed-preset': 'fast', 'bitrate': 4000}
            print(f"[WARNING] No hardware encoder found, using software: {selected_encoder}")
            logger.warning(f"Falling back to software encoder: {selected_encoder}")
        
        # Cache result
        result = (selected_encoder, selected_props)
        GStreamerPlatform._encoder_cache[cache_key] = result
        return result
    
    @staticmethod
    def get_decoder():
        """
        Get platform-specific H.265 decoder element with fallback.
        Tries hardware decoders first, falls back to software.
        """
        # Check cache
        if GStreamerPlatform._decoder_cache:
            return GStreamerPlatform._decoder_cache
        
        system = platform.system()
        
        # Define decoder preferences per platform
        decoder_options = []
        
        if system == 'Linux':
            # NVIDIA NVDEC
            decoder_options.append('nvh265dec')
            # Intel/AMD VAAPI
            decoder_options.append('vaapih265dec')
            # Software fallback (libav)
            decoder_options.append('avdec_h265')
        
        elif system == 'Windows':
            # NVIDIA NVDEC
            decoder_options.append('nvh265dec')
            # Microsoft Media Foundation
            decoder_options.append('mfh265dec')
            # Software fallback
            decoder_options.append('avdec_h265')
        
        elif system == 'Darwin':  # macOS
            # Apple VideoToolbox
            decoder_options.append('vtdec_h265')
            # Software fallback
            decoder_options.append('avdec_h265')
        
        else:
            # Unknown platform
            decoder_options.append('avdec_h265')
        
        # Try each decoder in order
        selected_decoder = None
        
        for decoder_name in decoder_options:
            if GStreamerPlatform._check_element_available(decoder_name):
                selected_decoder = decoder_name
                
                hw_type = "hardware" if decoder_name not in ['avdec_h265'] else "software"
                print(f"[INFO] Using {hw_type} H.265 decoder: {decoder_name}")
                logger.info(f"Selected {hw_type} decoder: {decoder_name}")
                break
        
        if not selected_decoder:
            # Ultimate fallback
            selected_decoder = 'avdec_h265'
            print(f"[WARNING] No hardware decoder found, using software: {selected_decoder}")
            logger.warning(f"Falling back to software decoder: {selected_decoder}")
        
        # Cache result
        GStreamerPlatform._decoder_cache = selected_decoder
        return selected_decoder


class MultiCameraVideoStreamer:
    """
    Multi-camera video streamer using RTP payload type multiplexing.
    
    **Architecture:**
    - All cameras (RGB + Depth) multiplexed on SINGLE UDP port
    - Uses different RTP payload types for stream identification:
        - Camera N RGB:   PT = 96 + (N * 2)
        - Camera N Depth: PT = 97 + (N * 2)
    
    **Example (3 cameras):**
        Camera 0: RGB=PT96,  Depth=PT97
        Camera 1: RGB=PT98,  Depth=PT99
        Camera 2: RGB=PT100, Depth=PT101
        
        All on port 5000 → client uses rtpptdemux to separate
    
    **Benefits:**
    - Single firewall port
    - Synchronized streams (same network path)
    - Easy client detection
    - Scalable to many cameras
    
    Automatically starts/stops streaming based on active client count.
    """
    
    def __init__(self, num_cameras=1, host='239.255.0.1', stream_port=5000,
                 camera_width=1280, camera_height=720, framerate=30,
                 enable_client_detection=True, client_timeout=5.0):
        """
        Initialize multi-camera video streamer with RTP multiplexing.
        
        Args:
            num_cameras: Number of cameras to multiplex (RGB + Depth each)
            host: Multicast group address (default: 239.255.0.1 - organization-local)
            stream_port: Single UDP port for ALL streams (default: 5000)
            camera_width: Width of each individual camera
            camera_height: Height of each individual camera
            framerate: Target framerate
            enable_client_detection: Auto start/stop streaming based on clients
            client_timeout: Seconds before inactive client is removed
        """
        self.num_cameras = num_cameras
        self.host = host
        self.stream_port = stream_port
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.framerate = framerate
        self.enable_client_detection = enable_client_detection
        self.client_timeout = client_timeout
        
        # Single pipeline with MPEG-TS mux containing both RGB and depth
        self.pipeline = None
        
        # Multiple appsrc elements (one per camera for RGB and depth)
        self.rgb_appsrcs = []
        self.depth_appsrcs = []
        
        self.is_streaming = False
        # Timing for appsrc buffers
        self._frame_duration_ns = int(1_000_000_000 // max(1, self.framerate))
        self._rgb_pts = [0] * self.num_cameras
        self._depth_pts = [0] * self.num_cameras
        
        # Client detection
        self.active_clients = {}  # {(ip, port): last_seen_timestamp}
        self.client_lock = threading.Lock()
        self.heartbeat_socket = None
        self.heartbeat_thread = None
        self.heartbeat_running = False
        
        if enable_client_detection:
            self._start_heartbeat_listener()
        
        logger.info(f"MultiCameraVideoStreamer initialized:")
        logger.info(f"  Cameras: {num_cameras}")
        logger.info(f"  Resolution: {camera_width}x{camera_height}@{framerate}fps")
        logger.info(f"  Single multiplexed stream: {host}:{stream_port}")
        logger.info(f"  Client detection: {'enabled' if enable_client_detection else 'disabled'}")
        
        print(f"[INFO] MultiCameraVideoStreamer initialized:")
        print(f"[INFO]   Cameras: {num_cameras}")
        print(f"[INFO]   Resolution: {camera_width}x{camera_height}@{framerate}fps")
        print(f"[INFO]   Single stream port: {stream_port}")
        print(f"[INFO]   Client detection: {'enabled' if enable_client_detection else 'disabled'}")
    
    def _start_heartbeat_listener(self):
        """Start UDP listener for client heartbeat messages"""
        try:
            # Use a separate port for heartbeats (stream port + 100)
            heartbeat_port = self.stream_port + 100
            self.heartbeat_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.heartbeat_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.heartbeat_socket.settimeout(1.0)
            self.heartbeat_socket.bind((self.host, heartbeat_port))
            
            self.heartbeat_running = True
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self.heartbeat_thread.start()
            
            logger.info(f"Heartbeat listener started on port {heartbeat_port}")
            print(f"[INFO] Heartbeat listener started on port {heartbeat_port}")
        except Exception as e:
            logger.error(f"Failed to start heartbeat listener: {e}")
            print(f"[ERROR] Failed to start heartbeat listener: {e}")
            self.enable_client_detection = False
    
    def _heartbeat_loop(self):
        """Background thread to receive client heartbeats"""
        heartbeat_count = 0
        while self.heartbeat_running:
            try:
                data, addr = self.heartbeat_socket.recvfrom(1024)
                if data == b"STREAMING_CLIENT":
                    heartbeat_count += 1
                    if heartbeat_count <= 3:
                        print(f"[DEBUG] Heartbeat #{heartbeat_count} received from {addr}")
                    self._register_client(addr)
                    # Send acknowledgment
                    self.heartbeat_socket.sendto(b"ACK", addr)
            except socket.timeout:
                pass
            except Exception as e:
                if self.heartbeat_running:
                    logger.warning(f"Heartbeat error: {e}")
                    print(f"[WARNING] Heartbeat error: {e}")
            
            # Periodically check for expired clients
            self._cleanup_inactive_clients()
    
    def _register_client(self, addr):
        """Register or update client activity"""
        with self.client_lock:
            was_empty = len(self.active_clients) == 0
            is_new_client = addr not in self.active_clients
            self.active_clients[addr] = time.time()
            
            # Start streaming if first client
            if was_empty and not self.is_streaming:
                logger.info(f"First client connected from {addr}, starting streaming")
                print(f"[INFO] First client connected from {addr}, starting streaming")
                self.start()
                # Add first client to multiudpsink
                self._add_client_to_sinks(addr[0])
            elif is_new_client and self.is_streaming:
                logger.info(f"Client connected from {addr} (total: {len(self.active_clients)})")
                print(f"[INFO] Client connected from {addr} (total: {len(self.active_clients)})")
                # Add new client to existing stream
                self._add_client_to_sinks(addr[0])
    
    def _add_client_to_sinks(self, client_ip):
        """Add client IP to RGB and depth sinks (idempotent)"""
        try:
            if hasattr(self, 'rgb_sink') and self.rgb_sink:
                # Check if client already exists to avoid duplicates
                current_clients = self.rgb_sink.get_property('clients')
                client_str = f"{client_ip}:{self.stream_port}"
                if client_str not in current_clients:
                    self.rgb_sink.emit('add', client_ip, self.stream_port)
                    clients = self.rgb_sink.get_property('clients')
                    print(f"[INFO] Added client {client_ip}:{self.stream_port} to RGB stream")
                    print(f"[DEBUG] RGB sink clients: {clients}")
                else:
                    print(f"[DEBUG] Client {client_str} already in RGB stream")
                    
            if hasattr(self, 'depth_sink') and self.depth_sink:
                depth_port = self.stream_port + 1
                current_clients = self.depth_sink.get_property('clients')
                client_str = f"{client_ip}:{depth_port}"
                if client_str not in current_clients:
                    self.depth_sink.emit('add', client_ip, depth_port)
                    clients = self.depth_sink.get_property('clients')
                    print(f"[INFO] Added client {client_ip}:{depth_port} to depth stream")
                    print(f"[DEBUG] Depth sink clients: {clients}")
                else:
                    print(f"[DEBUG] Client {client_str} already in depth stream")
        except Exception as e:
            print(f"[ERROR] Failed to add client: {e}")
            logger.error(f"Failed to add client: {e}")
    
    def _cleanup_inactive_clients(self):
        """Remove clients that haven't sent heartbeat recently"""
        current_time = time.time()
        
        with self.client_lock:
            expired = [
                addr for addr, last_seen in self.active_clients.items()
                if current_time - last_seen > self.client_timeout
            ]
            
            for addr in expired:
                logger.info(f"Client {addr} timed out, removing")
                del self.active_clients[addr]
            
            # Stop streaming if no more clients
            if len(self.active_clients) == 0 and self.is_streaming:
                logger.info("No active clients, stopping streaming")
                self.stop()
    
    def get_client_count(self):
        """Get number of active clients"""
        with self.client_lock:
            return len(self.active_clients)
    
    def start(self):
        """Start RGB streaming only (simplified for testing)"""
        if self.is_streaming:
            logger.warning("Streaming already active")
            return
        
        # Create RGB/Depth pipelines (multiple pipelines, one per camera per stream type)
        self._create_simple_rgb_pipeline()
        self._create_simple_depth_pipeline()
        
        # Start all RGB pipelines
        if hasattr(self, 'rgb_pipelines'):
            for pipeline in self.rgb_pipelines:
                if pipeline:
                    pipeline.set_state(Gst.State.PLAYING)
        
        # Start all Depth pipelines
        if hasattr(self, 'depth_pipelines'):
            for pipeline in self.depth_pipelines:
                if pipeline:
                    pipeline.set_state(Gst.State.PLAYING)
        
        self.is_streaming = True
        if hasattr(self, '_warned_not_streaming'):
            delattr(self, '_warned_not_streaming')
        logger.info("RGB/Depth streaming started")
        print("[INFO] RGB/Depth streaming started")
    
    def stop(self):
        """Stop streaming pipelines"""
        if not self.is_streaming:
            return
        
        self.is_streaming = False
        
        # Stop all RGB pipelines
        if hasattr(self, 'rgb_pipelines'):
            for pipeline in self.rgb_pipelines:
                if pipeline:
                    pipeline.set_state(Gst.State.NULL)
        
        # Stop all Depth pipelines
        if hasattr(self, 'depth_pipelines'):
            for pipeline in self.depth_pipelines:
                if pipeline:
                    pipeline.set_state(Gst.State.NULL)
        
        logger.info("Streaming stopped")
        print("[INFO] Streaming stopped")
    
    def shutdown(self):
        """Shutdown streamer and cleanup resources"""
        self.stop()
        
        # Stop heartbeat listener
        if self.heartbeat_running:
            self.heartbeat_running = False
            if self.heartbeat_thread:
                self.heartbeat_thread.join(timeout=2.0)
        
        if self.heartbeat_socket:
            self.heartbeat_socket.close()
        
        logger.info("MultiCameraVideoStreamer shutdown complete")
    
    def _on_server_bus_message(self, bus, message):
        """Handle GStreamer bus messages for server pipeline"""
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"[ERROR] Server GStreamer error: {err}: {debug}")
            logger.error(f"Server GStreamer error: {err}: {debug}")
        elif t == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            print(f"[WARNING] Server GStreamer warning: {warn}: {debug}")
            logger.warning(f"Server GStreamer warning: {warn}: {debug}")
        elif t == Gst.MessageType.STATE_CHANGED:
            # Only RGB pipeline exists now (contains both RGB and depth streams)
            if hasattr(self, 'rgb_pipeline') and self.rgb_pipeline and message.src == self.rgb_pipeline:
                old, new, pending = message.parse_state_changed()
                print(f"[DEBUG] Server pipeline state: {old.value_nick} -> {new.value_nick}")
        return True
    
    def _create_simple_rgb_pipeline(self):
        """
        Create separate RGB pipelines for each camera, all sending to same port with different payload types.
        Camera N uses PT = 96 + (N * 2) for RGB
        """
        try:
            encoder_name, encoder_props = GStreamerPlatform.get_encoder('rgb')
            props_str = " ".join([f"{k}={v}" for k, v in encoder_props.items()])
            framerate_int = int(self.framerate)
            
            # Create list to hold all RGB pipelines (one per camera)
            self.rgb_pipelines = []
            self.rgb_appsrcs = []
            
            for cam_idx in range(self.num_cameras):
                pt = 96 + (cam_idx * 2)  # PT: 96, 98, 100, ...
                
                pipeline_str = (
                    f"appsrc name=src format=time is-live=true do-timestamp=true "
                    f"caps=video/x-raw,format=BGR,width={self.camera_width},height={self.camera_height},"
                    f"framerate={framerate_int}/1 ! "
                    f"queue max-size-buffers=2 leaky=downstream ! "
                    f"videoconvert ! video/x-raw,format=NV12 ! "
                    f"{encoder_name} {props_str} ! "
                    f"h265parse ! "
                    f"rtph265pay config-interval=1 pt={pt} ! "
                    f"udpsink host={self.host} port={self.stream_port} "
                    f"auto-multicast=true ttl-mc=1 sync=false async=false"
                )
                
                pipeline = Gst.parse_launch(pipeline_str)
                appsrc = pipeline.get_by_name('src')
                appsrc.set_property('format', Gst.Format.TIME)
                
                # Add bus handler
                bus = pipeline.get_bus()
                bus.add_signal_watch()
                bus.connect('message', self._on_server_bus_message)
                
                self.rgb_pipelines.append(pipeline)
                self.rgb_appsrcs.append(appsrc)
                
                logger.info(f"RGB pipeline {cam_idx} created: PT={pt}, port={self.stream_port}")
            
            # Keep first pipeline as main reference for compatibility
            self.rgb_pipeline = self.rgb_pipelines[0] if self.rgb_pipelines else None
            
            print(f"[INFO] Created {len(self.rgb_pipelines)} RGB pipelines on port {self.stream_port} (PT 96, 98, 100...)")
            
        except Exception as e:
            logger.error(f"Failed to create RGB pipelines: {e}")
            print(f"[ERROR] Failed to create RGB pipelines: {e}")
            import traceback
            traceback.print_exc()
            self.rgb_pipeline = None
            self.rgb_pipelines = []
    
    def _create_simple_depth_pipeline(self):
        """
        Create separate depth pipelines for each camera, all sending to same port with different payload types.
        Camera N uses PT = 97 + (N * 2) for Depth
        
        Uses best-effort lossless H.265 encoding per platform:
        - NVIDIA nvh265enc: preset=lossless, qp=0 (truly lossless, bit-identical uint16)
        - Apple vtenc_h265: quality=1.0 (highest quality, near-lossless)
        - Software x265enc: qp-min=0 (near-lossless)
        - Others: Lowest QP possible
        """
        try:
            depth_encoder_name, depth_encoder_props = GStreamerPlatform.get_encoder('depth')
            
            # Platform-specific lossless/high-quality configuration
            if 'nvh265enc' in depth_encoder_name:
                # NVIDIA: True lossless mode (QP=0, bit-perfect)
                lossless_props = "preset=lossless rc-mode=constqp qp-const-i=0 qp-const-p=0 qp-const-b=0 gop-size=1"
                output_format = "Y444_16LE"  # 4:4:4 16-bit
                quality_desc = "lossless (QP=0)"
                
            elif 'vtenc_h265' in depth_encoder_name:
                # Apple VideoToolbox: Maximum quality
                # Note: vtenc doesn't support true lossless, but quality=1.0 is very high
                lossless_props = "quality=1.0 max-keyframe-interval=1 realtime=false"
                output_format = "I420_10LE"  # 10-bit
                quality_desc = "maximum quality (near-lossless)"
                
            elif 'vaapih265enc' in depth_encoder_name:
                # Intel/AMD VAAPI: Best quality mode
                lossless_props = "rate-control=1 quality-level=1"  # CQP mode, best quality
                output_format = "I420_10LE"
                quality_desc = "best quality (CQP)"
                
            elif 'mfh265enc' in depth_encoder_name:
                # Microsoft Media Foundation: Best quality
                lossless_props = "quality=100 low-latency=false"
                output_format = "I420_10LE"
                quality_desc = "maximum quality"
                
            elif 'x265enc' in depth_encoder_name:
                # Software x265: Near-lossless
                lossless_props = "speed-preset=veryslow tune=ssim qp-min=0 qp-max=5"
                output_format = "I420_10LE"
                quality_desc = "near-lossless (QP 0-5)"
                
            else:
                # Unknown encoder: conservative high quality
                lossless_props = "bitrate=50000"  # Very high bitrate
                output_format = "I420"
                quality_desc = "high bitrate"
            
            depth_encoder_props_str = ' '.join([f"{k}={v}" for k, v in depth_encoder_props.items()])
            framerate_int = int(self.framerate)
            
            # MTU optimization: 1400 for internet, could use 8192 for local networks (jumbo frames)
            mtu_size = 1400
            
            # Create list to hold all depth pipelines (one per camera)
            self.depth_pipelines = []
            self.depth_appsrcs = []
            
            print(f"[INFO] Depth encoding mode: {quality_desc}")
            logger.info(f"Depth encoding: {depth_encoder_name} with {quality_desc}")
            
            for cam_idx in range(self.num_cameras):
                pt = 97 + (cam_idx * 2)  # PT: 97, 99, 101, ...
                
                pipeline_str = (
                    f"appsrc name=src format=time is-live=true do-timestamp=true "
                    f"caps=video/x-raw,format=GRAY16_LE,width={self.camera_width},height={self.camera_height},"
                    f"framerate={framerate_int}/1 ! "
                    f"queue max-size-buffers=2 leaky=downstream ! "
                    f"videoconvert dither=none ! video/x-raw,format={output_format} ! "
                    f"{depth_encoder_name} {depth_encoder_props_str} {lossless_props} ! "
                    f"h265parse ! "
                    f"rtph265pay config-interval=1 mtu={mtu_size} pt={pt} ! "
                    f"udpsink host={self.host} port={self.stream_port} "
                    f"auto-multicast=true ttl-mc=1 sync=false async=false"
                )
                
                pipeline = Gst.parse_launch(pipeline_str)
                appsrc = pipeline.get_by_name('src')
                appsrc.set_property('format', Gst.Format.TIME)
                
                # Add bus handler
                bus = pipeline.get_bus()
                bus.add_signal_watch()
                bus.connect('message', self._on_server_bus_message)
                
                self.depth_pipelines.append(pipeline)
                self.depth_appsrcs.append(appsrc)
                
                logger.info(f"Depth pipeline {cam_idx} created: PT={pt}, port={self.stream_port} (lossless)")
            
            # Keep first pipeline as main reference for compatibility
            self.depth_pipeline = self.depth_pipelines[0] if self.depth_pipelines else None
            
            print(f"[INFO] Created {len(self.depth_pipelines)} lossless Depth pipelines on port {self.stream_port} (PT 97, 99, 101...)")
            
        except Exception as e:
            logger.error(f"Failed to create depth pipelines: {e}")
            print(f"[ERROR] Failed to create depth pipelines: {e}")
            import traceback
            traceback.print_exc()
            self.depth_pipeline = None
            self.depth_pipelines = []
    
    def push_camera_frames(self, rgb_frames: List[np.ndarray], depth_frames: List[np.ndarray]):
        """
        Push frames from multiple cameras to respective appsrc elements.
        
        Args:
            rgb_frames: List of RGB frames (H, W, 3) uint8 BGR, one per camera
            depth_frames: List of depth frames (H, W) uint16, one per camera
        """
        if not self.is_streaming:
            if not hasattr(self, '_warned_not_streaming'):
                self._warned_not_streaming = True
                print("[WARNING] push_camera_frames called but not streaming")
            return
        
        if not hasattr(self, '_push_count'):
            self._push_count = 0
        self._push_count += 1
        if self._push_count <= 3:
            print(f"[DEBUG] push_camera_frames called (#{self._push_count}): {len(rgb_frames)} RGB, {len(depth_frames)} depth")
        
        if len(rgb_frames) != self.num_cameras or len(depth_frames) != self.num_cameras:
            logger.warning(f"Expected {self.num_cameras} frames, got RGB:{len(rgb_frames)}, Depth:{len(depth_frames)}")
            return
        
        # Push each camera's frames to its respective appsrc
        for i in range(self.num_cameras):
            if i < len(self.rgb_appsrcs) and rgb_frames[i] is not None:
                self._push_rgb_frame(i, rgb_frames[i])
            
            # Depth: either via GStreamer appsrc or direct UDP (Zstd)
            if depth_frames[i] is not None:
                if hasattr(self, 'depth_compressor'):
                    # Using Zstd UDP - no appsrc check needed
                    self._push_depth_frame(i, depth_frames[i])
                elif i < len(self.depth_appsrcs):
                    # Using GStreamer appsrc
                    self._push_depth_frame(i, depth_frames[i])
    
    def _push_rgb_frame(self, camera_idx: int, frame: np.ndarray):
        """Push RGB frame for specific camera"""
        try:
            if frame.shape[2] != 3:
                logger.warning(f"RGB frame {camera_idx} has wrong channels: {frame.shape}")
                return
            
            data = frame.tobytes()
            buf = Gst.Buffer.new_wrapped(data)
            # Set timestamps for live pipeline
            buf.pts = self._rgb_pts[camera_idx]
            buf.dts = Gst.CLOCK_TIME_NONE
            buf.duration = self._frame_duration_ns
            self._rgb_pts[camera_idx] += self._frame_duration_ns
            ret = self.rgb_appsrcs[camera_idx].emit('push-buffer', buf)
            
            if ret != Gst.FlowReturn.OK:
                logger.warning(f"Failed to push RGB frame {camera_idx}: {ret}")
                print(f"[WARNING] Failed to push RGB frame {camera_idx}: {ret}")
            else:
                # Print first few successful pushes only - WITH CAMERA INDEX AND HASH
                if not hasattr(self, '_rgb_push_counts_per_cam'):
                    self._rgb_push_counts_per_cam = {}
                if camera_idx not in self._rgb_push_counts_per_cam:
                    self._rgb_push_counts_per_cam[camera_idx] = 0
                self._rgb_push_counts_per_cam[camera_idx] += 1
                if self._rgb_push_counts_per_cam[camera_idx] <= 5:
                    # Hash first 100 pixels to verify different data
                    frame_hash = hash(frame[:10, :10, :].tobytes())
                    pt = 96 + (camera_idx * 2)
                    print(f"[DEBUG] RGB CAM{camera_idx} frame #{self._rgb_push_counts_per_cam[camera_idx]} pushed to PT={pt} (shape={frame.shape}, hash={frame_hash})")
        except Exception as e:
            logger.error(f"Error pushing RGB frame {camera_idx}: {e}")
            print(f"[ERROR] Error pushing RGB frame {camera_idx}: {e}")
    
    def _push_depth_frame(self, camera_idx: int, frame: np.ndarray):
        """Push depth frame using Zstd compression over RTP
        
        OPTIMIZED for ZED cameras (configured with UNIT.MILLIMETER):
        - Input: float32 millimeters from ZED SDK
        - Convert: float32 mm → uint16 tenths-of-millimeter (0.1mm precision)
        - Sensor precision: ~1-3mm (0.1mm is 10x finer = lossless)
        - Range: 0-6553.5 mm with uint16 (0-65535 tenths-mm = 0-6.5 meters)
        - Compress uint16 with Zstd (60x+ compression typical)
        - Send via RTP (automatic fragmentation for large frames)
        - Result: ~8x smaller than float32 with identical perceptual quality
        
        Packet format: [magic:4bytes][width:2bytes][height:2bytes][compressed_data]
        RTP handles PTS/sequencing automatically
        """
        try:
            # Check if we have a valid depth appsrc for H.265 encoding
            if camera_idx >= len(self.depth_appsrcs):
                return
            
            if not self.depth_appsrcs[camera_idx]:
                return
                
            # Convert float32 millimeters to uint16 tenths-of-millimeter for GRAY16_LE encoding
            # ZED is configured with UNIT.MILLIMETER, so depth values are already in mm
            # We encode as uint16 tenths-mm (0.1mm precision) for H.265 lossless compression
            if frame.dtype == np.float32:
                # DEBUG: Check raw depth values
                if not hasattr(self, '_depth_debug_done'):
                    self._depth_debug_done = True
                    raw_valid = frame[np.isfinite(frame) & (frame > 0)]
                    if raw_valid.size > 0:
                        print(f"[DEBUG] RAW depth from ZED: min={raw_valid.min():.1f}mm, max={raw_valid.max():.1f}mm, mean={raw_valid.mean():.1f}mm")
                
                # Clean NaN/Inf values (ZED can produce these for invalid depth)
                frame_clean = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
                # Convert mm → tenths-mm, clip to uint16 range (0-6553.5mm)
                depth_uint16 = np.clip(frame_clean * 10.0, 0, 65535).astype(np.uint16)
            elif frame.dtype == np.uint16:
                depth_uint16 = frame
            else:
                frame_clean = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
                depth_uint16 = np.clip(frame_clean.astype(np.float32) * 10.0, 0, 65535).astype(np.uint16)
            
            h, w = depth_uint16.shape
            
            # Create GStreamer buffer from raw GRAY16_LE data
            raw_data = depth_uint16.tobytes()
            buf = Gst.Buffer.new_allocate(None, len(raw_data), None)
            buf.fill(0, raw_data)
            
            # Set timestamp
            if not hasattr(self, 'depth_pts_counters'):
                self.depth_pts_counters = [0] * self.num_cameras
            
            frame_duration_ns = int((1.0 / self.framerate) * Gst.SECOND)
            buf.pts = self.depth_pts_counters[camera_idx]
            buf.duration = frame_duration_ns
            self.depth_pts_counters[camera_idx] += frame_duration_ns
            
            # Push to appsrc (H.265 encoder will compress)
            appsrc = self.depth_appsrcs[camera_idx]
            ret = appsrc.emit('push-buffer', buf)
            
            # Debug output
            if not hasattr(self, '_depth_push_count'):
                self._depth_push_count = 0
            
            self._depth_push_count += 1
            
            if self._depth_push_count <= 5:
                valid_depth = depth_uint16[depth_uint16 > 0]
                if valid_depth.size > 0:
                    print(f"[DEBUG] Depth frame {self._depth_push_count}: {w}x{h} GRAY16_LE, "
                          f"range: {valid_depth.min()/10.0:.1f}mm-{valid_depth.max()/10.0:.1f}mm "
                          f"({valid_depth.size} valid pixels), raw size={len(raw_data)/1024:.1f}KB, ret={ret}")
                else:
                    print(f"[DEBUG] Depth frame {self._depth_push_count}: {w}x{h} GRAY16_LE, NO VALID DEPTH, ret={ret}")
                
        except Exception as e:
            logger.error(f"Error pushing depth frame {camera_idx}: {e}")
            print(f"[ERROR] Error pushing depth frame {camera_idx}: {e}")
            print(f"[ERROR] Error pushing depth frame {camera_idx}: {e}")
    
    def __del__(self):
        """Cleanup"""
        self.shutdown()


class VideoStreamer:
    """
    Server-side video streamer using GStreamer.
    Encodes and streams RGB and depth images via RTP.
    
    Automatically starts/stops streaming based on active client count.
    """
    
    def __init__(self, host='0.0.0.0', rgb_port=5000, depth_port=5001, 
                 width=1280, height=720, framerate=30,
                 enable_client_detection=True, client_timeout=5.0):
        """
        Initialize video streamer.
        
        Args:
            host: Host address to bind to (0.0.0.0 for all interfaces)
            rgb_port: RTP port for RGB stream
            depth_port: RTP port for depth stream
            width: Video width
            height: Video height
            framerate: Target framerate
            enable_client_detection: Auto start/stop streaming based on clients
            client_timeout: Seconds before inactive client is removed
        """
        self.host = host
        self.rgb_port = rgb_port
        self.depth_port = depth_port
        self.width = width
        self.height = height
        self.framerate = framerate
        self.enable_client_detection = enable_client_detection
        self.client_timeout = client_timeout
        
        self.rgb_pipeline = None
        self.depth_pipeline = None
        
        self.rgb_appsrc = None
        self.depth_appsrc = None
        
        self.is_streaming = False
        self.frame_count = 0
        
        # Client detection
        self.active_clients = {}  # {(ip, port): last_seen_timestamp}
        self.client_lock = threading.Lock()
        self.heartbeat_socket = None
        self.heartbeat_thread = None
        self.heartbeat_running = False
        
        if enable_client_detection:
            self._start_heartbeat_listener()
        
        logger.info(f"VideoStreamer initialized: {width}x{height}@{framerate}fps")
        logger.info(f"RGB stream: {host}:{rgb_port}, Depth stream: {host}:{depth_port}")
        logger.info(f"Client detection: {'enabled' if enable_client_detection else 'disabled'}")
    
    def _start_heartbeat_listener(self):
        """Start UDP listener for client heartbeat messages"""
        try:
            heartbeat_port = self.rgb_port + 100
            self.heartbeat_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.heartbeat_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.heartbeat_socket.settimeout(1.0)
            self.heartbeat_socket.bind((self.host, heartbeat_port))
            
            self.heartbeat_running = True
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self.heartbeat_thread.start()
            
            logger.info(f"Heartbeat listener started on port {heartbeat_port}")
        except Exception as e:
            logger.error(f"Failed to start heartbeat listener: {e}")
            self.enable_client_detection = False
    
    def _heartbeat_loop(self):
        """Background thread to receive client heartbeats"""
        while self.heartbeat_running:
            try:
                data, addr = self.heartbeat_socket.recvfrom(1024)
                if data == b"STREAMING_CLIENT":
                    self._register_client(addr)
                    self.heartbeat_socket.sendto(b"ACK", addr)
            except socket.timeout:
                pass
            except Exception as e:
                if self.heartbeat_running:
                    logger.warning(f"Heartbeat error: {e}")
            
            self._cleanup_inactive_clients()
    
    def _register_client(self, addr):
        """Register or update client activity"""
        with self.client_lock:
            was_empty = len(self.active_clients) == 0
            is_new = addr not in self.active_clients
            self.active_clients[addr] = time.time()
            
            if was_empty and not self.is_streaming:
                logger.info(f"First client connected from {addr}, starting streaming")
                self.start()
            elif is_new:
                logger.info(f"Client connected from {addr} (total: {len(self.active_clients)})")
    
    def _cleanup_inactive_clients(self):
        """Remove clients that haven't sent heartbeat recently"""
        current_time = time.time()
        
        with self.client_lock:
            expired = [
                addr for addr, last_seen in self.active_clients.items()
                if current_time - last_seen > self.client_timeout
            ]
            
            for addr in expired:
                logger.info(f"Client {addr} timed out, removing")
                del self.active_clients[addr]
            
            if len(self.active_clients) == 0 and self.is_streaming:
                logger.info("No active clients, stopping streaming")
                self.stop()
    
    def get_client_count(self):
        """Get number of active clients"""
        with self.client_lock:
            return len(self.active_clients)
    
    def start(self):
        """Start streaming pipelines"""
        if self.is_streaming:
            logger.warning("Streaming already active")
            return
        
        self._create_rgb_pipeline()
        self._create_depth_pipeline()
        
        # Start pipelines
        if self.rgb_pipeline:
            self.rgb_pipeline.set_state(Gst.State.PLAYING)
        if self.depth_pipeline:
            self.depth_pipeline.set_state(Gst.State.PLAYING)
        
        self.is_streaming = True
        logger.info("Video streaming started")
    
    def stop(self):
        """Stop streaming pipelines"""
        if not self.is_streaming:
            return
        
        self.is_streaming = False
        
        # Stop pipelines
        if self.rgb_pipeline:
            self.rgb_pipeline.set_state(Gst.State.NULL)
        if self.depth_pipeline:
            self.depth_pipeline.set_state(Gst.State.NULL)
        
        logger.info("Video streaming stopped")
    
    def shutdown(self):
        """Shutdown streamer and cleanup resources"""
        self.stop()
        
        if self.heartbeat_running:
            self.heartbeat_running = False
            if self.heartbeat_thread:
                self.heartbeat_thread.join(timeout=2.0)
        
        if self.heartbeat_socket:
            self.heartbeat_socket.close()
        
        logger.info("VideoStreamer shutdown complete")
    
    def _create_rgb_pipeline(self):
        """Create GStreamer pipeline for RGB streaming"""
        try:
            # Get platform-specific encoder
            encoder_name, encoder_props = GStreamerPlatform.get_encoder('rgb')
            
            # Build pipeline string
            # appsrc -> videoconvert -> encoder -> rtph265pay -> udpsink
            pipeline_str = (
                f"appsrc name=src format=time is-live=true do-timestamp=true "
                f"caps=video/x-raw,format=BGR,width={self.width},height={self.height},"
                f"framerate={self.framerate}/1 ! "
                f"videoconvert ! "
                f"{encoder_name} "
            )
            
            # Add encoder properties
            for key, val in encoder_props.items():
                pipeline_str += f"{key}={val} "
            
            pipeline_str += (
                f"! rtph265pay config-interval=1 pt=96 ! "
                f"udpsink host={self.host} port={self.rgb_port} sync=false"
            )
            
            logger.debug(f"RGB pipeline: {pipeline_str}")
            
            self.rgb_pipeline = Gst.parse_launch(pipeline_str)
            self.rgb_appsrc = self.rgb_pipeline.get_by_name('src')
            
            # Configure appsrc for manual pushing
            self.rgb_appsrc.set_property('format', Gst.Format.TIME)
            
        except Exception as e:
            logger.error(f"Failed to create RGB pipeline: {e}")
            self.rgb_pipeline = None
    
    def _create_depth_pipeline(self):
        """Create GStreamer pipeline for depth streaming (lossless)"""
        try:
            # Get platform-specific encoder for lossless
            encoder_name, encoder_props = GStreamerPlatform.get_encoder('depth')
            
            # Depth is typically 16-bit grayscale
            pipeline_str = (
                f"appsrc name=src format=time is-live=true do-timestamp=true "
                f"caps=video/x-raw,format=GRAY16_LE,width={self.width},height={self.height},"
                f"framerate={self.framerate}/1 ! "
                f"videoconvert ! "
                f"{encoder_name} "
            )
            
            # Add lossless encoding properties
            for key, val in encoder_props.items():
                pipeline_str += f"{key}={val} "
            
            pipeline_str += (
                f"! rtph265pay config-interval=1 pt=97 ! "
                f"udpsink host={self.host} port={self.depth_port} sync=false"
            )
            
            logger.debug(f"Depth pipeline: {pipeline_str}")
            
            self.depth_pipeline = Gst.parse_launch(pipeline_str)
            self.depth_appsrc = self.depth_pipeline.get_by_name('src')
            
            self.depth_appsrc.set_property('format', Gst.Format.TIME)
            
        except Exception as e:
            logger.error(f"Failed to create depth pipeline: {e}")
            self.depth_pipeline = None
    
    def push_rgb_frame(self, frame: np.ndarray):
        """
        Push RGB frame to stream.
        
        Args:
            frame: RGB image as numpy array (H, W, 3), dtype=uint8, BGR format
        """
        if not self.is_streaming or self.rgb_appsrc is None:
            return
        
        try:
            # Ensure correct shape and format
            if frame.shape[2] != 3:
                logger.warning(f"RGB frame has wrong channels: {frame.shape}")
                return
            
            # Create GStreamer buffer
            data = frame.tobytes()
            buf = Gst.Buffer.new_wrapped(data)
            
            # Push buffer
            ret = self.rgb_appsrc.emit('push-buffer', buf)
            
            if ret != Gst.FlowReturn.OK:
                logger.warning(f"Failed to push RGB buffer: {ret}")
            
        except Exception as e:
            logger.error(f"Error pushing RGB frame: {e}")
    
    def push_depth_frame(self, frame: np.ndarray):
        """
        Push depth frame to stream.
        
        Args:
            frame: Depth image as numpy array (H, W), dtype=uint16
        """
        if not self.is_streaming or self.depth_appsrc is None:
            return
        
        try:
            # Ensure correct dtype
            if frame.dtype != np.uint16:
                frame = frame.astype(np.uint16)
            
            # Create GStreamer buffer
            data = frame.tobytes()
            buf = Gst.Buffer.new_wrapped(data)
            
            # Push buffer
            ret = self.depth_appsrc.emit('push-buffer', buf)
            
            if ret != Gst.FlowReturn.OK:
                logger.warning(f"Failed to push depth buffer: {ret}")
            
        except Exception as e:
            logger.error(f"Error pushing depth frame: {e}")
    
    def __del__(self):
        """Cleanup on destruction"""
        self.shutdown()


class VideoReceiver:
    """
    Client-side video receiver using GStreamer.
    Receives and decodes multiplexed MPEG-TS stream containing RGB and depth.
    
    Sends periodic heartbeats to server to maintain streaming connection.
    """
    
    def __init__(self, server_ip, stream_port: Optional[int] = 5000, num_cameras=1,
                 rgb_callback: Optional[Callable] = None,
                 depth_callback: Optional[Callable] = None,
                 send_heartbeat=True, heartbeat_interval=2.0,
                 rgb_port: Optional[int] = None,
                 depth_port: Optional[int] = None):
        """
        Initialize video receiver.
        
        Args:
            server_ip: Server IP address
            stream_port: UDP port for RGB stream (defaults to 5000)
            num_cameras: Number of cameras in stream
            rgb_callback: Callback function for RGB frames: callback(camera_idx, numpy_array)
            depth_callback: Callback function for depth frames: callback(camera_idx, numpy_array)
            send_heartbeat: Send periodic heartbeats to server
            heartbeat_interval: Seconds between heartbeats
            rgb_port: Optional legacy alias for stream_port
            depth_port: Optional override for depth RTP port (defaults to stream_port + 1)
        """
        self.server_ip = server_ip

        if stream_port is None and rgb_port is not None:
            stream_port = rgb_port

        self.stream_port = stream_port if stream_port is not None else 5000
        self.rgb_port = self.stream_port
        self.num_cameras = num_cameras
        
        self.rgb_callback = rgb_callback
        self.depth_callback = depth_callback
        
        self.pipeline = None
        self.rgb_pipeline = None
        self.depth_pipeline = None
        if depth_port is not None:
            self.depth_port = depth_port
        else:
            self.depth_port = self.stream_port + 1
        
        self.is_receiving = False
        
        self.main_loop = None
        self.loop_thread = None
        
        # Heartbeat
        self.send_heartbeat = send_heartbeat
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_socket = None
        self.heartbeat_thread = None
        self.heartbeat_running = False
        
        logger.info(f"VideoReceiver initialized: {server_ip}:{self.stream_port}")
        logger.info(f"Cameras: {num_cameras}, Heartbeat: {'enabled' if send_heartbeat else 'disabled'}")
    
    def start(self):
        """Start receiving RGB and depth streams"""
        if self.is_receiving:
            logger.warning("Already receiving")
            return
        
        self._create_rgb_receiver()
        self._create_depth_receiver()
        
        # Start GLib main loop in separate thread
        self.main_loop = GLib.MainLoop()
        self.loop_thread = threading.Thread(target=self.main_loop.run, daemon=True)
        self.loop_thread.start()
        
        # Start pipeline first
        if self.rgb_pipeline:
            ret = self.rgb_pipeline.set_state(Gst.State.PLAYING)
            print(f"[DEBUG] Pipeline set_state result: {ret}")
            # Give pipeline time to start listening on UDP port
            time.sleep(0.2)
        if self.depth_pipeline:
            ret = self.depth_pipeline.set_state(Gst.State.PLAYING)
            print(f"[DEBUG] Depth pipeline set_state result: {ret}")
        
        # Start heartbeat AFTER pipeline is listening
        if self.send_heartbeat:
            self._start_heartbeat()
        
        self.is_receiving = True
    logger.info("RGB/Depth receiving started")
    print("[INFO] RGB/Depth receiving started")
    
    def _start_heartbeat(self):
        """Start sending heartbeat messages to server"""
        try:
            self.heartbeat_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            heartbeat_port = self.stream_port + 100
            
            self.heartbeat_running = True
            
            def heartbeat_loop():
                heartbeat_count = 0
                while self.heartbeat_running:
                    try:
                        self.heartbeat_socket.sendto(b"STREAMING_CLIENT", 
                                                     (self.server_ip, heartbeat_port))
                        heartbeat_count += 1
                        if heartbeat_count <= 3:
                            print(f"[DEBUG] Heartbeat #{heartbeat_count} sent to {self.server_ip}:{heartbeat_port}")
                        time.sleep(self.heartbeat_interval)
                    except Exception as e:
                        if self.heartbeat_running:
                            logger.warning(f"Heartbeat send error: {e}")
                            print(f"[WARNING] Heartbeat send error: {e}")
            
            self.heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
            self.heartbeat_thread.start()
            
            logger.info(f"Sending heartbeats to {self.server_ip}:{heartbeat_port}")
            print(f"[INFO] Sending heartbeats to {self.server_ip}:{heartbeat_port}")
        except Exception as e:
            logger.error(f"Failed to start heartbeat: {e}")
            print(f"[ERROR] Failed to start heartbeat: {e}")
    
    def _on_bus_message(self, bus, message):
        """Handle GStreamer bus messages"""
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"[ERROR] GStreamer error: {err}: {debug}")
            logger.error(f"GStreamer error: {err}: {debug}")
        elif t == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            print(f"[WARNING] GStreamer warning: {warn}: {debug}")
            logger.warning(f"GStreamer warning: {warn}: {debug}")
        elif t == Gst.MessageType.EOS:
            print("[INFO] End of stream")
        elif t == Gst.MessageType.STATE_CHANGED:
            if message.src == self.rgb_pipeline:
                old, new, pending = message.parse_state_changed()
                print(f"[DEBUG] RGB pipeline state: {old.value_nick} -> {new.value_nick}")
            elif message.src == self.depth_pipeline:
                old, new, pending = message.parse_state_changed()
                print(f"[DEBUG] Depth pipeline state: {old.value_nick} -> {new.value_nick}")
        elif t == Gst.MessageType.STREAM_START:
            print("[DEBUG] Stream started")
        elif t == Gst.MessageType.NEW_CLOCK:
            print("[DEBUG] New clock")
        elif t == Gst.MessageType.ASYNC_DONE:
            print("[DEBUG] Async done")
        return True
    
    def stop(self):
        """Stop receiving streams"""
        if not self.is_receiving:
            return
        
        self.is_receiving = False
        
        # Stop heartbeat
        if self.heartbeat_running:
            self.heartbeat_running = False
            if self.heartbeat_thread:
                self.heartbeat_thread.join(timeout=2.0)
        
        if self.heartbeat_socket:
            self.heartbeat_socket.close()
        
        # Stop pipelines
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        if self.rgb_pipeline:
            self.rgb_pipeline.set_state(Gst.State.NULL)
        if self.depth_pipeline:
            self.depth_pipeline.set_state(Gst.State.NULL)
        
        # Stop main loop
        if self.main_loop:
            self.main_loop.quit()
        
        if self.loop_thread:
            self.loop_thread.join(timeout=2.0)
        
        logger.info("Video receiving stopped")
        print("[INFO] Video receiving stopped")
    
    def _create_demux_receiver(self):
        """Create GStreamer pipeline to receive interleaved RTP streams"""
        try:
            decoder_name = GStreamerPlatform.get_decoder()
            
            # Receive on single port, use rtpptdemux to separate by payload type
            pipeline_str = (
                f"udpsrc address=239.0.0.1 port={self.stream_port} auto-multicast=true "
                f"caps=\"application/x-rtp\" ! "
                f"rtpptdemux name=demux "
                f"demux.src_96 ! queue ! rtph265depay ! h265parse ! {decoder_name} ! "
                f"videoconvert ! video/x-raw,format=BGR ! "
                f"appsink name=rgb_sink emit-signals=true sync=false max-buffers=1 drop=true "
                f"demux.src_97 ! queue ! rtph265depay ! h265parse ! {decoder_name} ! "
                f"videoconvert ! video/x-raw,format=GRAY16_LE ! "
                f"appsink name=depth_sink emit-signals=true sync=false max-buffers=1 drop=true"
            )
            
            logger.debug(f"Interleaved RTP receiver pipeline: {pipeline_str}")
            print(f"[DEBUG] Creating interleaved RTP receiver...")
            print(f"[DEBUG] Pipeline: {pipeline_str}")
            
            self.pipeline = Gst.parse_launch(pipeline_str)
            
            if not self.pipeline:
                raise Exception("Failed to parse demux pipeline")
            
            # Add bus message handler
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect('message', self._on_bus_message)
            
            # Connect to new-sample signals
            rgb_sink = self.pipeline.get_by_name('rgb_sink')
            if rgb_sink:
                rgb_sink.connect('new-sample', self._on_rgb_sample)
            else:
                logger.warning("Could not find rgb_sink")
            
            depth_sink = self.pipeline.get_by_name('depth_sink')
            if depth_sink:
                depth_sink.connect('new-sample', self._on_depth_sample)
            else:
                logger.warning("Could not find depth_sink")
            
            print(f"[DEBUG] MPEG-TS demux receiver pipeline created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create demux receiver: {e}")
            print(f"[ERROR] Failed to create demux receiver: {e}")
            import traceback
            traceback.print_exc()
            self.pipeline = None
            print(f"[ERROR] Failed to create demux receiver: {e}")
            import traceback
            traceback.print_exc()
            self.pipeline = None
    
    def _create_rgb_receiver(self):
        """
        Create RTP receiver with payload type demultiplexing.
        Uses rtpptdemux to separate RGB and Depth streams by payload type on single port.
        Dynamically connects pads as streams are detected.
        """
        try:
            decoder_name = GStreamerPlatform.get_decoder()
            
            # Single udpsrc → rtpptdemux (pads created dynamically)
            # CRITICAL: Must specify RTP caps with encoding-name=H265 for proper demuxing
            # Match the working test command caps exactly
            # IMPORTANT: Use multicast address (239.255.0.1), not server IP
            multicast_address = "239.255.0.1"  # Organization-local multicast group
            pipeline_str = (
                f"udpsrc address={multicast_address} port={self.stream_port} auto-multicast=true "
                f'caps="application/x-rtp, media=(string)video, encoding-name=(string)H265, '
                f'clock-rate=(int)90000" ! '
                f"rtpptdemux name=demux"
            )
            
            logger.debug(f"RTP demux receiver base pipeline: {pipeline_str}")
            print(f"[DEBUG] RTP demux receiver pipeline (single port {self.stream_port})")
            print(f"[DEBUG] Multicast address: {multicast_address}")
            
            self.rgb_pipeline = Gst.parse_launch(pipeline_str)
            
            # Get udpsrc element and add probe to monitor packet reception
            udpsrc = self.rgb_pipeline.get_by_name('udpsrc0')
            if udpsrc:
                srcpad = udpsrc.get_static_pad('src')
                if srcpad:
                    # Add probe to count packets
                    self._udp_packet_count = 0
                    def udp_probe_callback(pad, info):
                        self._udp_packet_count += 1
                        if self._udp_packet_count <= 5:
                            print(f"[DEBUG] UDP packet #{self._udp_packet_count} received from multicast {multicast_address}:{self.stream_port}")
                        elif self._udp_packet_count == 100:
                            print(f"[DEBUG] Received 100 UDP packets from multicast stream")
                        return Gst.PadProbeReturn.OK
                    srcpad.add_probe(Gst.PadProbeType.BUFFER, udp_probe_callback)
                    print(f"[DEBUG] Added UDP packet probe to monitor multicast reception")
            
            # Get demux element to connect pad-added signal
            demux = self.rgb_pipeline.get_by_name('demux')
            if demux:
                # Connect to pad-added signal to handle dynamic pads
                demux.connect('pad-added', self._on_demux_pad_added)
                print(f"[INFO] Connected to rtpptdemux pad-added signal")
            
            # Add bus message handler
            bus = self.rgb_pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect('message', self._on_bus_message)
            
            # Initialize sink tracking
            self.rgb_sinks = {}  # {cam_idx: appsink}
            self.depth_sinks = {}  # {cam_idx: appsink}
            
            print(f"[INFO] RTP demux receiver pipeline created on port {self.stream_port}")
            
        except Exception as e:
            logger.error(f"Failed to create RTP demux receiver: {e}")
            print(f"[ERROR] Failed to create RTP demux receiver: {e}")
            import traceback
            traceback.print_exc()
            self.rgb_pipeline = None
    
    def _on_demux_pad_added(self, demux, pad):
        """Handle dynamically added pads from rtpptdemux based on payload type"""
        try:
            # Get pad name (format: "src_96", "src_97", etc.)
            pad_name = pad.get_name()
            if not pad_name.startswith('src_'):
                return
            
            # Extract payload type
            pt = int(pad_name.split('_')[1])
            
            # Determine if RGB or Depth based on PT (even=RGB, odd=Depth)
            # PT 96, 98, 100... = RGB (cameras 0, 1, 2...)
            # PT 97, 99, 101... = Depth (cameras 0, 1, 2...)
            is_rgb = (pt % 2 == 0)
            cam_idx = (pt - 96) // 2
            
            decoder_name = GStreamerPlatform.get_decoder()
            
            if is_rgb:
                # Create RGB decoder chain with proper RTP caps
                elements_str = (
                    f"capsfilter caps=\"application/x-rtp,media=video,clock-rate=90000,encoding-name=H265,payload={pt}\" ! "
                    f"queue ! "
                    f"rtph265depay ! "
                    f"h265parse ! "
                    f"{decoder_name} ! "
                    f"videoconvert ! "
                    f"video/x-raw,format=BGR ! "
                    f"appsink name=rgb_sink_{cam_idx} emit-signals=true sync=false max-buffers=1 drop=true"
                )
                
                bin = Gst.parse_bin_from_description(elements_str, True)
                self.rgb_pipeline.add(bin)
                
                # Link demux pad to bin
                sink_pad = bin.get_static_pad('sink')
                if pad.link(sink_pad) == Gst.PadLinkReturn.OK:
                    bin.sync_state_with_parent()
                    
                    # Connect appsink callback
                    appsink = bin.get_by_name(f'rgb_sink_{cam_idx}')
                    if appsink:
                        appsink.connect('new-sample', self._on_rgb_sample)
                        self.rgb_sinks[cam_idx] = appsink
                        
                        # Add probe to appsink to monitor data flow
                        appsink_pad = appsink.get_static_pad('sink')
                        if appsink_pad:
                            def rgb_appsink_probe(pad, info, cam_idx=cam_idx):
                                if not hasattr(self, f'_rgb_appsink_probe_count_{cam_idx}'):
                                    setattr(self, f'_rgb_appsink_probe_count_{cam_idx}', 0)
                                count = getattr(self, f'_rgb_appsink_probe_count_{cam_idx}')
                                count += 1
                                setattr(self, f'_rgb_appsink_probe_count_{cam_idx}', count)
                                if count <= 3:
                                    print(f"[DEBUG] RGB CAM{cam_idx} appsink received buffer #{count}")
                                return Gst.PadProbeReturn.OK
                            appsink_pad.add_probe(Gst.PadProbeType.BUFFER, rgb_appsink_probe)
                        
                        print(f"[INFO] RGB receiver connected: Camera {cam_idx}, PT={pt}")
                else:
                    print(f"[ERROR] Failed to link RGB pad for camera {cam_idx}")
            else:
                # Create Depth decoder chain with proper RTP caps
                elements_str = (
                    f"capsfilter caps=\"application/x-rtp,media=video,clock-rate=90000,encoding-name=H265,payload={pt}\" ! "
                    f"queue ! "
                    f"rtph265depay ! "
                    f"h265parse ! "
                    f"{decoder_name} ! "
                    f"videoconvert ! "
                    f"video/x-raw,format=GRAY16_LE ! "
                    f"appsink name=depth_sink_{cam_idx} emit-signals=true sync=false max-buffers=2 drop=true"
                )
                
                bin = Gst.parse_bin_from_description(elements_str, True)
                self.rgb_pipeline.add(bin)
                
                # Link demux pad to bin
                sink_pad = bin.get_static_pad('sink')
                if pad.link(sink_pad) == Gst.PadLinkReturn.OK:
                    bin.sync_state_with_parent()
                    
                    # Connect appsink callback
                    appsink = bin.get_by_name(f'depth_sink_{cam_idx}')
                    if appsink:
                        appsink.connect('new-sample', self._on_depth_sample)
                        self.depth_sinks[cam_idx] = appsink
                        
                        # Add probe to appsink to monitor data flow
                        appsink_pad = appsink.get_static_pad('sink')
                        if appsink_pad:
                            def depth_appsink_probe(pad, info, cam_idx=cam_idx):
                                if not hasattr(self, f'_depth_appsink_probe_count_{cam_idx}'):
                                    setattr(self, f'_depth_appsink_probe_count_{cam_idx}', 0)
                                count = getattr(self, f'_depth_appsink_probe_count_{cam_idx}')
                                count += 1
                                setattr(self, f'_depth_appsink_probe_count_{cam_idx}', count)
                                if count <= 3:
                                    print(f"[DEBUG] Depth CAM{cam_idx} appsink received buffer #{count}")
                                return Gst.PadProbeReturn.OK
                            appsink_pad.add_probe(Gst.PadProbeType.BUFFER, depth_appsink_probe)
                        
                        print(f"[INFO] Depth receiver connected: Camera {cam_idx}, PT={pt}")
                else:
                    print(f"[ERROR] Failed to link Depth pad for camera {cam_idx}")
                    
        except Exception as e:
            print(f"[ERROR] Failed to handle demux pad: {e}")
            import traceback
            traceback.print_exc()
            
        except Exception as e:
            logger.error(f"Failed to create RTP demux receiver: {e}")
            print(f"[ERROR] Failed to create RTP demux receiver: {e}")
            import traceback
            traceback.print_exc()
            self.rgb_pipeline = None
    
    def _create_depth_receiver(self):
        """
        Depth receiver is now part of the RTP demux pipeline in _create_rgb_receiver().
        This method is a no-op to maintain API compatibility.
        """
        # Depth receivers are created dynamically in _on_demux_pad_added
        logger.info("Depth receiver will be created dynamically via rtpptdemux")
        print(f"[DEBUG] Depth receiver will be connected via pad-added signal")
        # Depth pipeline already created in _create_rgb_receiver
        pass
    
    def _on_depth_sample(self, appsink):
        """Callback for H.265 depth sample (GRAY16_LE format)"""
        try:
            sample = appsink.emit('pull-sample')
            if not sample:
                return Gst.FlowReturn.OK
            
            # Extract camera index from appsink name (format: "depth_sink_0", "depth_sink_1", etc.)
            appsink_name = appsink.get_name()
            camera_idx = 0  # default
            if '_' in appsink_name:
                try:
                    camera_idx = int(appsink_name.split('_')[-1])
                except (ValueError, IndexError):
                    pass
            
            buf = sample.get_buffer()
            caps = sample.get_caps()
            
            # Get dimensions from caps
            struct = caps.get_structure(0)
            width = struct.get_value('width')
            height = struct.get_value('height')
            
            # Extract GRAY16_LE data
            result, mapinfo = buf.map(Gst.MapFlags.READ)
            if not result:
                return Gst.FlowReturn.OK
            
            # Convert to uint16 array (GRAY16_LE = 16-bit grayscale, tenths-of-mm)
            depth_uint16 = np.frombuffer(mapinfo.data, dtype=np.uint16).reshape(height, width).copy()
            buf.unmap(mapinfo)
            
            # Convert uint16 tenths-of-mm → float32 mm
            depth_mm = depth_uint16.astype(np.float32) / 10.0
            
            # Debug
            if not hasattr(self, '_depth_recv_count'):
                self._depth_recv_count = {}
            
            if camera_idx not in self._depth_recv_count:
                self._depth_recv_count[camera_idx] = 0
                if not hasattr(self, '_depth_recv_start_time'):
                    self._depth_recv_start_time = {}
                self._depth_recv_start_time[camera_idx] = None
            
            self._depth_recv_count[camera_idx] += 1
            
            # FPS monitoring
            import time
            current_time = time.time()
            if self._depth_recv_start_time[camera_idx] is None:
                self._depth_recv_start_time[camera_idx] = current_time
            
            elapsed = current_time - self._depth_recv_start_time[camera_idx]
            fps = self._depth_recv_count[camera_idx] / elapsed if elapsed > 0 else 0
            
            if self._depth_recv_count[camera_idx] <= 5:
                valid_depth = depth_mm[depth_mm > 0]
                if valid_depth.size > 0:
                    print(f"[DEBUG] Depth frame CAM{camera_idx} #{self._depth_recv_count[camera_idx]}: {width}x{height} H.265, "
                          f"range: {valid_depth.min():.1f}mm-{valid_depth.max():.1f}mm, FPS: {fps:.1f}")
                else:
                    print(f"[DEBUG] Depth frame CAM{camera_idx} #{self._depth_recv_count[camera_idx]}: {width}x{height} H.265, NO VALID DEPTH, FPS: {fps:.1f}")
            elif self._depth_recv_count[camera_idx] % 60 == 0:
                print(f"[INFO] Depth CAM{camera_idx} H.265 streaming: {self._depth_recv_count[camera_idx]} frames, avg FPS: {fps:.1f}")
            
            # Call user callback with camera index
            if self.depth_callback:
                self.depth_callback(depth_mm, camera_idx)
            
            return Gst.FlowReturn.OK
            
        except Exception as e:
            logger.error(f"Error in depth H.265 callback: {e}")
            print(f"[ERROR] Error in depth H.265 callback: {e}")
            import traceback
            traceback.print_exc()
            return Gst.FlowReturn.ERROR
    
    def _on_rgb_sample(self, appsink):
        """Callback for new RGB sample"""
        sample = appsink.emit('pull-sample')
        if sample is None:
            print("[DEBUG] _on_rgb_sample: No sample available")
            return Gst.FlowReturn.OK
        
        try:
            # Extract camera index from appsink name (format: "rgb_sink_0", "rgb_sink_1", etc.)
            appsink_name = appsink.get_name()
            camera_idx = 0  # default
            if '_' in appsink_name:
                try:
                    camera_idx = int(appsink_name.split('_')[-1])
                except (ValueError, IndexError):
                    pass
            
            buf = sample.get_buffer()
            caps = sample.get_caps()
            
            # Get dimensions from caps
            struct = caps.get_structure(0)
            width = struct.get_value('width')
            height = struct.get_value('height')
            
            # Extract data
            success, map_info = buf.map(Gst.MapFlags.READ)
            if not success:
                print(f"[DEBUG] RGB CAM{camera_idx}: Failed to map buffer")
                return Gst.FlowReturn.OK
            
            # Convert to numpy array
            frame = np.ndarray(shape=(height, width, 3), dtype=np.uint8, buffer=map_info.data)
            frame = frame.copy()  # Make a copy since buffer will be unmapped
            
            buf.unmap(map_info)
            
            # Debug counter
            if not hasattr(self, '_rgb_recv_count'):
                self._rgb_recv_count = {}
            if camera_idx not in self._rgb_recv_count:
                self._rgb_recv_count[camera_idx] = 0
            self._rgb_recv_count[camera_idx] += 1
            
            if self._rgb_recv_count[camera_idx] <= 5:
                print(f"[DEBUG] RGB frame CAM{camera_idx} #{self._rgb_recv_count[camera_idx]} received: {width}x{height}")
            
            # Call callback with camera index
            if self.rgb_callback:
                self.rgb_callback(frame, camera_idx)
            else:
                print(f"[WARNING] RGB frame CAM{camera_idx} received but no callback set!")
            
        except Exception as e:
            logger.error(f"Error processing RGB sample: {e}")
            print(f"[ERROR] Error processing RGB sample: {e}")
        
        return Gst.FlowReturn.OK
    
    def __del__(self):
        """Cleanup"""
        self.stop()


class MultiCameraVideoReceiver:
    """
    Client-side multi-camera video receiver using GStreamer demuxing.
    Receives multiplexed MPEG-TS streams and demuxes to individual camera callbacks.
    
    Sends periodic heartbeats to server to maintain streaming connection.
    """
    
    def __init__(self, num_cameras, server_ip, rgb_port=5000, depth_port=5001,
                 rgb_callbacks: Optional[List[Callable]] = None,
                 depth_callbacks: Optional[List[Callable]] = None,
                 send_heartbeat=True, heartbeat_interval=2.0):
        """
        Initialize multi-camera video receiver.
        
        Args:
            num_cameras: Number of cameras in multiplexed stream
            server_ip: Server IP address
            rgb_port: RTP port for multiplexed RGB stream
            depth_port: RTP port for multiplexed depth stream
            rgb_callbacks: List of callback functions for RGB frames (one per camera)
            depth_callbacks: List of callback functions for depth frames (one per camera)
            send_heartbeat: Send periodic heartbeats to server
            heartbeat_interval: Seconds between heartbeats
        """
        self.num_cameras = num_cameras
        self.server_ip = server_ip
        self.rgb_port = rgb_port
        self.depth_port = depth_port
        
        self.rgb_callbacks = rgb_callbacks or [None] * num_cameras
        self.depth_callbacks = depth_callbacks or [None] * num_cameras
        
        self.rgb_pipeline = None
        self.depth_pipeline = None
        
        self.is_receiving = False
        
        self.main_loop = None
        self.loop_thread = None
        
        # Track which camera stream each pad corresponds to
        self.rgb_pad_to_camera = {}
        self.depth_pad_to_camera = {}
        
        # Heartbeat
        self.send_heartbeat = send_heartbeat
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_socket = None
        self.heartbeat_thread = None
        self.heartbeat_running = False
        
        logger.info(f"MultiCameraVideoReceiver initialized: {num_cameras} cameras from {server_ip}:{rgb_port}/{depth_port}")
        logger.info(f"Heartbeat: {'enabled' if send_heartbeat else 'disabled'}")
    
    def start(self):
        """Start receiving demultiplexed streams"""
        if self.is_receiving:
            logger.warning("Already receiving")
            return
        
        self._create_demux_rgb_receiver()
        self._create_demux_depth_receiver()
        
        # Start heartbeat if enabled
        if self.send_heartbeat:
            self._start_heartbeat()
        
        # Start GLib main loop
        self.main_loop = GLib.MainLoop()
        self.loop_thread = threading.Thread(target=self.main_loop.run, daemon=True)
        self.loop_thread.start()
        
        # Start pipelines
        if self.rgb_pipeline:
            self.rgb_pipeline.set_state(Gst.State.PLAYING)
        if self.depth_pipeline:
            self.depth_pipeline.set_state(Gst.State.PLAYING)
        
        self.is_receiving = True
        logger.info("Multi-camera video receiving started")
    
    def _start_heartbeat(self):
        """Start sending heartbeat messages to server"""
        try:
            self.heartbeat_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            heartbeat_port = self.rgb_port + 100
            
            self.heartbeat_running = True
            
            def heartbeat_loop():
                while self.heartbeat_running:
                    try:
                        self.heartbeat_socket.sendto(b"STREAMING_CLIENT", 
                                                     (self.server_ip, heartbeat_port))
                        time.sleep(self.heartbeat_interval)
                    except Exception as e:
                        if self.heartbeat_running:
                            logger.warning(f"Heartbeat send error: {e}")
            
            self.heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
            self.heartbeat_thread.start()
            
            logger.info(f"Sending heartbeats to {self.server_ip}:{heartbeat_port}")
        except Exception as e:
            logger.error(f"Failed to start heartbeat: {e}")
    
    def stop(self):
        """Stop receiving"""
        if not self.is_receiving:
            return
        
        self.is_receiving = False
        
        # Stop heartbeat
        if self.heartbeat_running:
            self.heartbeat_running = False
            if self.heartbeat_thread:
                self.heartbeat_thread.join(timeout=2.0)
        
        if self.heartbeat_socket:
            self.heartbeat_socket.close()
        
        if self.rgb_pipeline:
            self.rgb_pipeline.set_state(Gst.State.NULL)
        if self.depth_pipeline:
            self.depth_pipeline.set_state(Gst.State.NULL)
        
        if self.main_loop:
            self.main_loop.quit()
        
        if self.loop_thread:
            self.loop_thread.join(timeout=2.0)
        
        logger.info("Multi-camera receiving stopped")
    
    def _create_demux_rgb_receiver(self):
        """Create RGB receiver pipeline with demuxer"""
        try:
            decoder_name = GStreamerPlatform.get_decoder()
            
            # udpsrc -> rtpmp2tdepay -> tsdemux -> (dynamically link to decoders)
            pipeline_str = (
                f"udpsrc port={self.rgb_port} ! "
                f"application/x-rtp ! "
                f"rtpmp2tdepay ! "
                f"tsdemux name=demux "
            )
            
            logger.debug(f"Demux RGB receiver pipeline: {pipeline_str}")
            
            self.rgb_pipeline = Gst.parse_launch(pipeline_str)
            
            # Connect to pad-added signal for dynamic linking
            demux = self.rgb_pipeline.get_by_name('demux')
            demux.connect('pad-added', self._on_rgb_pad_added)
            
        except Exception as e:
            logger.error(f"Failed to create demux RGB receiver: {e}")
            self.rgb_pipeline = None
    
    def _create_demux_depth_receiver(self):
        """Create depth receiver pipeline with demuxer"""
        try:
            decoder_name = GStreamerPlatform.get_decoder()
            
            pipeline_str = (
                f"udpsrc port={self.depth_port} ! "
                f"application/x-rtp ! "
                f"rtpmp2tdepay ! "
                f"tsdemux name=demux "
            )
            
            logger.debug(f"Demux depth receiver pipeline: {pipeline_str}")
            
            self.depth_pipeline = Gst.parse_launch(pipeline_str)
            
            demux = self.depth_pipeline.get_by_name('demux')
            demux.connect('pad-added', self._on_depth_pad_added)
            
        except Exception as e:
            logger.error(f"Failed to create demux depth receiver: {e}")
            self.depth_pipeline = None
    
    def _on_rgb_pad_added(self, demux, pad):
        """Handle new demuxed RGB stream pad"""
        camera_idx = len(self.rgb_pad_to_camera)
        if camera_idx >= self.num_cameras:
            logger.warning(f"Unexpected RGB stream {camera_idx}")
            return
        
        self.rgb_pad_to_camera[pad.get_name()] = camera_idx
        
        decoder_name = GStreamerPlatform.get_decoder()
        
        # Create decoder branch for this stream
        try:
            decoder = Gst.ElementFactory.make(decoder_name, f"rgb_dec_{camera_idx}")
            videoconvert = Gst.ElementFactory.make("videoconvert", f"rgb_convert_{camera_idx}")
            appsink = Gst.ElementFactory.make("appsink", f"rgb_sink_{camera_idx}")
            
            appsink.set_property('emit-signals', True)
            appsink.set_property('sync', False)
            appsink.set_property('max-buffers', 1)
            appsink.set_property('drop', True)
            
            # Set caps for BGR output
            caps = Gst.Caps.from_string("video/x-raw,format=BGR")
            appsink.set_property('caps', caps)
            
            # Add to pipeline
            self.rgb_pipeline.add(decoder)
            self.rgb_pipeline.add(videoconvert)
            self.rgb_pipeline.add(appsink)
            
            # Link elements
            decoder.link(videoconvert)
            videoconvert.link(appsink)
            
            # Link demux pad to decoder
            sink_pad = decoder.get_static_pad('sink')
            pad.link(sink_pad)
            
            # Connect callback
            appsink.connect('new-sample', self._on_rgb_sample, camera_idx)
            
            # Sync state
            decoder.sync_state_with_parent()
            videoconvert.sync_state_with_parent()
            appsink.sync_state_with_parent()
            
            logger.info(f"RGB stream {camera_idx} connected")
            
        except Exception as e:
            logger.error(f"Failed to create RGB decoder branch {camera_idx}: {e}")
    
    def _on_depth_pad_added(self, demux, pad):
        """Handle new demuxed depth stream pad"""
        camera_idx = len(self.depth_pad_to_camera)
        if camera_idx >= self.num_cameras:
            logger.warning(f"Unexpected depth stream {camera_idx}")
            return
        
        self.depth_pad_to_camera[pad.get_name()] = camera_idx
        
        decoder_name = GStreamerPlatform.get_decoder()
        
        try:
            decoder = Gst.ElementFactory.make(decoder_name, f"depth_dec_{camera_idx}")
            videoconvert = Gst.ElementFactory.make("videoconvert", f"depth_convert_{camera_idx}")
            appsink = Gst.ElementFactory.make("appsink", f"depth_sink_{camera_idx}")
            
            appsink.set_property('emit-signals', True)
            appsink.set_property('sync', False)
            appsink.set_property('max-buffers', 1)
            appsink.set_property('drop', True)
            
            caps = Gst.Caps.from_string("video/x-raw,format=GRAY16_LE")
            appsink.set_property('caps', caps)
            
            self.depth_pipeline.add(decoder)
            self.depth_pipeline.add(videoconvert)
            self.depth_pipeline.add(appsink)
            
            decoder.link(videoconvert)
            videoconvert.link(appsink)
            
            sink_pad = decoder.get_static_pad('sink')
            pad.link(sink_pad)
            
            appsink.connect('new-sample', self._on_depth_sample, camera_idx)
            
            decoder.sync_state_with_parent()
            videoconvert.sync_state_with_parent()
            appsink.sync_state_with_parent()
            
            logger.info(f"Depth stream {camera_idx} connected")
            
        except Exception as e:
            logger.error(f"Failed to create depth decoder branch {camera_idx}: {e}")
    
    def _on_rgb_sample(self, appsink, camera_idx):
        """Callback for RGB sample from specific camera"""
        sample = appsink.emit('pull-sample')
        if sample is None:
            return Gst.FlowReturn.OK
        
        try:
            buf = sample.get_buffer()
            caps = sample.get_caps()
            
            struct = caps.get_structure(0)
            width = struct.get_value('width')
            height = struct.get_value('height')
            
            success, map_info = buf.map(Gst.MapFlags.READ)
            if not success:
                return Gst.FlowReturn.OK
            
            frame = np.ndarray(shape=(height, width, 3), dtype=np.uint8, buffer=map_info.data)
            frame = frame.copy()
            
            buf.unmap(map_info)
            
            if camera_idx < len(self.rgb_callbacks) and self.rgb_callbacks[camera_idx]:
                self.rgb_callbacks[camera_idx](frame)
            
        except Exception as e:
            logger.error(f"Error processing RGB sample from camera {camera_idx}: {e}")
        
        return Gst.FlowReturn.OK
    
    def _on_depth_sample(self, appsink, camera_idx):
        """Callback for depth sample from specific camera"""
        sample = appsink.emit('pull-sample')
        if sample is None:
            return Gst.FlowReturn.OK
        
        try:
            buf = sample.get_buffer()
            caps = sample.get_caps()
            
            struct = caps.get_structure(0)
            width = struct.get_value('width')
            height = struct.get_value('height')
            
            success, map_info = buf.map(Gst.MapFlags.READ)
            if not success:
                return Gst.FlowReturn.OK
            
            frame = np.ndarray(shape=(height, width), dtype=np.uint16, buffer=map_info.data)
            frame = frame.copy()
            
            buf.unmap(map_info)
            
            if camera_idx < len(self.depth_callbacks) and self.depth_callbacks[camera_idx]:
                self.depth_callbacks[camera_idx](frame)
            
        except Exception as e:
            logger.error(f"Error processing depth sample from camera {camera_idx}: {e}")
        
        return Gst.FlowReturn.OK
    
    def __del__(self):
        """Cleanup"""
        self.stop()


# Convenience functions
def create_streamer(host='0.0.0.0', rgb_port=5000, depth_port=5001, **kwargs):
    """Create and return a VideoStreamer instance (single camera)"""
    return VideoStreamer(host=host, rgb_port=rgb_port, depth_port=depth_port, **kwargs)


def create_multi_camera_streamer(num_cameras, host='0.0.0.0', stream_port=5000, **kwargs):
    """
    Create and return a MultiCameraVideoStreamer instance with RTP multiplexing.
    
    Args:
        num_cameras: Number of cameras to multiplex into single stream
        host: Host address to bind to
        stream_port: Single UDP port for ALL streams (RGB+Depth from all cameras)
        **kwargs: Additional arguments (camera_width, camera_height, framerate)
    
    Returns:
        MultiCameraVideoStreamer instance
    """
    return MultiCameraVideoStreamer(
        num_cameras=num_cameras,
        host=host,
        stream_port=stream_port,
        **kwargs
    )


def create_receiver(server_ip, rgb_port=5000, depth_port=5001, **kwargs):
    """Create and return a VideoReceiver instance (single camera)"""
    return VideoReceiver(server_ip=server_ip, rgb_port=rgb_port, depth_port=depth_port, **kwargs)


def create_multi_camera_receiver(num_cameras, server_ip, rgb_port=5000, depth_port=5001, **kwargs):
    """
    Create and return a MultiCameraVideoReceiver instance with demultiplexing.
    
    Args:
        num_cameras: Number of cameras in multiplexed stream
        server_ip: Server IP address
        rgb_port: RTP port for multiplexed RGB stream
        depth_port: RTP port for multiplexed depth stream
        **kwargs: Additional arguments (rgb_callbacks, depth_callbacks)
    
    Returns:
        MultiCameraVideoReceiver instance
    """
    return MultiCameraVideoReceiver(
        num_cameras=num_cameras,
        server_ip=server_ip,
        rgb_port=rgb_port,
        depth_port=depth_port,
        **kwargs
    )

