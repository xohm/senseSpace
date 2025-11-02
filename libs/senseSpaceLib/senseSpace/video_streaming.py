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
    
    @staticmethod
    def get_encoder(encoder_type='rgb'):
        """
        Get platform-specific H.265 encoder element.
        
        Args:
            encoder_type: 'rgb' for standard H.265, 'depth' for lossless H.265
        
        Returns:
            tuple: (encoder_name, properties_dict)
        """
        system = platform.system()
        
        # Lossless encoding for depth
        lossless_props = {'tune': 'lossless', 'speed-preset': 'fast'}
        
        # Standard encoding for RGB
        rgb_props = {'speed-preset': 'fast', 'bitrate': 4000}  # 4 Mbps
        
        props = lossless_props if encoder_type == 'depth' else rgb_props
        
        if system == 'Linux':
            # Try NVENC (NVIDIA), VAAPI (Intel/AMD), then software
            # NVENC: nvh265enc
            # VAAPI: vaapih265enc
            # Software: x265enc
            return 'nvh265enc', {'preset': 'low-latency-hq', 'bitrate': props.get('bitrate', 4000)}
        
        elif system == 'Windows':
            # Try NVENC, then Media Foundation, then software
            return 'nvh265enc', {'preset': 'low-latency-hq', 'bitrate': props.get('bitrate', 4000)}
        
        elif system == 'Darwin':  # macOS
            # Try VideoToolbox (Apple hardware), then software
            return 'vtenc_h265', {'bitrate': props.get('bitrate', 4000)}
        
        # Fallback to software encoder
        return 'x265enc', props
    
    @staticmethod
    def get_decoder():
        """Get platform-specific H.265 decoder element"""
        system = platform.system()
        
        if system == 'Linux':
            # Try NVDEC, VAAPI, then software
            return 'nvh265dec'
        
        elif system == 'Windows':
            return 'nvh265dec'
        
        elif system == 'Darwin':
            return 'vtdec_h265'
        
        # Fallback
        return 'avdec_h265'


class MultiCameraVideoStreamer:
    """
    Multi-camera video streamer using GStreamer multiplexing.
    Encodes multiple camera streams and multiplexes them into single transport streams.
    Uses MPEG-TS container for efficient multi-stream multiplexing over RTP.
    
    Automatically starts/stops streaming based on active client count.
    """
    
    def __init__(self, num_cameras=1, host='0.0.0.0', stream_port=5000,
                 camera_width=1280, camera_height=720, framerate=30,
                 enable_client_detection=True, client_timeout=5.0):
        """
        Initialize multi-camera video streamer with single multiplexed stream.
        
        Args:
            num_cameras: Number of cameras to multiplex
            host: Host address to bind to
            stream_port: UDP port for single multiplexed MPEG-TS stream (RGB+Depth)
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
        
        # Create RGB/Depth pipelines
        self._create_simple_rgb_pipeline()
        self._create_simple_depth_pipeline()
        
        # Start pipeline
        if self.rgb_pipeline:
            self.rgb_pipeline.set_state(Gst.State.PLAYING)
        if self.depth_pipeline:
            self.depth_pipeline.set_state(Gst.State.PLAYING)
        
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
        
        if self.rgb_pipeline:
            self.rgb_pipeline.set_state(Gst.State.NULL)
        if self.depth_pipeline:
            self.depth_pipeline.set_state(Gst.State.NULL)
        
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
            watched_pipelines = [p for p in (self.rgb_pipeline, self.depth_pipeline) if p is not None]
            if message.src in watched_pipelines:
                old, new, pending = message.parse_state_changed()
                label = 'RGB' if message.src == self.rgb_pipeline else 'Depth'
                print(f"[DEBUG] Server {label} pipeline state: {old.value_nick} -> {new.value_nick}")
        return True
    
    def _create_multiplexed_pipeline(self):
        """Create single pipeline with interleaved RTP streams for RGB and depth"""
        try:
            encoder_name, encoder_props = GStreamerPlatform.get_encoder('rgb')
            props_str = " ".join([f"{k}={v}" for k, v in encoder_props.items()])
            
            # Create interleaved RTP pipeline
            pipeline_str = (
                f"appsrc name=rgb_src_0 format=time is-live=true do-timestamp=true "
                f"caps=video/x-raw,format=BGR,width={self.camera_width},height={self.camera_height},"
                f"framerate={self.framerate}/1 ! "
                f"queue max-size-buffers=2 leaky=downstream ! "
                f"videoconvert ! "
                f"{encoder_name} {props_str} ! "
                f"h265parse config-interval=-1 ! "
                f"rtph265pay pt=96 config-interval=-1 ! "
                f"funnel name=f ! "
                f"multiudpsink name=stream_sink sync=false async=false "
                f"appsrc name=depth_src_0 format=time is-live=true do-timestamp=true "
                f"caps=video/x-raw,format=GRAY16_LE,width={self.camera_width},height={self.camera_height},"
                f"framerate={self.framerate}/1 ! "
                f"queue max-size-buffers=2 leaky=downstream ! "
                f"videoconvert ! "
                f"video/x-raw,format=I420 ! "
                f"{encoder_name} {props_str} ! "
                f"h265parse config-interval=-1 ! "
                f"rtph265pay pt=97 config-interval=-1 ! "
                f"f."
            )
            
            logger.debug(f"Interleaved RTP pipeline: {pipeline_str}")
            print(f"[DEBUG] Creating interleaved RTP pipeline...")
            print(f"[DEBUG] Pipeline: {pipeline_str}")
            
            self.pipeline = Gst.parse_launch(pipeline_str)
            
            if not self.pipeline:
                raise Exception("Failed to parse pipeline")
            
            # Store all appsrc elements
            self.rgb_appsrcs = []
            self.depth_appsrcs = []
            
            for cam_idx in range(self.num_cameras):
                rgb_src = self.pipeline.get_by_name(f'rgb_src_{cam_idx}')
                if not rgb_src:
                    raise Exception(f"Failed to get rgb_src_{cam_idx}")
                rgb_src.set_property('format', Gst.Format.TIME)
                self.rgb_appsrcs.append(rgb_src)
                
                depth_src = self.pipeline.get_by_name(f'depth_src_{cam_idx}')
                if not depth_src:
                    raise Exception(f"Failed to get depth_src_{cam_idx}")
                depth_src.set_property('format', Gst.Format.TIME)
                self.depth_appsrcs.append(depth_src)
            
            # Store multiudpsink for adding clients later
            self.stream_sink = self.pipeline.get_by_name('stream_sink')
            if not self.stream_sink:
                raise Exception("Failed to get stream_sink")
            
            # Add bus message handler to catch errors
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect('message', self._on_server_bus_message)
            
            print(f"[DEBUG] Created {len(self.rgb_appsrcs)} RGB appsrcs and {len(self.depth_appsrcs)} depth appsrcs")
            logger.info(f"Created multiplexed pipeline with {self.num_cameras} camera(s)")
            print(f"[INFO] Single MPEG-TS multiplexed pipeline created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create multiplexed pipeline: {e}")
            print(f"[ERROR] Failed to create multiplexed pipeline: {e}")
            import traceback
            traceback.print_exc()
            self.pipeline = None
    
    def _create_simple_rgb_pipeline(self):
        """Create simple single-camera RGB pipeline without muxing"""
        try:
            encoder_name, encoder_props = GStreamerPlatform.get_encoder('rgb')
            
            props_str = " ".join([f"{k}={v}" for k, v in encoder_props.items()])
            
            # Convert framerate to integer for proper caps format
            framerate_int = int(self.framerate)
            
            # Use udpsink with localhost for testing (simpler than multiudpsink)
            pipeline_str = (
                f"appsrc name=rgb_src_0 format=time is-live=true do-timestamp=true "
                f"caps=video/x-raw,format=BGR,width={self.camera_width},height={self.camera_height},"
                f"framerate={framerate_int}/1 ! "
                f"queue max-size-buffers=2 leaky=downstream ! "
                f"videoconvert ! "
                f"video/x-raw,format=NV12,width={self.camera_width},height={self.camera_height},"
                f"framerate={framerate_int}/1 ! "
                f"{encoder_name} {props_str} ! "
                f"h265parse ! "
                f"rtph265pay config-interval=1 pt=96 ! "
                f"udpsink host=127.0.0.1 port={self.stream_port} sync=false async=false"
            )
            
            logger.debug(f"Simple RGB pipeline: {pipeline_str}")
            print(f"[DEBUG] Simple RGB pipeline with udpsink")
            
            self.rgb_pipeline = Gst.parse_launch(pipeline_str)
            appsrc = self.rgb_pipeline.get_by_name('rgb_src_0')
            appsrc.set_property('format', Gst.Format.TIME)
            self.rgb_appsrcs = [appsrc]
            
            # No sink reference needed for udpsink (fixed destination)
            self.rgb_sink = None

            if self.rgb_pipeline:
                bus = self.rgb_pipeline.get_bus()
                bus.add_signal_watch()
                bus.connect('message', self._on_server_bus_message)
            
        except Exception as e:
            logger.error(f"Failed to create simple RGB pipeline: {e}")
            print(f"[ERROR] Failed to create simple RGB pipeline: {e}")
            self.rgb_pipeline = None
    
    def _create_simple_depth_pipeline(self):
        """Create Zstd-compressed depth streaming over RTP
        
        OPTIMIZED for ZED camera depth precision:
        
        Pipeline: float32 mm → uint16 tenths-mm → Zstd compress → RTP → UDP
        
        Why uint16 tenths-of-millimeters?
        - ZED configured with UNIT.MILLIMETER (values already in mm)
        - ZED sensor precision: ~1-3mm (0.1mm is 10x finer = lossless)
        - uint16 range: 0-6553.5 mm (0-6.5 meters) with 0.1mm precision
        - Data size: 2 bytes/pixel (vs 4 bytes for float32)
        - Zstd compression: ~60-70x typical on depth data
        - Final bandwidth: ~8x smaller than float32
        - Quality: LOSSLESS within sensor noise floor
        
        Why RTP?
        - Automatic packet fragmentation (no manual chunking needed)
        - Handles large frames (>65KB compressed)
        - Built-in sequencing and reassembly
        - Same infrastructure as H.265 RGB stream
        
        Pipeline: appsrc → rtpgstpay pt=97 → udpsink
        """
        try:
            depth_port = self.stream_port + 1
            
            # Use Zstd compression (LOSSLESS within sensor limits)
            # No H.265 encoding - avoids YUV conversion, better compression
            if not ZSTD_AVAILABLE:
                logger.error("zstandard module required for depth streaming")
            
            # Create H.265 pipeline for depth (16-bit grayscale → YUV444 for lossless encoding)
            # Encode depth as GRAY16_LE, convert to I420_10LE (10-bit YUV), use H.265 lossless/near-lossless
            # Note: NVENC supports 10-bit better than 16-bit, so we'll use I420_10LE
            # This gives us 0-1023 range for depth values (we'll scale from uint16)
            encoder_name, encoder_props = GStreamerPlatform.get_encoder('depth')
            
            # For lossless/near-lossless depth encoding:
            # - Use constqp mode with qp-const-i/p/b=0 for lossless (or 1-5 for near-lossless)
            # - Use preset=lossless or preset=losslesshp
            # - gop-size=1 for all-intra (reduces latency)
            # - Y444_16LE preserves 16-bit depth precision (I420_10LE is 4:2:0 subsampled)
            
            if 'nvh265enc' in encoder_name:
                # NVIDIA hardware encoder - use lossless preset with constqp
                # Note: This GStreamer version uses qp-const-i/p/b (not qp)
                lossless_props = "preset=lossless rc-mode=constqp qp-const-i=0 qp-const-p=0 qp-const-b=0 gop-size=1"
                output_format = "Y444_16LE"
            else:
                # Software encoder fallback - use very high quality
                lossless_props = "speed-preset=veryslow tune=ssim qp-min=0 qp-max=5"
                output_format = "I420_10LE"
            
            # Build encoder properties
            encoder_props_str = ' '.join([f"{k}={v}" for k, v in encoder_props.items()])
            
            # Pipeline: GRAY16_LE → videoconvert → Y444_16LE (16-bit) → H.265 lossless → RTP
            pipeline_str = (
                f"appsrc name=depth_src_0 format=time is-live=true do-timestamp=true "
                f"caps=video/x-raw,format=GRAY16_LE,width={self.camera_width},height={self.camera_height},"
                f"framerate={int(self.framerate)}/1 ! "
                f"videoconvert dither=none ! video/x-raw,format={output_format} ! "
                f"{encoder_name} {encoder_props_str} {lossless_props} ! "
                f"h265parse ! "
                f"rtph265pay pt=98 config-interval=1 mtu=1400 ! "
                f"udpsink host=127.0.0.1 port={depth_port} sync=false async=false"
            )
            
            logger.debug(f"Depth H.265 pipeline: {pipeline_str}")
            print(f"[DEBUG] Depth streaming via H.265 (GRAY16→{output_format}, lossless) on port {depth_port}")
            
            self.depth_pipeline = Gst.parse_launch(pipeline_str)
            appsrc = self.depth_pipeline.get_by_name('depth_src_0')
            appsrc.set_property('format', Gst.Format.TIME)
            self.depth_appsrcs = [appsrc]
            
            self.depth_sink = None
            
            if self.depth_pipeline:
                bus = self.depth_pipeline.get_bus()
                bus.add_signal_watch()
                bus.connect('message', self._on_server_bus_message)
            
            logger.info(f"Depth streaming: H.265 10-bit lossless on port {depth_port}")
            print(f"[INFO] Depth streaming: H.265 10-bit lossless on port {depth_port}")
            
        except Exception as e:
            logger.error(f"Failed to create depth pipeline: {e}")
            print(f"[ERROR] Failed to create depth pipeline: {e}")
            self.depth_pipeline = None
    
    def _create_muxed_rgb_pipeline(self):
        """
        Create GStreamer pipeline with multiplexing for RGB streams.
        Uses mpegtsmux to multiplex multiple H.265 streams into MPEG-TS container.
        """
        try:
            encoder_name, encoder_props = GStreamerPlatform.get_encoder('rgb')
            
            # Build pipeline with multiple appsrc -> encode -> mux
            pipeline_desc = []
            
            # Create encoding branch for each camera
            for i in range(self.num_cameras):
                branch = (
                    f"appsrc name=rgb_src_{i} format=time is-live=true do-timestamp=true "
                    f"caps=video/x-raw,format=BGR,width={self.camera_width},height={self.camera_height},"
                    f"framerate={self.framerate}/1 ! "
                    f"queue max-size-buffers=2 leaky=downstream ! "
                    f"videoconvert ! "
                    f"{encoder_name} "
                )
                
                for key, val in encoder_props.items():
                    branch += f"{key}={val} "
                
                branch += (
                    f"! h265parse ! "
                    f"queue ! "
                    f"mux.sink_{i} "
                )
                
                pipeline_desc.append(branch)
            
            # Add muxer and network sink
            mux_part = (
                f"mpegtsmux name=mux ! "
                f"rtpmp2tpay ! "
                f"udpsink host={self.host} port={self.rgb_port} sync=false"
            )
            
            pipeline_str = " ".join(pipeline_desc) + " " + mux_part
            
            logger.debug(f"Muxed RGB pipeline: {pipeline_str}")
            
            self.rgb_pipeline = Gst.parse_launch(pipeline_str)
            
            # Get all appsrc elements
            self.rgb_appsrcs = []
            for i in range(self.num_cameras):
                appsrc = self.rgb_pipeline.get_by_name(f'rgb_src_{i}')
                appsrc.set_property('format', Gst.Format.TIME)
                self.rgb_appsrcs.append(appsrc)
            
        except Exception as e:
            logger.error(f"Failed to create muxed RGB pipeline: {e}")
            self.rgb_pipeline = None
    
    def _create_muxed_depth_pipeline(self):
        """
        Create GStreamer pipeline with multiplexing for depth streams.
        """
        try:
            encoder_name, encoder_props = GStreamerPlatform.get_encoder('depth')
            
            pipeline_desc = []
            
            for i in range(self.num_cameras):
                branch = (
                    f"appsrc name=depth_src_{i} format=time is-live=true do-timestamp=true "
                    f"caps=video/x-raw,format=GRAY16_LE,width={self.camera_width},height={self.camera_height},"
                    f"framerate={self.framerate}/1 ! "
                    f"queue max-size-buffers=2 leaky=downstream ! "
                    f"videoconvert ! "
                    f"{encoder_name} "
                )
                
                for key, val in encoder_props.items():
                    branch += f"{key}={val} "
                
                branch += (
                    f"! h265parse ! "
                    f"queue ! "
                    f"mux.sink_{i} "
                )
                
                pipeline_desc.append(branch)
            
            mux_part = (
                f"mpegtsmux name=mux ! "
                f"rtpmp2tpay ! "
                f"udpsink host={self.host} port={self.depth_port} sync=false"
            )
            
            pipeline_str = " ".join(pipeline_desc) + " " + mux_part
            
            logger.debug(f"Muxed depth pipeline: {pipeline_str}")
            
            self.depth_pipeline = Gst.parse_launch(pipeline_str)
            
            self.depth_appsrcs = []
            for i in range(self.num_cameras):
                appsrc = self.depth_pipeline.get_by_name(f'depth_src_{i}')
                appsrc.set_property('format', Gst.Format.TIME)
                self.depth_appsrcs.append(appsrc)
            
        except Exception as e:
            logger.error(f"Failed to create muxed depth pipeline: {e}")
            self.depth_pipeline = None
    
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
                # Print first few successful pushes only
                if not hasattr(self, '_rgb_push_count'):
                    self._rgb_push_count = 0
                self._rgb_push_count += 1
                if self._rgb_push_count <= 5:
                    print(f"[DEBUG] RGB frame {self._rgb_push_count} pushed successfully ({frame.shape})")
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
                f"udpsrc port={self.stream_port} caps=\"application/x-rtp\" ! "
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
        """Create GStreamer pipeline for RGB reception"""
        try:
            # Use NVIDIA hardware decoder for GPU acceleration
            decoder_name = GStreamerPlatform.get_decoder()
            
            # udpsrc -> rtph265depay -> h265parse -> nvh265dec (GPU) -> videoconvert -> appsink
            # h265parse converts stream-format from hvc1 (RTP) to byte-stream (nvh265dec requirement)
            # Caps MUST match server: payload=96, clock-rate=90000, encoding-name=H265
            pipeline_str = (
                f"udpsrc port={self.stream_port} "
                f"caps=\"application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)H265,payload=(int)96\" ! "
                f"rtph265depay ! "
                f"h265parse ! "
                f"{decoder_name} ! "
                f"videoconvert ! "
                f"video/x-raw,format=(string)BGR ! "
                f"appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true"
            )
            
            logger.debug(f"RGB receiver pipeline: {pipeline_str}")
            print(f"[DEBUG] RGB receiver pipeline: {pipeline_str}")
            
            self.rgb_pipeline = Gst.parse_launch(pipeline_str)
            
            # Add bus message handler
            bus = self.rgb_pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect('message', self._on_bus_message)
            
            # Connect to new-sample signal
            appsink = self.rgb_pipeline.get_by_name('sink')
            appsink.connect('new-sample', self._on_rgb_sample)
            
            print(f"[INFO] RGB receiver pipeline created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create RGB receiver: {e}")
            print(f"[ERROR] Failed to create RGB receiver: {e}")
            self.rgb_pipeline = None
    
    def _create_depth_receiver(self):
        """Create H.265 RTP receiver for depth stream"""
        try:
            # Use NVIDIA hardware decoder for GPU acceleration
            decoder_name = GStreamerPlatform.get_decoder()
            
            # Create H.265 RTP pipeline for depth (16-bit lossless via Y444_16LE)
            # udpsrc → rtph265depay → h265parse → nvh265dec (GPU) → videoconvert → GRAY16_LE → appsink
            # h265parse converts stream-format from hvc1 (RTP) to byte-stream (nvh265dec requirement)
            pipeline_str = (
                f"udpsrc port={self.depth_port} "
                f"caps=\"application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)H265,payload=(int)98\" ! "
                f"rtph265depay ! "
                f"h265parse ! "
                f"{decoder_name} ! "
                f"videoconvert ! video/x-raw,format=GRAY16_LE ! "
                f"appsink name=depth_sink emit-signals=true sync=false max-buffers=2 drop=true"
            )
            
            logger.debug(f"Depth H.265 receiver pipeline: {pipeline_str}")
            print(f"[DEBUG] Depth H.265 receiver pipeline: {pipeline_str}")
            
            self.depth_pipeline = Gst.parse_launch(pipeline_str)
            
            # Add bus message handler
            bus = self.depth_pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect('message', self._on_bus_message)
            
            # Connect depth appsink callback
            depth_sink = self.depth_pipeline.get_by_name('depth_sink')
            if depth_sink:
                depth_sink.connect('new-sample', self._on_depth_sample)
                print(f"[INFO] Depth receiver created: H.265 RTP port {self.depth_port}")
            else:
                logger.error("Could not find depth_sink in pipeline")
                print(f"[ERROR] Could not find depth_sink in pipeline")
                self.depth_pipeline = None
                return
            
            logger.info(f"Depth receiver: H.265 16-bit lossless (Y444_16LE) on port {self.depth_port}")
            print(f"[INFO] Depth receiver: H.265 16-bit lossless (Y444_16LE) on port {self.depth_port}")
            
        except Exception as e:
            logger.error(f"Failed to create depth receiver: {e}")
            print(f"[ERROR] Failed to create depth receiver: {e}")
            import traceback
            traceback.print_exc()
            self.depth_pipeline = None
    
    def _on_depth_sample(self, appsink):
        """Callback for H.265 depth sample (GRAY16_LE format)"""
        try:
            sample = appsink.emit('pull-sample')
            if not sample:
                return Gst.FlowReturn.OK
            
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
                self._depth_recv_count = 0
                self._depth_recv_start_time = None
            
            self._depth_recv_count += 1
            
            # FPS monitoring
            import time
            current_time = time.time()
            if self._depth_recv_start_time is None:
                self._depth_recv_start_time = current_time
            
            elapsed = current_time - self._depth_recv_start_time
            fps = self._depth_recv_count / elapsed if elapsed > 0 else 0
            
            if self._depth_recv_count <= 5:
                valid_depth = depth_mm[depth_mm > 0]
                if valid_depth.size > 0:
                    print(f"[DEBUG] Depth frame {self._depth_recv_count}: {width}x{height} H.265, "
                          f"range: {valid_depth.min():.1f}mm-{valid_depth.max():.1f}mm, FPS: {fps:.1f}")
                else:
                    print(f"[DEBUG] Depth frame {self._depth_recv_count}: {width}x{height} H.265, NO VALID DEPTH, FPS: {fps:.1f}")
            elif self._depth_recv_count % 60 == 0:
                print(f"[INFO] Depth H.265 streaming: {self._depth_recv_count} frames, avg FPS: {fps:.1f}")
            
            # Call user callback
            if self.depth_callback:
                self.depth_callback(depth_mm)
            
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
            return Gst.FlowReturn.OK
        
        try:
            buf = sample.get_buffer()
            caps = sample.get_caps()
            
            # Get dimensions from caps
            struct = caps.get_structure(0)
            width = struct.get_value('width')
            height = struct.get_value('height')
            
            # Extract data
            success, map_info = buf.map(Gst.MapFlags.READ)
            if not success:
                return Gst.FlowReturn.OK
            
            # Convert to numpy array
            frame = np.ndarray(shape=(height, width, 3), dtype=np.uint8, buffer=map_info.data)
            frame = frame.copy()  # Make a copy since buffer will be unmapped
            
            buf.unmap(map_info)
            
            #print(f"[DEBUG] RGB frame received: {width}x{height}")
            
            # Call callback
            if self.rgb_callback:
                self.rgb_callback(frame)
            
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


def create_multi_camera_streamer(num_cameras, host='0.0.0.0', rgb_port=5000, depth_port=5001, **kwargs):
    """
    Create and return a MultiCameraVideoStreamer instance with multiplexing.
    
    Args:
        num_cameras: Number of cameras to multiplex into single stream
        host: Host address to bind to
        rgb_port: RTP port for multiplexed RGB stream (MPEG-TS)
        depth_port: RTP port for multiplexed depth stream (MPEG-TS)
        **kwargs: Additional arguments (camera_width, camera_height, framerate)
    
    Returns:
        MultiCameraVideoStreamer instance
    """
    return MultiCameraVideoStreamer(
        num_cameras=num_cameras,
        host=host,
        rgb_port=rgb_port,
        depth_port=depth_port,
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

