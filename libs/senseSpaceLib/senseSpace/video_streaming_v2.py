"""
Per-Camera Video Streaming (Version 2)

New architecture where each camera gets its own RTP stream with multiplexed RGB+Depth.
- Server sends unicast streams per client
- Client requests specific cameras with fps decimation factor
- Uses rtpbin for proper RTCP heartbeat and synchronization
- Backward compatible - doesn't break existing skeleton-only clients

Design:
- Each camera = 1 RTP session with 2 payload types (RGB=96+cam*2, Depth=97+cam*2)
- Client connects via TCP, requests cameras + fpsFactor
- Server creates per-client unicast streams
- RTCP provides automatic heartbeat/keepalive
"""

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GLib', '2.0')
from gi.repository import Gst, GLib

import numpy as np
import threading
import time
import socket
import json
import logging
from typing import Optional, Callable, List, Dict, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Initialize GStreamer
Gst.init(None)


@dataclass
class StreamRequest:
    """Client's stream request"""
    cameras: List[int]  # Which cameras to stream (e.g., [0, 1, 2])
    fps_factor: int = 1  # FPS decimation: 1=full fps, 2=half, 3=third, etc.
    
    def to_dict(self):
        return {
            'cameras': self.cameras,
            'fps_factor': self.fps_factor
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            cameras=data.get('cameras', []),
            fps_factor=data.get('fps_factor', 1)
        )


class PerCameraStreamerClient:
    """
    Manages streaming for a single client.
    Each camera gets its own RTP session with multiplexed RGB+Depth.
    """
    
    def __init__(self, client_id: str, client_ip: str, base_port: int,
                 num_cameras: int, camera_width: int, camera_height: int,
                 camera_fps: int, stream_request: StreamRequest):
        """
        Args:
            client_id: Unique client identifier
            client_ip: Client IP address for unicast
            base_port: Base UDP port (camera 0 uses base_port, camera 1 uses base_port+10, etc.)
            num_cameras: Total number of cameras available
            camera_width: Frame width
            camera_height: Frame height
            camera_fps: Original camera framerate
            stream_request: Client's stream request (which cameras, fps factor)
        """
        self.client_id = client_id
        self.client_ip = client_ip
        self.base_port = base_port
        self.num_cameras = num_cameras
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.stream_request = stream_request
        
        # Calculate effective fps after decimation
        self.effective_fps = camera_fps // stream_request.fps_factor
        
        # Pipelines per camera: {camera_idx: {'rgb': pipeline, 'depth': pipeline}}
        self.pipelines = {}
        self.appsrcs = {}  # {camera_idx: {'rgb': appsrc, 'depth': appsrc}}
        self.frame_counters = {}  # {camera_idx: frame_counter} - for fps decimation
        
        self.is_streaming = False
        
        print(f"[INFO] PerCameraStreamerClient created for {client_id} @ {client_ip}")
        print(f"[INFO] Requested cameras: {stream_request.cameras}, FPS factor: {stream_request.fps_factor}")
    
    def start(self):
        """Start streaming requested cameras to client"""
        if self.is_streaming:
            return
        
        # Create pipelines only for requested cameras
        for cam_idx in self.stream_request.cameras:
            if cam_idx >= self.num_cameras:
                logger.warning(f"Camera {cam_idx} requested but only {self.num_cameras} available")
                continue
            
            self._create_camera_pipeline(cam_idx)
        
        # Start all pipelines
        for cam_idx, pipes in self.pipelines.items():
            if pipes.get('pipeline'):
                pipes['pipeline'].set_state(Gst.State.PLAYING)
        
        self.is_streaming = True
        print(f"[INFO] Started streaming for client {self.client_id}")
    
    def stop(self):
        """Stop all streaming"""
        if not self.is_streaming:
            return
        
        self.is_streaming = False
        
        # Stop all pipelines
        print(f"[INFO] Stopping pipelines for client {self.client_id}...")
        for cam_idx, pipes in self.pipelines.items():
            try:
                if pipes.get('pipeline'):
                    pipes['pipeline'].set_state(Gst.State.NULL)
            except Exception as e:
                print(f"[WARNING] Error stopping pipeline for cam {cam_idx}: {e}")
        
        self.pipelines.clear()
        self.appsrcs.clear()
        
        print(f"[INFO] Stopped streaming for client {self.client_id}")
    
    def _create_camera_pipeline(self, camera_idx: int):
        """
        Create multiplexed RGB+Depth pipeline for one camera.
        Uses single rtpbin with two sessions (PT 96=RGB, PT 97=Depth).
        Both streams share the same RTP port for multiplexing with synchronized timestamps.
        
        Camera streams use sequential ports (2 ports per camera):
        - Camera 0: RTP=5000, RTCP=5001
        - Camera 1: RTP=5010, RTCP=5011
        - Camera 2: RTP=5020, RTCP=5021
        """
        rtp_port = self.base_port + (camera_idx * 10)
        rtcp_port = rtp_port + 1
        
        # Fixed payload types for all cameras: RGB=96, Depth=97
        rgb_pt = 96
        depth_pt = 97
        
        # Get hardware encoder
        from .video_streaming import GStreamerPlatform
        encoder_name, encoder_props = GStreamerPlatform.get_encoder()
        
        # Create SINGLE pipeline with multiplexed RGB+Depth using rtpbin
        # Both streams go through the same rtpbin, different sessions, same RTP port
        # Simple approach like v1: separate pipelines to same port
        # rtpptdemux on receiver will demultiplex by PT
        pipeline_str = (
            # RGB branch - PT 96, to same port
            f"appsrc name=src_rgb format=time is-live=true do-timestamp=true "
            f"caps=video/x-raw,format=BGR,width={self.camera_width},height={self.camera_height},"
            f"framerate=(fraction){int(self.effective_fps)}/1 ! "
            f"queue max-size-buffers=2 leaky=downstream ! "
            f"videoconvert ! video/x-raw,format=NV12 ! "
            f"{encoder_name} bitrate=3000 ! "
            f"h265parse config-interval=1 ! "
            f"rtph265pay pt={rgb_pt} config-interval=1 ! "
            f"udpsink host={self.client_ip} port={rtp_port} sync=false async=false "
            
            # Depth branch - PT 97, to SAME port (multiplexed by PT)
            f"appsrc name=src_depth format=time is-live=true do-timestamp=true "
            f"caps=video/x-raw,format=GRAY16_LE,width={self.camera_width},height={self.camera_height},"
            f"framerate=(fraction){int(self.effective_fps)}/1 ! "
            f"queue max-size-buffers=2 leaky=downstream ! "
            f"videoconvert ! "
            f"{encoder_name} bitrate=3000 ! "
            f"h265parse config-interval=1 ! "
            f"rtph265pay pt={depth_pt} config-interval=1 ! "
            f"udpsink host={self.client_ip} port={rtp_port} sync=false async=false"
        )
        
        print(f"[DEBUG] Camera {camera_idx} multiplexed pipeline: RTP={rtp_port} (PT {rgb_pt}/{depth_pt}), RTCP={rtcp_port}")
        
        try:
            pipeline = Gst.parse_launch(pipeline_str)
            
            # Add bus message handler to catch errors
            bus = pipeline.get_bus()
            bus.add_signal_watch()
            
            def on_bus_message(bus, message):
                t = message.type
                if t == Gst.MessageType.ERROR:
                    err, debug_info = message.parse_error()
                    print(f"[GSTREAMER ERROR] Camera {camera_idx}: {err.message}")
                    print(f"[GSTREAMER DEBUG] {debug_info}")
                elif t == Gst.MessageType.WARNING:
                    warn, debug_info = message.parse_warning()
                    print(f"[GSTREAMER WARNING] Camera {camera_idx}: {warn.message}")
                elif t == Gst.MessageType.EOS:
                    print(f"[GSTREAMER] Camera {camera_idx} End-Of-Stream")
            
            bus.connect('message', on_bus_message)
            
            rgb_appsrc = pipeline.get_by_name('src_rgb')
            depth_appsrc = pipeline.get_by_name('src_depth')
            
            if not rgb_appsrc or not depth_appsrc:
                print(f"[ERROR] Failed to get appsrcs for camera {camera_idx}")
                return
            
            rgb_appsrc.set_property('format', Gst.Format.TIME)
            depth_appsrc.set_property('format', Gst.Format.TIME)
            
            self.pipelines[camera_idx] = {'pipeline': pipeline, 'bus': bus}
            self.appsrcs[camera_idx] = {'rgb': rgb_appsrc, 'depth': depth_appsrc}
            self.frame_counters[camera_idx] = 0  # Initialize frame counter for this camera
            
            print(f"[INFO] Created multiplexed pipeline for camera {camera_idx}")
            
        except Exception as e:
            print(f"[ERROR] Failed to create camera {camera_idx} pipeline: {e}")
            import traceback
            traceback.print_exc()
    
    def push_frame(self, camera_idx: int, rgb_frame: np.ndarray, depth_frame: np.ndarray):
        """Push RGB and depth frames for a camera"""
        if not self.is_streaming or camera_idx not in self.appsrcs:
            return
        
        # Check pipeline state
        if camera_idx not in self.pipelines:
            logger.error(f"No pipeline for camera {camera_idx}")
            return
        
        pipeline = self.pipelines[camera_idx]['pipeline']
        state = pipeline.get_state(0)[1]  # Get current state without waiting
        if state != Gst.State.PLAYING:
            if self.frame_counters[camera_idx] % 100 == 0:
                logger.warning(f"Pipeline for cam {camera_idx} not in PLAYING state: {state}")
            return
        
        # Frame decimation: only push every Nth frame based on fps_factor
        self.frame_counters[camera_idx] += 1
        
        # Debug first few frames
        if self.frame_counters[camera_idx] <= 10:
            print(f"[DEBUG] Client {self.client_id} cam {camera_idx} frame {self.frame_counters[camera_idx]}, "
                  f"fps_factor={self.stream_request.fps_factor}, "
                  f"skip={(self.frame_counters[camera_idx] % self.stream_request.fps_factor != 0)}")
        
        if self.frame_counters[camera_idx] % self.stream_request.fps_factor != 0:
            return  # Skip this frame
        
        # Push RGB
        try:
            if rgb_frame.dtype != np.uint8:
                rgb_frame = rgb_frame.astype(np.uint8)
            
            data = rgb_frame.tobytes()
            buf = Gst.Buffer.new_wrapped(data)
            ret = self.appsrcs[camera_idx]['rgb'].emit('push-buffer', buf)
            
            if ret != Gst.FlowReturn.OK:
                if self.frame_counters[camera_idx] <= 10 or self.frame_counters[camera_idx] % 100 == 0:
                    logger.warning(f"RGB push failed for cam {camera_idx}: {ret}")
        except Exception as e:
            logger.error(f"Error pushing RGB frame cam {camera_idx}: {e}")
            import traceback
            traceback.print_exc()
        
        # Push Depth
        try:
            # Convert depth mm (float32) to tenths-of-mm (uint16) for GRAY16_LE
            if depth_frame.dtype == np.float32:
                depth_uint16 = (depth_frame * 10.0).astype(np.uint16)
            else:
                depth_uint16 = depth_frame.astype(np.uint16)
            
            data = depth_uint16.tobytes()
            buf = Gst.Buffer.new_wrapped(data)
            ret = self.appsrcs[camera_idx]['depth'].emit('push-buffer', buf)
            
            if ret != Gst.FlowReturn.OK:
                if self.frame_counters[camera_idx] <= 10 or self.frame_counters[camera_idx] % 100 == 0:
                    logger.warning(f"Depth push failed for cam {camera_idx}: {ret}")
        except Exception as e:
            logger.error(f"Error pushing depth frame cam {camera_idx}: {e}")
            import traceback
            traceback.print_exc()



class PerCameraVideoStreamer:
    """
    Server-side per-camera video streamer.
    Manages multiple clients, each with their own unicast streams.
    """
    
    def __init__(self, num_cameras: int, camera_width: int, camera_height: int,
                 camera_fps: int = 30, base_port: int = 5000):
        """
        Args:
            num_cameras: Number of cameras available
            camera_width: Frame width
            camera_height: Frame height  
            camera_fps: Original camera framerate
            base_port: Base UDP port for RTP streams
        """
        self.num_cameras = num_cameras
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.base_port = base_port
        
        # Active clients: {client_id: PerCameraStreamerClient}
        self.clients = {}
        self.clients_lock = threading.Lock()
        
        self._frame_count = 0  # Debug counter
        
        logger.info(f"PerCameraVideoStreamer initialized: {num_cameras} cameras, "
                   f"{camera_width}x{camera_height}@{camera_fps}fps")
    
    def add_client(self, client_id: str, client_ip: str, stream_request: StreamRequest) -> bool:
        """
        Add a client and start streaming to them.
        
        Args:
            client_id: Unique client identifier
            client_ip: Client IP address
            stream_request: Client's stream request
        
        Returns:
            True if client added successfully
        """
        with self.clients_lock:
            if client_id in self.clients:
                logger.warning(f"Client {client_id} already exists")
                return False
            
            # Validate request
            for cam_idx in stream_request.cameras:
                if cam_idx >= self.num_cameras:
                    logger.error(f"Client requested camera {cam_idx} but only {self.num_cameras} available")
                    return False
            
            if stream_request.fps_factor < 1:
                logger.error(f"Invalid fps_factor: {stream_request.fps_factor}")
                return False
            
            # Create client streamer
            client = PerCameraStreamerClient(
                client_id=client_id,
                client_ip=client_ip,
                base_port=self.base_port,
                num_cameras=self.num_cameras,
                camera_width=self.camera_width,
                camera_height=self.camera_height,
                camera_fps=self.camera_fps,
                stream_request=stream_request
            )
            
            client.start()
            self.clients[client_id] = client
            
            logger.info(f"Added client {client_id} ({client_ip}), streaming cameras: {stream_request.cameras}")
            return True
    
    def remove_client(self, client_id: str):
        """Remove a client and stop streaming to them"""
        with self.clients_lock:
            if client_id not in self.clients:
                logger.warning(f"Client {client_id} not found")
                return
            
            client = self.clients.pop(client_id)
            client.stop()
            
            logger.info(f"Removed client {client_id}")
    
    def push_frames(self, rgb_frames: List[np.ndarray], depth_frames: List[np.ndarray]):
        """
        Push frames from all cameras.
        
        Args:
            rgb_frames: List of RGB frames [cam0, cam1, cam2, ...]
            depth_frames: List of depth frames [cam0, cam1, cam2, ...]
        """
        with self.clients_lock:
            if len(self.clients) == 0:
                return  # No clients, skip silently
            
            # Debug first few calls
            self._frame_count += 1
            if self._frame_count <= 5:
                print(f"[DEBUG] PerCameraVideoStreamer.push_frames called (frame {self._frame_count}), "
                      f"clients={len(self.clients)}, rgb_frames={len(rgb_frames)}, depth_frames={len(depth_frames)}")
            
            for client_id, client in list(self.clients.items()):
                for cam_idx in client.stream_request.cameras:
                    if cam_idx < len(rgb_frames) and cam_idx < len(depth_frames):
                        client.push_frame(cam_idx, rgb_frames[cam_idx], depth_frames[cam_idx])
    
    def get_client_count(self) -> int:
        """Get number of active clients"""
        with self.clients_lock:
            return len(self.clients)
    
    def shutdown(self):
        """Shutdown all client streams"""
        with self.clients_lock:
            for client_id in list(self.clients.keys()):
                self.remove_client(client_id)
        
        logger.info("PerCameraVideoStreamer shutdown complete")


class PerCameraVideoReceiver:
    """
    Client-side per-camera video receiver.
    Receives RGB+Depth streams for requested cameras using rtpbin.
    """
    
    def __init__(self, server_ip: str, base_port: int, cameras: List[int],
                 rgb_callback: Optional[Callable] = None,
                 depth_callback: Optional[Callable] = None):
        """
        Args:
            server_ip: Server IP address
            base_port: Base UDP port (must match server)
            cameras: List of camera indices to receive
            rgb_callback: Callback for RGB frames: callback(frame, camera_idx)
            depth_callback: Callback for depth frames: callback(frame, camera_idx)
        """
        self.server_ip = server_ip
        self.base_port = base_port
        self.cameras = cameras
        self.rgb_callback = rgb_callback
        self.depth_callback = depth_callback
        
        # Pipelines per camera: {camera_idx: {'rgb': pipeline, 'depth': pipeline}}
        self.pipelines = {}
        self.is_receiving = False
        
        self.main_loop = None
        self.loop_thread = None
        
        logger.info(f"PerCameraVideoReceiver initialized for cameras {cameras}")
    
    def start(self):
        """Start receiving streams"""
        if self.is_receiving:
            return
        
        # Create pipelines for each requested camera
        for cam_idx in self.cameras:
            self._create_camera_receiver(cam_idx)
        
        # Start GLib main loop
        self.main_loop = GLib.MainLoop()
        self.loop_thread = threading.Thread(target=self.main_loop.run, daemon=True)
        self.loop_thread.start()
        
        # Start all pipelines
        for cam_idx, pipes in self.pipelines.items():
            if pipes.get('pipeline'):
                pipes['pipeline'].set_state(Gst.State.PLAYING)
        
        self.is_receiving = True
        logger.info("Started receiving video streams")
    
    def stop(self):
        """Stop receiving streams"""
        if not self.is_receiving:
            return
        
        self.is_receiving = False
        
        # Stop all pipelines
        for cam_idx, pipes in self.pipelines.items():
            if pipes.get('pipeline'):
                pipes['pipeline'].set_state(Gst.State.NULL)
        
        # Stop main loop
        if self.main_loop:
            self.main_loop.quit()
        
        if self.loop_thread:
            self.loop_thread.join(timeout=2.0)
        
        logger.info("Stopped receiving video streams")
    
    def _create_camera_receiver(self, camera_idx: int):
        """
        Create multiplexed RGB+Depth receiver for one camera.
        Receives both PT 96 (RGB) and PT 97 (Depth) from the same RTP port.
        """
        rtp_port = self.base_port + (camera_idx * 10)
        rtcp_port = rtp_port + 1
        
        # Fixed payload types: RGB=96, Depth=97
        rgb_pt = 96
        depth_pt = 97
        
        # Get hardware decoder
        from .video_streaming import GStreamerPlatform
        decoder_name = GStreamerPlatform.get_decoder()
        
        # Use rtpptdemux like v1 - much simpler than rtpbin!
        # Single udpsrc receives both PT 96 and PT 97, rtpptdemux separates them
        pipeline_str = (
            f"udpsrc port={rtp_port} "
            f'caps="application/x-rtp, media=(string)video, encoding-name=(string)H265, '
            f'clock-rate=(int)90000" ! '
            f"rtpptdemux name=demux "
            
            # RGB stream (PT 96)
            f"demux.src_{rgb_pt} ! queue ! rtph265depay ! h265parse ! {decoder_name} ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"appsink name=sink_rgb emit-signals=true sync=false max-buffers=1 drop=true "
            
            # Depth stream (PT 97)
            f"demux.src_{depth_pt} ! queue ! rtph265depay ! h265parse ! {decoder_name} ! "
            f"videoconvert ! video/x-raw,format=GRAY16_LE ! "
            f"appsink name=sink_depth emit-signals=true sync=false max-buffers=1 drop=true"
        )
        
        print(f"[DEBUG] Camera {camera_idx} receiver: RTP={rtp_port} (PT {rgb_pt}/{depth_pt}), RTCP={rtcp_port}")
        
        try:
            pipeline = Gst.parse_launch(pipeline_str)
            
            # Connect appsink callbacks
            rgb_sink = pipeline.get_by_name('sink_rgb')
            if rgb_sink:
                rgb_sink.connect('new-sample', self._on_rgb_sample, camera_idx)
            
            depth_sink = pipeline.get_by_name('sink_depth')
            if depth_sink:
                depth_sink.connect('new-sample', self._on_depth_sample, camera_idx)
            
            self.pipelines[camera_idx] = {'pipeline': pipeline}
            
            print(f"[INFO] Created receiver for camera {camera_idx}: ports={rtp_port}-{rtcp_port}")
            
        except Exception as e:
            print(f"[ERROR] Failed to create receiver for camera {camera_idx}: {e}")
            import traceback
            traceback.print_exc()
    
    
    def _on_rgb_sample(self, appsink, camera_idx):
        """RGB frame callback"""
        sample = appsink.emit('pull-sample')
        if not sample:
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
            
            frame = np.ndarray(shape=(height, width, 3), dtype=np.uint8, buffer=map_info.data).copy()
            buf.unmap(map_info)
            
            if self.rgb_callback:
                self.rgb_callback(frame, camera_idx)
            
        except Exception as e:
            logger.error(f"Error in RGB callback cam {camera_idx}: {e}")
        
        return Gst.FlowReturn.OK
    
    def _on_depth_sample(self, appsink, camera_idx):
        """Depth frame callback"""
        sample = appsink.emit('pull-sample')
        if not sample:
            return Gst.FlowReturn.OK
        
        try:
            buf = sample.get_buffer()
            caps = sample.get_caps()
            
            struct = caps.get_structure(0)
            width = struct.get_value('width')
            height = struct.get_value('height')
            
            result, mapinfo = buf.map(Gst.MapFlags.READ)
            if not result:
                return Gst.FlowReturn.OK
            
            # Convert GRAY16_LE (tenths-of-mm) to float32 (mm)
            depth_uint16 = np.frombuffer(mapinfo.data, dtype=np.uint16).reshape(height, width).copy()
            buf.unmap(mapinfo)
            
            depth_mm = depth_uint16.astype(np.float32) / 10.0
            
            if self.depth_callback:
                self.depth_callback(depth_mm, camera_idx)
            
        except Exception as e:
            logger.error(f"Error in depth callback cam {camera_idx}: {e}")
        
        return Gst.FlowReturn.OK
    
    def __del__(self):
        """Cleanup"""
        self.stop()
