#!/usr/bin/env python3
"""
SenseSpace Frame Recorder

Records and plays back skeleton tracking frames with optional point cloud data.
Uses streaming compression for memory efficiency and performance.
"""

import json
import time
import threading
import os
import struct
from typing import Optional, Callable, Dict, List, Tuple
import numpy as np

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False
    print("[WARNING] zstandard not available - recordings will not be compressed")

from .protocol import Frame


class FrameRecorder:
    """
    Records frames with optional point cloud data using streaming compression.
    
    File format (.ssrec):
    - Header: JSON metadata line
    - Frames: Stream of records (skeleton + optional point cloud)
    - Compressed with zstandard streaming for memory efficiency
    
    Each frame record:
    {
        "timestamp": float,
        "frame_number": int,
        "frame": Frame dict,
        "has_pointcloud": bool,
        "pointcloud_offset": int (file position of point cloud data, if present)
    }
    
    Point cloud data (if present):
    - Stored as binary: [num_people: uint32] + for each person: [person_id: int32][num_points: uint32][points: float32 array][colors: uint8 array]
    """
    
    def __init__(self, filepath: str, compression_level: int = 3, record_pointcloud: bool = False, 
                 record_video: bool = False):
        """
        Initialize frame recorder.
        
        Args:
            filepath: Path to save recording file (.ssrec extension recommended)
            compression_level: Zstandard compression level (1-22, default 3)
            record_pointcloud: Whether to record point cloud data (increases file size significantly)
            record_video: Whether to record H.265 video streams (RGB + depth)
        """
        self.filepath = filepath
        self.compression_level = compression_level
        self.record_pointcloud = record_pointcloud
        self.record_video = record_video
        self.recording = False
        self.frame_count = 0
        self.start_time = None
        self._file = None
        self._stream_writer = None
        self._lock = threading.Lock()
        
        # Statistics
        self.total_bytes_written = 0
        self.total_pointcloud_points = 0
        self.total_video_bytes = 0
        
        # Video recording state
        self.video_cameras = []  # List of camera metadata dicts
        self.video_buffers = {}  # {camera_idx: {'rgb': bytes, 'depth': bytes}}
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    def start(self):
        """Start recording with streaming compression"""
        with self._lock:
            if self.recording:
                print("[WARNING] Already recording")
                return False
            
            try:
                self._file = open(self.filepath, 'wb')
                
                if ZSTD_AVAILABLE:
                    # Use streaming compression for memory efficiency
                    compressor = zstd.ZstdCompressor(level=self.compression_level)
                    self._stream_writer = compressor.stream_writer(self._file, closefd=False)
                else:
                    self._stream_writer = self._file
                
                # Write header with metadata
                header = {
                    'version': '2.0',  # Updated version for video support
                    'record_pointcloud': self.record_pointcloud,
                    'record_video': self.record_video,
                    'video_cameras': self.video_cameras,  # Camera metadata (populated when video sources register)
                    'compression': 'zstd' if ZSTD_AVAILABLE else 'none',
                    'compression_level': self.compression_level if ZSTD_AVAILABLE else 0,
                    'start_time': time.time(),
                    # Placeholder values - will be updated in metadata line when recording stops
                    'duration': 0.0,
                    'total_frames': 0,
                    'framerate': 0.0
                }
                header_json = json.dumps(header, separators=(',', ':')) + '\n'
                self._stream_writer.write(header_json.encode('utf-8'))
                self._stream_writer.flush()
                
                self.recording = True
                self.frame_count = 0
                self.total_bytes_written = len(header_json)
                self.total_pointcloud_points = 0
                self.total_video_bytes = 0
                self.start_time = time.time()
                
                features = []
                if self.record_pointcloud:
                    features.append("point clouds")
                if self.record_video:
                    features.append("video")
                mode = f"with {', '.join(features)}" if features else "skeleton only"
                print(f"[RECORDER] Started recording ({mode}) to: {self.filepath}")
                return True
                
            except Exception as e:
                print(f"[ERROR] Failed to start recording: {e}")
                if self._stream_writer and self._stream_writer != self._file:
                    try:
                        self._stream_writer.close()
                    except:
                        pass
                if self._file:
                    self._file.close()
                    self._file = None
                return False
    
    def stop(self):
        """Stop recording and flush streams"""
        with self._lock:
            if not self.recording:
                return
            
            try:
                # Calculate final statistics
                duration = time.time() - self.start_time if self.start_time else 0
                framerate = self.frame_count / duration if duration > 0 else 0
                
                # Write metadata line with actual recording statistics
                # This line is written at the END of the file (after all frames)
                # During playback, it will be skipped (fails JSON parsing due to "META:" prefix)
                # It can be read separately by load_header() for displaying file info
                metadata = {
                    'type': 'recording_metadata',
                    'duration': round(duration, 2),
                    'total_frames': self.frame_count,
                    'framerate': round(framerate, 2),
                    'file_size_bytes': self.total_bytes_written
                }
                
                if self.record_pointcloud:
                    metadata['total_pointcloud_points'] = self.total_pointcloud_points
                
                if self.record_video:
                    metadata['total_video_bytes'] = self.total_video_bytes
                    metadata['num_cameras'] = len(self.video_cameras)
                
                # Write metadata line (prefixed so it's easy to find and skip during playback)
                metadata_line = 'META:' + json.dumps(metadata, separators=(',', ':')) + '\n'
                self._stream_writer.write(metadata_line.encode('utf-8'))
                
                # Flush and close stream writer
                if self._stream_writer:
                    self._stream_writer.flush()
                    if self._stream_writer != self._file:
                        self._stream_writer.close()
                
                if self._file:
                    self._file.close()
                
                file_size = os.path.getsize(self.filepath) / (1024 * 1024)  # MB
                
                print(f"[RECORDER] Stopped recording:")
                print(f"  Frames: {self.frame_count}")
                print(f"  Duration: {duration:.1f}s")
                print(f"  File size: {file_size:.2f} MB")
                if self.record_pointcloud:
                    print(f"  Total points: {self.total_pointcloud_points:,}")
                if self.record_video:
                    print(f"  Video data: {self.total_video_bytes / (1024*1024):.2f} MB")
                    print(f"  Cameras: {len(self.video_cameras)}")
                print(f"  Saved to: {self.filepath}")
                
            except Exception as e:
                print(f"[ERROR] Error stopping recording: {e}")
            finally:
                self.recording = False
                self._file = None
                self._stream_writer = None
    
    def record_frame(self, frame: Frame, pointcloud_data: Optional[List[Dict]] = None):
        """
        Record a single frame with optional point cloud and video data.
        
        Args:
            frame: Frame object to record
            pointcloud_data: Optional list of per-person point clouds
                            Each item: {"person_id": int, "points": ndarray(N,3), "colors": ndarray(N,3)}
        
        Note: Video data must be provided separately via record_video_data() before calling this method.
        """
        if not self.recording:
            return
        
        with self._lock:
            try:
                # Convert frame to dict
                frame_dict = frame.to_dict() if hasattr(frame, 'to_dict') else frame
                
                # Use current time for recording timestamp
                # NOTE: We use time.time() here instead of frame.timestamp because:
                # 1. Frame timestamp from ZED may not be precise (often rounded to seconds)
                # 2. Network delays mean frame.timestamp != when we receive it
                # 3. time.time() gives us accurate inter-frame timing for playback
                record_timestamp = time.time()
                
                # Prepare frame record
                record = {
                    'timestamp': record_timestamp,
                    'frame_number': self.frame_count,
                    'frame': frame_dict,
                    'has_pointcloud': self.record_pointcloud and pointcloud_data is not None and len(pointcloud_data) > 0,
                    'has_video': self.record_video and len(self.video_buffers) > 0
                }
                
                # Write frame metadata as JSON line
                json_line = json.dumps(record, separators=(',', ':')) + '\n'
                self._stream_writer.write(json_line.encode('utf-8'))
                self.total_bytes_written += len(json_line)
                
                # Write point cloud data as binary if present
                if record['has_pointcloud']:
                    pc_bytes = self._serialize_pointcloud(pointcloud_data)
                    self._stream_writer.write(pc_bytes)
                    self.total_bytes_written += len(pc_bytes)
                    
                    # Update statistics
                    for pc in pointcloud_data:
                        if 'points' in pc:
                            self.total_pointcloud_points += len(pc['points'])
                
                # Write video data as binary if present
                if record['has_video']:
                    video_bytes = self._serialize_video()
                    self._stream_writer.write(video_bytes)
                    self.total_bytes_written += len(video_bytes)
                    self.total_video_bytes += len(video_bytes)
                    # Clear video buffers for next frame
                    self.video_buffers = {}
                
                # Flush every 10 frames to ensure data is written
                if self.frame_count % 10 == 0:
                    self._stream_writer.flush()
                
                self.frame_count += 1
                
            except Exception as e:
                print(f"[ERROR] Failed to record frame: {e}")
                import traceback
                traceback.print_exc()
    
    def _serialize_pointcloud(self, pointcloud_data: List[Dict]) -> bytes:
        """
        Serialize point cloud data to binary format for efficient storage.
        
        Format: [num_people: uint32]
                For each person:
                  [person_id: int32]
                  [num_points: uint32]
                  [points: float32 array (num_points * 3)]
                  [colors: uint8 array (num_points * 3)]
        """
        chunks = []
        
        # Number of people
        chunks.append(struct.pack('<I', len(pointcloud_data)))
        
        for pc in pointcloud_data:
            person_id = pc.get('person_id', 0)
            points = pc.get('points')
            colors = pc.get('colors')
            
            if points is None or len(points) == 0:
                continue
            
            # Ensure numpy arrays
            if not isinstance(points, np.ndarray):
                points = np.array(points, dtype=np.float32)
            if not isinstance(colors, np.ndarray):
                colors = np.array(colors, dtype=np.uint8)
            
            # Convert to correct dtypes
            points = points.astype(np.float32)
            colors = colors.astype(np.uint8)
            
            num_points = len(points)
            
            # Write person header
            chunks.append(struct.pack('<i', person_id))  # person_id (int32)
            chunks.append(struct.pack('<I', num_points))  # num_points (uint32)
            
            # Write points and colors as binary
            chunks.append(points.tobytes())  # float32 array
            chunks.append(colors.tobytes())  # uint8 array
        
        return b''.join(chunks)
    
    def register_video_camera(self, camera_idx: int, width: int, height: int, fps: int, codec: str = 'h265'):
        """
        Register a video camera for recording.
        Should be called before starting recording or shortly after.
        
        Args:
            camera_idx: Camera index (0, 1, 2, ...)
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second
            codec: Video codec (default: 'h265')
        """
        camera_info = {
            'camera_idx': camera_idx,
            'width': width,
            'height': height,
            'fps': fps,
            'codec': codec
        }
        
        # Check if camera already registered
        for cam in self.video_cameras:
            if cam['camera_idx'] == camera_idx:
                # Update existing
                cam.update(camera_info)
                return
        
        # Add new camera
        self.video_cameras.append(camera_info)
        print(f"[RECORDER] Registered video camera {camera_idx}: {width}x{height} @ {fps}fps ({codec})")
    
    def record_video_data(self, camera_idx: int, rgb_nal_units: bytes, depth_nal_units: bytes):
        """
        Store H.265 NAL units for the current frame.
        Must be called before record_frame() to include video in that frame.
        
        Args:
            camera_idx: Camera index
            rgb_nal_units: Raw H.265 NAL units for RGB stream
            depth_nal_units: Raw H.265 NAL units for depth stream
        """
        if not self.recording or not self.record_video:
            return
        
        self.video_buffers[camera_idx] = {
            'rgb': rgb_nal_units,
            'depth': depth_nal_units
        }
    
    def _serialize_video(self) -> bytes:
        """
        Serialize video data to binary format for efficient storage.
        
        Format: [num_cameras: uint32]
                For each camera:
                  [camera_idx: uint32]
                  [rgb_size: uint32]
                  [depth_size: uint32]
                  [rgb_data: bytes]
                  [depth_data: bytes]
        """
        chunks = []
        
        # Number of cameras with video data
        chunks.append(struct.pack('<I', len(self.video_buffers)))
        
        for camera_idx in sorted(self.video_buffers.keys()):
            video_data = self.video_buffers[camera_idx]
            rgb_data = video_data.get('rgb', b'')
            depth_data = video_data.get('depth', b'')
            
            # Write camera header
            chunks.append(struct.pack('<I', camera_idx))       # camera_idx (uint32)
            chunks.append(struct.pack('<I', len(rgb_data)))    # rgb_size (uint32)
            chunks.append(struct.pack('<I', len(depth_data)))  # depth_size (uint32)
            
            # Write video data
            chunks.append(rgb_data)    # RGB H.265 NAL units
            chunks.append(depth_data)  # Depth H.265 NAL units
        
        return b''.join(chunks)
    
    def is_recording(self) -> bool:
        """Check if currently recording"""
        return self.recording


class FramePlayer:
    """
    Plays back recorded frames with optional point cloud data using streaming decompression.
    Memory efficient - doesn't load entire file into memory at once.
    """
    
    def __init__(self, filepath: str, loop: bool = False, speed: float = 1.0):
        """
        Initialize frame player with streaming support.
        
        Args:
            filepath: Path to recording file
            loop: Whether to loop playback
            speed: Playback speed multiplier (1.0 = realtime, 2.0 = 2x speed)
        """
        self.filepath = filepath
        self.loop = loop
        self.speed = max(0.1, min(10.0, speed))  # Clamp between 0.1x and 10x
        self.playing = False
        self.paused = False
        self.pause_start_time = None  # Track when pause started
        self.total_pause_time = 0.0   # Accumulated pause time
        self.frame_count = 0
        self.header = None
        self._playback_thread = None
        self._lock = threading.Lock()
        
        # Callbacks
        self.frame_callback: Optional[Callable[[Frame], None]] = None
        self.pointcloud_callback: Optional[Callable[[List[Dict]], None]] = None
        self.video_callback: Optional[Callable[[int, np.ndarray, np.ndarray], None]] = None  # (camera_idx, rgb_frame, depth_frame)
        self.playback_finished_callback: Optional[Callable[[], None]] = None
    
    def set_frame_callback(self, callback: Callable[[Frame], None]):
        """Set callback to be called for each frame"""
        self.frame_callback = callback
    
    def set_pointcloud_callback(self, callback: Callable[[List[Dict]], None]):
        """Set callback to be called for each point cloud (if present)"""
        self.pointcloud_callback = callback
    
    def set_video_callback(self, callback: Callable[[int, np.ndarray, np.ndarray], None]):
        """
        Set callback to be called for each video frame (if present).
        
        Args:
            callback: Function(camera_idx, rgb_frame, depth_frame)
                     rgb_frame: numpy array (H,W,3) BGR uint8
                     depth_frame: numpy array (H,W) float32 millimeters
        """
        self.video_callback = callback
    
    def set_playback_finished_callback(self, callback: Callable[[], None]):
        """Set callback to be called when playback finishes"""
        self.playback_finished_callback = callback
    
    @staticmethod
    def get_recording_info(filepath: str) -> Optional[Dict]:
        """
        Quick method to get recording information without loading entire file.
        Returns metadata dict or None if file cannot be read.
        
        Returns dict with keys:
            - version: Format version
            - duration: Recording duration in seconds
            - total_frames: Number of frames
            - framerate: Average framerate
            - has_pointcloud: Whether point clouds are included
            - has_video: Whether video is included
            - num_cameras: Number of video cameras (if video enabled)
            - file_size_mb: File size in megabytes
        """
        try:
            import os
            player = FramePlayer(filepath)
            if not player.load_header():
                return None
            
            info = {
                'version': player.header.get('version', 'unknown'),
                'has_pointcloud': player.header.get('record_pointcloud', False),
                'has_video': player.header.get('record_video', False),
                'compression': player.header.get('compression', 'none'),
                'file_size_mb': round(os.path.getsize(filepath) / (1024 * 1024), 2)
            }
            
            # Add metadata if available
            if 'metadata' in player.header:
                meta = player.header['metadata']
                info['duration'] = meta.get('duration', 0)
                info['total_frames'] = meta.get('total_frames', 0)
                info['framerate'] = meta.get('framerate', 0)
                
                if info['has_video']:
                    info['num_cameras'] = meta.get('num_cameras', 0)
                    info['video_size_mb'] = round(meta.get('total_video_bytes', 0) / (1024 * 1024), 2)
            
            return info
            
        except Exception as e:
            print(f"[ERROR] Failed to get recording info: {e}")
            return None
    
    def load_header(self) -> bool:
        """
        Load only the header to get metadata without loading entire file.
        Uses streaming decompression. Handles both old format (no header) and new format (with header).
        """
        try:
            print(f"[PLAYER] Loading recording header: {self.filepath}")
            
            with open(self.filepath, 'rb') as f:
                if ZSTD_AVAILABLE:
                    # Use streaming decompressor
                    decompressor = zstd.ZstdDecompressor()
                    with decompressor.stream_reader(f) as reader:
                        # Read bytes until we hit newline
                        first_line = b''
                        while True:
                            byte = reader.read(1)
                            if not byte or byte == b'\n':
                                break
                            first_line += byte
                else:
                    first_line = f.readline().rstrip(b'\n')
                
                if not first_line:
                    print("[ERROR] Empty recording file")
                    return False
                
                # Parse first line to detect format
                first_record = json.loads(first_line.decode('utf-8'))
                
                # Check if this is a header (new format) or a frame (old format)
                if 'version' in first_record and 'compression' in first_record:
                    # New format with header
                    self.header = first_record
                    
                    # Try to read metadata from end of file (added in v2.0+)
                    # The metadata line is the LAST line in the file, prefixed with "META:"
                    # We need to read it separately since it's after all frames
                    try:
                        # Read last few lines of file to find metadata
                        # Open file again and read from end
                        with open(self.filepath, 'rb') as f_end:
                            # Seek to end and read last chunk
                            f_end.seek(0, 2)  # Seek to end
                            file_size = f_end.tell()
                            
                            # Read last 4KB (should contain metadata line)
                            chunk_size = min(4096, file_size)
                            f_end.seek(file_size - chunk_size)
                            chunk = f_end.read()
                            
                            # Decompress if needed
                            if ZSTD_AVAILABLE and self.header.get('compression') == 'zstd':
                                # For compressed files, we can't easily read from end
                                # Skip metadata reading for now - not critical
                                pass
                            else:
                                # Look for META: line in the chunk
                                lines = chunk.split(b'\n')
                                for line in reversed(lines):
                                    if line.startswith(b'META:'):
                                        metadata = json.loads(line[5:].decode('utf-8'))
                                        self.header['metadata'] = metadata
                                        break
                    except Exception as e:
                        # Metadata reading is optional - don't fail if it's not there
                        pass
                    
                    # Display recording info
                    print(f"[PLAYER] Recording info:")
                    print(f"  Version: {self.header.get('version', 'unknown')}")
                    
                    # Display duration/frames if available from metadata
                    if 'metadata' in self.header:
                        meta = self.header['metadata']
                        duration = meta.get('duration', 0)
                        frames = meta.get('total_frames', 0)
                        fps = meta.get('framerate', 0)
                        print(f"  Duration: {duration:.1f}s ({frames} frames @ {fps:.1f} fps)")
                    
                    print(f"  Point clouds: {self.header.get('record_pointcloud', False)}")
                    print(f"  Video: {self.header.get('record_video', False)}")
                    if self.header.get('record_video'):
                        num_cams = len(self.header.get('video_cameras', []))
                        print(f"  Video cameras: {num_cams}")
                    print(f"  Compression: {self.header.get('compression', 'none')}")
                else:
                    # Old format without header (just frames)
                    print("[PLAYER] Old format recording detected (no header)")
                    self.header = {
                        'version': 0,  # Version 0 = old format
                        'record_pointcloud': False,
                        'record_video': False,
                        'video_cameras': [],
                        'compression': 'zstd' if ZSTD_AVAILABLE else 'none'
                    }
                
                return True
                
        except Exception as e:
            print(f"[ERROR] Failed to load header: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def start(self):
        """Start playback with streaming decompression"""
        with self._lock:
            if self.playing:
                print("[WARNING] Already playing")
                return False
            
            # Load header if not already loaded
            if self.header is None:
                if not self.load_header():
                    return False
            
            self.playing = True
            self.paused = False
            self.frame_count = 0
            
            # Start playback thread
            self._playback_thread = threading.Thread(target=self._playback_loop_streaming, daemon=True)
            self._playback_thread.start()
            
            print(f"[PLAYER] Started playback (speed={self.speed}x, loop={self.loop})")
            return True
    
    def stop(self):
        """Stop playback"""
        with self._lock:
            if not self.playing:
                return
            
            self.playing = False
            self.paused = False
            print("[PLAYER] Stopped playback")
    
    def pause(self):
        """Pause playback"""
        if not self.paused:
            self.paused = True
            self.pause_start_time = time.time()
            print("[PLAYER] Paused")
    
    def resume(self):
        """Resume playback"""
        if self.paused:
            self.paused = False
            # Accumulate the time spent paused
            if self.pause_start_time is not None:
                pause_duration = time.time() - self.pause_start_time
                self.total_pause_time += pause_duration
                self.pause_start_time = None
            print("[PLAYER] Resumed")
    
    def _read_line(self, reader) -> bytes:
        """Read a line from reader (handles both file and stream_reader)"""
        line = b''
        while True:
            byte = reader.read(1)
            if not byte:
                return line
            if byte == b'\n':
                return line
            line += byte
    
    def _playback_loop_streaming(self):
        """
        Main playback loop using streaming decompression.
        Memory efficient - processes one frame at a time.
        Handles both old format (no header) and new format (with header).
        """
        try:
            while self.playing:
                # Reset pause time tracking for this loop iteration
                self.total_pause_time = 0.0
                self.pause_start_time = None
                
                # Open file for streaming
                with open(self.filepath, 'rb') as f:
                    if ZSTD_AVAILABLE and self.header.get('compression') == 'zstd':
                        decompressor = zstd.ZstdDecompressor()
                        reader = decompressor.stream_reader(f)
                    else:
                        reader = f
                    
                    # Skip header line if new format (version >= 1)
                    # Handle both string and numeric versions
                    version = self.header.get('version', 0)
                    if isinstance(version, str):
                        try:
                            version = float(version)
                        except:
                            version = 0
                    
                    if version >= 1:
                        self._read_line(reader)  # Skip header
                    
                    # Get first frame timestamp for timing
                    first_line = self._read_line(reader)
                    if not first_line:
                        print("[WARNING] No frames in recording")
                        break
                    
                    first_record = json.loads(first_line.decode('utf-8'))
                    start_ts = first_record.get('timestamp', 0)
                    playback_start = time.time()
                    
                    # Process first frame
                    self._process_frame_record(first_record, reader, start_ts, playback_start)
                    
                    # Process remaining frames
                    while self.playing:
                        line = self._read_line(reader)
                        if not line:
                            break  # End of file
                        
                        # Handle pause
                        while self.paused and self.playing:
                            time.sleep(0.1)
                        
                        if not self.playing:
                            break
                        
                        # Decode JSON line
                        try:
                            line_str = line.decode('utf-8').strip()
                            if not line_str:
                                continue
                            
                            record = json.loads(line_str)
                            self._process_frame_record(record, reader, start_ts, playback_start)
                            
                        except json.JSONDecodeError:
                            # Might be binary point cloud data, skip
                            continue
                        except Exception as e:
                            print(f"[WARNING] Error processing frame: {e}")
                            continue
                
                # Check if we should loop
                if not self.loop:
                    break
                
                # Small delay before looping
                if self.loop and self.playing:
                    time.sleep(0.1)
            
            # Playback finished
            if self.playback_finished_callback:
                self.playback_finished_callback()
                
        except Exception as e:
            print(f"[ERROR] Playback error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.playing = False
    
    def _process_frame_record(self, record: dict, reader, start_ts: float, playback_start: float):
        """Process a single frame record with timing"""
        # Calculate when this frame should be displayed
        record_ts = record.get('timestamp', 0)
        relative_time = (record_ts - start_ts) / self.speed
        # Adjust playback start by accumulated pause time
        adjusted_playback_start = playback_start + self.total_pause_time
        target_time = adjusted_playback_start + relative_time
        
        # Send frame via callback FIRST
        frame_data = record.get('frame')
        if frame_data and self.frame_callback:
            try:
                frame = Frame.from_dict(frame_data)
                self.frame_callback(frame)
                self.frame_count += 1
            except Exception as e:
                print(f"[WARNING] Failed to parse frame: {e}")
        
        # Load point cloud if present
        if record.get('has_pointcloud') and self.pointcloud_callback:
            try:
                pointcloud_data = self._deserialize_pointcloud(reader)
                self.pointcloud_callback(pointcloud_data)
            except Exception as e:
                print(f"[WARNING] Failed to load point cloud: {e}")
        
        # Load video if present
        if record.get('has_video') and self.video_callback:
            try:
                video_data = self._deserialize_video(reader)
                # Decode and send via callback
                for camera_idx, rgb_nal, depth_nal in video_data:
                    rgb_frame, depth_frame = self._decode_video_frame(rgb_nal, depth_nal, camera_idx)
                    if rgb_frame is not None and depth_frame is not None:
                        self.video_callback(camera_idx, rgb_frame, depth_frame)
            except Exception as e:
                print(f"[WARNING] Failed to load/decode video: {e}")
                import traceback
                traceback.print_exc()
        
        # NOW wait until target time (after processing)
        # This accounts for the time spent processing the frame
        sleep_duration = target_time - time.time()
        if sleep_duration > 0:
            time.sleep(sleep_duration)
        elif sleep_duration < -0.5:  # More than 500ms behind
            # We're significantly behind - processing is too slow
            pass  # Just continue without sleeping
    
    def _deserialize_pointcloud(self, reader) -> List[Dict]:
        """
        Deserialize point cloud data from binary format.
        
        Format: [num_people: uint32]
                For each person:
                  [person_id: int32]
                  [num_points: uint32]
                  [points: float32 array (num_points * 3)]
                  [colors: uint8 array (num_points * 3)]
        """
        pointcloud_data = []
        
        # Read number of people
        num_people_bytes = reader.read(4)
        if len(num_people_bytes) < 4:
            return pointcloud_data
        
        num_people = struct.unpack('<I', num_people_bytes)[0]
        
        for _ in range(num_people):
            # Read person header
            person_id_bytes = reader.read(4)
            num_points_bytes = reader.read(4)
            
            if len(person_id_bytes) < 4 or len(num_points_bytes) < 4:
                break
            
            person_id = struct.unpack('<i', person_id_bytes)[0]
            num_points = struct.unpack('<I', num_points_bytes)[0]
            
            # Read points (float32 array)
            points_bytes = reader.read(num_points * 3 * 4)  # 3 floats per point, 4 bytes per float
            if len(points_bytes) < num_points * 3 * 4:
                break
            
            points = np.frombuffer(points_bytes, dtype=np.float32).reshape(num_points, 3)
            
            # Read colors (uint8 array)
            colors_bytes = reader.read(num_points * 3)  # 3 bytes per point
            if len(colors_bytes) < num_points * 3:
                break
            
            colors = np.frombuffer(colors_bytes, dtype=np.uint8).reshape(num_points, 3)
            
            pointcloud_data.append({
                'person_id': person_id,
                'points': points,
                'colors': colors
            })
        
        return pointcloud_data
    
    def _deserialize_video(self, reader) -> List[Tuple[int, bytes, bytes]]:
        """
        Deserialize video data from binary format.
        
        Format: [num_cameras: uint32]
                For each camera:
                  [camera_idx: uint32]
                  [rgb_size: uint32]
                  [depth_size: uint32]
                  [rgb_data: bytes]
                  [depth_data: bytes]
        
        Returns:
            List of tuples: (camera_idx, rgb_nal_units, depth_nal_units)
        """
        video_data = []
        
        # Read number of cameras
        num_cameras_bytes = reader.read(4)
        if len(num_cameras_bytes) < 4:
            return video_data
        
        num_cameras = struct.unpack('<I', num_cameras_bytes)[0]
        
        for _ in range(num_cameras):
            # Read camera header
            camera_idx_bytes = reader.read(4)
            rgb_size_bytes = reader.read(4)
            depth_size_bytes = reader.read(4)
            
            if len(camera_idx_bytes) < 4 or len(rgb_size_bytes) < 4 or len(depth_size_bytes) < 4:
                break
            
            camera_idx = struct.unpack('<I', camera_idx_bytes)[0]
            rgb_size = struct.unpack('<I', rgb_size_bytes)[0]
            depth_size = struct.unpack('<I', depth_size_bytes)[0]
            
            # Read RGB NAL units
            rgb_data = reader.read(rgb_size) if rgb_size > 0 else b''
            if len(rgb_data) < rgb_size:
                break
            
            # Read depth NAL units
            depth_data = reader.read(depth_size) if depth_size > 0 else b''
            if len(depth_data) < depth_size:
                break
            
            video_data.append((camera_idx, rgb_data, depth_data))
        
        return video_data
    
    def _decode_video_frame(self, rgb_nal_units: bytes, depth_nal_units: bytes, 
                           camera_idx: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Decode video NAL units back to numpy frames using GStreamer.
        Supports multiple codecs (h265, h264, etc.) based on metadata.
        
        Args:
            rgb_nal_units: Raw NAL units for RGB stream
            depth_nal_units: Raw NAL units for depth stream
            camera_idx: Camera index (for logging)
        
        Returns:
            (rgb_frame, depth_frame) as numpy arrays, or (None, None) on error
            rgb_frame: (H,W,3) BGR uint8
            depth_frame: (H,W) float32 millimeters
        """
        try:
            import gi
            gi.require_version('Gst', '1.0')
            from gi.repository import Gst
            
            # Initialize GStreamer (safe to call multiple times)
            Gst.init(None)
            
            # Get camera metadata
            camera_info = None
            for cam in self.header.get('video_cameras', []):
                if cam['camera_idx'] == camera_idx:
                    camera_info = cam
                    break
            
            if not camera_info:
                print(f"[WARNING] No video metadata for camera {camera_idx}")
                return None, None
            
            width = camera_info['width']
            height = camera_info['height']
            codec = camera_info.get('codec', 'h265')  # Default to h265 for backward compatibility
            
            # Decode RGB
            rgb_frame = None
            if rgb_nal_units:
                rgb_frame = self._decode_video_buffer(rgb_nal_units, width, height, codec, is_rgb=True)
            
            # Decode depth
            depth_frame = None
            if depth_nal_units:
                depth_frame = self._decode_video_buffer(depth_nal_units, width, height, codec, is_rgb=False)
            
            return rgb_frame, depth_frame
            
        except Exception as e:
            print(f"[ERROR] Failed to decode video frame: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _decode_video_buffer(self, nal_units: bytes, width: int, height: int, 
                            codec: str, is_rgb: bool) -> Optional[np.ndarray]:
        """
        Decode a video buffer to numpy array using GStreamer.
        Supports multiple codecs: h265, h264, vp8, vp9, av1, etc.
        
        Args:
            nal_units: Raw NAL units (or codec-specific bitstream)
            width: Frame width
            height: Frame height
            codec: Codec name ('h265', 'h264', 'vp8', 'vp9', 'av1', etc.)
            is_rgb: True for RGB (3 channels), False for depth (1 channel grayscale)
        
        Returns:
            Decoded frame as numpy array or None on error
        """
        try:
            import gi
            gi.require_version('Gst', '1.0')
            from gi.repository import Gst
            
            # Map codec names to GStreamer elements
            codec_map = {
                'h265': ('h265parse', 'avdec_h265'),
                'h264': ('h264parse', 'avdec_h264'),
                'vp8': ('', 'vp8dec'),
                'vp9': ('', 'vp9dec'),
                'av1': ('av1parse', 'av1dec'),
                'mpeg4': ('mpeg4videoparse', 'avdec_mpeg4'),
            }
            
            # Get parser and decoder for this codec
            codec_lower = codec.lower()
            if codec_lower not in codec_map:
                print(f"[WARNING] Unsupported codec '{codec}', trying h265 as fallback")
                codec_lower = 'h265'
            
            parser, decoder = codec_map[codec_lower]
            
            # Build pipeline based on codec
            if is_rgb:
                format_str = "BGR"
                channels = 3
            else:
                format_str = "GRAY8"
                channels = 1
            
            # Build pipeline string
            pipeline_parts = ["appsrc name=src"]
            
            # Add parser if codec needs it (some codecs like VP8 don't need a parser)
            if parser:
                pipeline_parts.append(parser)
            
            # Add decoder and rest of pipeline
            pipeline_parts.extend([
                decoder,
                "videoconvert",
                f"video/x-raw,format={format_str}",
                "appsink name=sink emit-signals=true sync=false"
            ])
            
            pipeline_str = " ! ".join(pipeline_parts)
            
            pipeline = Gst.parse_launch(pipeline_str)
            appsrc = pipeline.get_by_name('src')
            appsink = pipeline.get_by_name('sink')
            
            # Configure appsrc
            appsrc.set_property('format', Gst.Format.TIME)
            appsrc.set_property('is-live', False)
            
            # Start pipeline
            pipeline.set_state(Gst.State.PLAYING)
            
            # Push NAL units to pipeline
            buffer = Gst.Buffer.new_wrapped(nal_units)
            appsrc.emit('push-buffer', buffer)
            appsrc.emit('end-of-stream')
            
            # Pull decoded frame
            sample = appsink.emit('pull-sample')
            
            # Stop pipeline
            pipeline.set_state(Gst.State.NULL)
            
            if not sample:
                return None
            
            # Extract frame data
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            
            # Get frame dimensions from caps
            structure = caps.get_structure(0)
            frame_width = structure.get_value('width')
            frame_height = structure.get_value('height')
            
            # Map buffer to numpy array
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if not success:
                return None
            
            # Create numpy array
            if is_rgb:
                frame = np.ndarray(
                    shape=(frame_height, frame_width, channels),
                    dtype=np.uint8,
                    buffer=map_info.data
                ).copy()
            else:
                # Depth is encoded as grayscale, convert back to float32 millimeters
                # Assuming depth was normalized to 0-255 range during encoding
                # This needs to match the encoding scheme used
                frame_gray = np.ndarray(
                    shape=(frame_height, frame_width),
                    dtype=np.uint8,
                    buffer=map_info.data
                ).copy()
                
                # Convert back to depth values (this is a placeholder - actual conversion depends on encoding)
                # For now, just return the grayscale values
                frame = frame_gray.astype(np.float32)
            
            buffer.unmap(map_info)
            
            return frame
            
        except Exception as e:
            print(f"[ERROR] Failed to decode H.265 buffer: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def is_playing(self) -> bool:
        """Check if currently playing"""
        return self.playing
    
    def is_paused(self) -> bool:
        """Check if currently paused"""
        return self.paused
    
    def get_info(self) -> dict:
        """Get playback information"""
        info = {
            'filepath': self.filepath,
            'speed': self.speed,
            'loop': self.loop,
            'playing': self.playing,
            'paused': self.paused,
            'current_frame': self.frame_count
        }
        
        if self.header:
            info.update({
                'version': self.header.get('version'),
                'has_pointcloud': self.header.get('record_pointcloud', False),
                'compression': self.header.get('compression', 'none')
            })
        
        return info
    
    def set_frame_callback(self, callback: Callable[[Frame], None]):
        """Set callback to be called for each frame"""
        self.frame_callback = callback
    
    def set_playback_finished_callback(self, callback: Callable[[], None]):
        """Set callback to be called when playback finishes"""
        self.playback_finished_callback = callback
    
    def set_pointcloud_callback(self, callback: Callable[[list], None]):
        """Set callback to be called for point cloud data"""
        self.pointcloud_callback = callback

