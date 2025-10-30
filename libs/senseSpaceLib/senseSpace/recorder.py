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
from typing import Optional, Callable, Dict, List
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
    
    def __init__(self, filepath: str, compression_level: int = 3, record_pointcloud: bool = False):
        """
        Initialize frame recorder.
        
        Args:
            filepath: Path to save recording file (.ssrec extension recommended)
            compression_level: Zstandard compression level (1-22, default 3)
            record_pointcloud: Whether to record point cloud data (increases file size significantly)
        """
        self.filepath = filepath
        self.compression_level = compression_level
        self.record_pointcloud = record_pointcloud
        self.recording = False
        self.frame_count = 0
        self.start_time = None
        self._file = None
        self._stream_writer = None
        self._lock = threading.Lock()
        
        # Statistics
        self.total_bytes_written = 0
        self.total_pointcloud_points = 0
        
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
                    'version': '1.0',
                    'record_pointcloud': self.record_pointcloud,
                    'compression': 'zstd' if ZSTD_AVAILABLE else 'none',
                    'compression_level': self.compression_level if ZSTD_AVAILABLE else 0,
                    'start_time': time.time()
                }
                header_json = json.dumps(header, separators=(',', ':')) + '\n'
                self._stream_writer.write(header_json.encode('utf-8'))
                self._stream_writer.flush()
                
                self.recording = True
                self.frame_count = 0
                self.total_bytes_written = len(header_json)
                self.total_pointcloud_points = 0
                self.start_time = time.time()
                
                mode = "with point clouds" if self.record_pointcloud else "skeleton only"
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
                # Flush and close stream writer
                if self._stream_writer:
                    self._stream_writer.flush()
                    if self._stream_writer != self._file:
                        self._stream_writer.close()
                
                if self._file:
                    self._file.close()
                
                duration = time.time() - self.start_time if self.start_time else 0
                file_size = os.path.getsize(self.filepath) / (1024 * 1024)  # MB
                
                print(f"[RECORDER] Stopped recording:")
                print(f"  Frames: {self.frame_count}")
                print(f"  Duration: {duration:.1f}s")
                print(f"  File size: {file_size:.2f} MB")
                if self.record_pointcloud:
                    print(f"  Total points: {self.total_pointcloud_points:,}")
                print(f"  Saved to: {self.filepath}")
                
            except Exception as e:
                print(f"[ERROR] Error stopping recording: {e}")
            finally:
                self.recording = False
                self._file = None
                self._stream_writer = None
    
    def record_frame(self, frame: Frame, pointcloud_data: Optional[List[Dict]] = None):
        """
        Record a single frame with optional point cloud data.
        
        Args:
            frame: Frame object to record
            pointcloud_data: Optional list of per-person point clouds
                            Each item: {"person_id": int, "points": ndarray(N,3), "colors": ndarray(N,3)}
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
                    'has_pointcloud': self.record_pointcloud and pointcloud_data is not None and len(pointcloud_data) > 0
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
        self.playback_finished_callback: Optional[Callable[[], None]] = None
    
    def set_frame_callback(self, callback: Callable[[Frame], None]):
        """Set callback to be called for each frame"""
        self.frame_callback = callback
    
    def set_pointcloud_callback(self, callback: Callable[[List[Dict]], None]):
        """Set callback to be called for each point cloud (if present)"""
        self.pointcloud_callback = callback
    
    def set_playback_finished_callback(self, callback: Callable[[], None]):
        """Set callback to be called when playback finishes"""
        self.playback_finished_callback = callback
    
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
                    print(f"[PLAYER] Recording info:")
                    print(f"  Version: {self.header.get('version', 'unknown')}")
                    print(f"  Point clouds: {self.header.get('record_pointcloud', False)}")
                    print(f"  Compression: {self.header.get('compression', 'none')}")
                else:
                    # Old format without header (just frames)
                    print("[PLAYER] Old format recording detected (no header)")
                    self.header = {
                        'version': 0,  # Version 0 = old format
                        'record_pointcloud': False,
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
                        self._read_line(reader)
                    
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

