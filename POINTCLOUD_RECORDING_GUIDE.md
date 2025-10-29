# Point Cloud Recording - Technical Guide

This guide explains how to record and play back point cloud data along with skeleton tracking data in senseSpace.

## Overview

The recording system has been extended to support **per-person point cloud data** with **streaming compression** for memory efficiency. This allows you to record complete sessions including both skeleton tracking and 3D point cloud data.

## Features

✅ **Per-Person Point Clouds**: Records point cloud data separately for each tracked person  
✅ **Streaming Compression**: Memory-efficient recording/playback (constant ~10-50 MB usage)  
✅ **Binary Format**: Efficient storage (~15 bytes/point vs ~60+ bytes with JSON)  
✅ **Synchronized Data**: Point clouds are timestamped and synchronized with skeleton frames  
✅ **Optional**: Can record skeleton-only or skeleton+pointcloud  

## Architecture

### Data Flow

```
Server (ZED Cameras)
    ↓
Skeleton Server (port 12345) → SenseSpaceClient → FrameRecorder
    ↓                                                    ↓
Point Cloud Server (port 12346) → PointCloudClient ─────┘
                                                         ↓
                                                  .ssrec file
                                                         ↓
                                                   FramePlayer
                                                         ↓
                                        frame_callback + pointcloud_callback
```

### File Format (.ssrec)

```
┌─────────────────────────────────────────────────┐
│ HEADER (JSON)                                   │
│ - version: 1                                    │
│ - record_pointcloud: true/false                 │
│ - compression: "zstd"                           │
├─────────────────────────────────────────────────┤
│ FRAME 1 (JSON)                                  │
│ - timestamp, skeleton data, has_pointcloud flag │
├─────────────────────────────────────────────────┤
│ POINT CLOUD 1 (BINARY) [if has_pointcloud]      │
│ - num_people                                    │
│ - per-person: id, num_points, xyz[], rgb[]      │
├─────────────────────────────────────────────────┤
│ FRAME 2 (JSON)                                  │
├─────────────────────────────────────────────────┤
│ POINT CLOUD 2 (BINARY) [if has_pointcloud]      │
├─────────────────────────────────────────────────┤
│ ...                                             │
└─────────────────────────────────────────────────┘
           ↑
    Entire file compressed with zstd streaming
```

## Implementation Details

### Point Cloud Binary Format

For each frame with point cloud data:

```python
# Header
num_people: uint32  # Number of people in this frame

# For each person:
person_id: int32           # Person tracking ID
num_points: uint32         # Number of points for this person
points: float32[num_points][3]  # XYZ coordinates (12 bytes/point)
colors: uint8[num_points][3]    # RGB colors (3 bytes/point)
```

Total size per point: **15 bytes** (vs ~60+ bytes with JSON encoding)

### Memory Efficiency

**Recording (FrameRecorder)**:
- Uses `zstd.ZstdCompressor.stream_writer(file)` 
- Writes frames incrementally as they arrive
- No buffering in memory
- Memory usage: ~10 MB constant

**Playback (FramePlayer)**:
- Uses `zstd.ZstdDecompressor.stream_reader(file)`
- Reads and processes one frame at a time
- Never loads entire file into memory
- Memory usage: ~10-50 MB regardless of file size

This approach allows recording and playing back **hours-long sessions** without memory issues.

### Timestamp Synchronization

Point clouds and skeleton frames arrive independently and may not be perfectly synchronized. The recording system handles this:

```python
# Store latest point cloud
latest_pointcloud = None
latest_pc_timestamp = 0.0

def on_pointcloud(pointcloud_data, timestamp):
    global latest_pointcloud, latest_pc_timestamp
    latest_pointcloud = pointcloud_data
    latest_pc_timestamp = timestamp

def on_frame(frame):
    # Match point cloud within 100ms threshold
    pointcloud_data = None
    if latest_pointcloud:
        time_diff = abs(frame.timestamp - latest_pc_timestamp)
        if time_diff < 0.1:  # 100ms threshold
            pointcloud_data = latest_pointcloud
    
    # Record frame with synchronized point cloud
    recorder.record_frame(frame, pointcloud_data)
```

## Code Examples

### Basic Recording (Skeleton Only)

```python
from senseSpaceLib.senseSpace import SenseSpaceClient, FrameRecorder

# Create client and recorder
client = SenseSpaceClient("192.168.1.100", 12345)
recorder = FrameRecorder("recording.ssrec", record_pointcloud=False)

def on_frame(frame):
    if recorder.is_recording():
        recorder.record_frame(frame)

client.set_frame_callback(on_frame)
client.connect()

# Start recording
recorder.start()

# ... frames are recorded automatically ...

# Stop recording
recorder.stop()
```

### Advanced Recording (Skeleton + Point Clouds)

```python
from senseSpaceLib.senseSpace import SenseSpaceClient, FrameRecorder
from pointCloudClient import PointCloudClient

# Create clients
skeleton_client = SenseSpaceClient("192.168.1.100", 12345)
pc_client = PointCloudClient("192.168.1.100", 12346)

# Create recorder with point cloud support
recorder = FrameRecorder("recording.ssrec", record_pointcloud=True)

# Synchronization storage
latest_pointcloud = None
latest_pc_timestamp = 0.0

def on_pointcloud_perperson(pointcloud_data, timestamp):
    """Called when point cloud data arrives"""
    global latest_pointcloud, latest_pc_timestamp
    latest_pointcloud = pointcloud_data
    latest_pc_timestamp = timestamp

def on_frame(frame):
    """Called when skeleton frame arrives"""
    # Find matching point cloud (within 100ms)
    pointcloud_data = None
    if latest_pointcloud:
        time_diff = abs(frame.timestamp - latest_pc_timestamp)
        if time_diff < 0.1:
            pointcloud_data = latest_pointcloud
    
    # Record frame with point cloud
    if recorder.is_recording():
        recorder.record_frame(frame, pointcloud_data)

# Set up callbacks
skeleton_client.set_frame_callback(on_frame)
pc_client.on_pointcloud_perperson_received = on_pointcloud_perperson

# Connect both clients
skeleton_client.connect()
pc_client.connect()

# Start recording
recorder.start()

# ... recording happens automatically ...

# Stop recording
recorder.stop()
```

### Playback

```python
from senseSpaceLib.senseSpace import FramePlayer

# Create player
player = FramePlayer("recording.ssrec", loop=False, speed=1.0)

# Define callbacks
def on_frame(frame):
    print(f"Frame: {len(frame.persons)} persons detected")

def on_pointcloud(pointcloud_data):
    """Called when point cloud data is available"""
    for person_data in pointcloud_data:
        person_id = person_data['person_id']
        points = person_data['points']  # numpy array Nx3
        colors = person_data['colors']  # numpy array Nx3
        print(f"  Person {person_id}: {len(points)} points")

# Set callbacks
player.set_frame_callback(on_frame)
player.set_pointcloud_callback(on_pointcloud)

# Load and check format
if player.load_header():
    info = player.get_info()
    print(f"Version: {info['version']}")
    print(f"Has point clouds: {info['has_pointcloud']}")
    print(f"Compression: {info['compression']}")
    
    # Start playback
    player.start()
    
    # Wait for completion
    while player.is_playing():
        time.sleep(0.1)
```

## API Reference

### FrameRecorder

```python
class FrameRecorder:
    def __init__(self, filepath: str, record_pointcloud: bool = False)
    def start() -> bool
    def stop()
    def is_recording() -> bool
    def record_frame(self, frame: Frame, pointcloud_data: Optional[List[Dict]] = None)
```

**Parameters for record_frame()**:
- `frame`: Skeleton frame object
- `pointcloud_data`: Optional list of per-person point clouds
  ```python
  [
      {
          'person_id': int,
          'points': np.ndarray,  # Nx3 float32
          'colors': np.ndarray   # Nx3 uint8
      },
      ...
  ]
  ```

### FramePlayer

```python
class FramePlayer:
    def __init__(self, filepath: str, loop: bool = False, speed: float = 1.0)
    def load_header() -> bool
    def start() -> bool
    def stop()
    def pause()
    def resume()
    def is_playing() -> bool
    def get_info() -> dict
    def set_frame_callback(self, callback: Callable[[Frame], None])
    def set_pointcloud_callback(self, callback: Callable[[List[Dict]], None])
```

### PointCloudClient

Extended with per-person callback:

```python
class PointCloudClient:
    # New callback for per-person data
    on_pointcloud_perperson_received: Callable[[List[Dict], float], None]
    
    # Callback signature:
    # on_pointcloud_perperson_received(pointcloud_data, timestamp)
    # pointcloud_data: List of dicts with 'person_id', 'points', 'colors'
    # timestamp: Frame timestamp
```

## Performance Characteristics

### File Sizes (Compressed)

**Skeleton Only**:
- ~1-5 MB per minute
- Compression ratio: ~10-20x

**Skeleton + Point Clouds**:
- Skeleton: ~1-5 MB per minute
- Point clouds: ~50-200 MB per minute (varies with scene complexity)
- Total: ~50-200 MB per minute
- Compression ratio: ~3-5x for point clouds

### Memory Usage

**Recording**:
- Skeleton only: ~5-10 MB constant
- With point clouds: ~10-50 MB constant
- Independent of recording duration

**Playback**:
- Skeleton only: ~5-10 MB constant
- With point clouds: ~10-50 MB constant
- Independent of file size

### CPU Usage

**Compression (zstd)**:
- Low CPU usage (~5-15% on modern CPU)
- Fast compression speed (~200-500 MB/s)
- Good compression ratio

**Decompression**:
- Very low CPU usage (~2-5%)
- Fast decompression speed (~500-1000 MB/s)

## Troubleshooting

### "zstandard not available"
```bash
pip install zstandard
```

### Point cloud callback never called
- Check `record_pointcloud=True` when creating recorder
- Verify point cloud server is running and sending data
- Check PointCloudClient is connected and receiving data

### Point clouds not synchronized
- Increase sync threshold (default 100ms) if network latency high
- Check both servers are sending timestamps
- Verify clocks are synchronized between server and client

### File size too large
- Point clouds are large - this is normal
- Consider reducing point cloud resolution on server
- Consider recording only skeleton data
- Verify zstd compression is enabled

### Memory usage grows during playback
- Should NOT happen with streaming!
- Ensure using latest version of recorder.py
- Check for resource leaks in your callbacks

## Technical Notes

### Why Streaming?

Previous approaches that loaded entire files into memory:
- ❌ Crashes with large files (OOM errors)
- ❌ Long load times (30+ seconds for large recordings)
- ❌ Memory usage proportional to file size

Streaming approach:
- ✅ Handles any file size
- ✅ Instant startup (only reads header)
- ✅ Constant memory usage
- ✅ Can record for hours without issues

### Binary vs JSON

**JSON Encoding** (old approach):
```json
{
    "points": [[1.5, 2.3, 0.8], [1.6, 2.4, 0.9], ...],
    "colors": [[255, 128, 64], [255, 129, 65], ...]
}
```
Size: ~60+ bytes per point

**Binary Encoding** (new approach):
```
float32[x,y,z] + uint8[r,g,b] = 12 + 3 = 15 bytes per point
```
Size: **15 bytes per point** (4x smaller!)

### Compression Performance

Zstandard (zstd) offers excellent balance:
- Fast compression (~200-500 MB/s)
- Fast decompression (~500-1000 MB/s)
- Good compression ratio (~3-5x for point clouds)
- Streaming support (compress/decompress incrementally)

Alternatives considered:
- gzip: Slower, slightly better compression
- lz4: Faster, worse compression
- No compression: 3-5x larger files

## See Also

- `libs/senseSpaceLib/senseSpace/recorder.py` - Core implementation
- `client/pointCloudClient.py` - Point cloud client with per-person callback
- `client/examples/recording/pointcloud_recording_example.py` - Complete example
- `RECORDING_IMPLEMENTATION.md` - High-level implementation summary
