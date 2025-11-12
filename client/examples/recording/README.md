# Frame Recording and Playback

Record skeleton tracking frames, point clouds, and video for offline playback and testing.

## Features

- **Compressed Storage**: Uses zstandard compression for efficient file sizes
- **Video Recording**: Capture H.265 video streams with zero re-encoding overhead
- **Point Cloud Support**: Record 3D point clouds alongside skeleton data
- **Seamless Playback**: Replay recordings with original timing
- **Loop Support**: Continuous playback for testing
- **Easy Integration**: Works with all client examples

## File Format

### Version 1.0 (.ssrec)
- Skeleton data (JSON)
- Optional point cloud data (binary)

### Version 2.0 (.ssrec) - NEW!
- Skeleton data (JSON)
- Optional point cloud data (binary)
- **Optional video data (H.265 NAL units)**
- Multiple camera support
- Backward compatible with v1.0 readers

## Recording

### Video Recording (NEW!)

Record skeleton, point clouds, AND video streams:

```bash
# Record with video for 60 seconds
python video_recording_example.py \
    --server-ip 127.0.0.1 \
    --cameras 33253574 34893077 \
    --duration 60

# Record manually (press 'r' to start, 's' to stop)
python video_recording_example.py \
    --server-ip 127.0.0.1 \
    --cameras 33253574 34893077
```

**Requirements:**
- GStreamer with H.265 support
- Video streaming enabled on server
- Camera serial numbers

**Options:**
- `--no-pointcloud`: Disable point cloud recording
- `--no-video`: Disable video recording
- `--rgb-width`, `--rgb-height`: RGB resolution (default: 1280x720)
- `--depth-width`, `--depth-height`: Depth resolution (default: 640x480)
- `--framerate`: Video framerate (default: 30)

### Interactive Recording (Keyboard)

Press **'R'** key in any visualization client to toggle recording:

```bash
# Start client with visualization
python ../senseSpaceClient.py --viz --server localhost

# Press 'R' to start recording
# Press 'R' again to stop recording
# Files saved to: recordings/recording_TIMESTAMP.ssrec
```

### Programmatic Recording

```python
from senseSpaceLib.senseSpace.vizClient import VisualizationClient
from senseSpaceLib.senseSpace.vizWidget import SkeletonGLWidget

client = VisualizationClient(
    viewer_class=SkeletonGLWidget,
    server_ip="localhost",
    server_port=12345
)

# Start recording to specific file
client.start_recording("my_session.ssrec")

# ... recording happens automatically ...

# Stop recording
client.stop_recording()
```

## Playback

### Video Playback (NEW!)

Play back recordings with video:

```bash
# Display video in OpenCV windows
python video_playback_example.py recordings/my_session.ssrec --show-video

# Export video to MP4 files
python video_playback_example.py recordings/my_session.ssrec --export-video output/

# Play at 2x speed
python video_playback_example.py recordings/my_session.ssrec --speed 2.0

# Loop playback
python video_playback_example.py recordings/my_session.ssrec --show-video --loop
```

**Features:**
- Decodes H.265 video in real-time using GStreamer
- Displays RGB and depth streams
- Export to MP4 format
- Speed control and looping

**Controls (when --show-video):**
- Press `SPACE` to pause/resume
- Press `q` to quit

### Command Line Playback

```bash
# Play recording in visualization mode
python ../senseSpaceClient.py --viz --rec recordings/recording_20241029_143022.ssrec

# Works with any example that supports --viz
python ../llm/llm_ollama.py --viz --rec recordings/recording_20241029_143022.ssrec
```

### Programmatic Playback

```python
from senseSpaceLib.senseSpace.recorder import FramePlayer
from senseSpaceLib.senseSpace.protocol import Frame

def on_frame(frame: Frame):
    print(f"Frame: {len(frame.people)} people")

player = FramePlayer("recording.ssrec", loop=True, speed=1.0)
player.set_frame_callback(on_frame)

if player.load():
    player.start()
    
    # Playback runs in background thread
    import time
    time.sleep(10)
    
    player.stop()
```

## File Format

**.ssrec files** (SenseSpace Recording):
- Zstandard compressed
- Each frame contains:
  - JSON skeleton data
  - Binary point cloud data (optional)
  - Binary H.265 video data (optional, v2.0+)

### Version 1.0 Format
```
Header (JSON): {version, timestamp, framerate}
---
Frame 1: JSON skeleton + optional binary point cloud
Frame 2: JSON skeleton + optional binary point cloud
...
```

### Version 2.0 Format (with video)
```
Header (JSON): {version, timestamp, framerate, has_video, video_cameras}
---
Frame 1: JSON skeleton + optional binary point cloud + binary H.265 NAL units
Frame 2: JSON skeleton + optional binary point cloud + binary H.265 NAL units
...
```

**Video binary format:**
```
[num_cameras: uint32]
For each camera:
  [camera_idx: uint32]
  [rgb_size: uint32]
  [depth_size: uint32]
  [rgb_nal_units: bytes]
  [depth_nal_units: bytes]
```

### File Size Estimates

| Configuration | Per Frame | 10 min @ 30fps |
|--------------|-----------|----------------|
| Skeleton only | ~1 KB | ~54 MB |
| + Point clouds (3 cams) | ~100 KB | ~5.4 GB |
| + Video (3 cams, RGB+Depth) | ~200 KB | ~10.8 GB |

**Video overhead:**
- H.265 encoding: ~2-4 MB/sec per camera (RGB + depth)
- Zero re-encoding cost (uses existing stream)
- File size ≈ streaming bandwidth × duration

## Use Cases

### Testing & Development
Record sessions for reproducible testing without live camera setup.

### Offline Analysis
Analyze recorded sessions without server connection.

### Demo & Presentation
Create polished recordings for demonstrations.

### Algorithm Development
Test pose analysis algorithms on consistent recorded data.

## Examples

### Record a Session
```bash
# 1. Start server
cd ../../server
python seg_pointcloud_server.py --viz

# 2. Start client and press 'R' to record
cd ../client
python senseSpaceClient.py --viz --server localhost

# 3. Recording saved to: recordings/recording_TIMESTAMP.ssrec
```

### Playback Session
```bash
# Play most recent recording
python senseSpaceClient.py --viz --rec recordings/recording_20241029_143022.ssrec

# Use with LLM analysis
python examples/llm/llm_ollama.py --viz --rec recordings/recording_20241029_143022.ssrec
```

### Loop Playback for Testing
```python
from senseSpaceLib.senseSpace.vizClient import VisualizationClient

client = VisualizationClient(
    playback_file="test_session.ssrec"
)

client.run()  # Loops automatically
```

## Keyboard Controls

All visualization clients support:
- **'R'**: Toggle recording on/off
- **'P'**: Toggle point cloud display

During playback:
- Playback loops automatically
- Close window to stop

## Tips

- **Compression**: Recordings compress well (~10-20x) due to zstandard
- **File Names**: Auto-generated with timestamp: `recording_YYYYMMDD_HHMMSS.ssrec`
- **Storage**: 1 minute ≈ 1-5MB depending on number of people
- **Speed Control**: Adjust playback speed programmatically with `speed` parameter

## Troubleshooting

**Recording not starting:**
- Check `recordings/` directory is writable
- Ensure zstandard is installed: `pip install zstandard`

**Playback fails:**
- Verify `.ssrec` file exists and is not corrupted
- Check file was recorded with same protocol version

**No frames in recording:**
- Ensure server was sending frames during recording
- Check recording was stopped properly (press 'R' again)
