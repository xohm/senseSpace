# Video Recording Integration Guide

## Overview

The senseSpace recording system now supports embedding H.265 video streams directly into `.ssrec` files. This feature captures both RGB and depth video streams from all cameras without re-encoding, providing zero-overhead recording.

## File Format (.ssrec v2.0)

The `.ssrec` format has been extended to version 2.0:

### Header Structure
```json
{
  "version": "2.0",
  "timestamp": "2025-01-30T10:30:00",
  "framerate": 30,
  "has_video": true,
  "video_cameras": [
    {
      "camera_idx": 0,
      "width": 1280,
      "height": 720,
      "fps": 30,
      "codec": "h265"
    }
  ]
}
```

### Binary Data Format
Each frame contains:
- JSON skeleton data (as before)
- Optional point cloud binary data (as before)
- **NEW**: Video binary data with H.265 NAL units

Video binary format:
```
[num_cameras: uint32]
For each camera:
  [camera_idx: uint32]
  [rgb_size: uint32]
  [depth_size: uint32]
  [rgb_nal_units: bytes (rgb_size)]
  [depth_nal_units: bytes (depth_size)]
```

## Usage

### 1. Recording with Video

```python
from senseSpace.recorder import FrameRecorder
from senseSpace.video_streaming import MultiCameraVideoStreamer

# Create recorder with video recording enabled
recorder = FrameRecorder(
    filename="my_session.ssrec",
    record_pointclouds=True,
    record_video=True  # Enable video recording
)

# Create video streamer
streamer = MultiCameraVideoStreamer(
    camera_serials=["33253574", "34893077"],
    server_ip="127.0.0.1",
    server_port=5000,
    rgb_width=1280,
    rgb_height=720,
    depth_width=640,
    depth_height=480,
    framerate=30
)

# Enable recording on the streamer
streamer.enable_recording(recorder)

# Start recording
recorder.start_recording()
streamer.start()

# Record frames
for frame_data in your_skeleton_stream:
    # Optionally record point clouds
    recorder.record_frame(
        skeleton_data=frame_data['skeleton'],
        pointcloud_data=frame_data.get('pointclouds')
    )
    # Video is automatically captured by GStreamer callbacks

# Stop recording
recorder.stop_recording()
streamer.disable_recording()
streamer.stop()
recorder.close()
```

### 2. Check Recording Info (Quick Preview)

```python
from senseSpace.recorder import FramePlayer

# Get recording info without loading entire file
info = FramePlayer.get_recording_info("my_session.ssrec")

if info:
    print(f"Duration: {info['duration']}s")
    print(f"Frames: {info['total_frames']}")
    print(f"Framerate: {info['framerate']} fps")
    print(f"Has video: {info['has_video']}")
    print(f"Cameras: {info.get('num_cameras', 0)}")
    print(f"File size: {info['file_size_mb']} MB")
```

Or use the command-line utility:
```bash
python check_recording.py my_session.ssrec
```

### 3. Playback with Video

```python
from senseSpace.recorder import FramePlayer

# Create player
player = FramePlayer(filename="my_session.ssrec")

# Optional: Set video callback to receive decoded frames
def on_video_frame(camera_idx, rgb_frame, depth_frame):
    """
    Callback for decoded video frames.
    
    Args:
        camera_idx: Camera index (0, 1, 2, ...)
        rgb_frame: numpy array (H, W, 3) BGR uint8
        depth_frame: numpy array (H, W) float32 millimeters
    """
    # Display or process video frames
    cv2.imshow(f"RGB Camera {camera_idx}", rgb_frame)
    if depth_frame is not None:
        cv2.imshow(f"Depth Camera {camera_idx}", depth_frame)

player.set_video_callback(on_video_frame)

# Play recording
player.play()

# Get frames
for skeleton, pointclouds in player.get_next_frame():
    # Video frames are delivered via callback
    # Process skeleton and point clouds
    pass

player.close()
```

## Technical Details

### GStreamer Pipeline Integration

The video recording works by forking the H.265-encoded stream using a `tee` element:

**RGB Pipeline:**
```
v4l2src → videoconvert → encoder → h265parse → tee
                                                 ├→ rtph265pay → udpsink (streaming)
                                                 └→ appsink (recording)
```

**Recording Callback:**
```python
def _on_rgb_sample(self, sink, camera_idx):
    sample = sink.emit('pull-sample')
    buffer = sample.get_buffer()
    # Extract NAL units
    success, map_info = buffer.map(Gst.MapFlags.READ)
    nal_units = bytes(map_info.data)
    # Send to recorder
    self.recorder.record_video_data(camera_idx, rgb_nal_units, depth_nal_units)
```

### Decoder Implementation

During playback, NAL units are decoded using GStreamer:

```
appsrc → h265parse → avdec_h265 → videoconvert → appsink
```

The decoder creates a new pipeline for each frame and pulls the decoded image data as a numpy array.

### Performance Characteristics

- **Zero encoding overhead**: Uses already-encoded H.265 stream
- **File size**: ~2.5-4 MB/sec per camera (RGB + depth)
- **3 cameras + point clouds**: ~10-15 GB per 10 minutes
- **Decoding**: Real-time playback at 30 fps on modern hardware

## Backward Compatibility

Files are forward and backward compatible:
- **Old readers** (v1.0): Will skip video data sections, read skeleton/point clouds normally
- **New readers** (v2.0): Detect version, handle both v1.0 and v2.0 files

## File Size Estimates

| Configuration | Bitrate/Camera | Total (3 cams) | 10 min |
|--------------|----------------|----------------|--------|
| Skeleton only | ~1 KB/frame | ~3 KB/frame | ~54 MB |
| + Point clouds | ~100 KB/frame | ~300 KB/frame | ~5.4 GB |
| + Video (RGB+Depth) | ~100 KB/frame | ~300 KB/frame | ~5.4 GB |
| All combined | ~200 KB/frame | ~600 KB/frame | ~10.8 GB |

## Limitations & Future Work

### Current Limitations
1. Depth encoding uses grayscale conversion (lossy)
2. Decoder creates new pipeline per frame (could be optimized)
3. No support for variable framerate
4. H.265 only (no H.264 fallback)

### Future Enhancements
- Persistent decoder pipelines for better performance
- Support for H.264 fallback
- Better depth encoding (lossless or 16-bit)
- Streaming playback (decode on demand)
- GPU-accelerated decoding
- Multi-threaded decoding for multiple cameras

## Troubleshooting

### Recording Issues

**Video not being captured:**
- Ensure `record_video=True` when creating FrameRecorder
- Call `streamer.enable_recording(recorder)` before streaming
- Verify GStreamer appsink is receiving samples

**File too large:**
- Adjust H.265 encoder bitrate (modify encoder properties)
- Reduce resolution or framerate
- Disable point cloud recording if not needed

### Playback Issues

**Video not decoding:**
- Check GStreamer H.265 decoder is installed (`gst-inspect-1.0 avdec_h265`)
- Verify video metadata in header (`player.header['video_cameras']`)
- Check for error messages during decode

**Poor playback performance:**
- Use hardware-accelerated decoder (NVDEC, VideoToolbox, VAAPI)
- Reduce playback framerate
- Pre-decode frames in background thread

## Example Integration

See `client/examples/recording/` for complete examples:
- `record_with_video.py`: Full recording session with video
- `playback_video.py`: Playback with video visualization
- `export_video.py`: Export video to separate MP4 files

## Support

For issues or questions:
1. Check the console output for GStreamer errors
2. Verify `.ssrec` file header with `player.load_header()`
3. Test with minimal example (single camera, short duration)
4. Report issues with full error log and system info
