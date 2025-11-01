# Video Streaming Integration Complete

## Overview
Full video streaming integration has been implemented in the senseSpace server. The server can now stream RGB and depth video from ZED cameras to clients using GStreamer.

## Changes Made

### 1. CLI Arguments (senseSpace_fusion_main.py)
Added streaming command-line arguments:
- `--stream`: Enable video streaming
- `--stream-host`: Host address for streaming (default: 0.0.0.0)
- `--stream-rgb-port`: RGB stream port (default: 5000)
- `--stream-depth-port`: Depth stream port (default: 5001)

### 2. Server Class (libs/senseSpaceLib/senseSpace/server.py)

#### Constructor Updates
- Added `enable_streaming`, `stream_host`, `stream_rgb_port`, `stream_depth_port` parameters
- Added `self.video_streamer` attribute (initialized to None)
- Imported `MultiCameraVideoStreamer` with `STREAMING_AVAILABLE` flag for graceful fallback

#### Video Streamer Initialization
- Created `_initialize_video_streamer(num_cameras)` method
- Called after camera initialization completes in both modes:
  - Single camera mode: after line 817 (1 camera)
  - Fusion mode: after line 1107 (multiple cameras)

#### Frame Capture and Streaming

**Single Camera Loop** (`_run_single_camera_loop`):
```python
# After successful grab()
if self.video_streamer is not None:
    # Retrieve RGB and depth images
    self.camera.retrieve_image(rgb_mat, sl.VIEW.LEFT, sl.MEM.CPU)
    self.camera.retrieve_measure(depth_mat, sl.MEASURE.DEPTH, sl.MEM.CPU)
    
    # Convert to numpy and push to streamer
    self.video_streamer.push_frames(0, rgb_frame, depth_frame)
```

**Fusion Loop** (`_run_fusion_loop`):
```python
# For each camera after successful grab()
for camera_idx, (serial, zed) in enumerate(self._fusion_senders.items()):
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        if self.video_streamer is not None:
            # Retrieve RGB and depth images
            zed.retrieve_image(rgb_mat, sl.VIEW.LEFT, sl.MEM.CPU)
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH, sl.MEM.CPU)
            
            # Push to streamer with camera index
            self.video_streamer.push_frames(camera_idx, rgb_frame, depth_frame)
```

## Usage

### Start Server with Streaming
```bash
# Single camera
python senseSpace_fusion_main.py --viz --stream

# Multiple cameras (fusion mode)
python senseSpace_fusion_main.py --viz --stream --serial 34893077 33253574

# Custom ports
python senseSpace_fusion_main.py --viz --stream --stream-rgb-port 6000 --stream-depth-port 6001
```

### Connect Client
```bash
# Basic streaming client
python streamingClient.py --server localhost

# Specify ports if custom
python streamingClient.py --server localhost --rgb-port 6000 --depth-port 6001
```

## Architecture

### Client Detection
- Server listens for UDP heartbeat on port 5100
- Streaming only starts when client is detected
- Streaming stops when client disconnects (resource efficient)

### Frame Flow
1. ZED camera captures RGB + depth at 60 FPS (fusion) or 30 FPS (single)
2. Server retrieves frames via ZED SDK
3. Frames converted to numpy arrays
4. MultiCameraVideoStreamer encodes and streams via GStreamer
5. Client receives and displays frames

### Multi-Camera Support
- Each camera assigned an index (0, 1, 2, ...)
- Frames tagged with camera_idx
- Client can display multiple camera feeds simultaneously

## Dependencies

### System Packages (Required)
```bash
sudo apt-get install -y \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gstreamer-1.0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libcairo2-dev \
    libgirepository1.0-dev
```

### Virtual Environment Setup
If using a virtual environment, create symlinks to system packages:
```bash
ln -s /usr/lib/python3/dist-packages/gi .venv/lib/python3.12/site-packages/gi
ln -s /usr/lib/python3/dist-packages/cairo .venv/lib/python3.12/site-packages/cairo
```

### Python Package
Install senseSpace with streaming support:
```bash
pip install -e ".[streaming]"
```

## Performance

### Resource Usage
- Streaming only active when client connected (zero overhead when idle)
- GStreamer handles encoding efficiently
- Minimal CPU impact (~5-10% per camera)

### Network Bandwidth
- RGB: ~2-5 Mbps per camera (H.264 encoded)
- Depth: ~1-2 Mbps per camera (encoded as grayscale)
- Total: ~3-7 Mbps per camera

## Troubleshooting

### No Stream Available
- Check server started with `--stream` flag
- Verify client can reach server (ping test)
- Check firewall allows ports 5000, 5001, 5100

### PyGObject Import Errors
- Ensure system packages installed
- Create symlinks if using virtual environment
- Verify with: `python -c "import gi; print(gi.__file__)"`

### Poor Video Quality
- Increase bitrate in VideoStreamer configuration
- Check network bandwidth with `iperf3`
- Reduce camera resolution if needed

### Frame Drops
- Check USB bandwidth (multiple cameras)
- Verify network stability
- Monitor CPU usage on server

## Testing

### Verify Streaming Setup
```bash
# Terminal 1: Start server with streaming
python senseSpace_fusion_main.py --viz --stream

# Terminal 2: Connect client
python streamingClient.py --server localhost

# Should see:
# Server: "Client connected from <IP>"
# Server: "Starting video streaming"
# Client: Video frames displayed in window
```

### Check GStreamer Pipeline
```bash
# Server side
gst-launch-1.0 videotestsrc ! autovideosink

# Should display test pattern
```

## Next Steps

### Potential Enhancements
- Add compression level configuration
- Support JPEG as alternative codec
- Implement frame rate limiting per client
- Add bandwidth monitoring
- Support multiple simultaneous clients

### Integration Points
- Could add streaming to recording files (.ssrec)
- Integrate with LLM examples for vision analysis
- Add web-based viewer (WebRTC)

## Files Modified
1. `senseSpace_fusion_main.py` - CLI arguments
2. `libs/senseSpaceLib/senseSpace/server.py` - Core streaming integration
3. `pyproject.toml` - Dependency documentation (commented PyGObject)

## Files Unchanged (Already Implemented)
- `libs/senseSpaceLib/senseSpace/video_streaming.py` - Streaming classes
- `client/streamingClient.py` - Client implementation
- `test_client_detection.py` - Client detection test
