# Video Streaming Guide

## Overview

senseSpace supports real-time RGB and depth video streaming from ZED cameras to clients using GStreamer and RTP protocol.

## Features

- **H.265 Encoding**: Hardware-accelerated H.265 encoding for RGB streams
- **Lossless Depth**: H.265 lossless encoding for depth data
- **Multi-Camera Support**: Stream from multiple cameras simultaneously
- **Low Latency**: RTP streaming with optimized pipelines
- **Cross-Platform**: GPU encoding/decoding on Linux, Windows, and macOS
- **Network Efficient**: Compressed streams over UDP
- **Smart Resource Management**: Automatic start/stop based on active clients (saves CPU/GPU when no one is watching)

## Requirements

### GStreamer Installation

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get install \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-vaapi \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gstreamer-1.0
```

For NVIDIA GPU support:
```bash
sudo apt-get install gstreamer1.0-plugins-nvcodec
```

#### macOS
```bash
brew install gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly gst-libav
brew install pygobject3 gtk+3
```

####Windows
Download and install GStreamer from: https://gstreamer.freedesktop.org/download/

Install both runtime and development installers. Add GStreamer bin directory to PATH.

Install PyGObject:
```bash
pip install PyGObject
```

### Python Dependencies
```bash
pip install pygobject numpy
```

## Architecture

### Server Side
- Captures RGB and depth images from ZED cameras
- Encodes using H.265 (RGB) and H.265 lossless (depth)
- Streams via RTP over UDP
- Supports multiple cameras (composite stream)

### Client Side
- Receives RTP streams
- Decodes using hardware acceleration
- Provides callbacks for processed frames
- Displays in visualization widget (optional)

## Usage

### Server (Streaming)

```python
from senseSpace.video_streaming import VideoStreamer

# Create streamer with automatic client detection (default)
streamer = VideoStreamer(
    host='0.0.0.0',              # Bind to all interfaces
    rgb_port=5000,               # RGB stream port
    depth_port=5001,             # Depth stream port
    width=1280,
    height=720,
    framerate=30,
    enable_client_detection=True,  # Auto start/stop based on clients (default)
    client_timeout=5.0             # Remove inactive clients after 5 seconds
)

# Streaming starts automatically when first client connects
# No need to manually call streamer.start() with client detection enabled

# In your camera capture loop:
rgb_image = get_rgb_from_camera()    # BGR format, uint8
depth_image = get_depth_from_camera()  # uint16

streamer.push_rgb_frame(rgb_image)
streamer.push_depth_frame(depth_image)

# Check active client count
print(f"Active clients: {streamer.get_client_count()}")

# When done:
streamer.shutdown()  # Stops streaming and cleanup
```

**Client Detection Mechanism:**
- Server listens for client heartbeat messages on `rgb_port + 100` (default: port 5100)
- When first client connects → streaming starts automatically
- When all clients disconnect → streaming stops to save resources
- Clients send heartbeat every 2 seconds
- Server removes clients that haven't sent heartbeat in 5 seconds

### Client (Receiving)

```python
from senseSpace.video_streaming import VideoReceiver

def on_rgb_frame(frame):
    """Called when RGB frame is received"""
    print(f"RGB frame: {frame.shape}")
    # Display or process frame

def on_depth_frame(frame):
    """Called when depth frame is received"""
    print(f"Depth frame: {frame.shape}")
    # Display or process frame

# Create receiver with automatic heartbeat (default)
receiver = VideoReceiver(
    server_ip='192.168.1.100',
    rgb_port=5000,
    depth_port=5001,
    rgb_callback=on_rgb_frame,
    depth_callback=on_depth_frame,
    send_heartbeat=True,       # Sends heartbeat to server (default)
    heartbeat_interval=2.0     # Heartbeat every 2 seconds
)

# Start receiving (automatically triggers server to start streaming)
receiver.start()

# Keep running...
# Streaming will stop on server if this client disconnects
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    receiver.stop()
```

### Command Line Examples

#### Start Server with Streaming
```bash
cd server
python senseSpace_fusion_main.py --viz --stream
```

#### Start Client with Video Display
```bash
cd client/examples/streaming
python streamingClient.py --server 192.168.1.100
```

## Network Configuration

### Firewall Rules
Allow UDP traffic on streaming ports:

**Linux (iptables)**:
```bash
sudo iptables -A INPUT -p udp --dport 5000:5001 -j ACCEPT
```

**Linux (ufw)**:
```bash
sudo ufw allow 5000:5001/udp
```

**Windows Firewall**:
- Open Windows Defender Firewall
- New Inbound Rule → Port → UDP → 5000-5001

**macOS**:
```bash
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /path/to/python
```

### Port Configuration
Default ports:
- RGB stream: 5000 (UDP)
- Depth stream: 5001 (UDP)

For multiple camera setups, additional ports may be used (5002, 5003, etc.)

## Resource Management

### Automatic Start/Stop with Client Detection

By default, the streaming system **only encodes and transmits when clients are actively receiving**. This saves significant CPU/GPU resources when no one is watching:

**Without Client Detection (wastes resources):**
- Server encodes continuously even with 0 clients
- Network bandwidth consumed unnecessarily
- Higher power consumption

**With Client Detection (default, efficient):**
- Server idle when no clients connected
- Encoding starts automatically when first client connects
- Stops encoding when last client disconnects
- Typical CPU/GPU savings: **60-80%** when idle

**Manual Control:**
If you need streaming to always run, disable client detection:
```python
streamer = VideoStreamer(
    enable_client_detection=False,  # Always stream
    ...
)
streamer.start()  # Must manually start
```

### Multi-Client Efficiency

Multiple clients receive the **same encoded stream** via RTP multicast:
- 1 client = 1× encoding cost
- 10 clients = still 1× encoding cost (no increase!)
- Network handles packet distribution
- Zero additional CPU/GPU load per client

## Performance Optimization

### GPU Encoding

The system automatically selects GPU encoders when available:

- **Linux**: NVENC (NVIDIA) or VAAPI (Intel/AMD)
- **Windows**: NVENC or Media Foundation
- **macOS**: VideoToolbox (Apple Silicon/Intel)

### Bandwidth Considerations

Typical bandwidth usage (1280x720 @ 30fps):
- RGB stream: ~4-6 Mbps
- Depth stream: ~2-4 Mbps (lossless)
- Total per camera: ~6-10 Mbps

For multiple cameras or higher resolutions, ensure sufficient network bandwidth.

### Latency Tuning

Reduce latency by:
1. Using wired Ethernet instead of Wi-Fi
2. Enabling low-latency encoder presets
3. Reducing resolution/framerate if needed
4. Minimizing network hops between server and client

## Troubleshooting

### No Video Received
1. Check firewall allows UDP on streaming ports
2. Verify server IP address is correct
3. Ensure GStreamer is properly installed
4. Check network connectivity with `ping`

### Poor Video Quality
1. Check available bandwidth
2. Adjust encoder bitrate settings
3. Reduce resolution or framerate
4. Verify GPU encoding is being used

### High Latency
1. Use wired connection
2. Enable low-latency encoder preset
3. Reduce buffer sizes in appsink
4. Check for network congestion

### Checking GPU Encoding

Run server with debug logging to see encoder selection:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Look for encoder names in output:
- `nvh265enc` = NVIDIA GPU
- `vaapih265enc` = Intel/AMD GPU (Linux)
- `vtenc_h265` = Apple hardware (macOS)
- `x265enc` = Software fallback

## Multi-Camera Streaming

For multiple cameras, each camera can use separate port pairs:

```python
# Camera 1
streamer1 = VideoStreamer(rgb_port=5000, depth_port=5001, ...)

# Camera 2
streamer2 = VideoStreamer(rgb_port=5002, depth_port=5003, ...)

# Client receives from both
receiver1 = VideoReceiver(server_ip=ip, rgb_port=5000, depth_port=5001, ...)
receiver2 = VideoReceiver(server_ip=ip, rgb_port=5002, depth_port=5003, ...)
```

## Advanced Configuration

### Custom Encoder Settings

Modify encoder properties in `video_streaming.py`:

```python
# Example: Increase RGB bitrate
rgb_props = {'speed-preset': 'fast', 'bitrate': 8000}  # 8 Mbps
```

### Buffer Management

Adjust appsink buffer settings for memory/latency trade-offs:

```python
# In pipeline string:
appsink max-buffers=1 drop=true  # Low latency, drops frames
appsink max-buffers=10 drop=false  # Higher latency, no drops
```

## API Reference

### VideoStreamer

**Methods**:
- `start()`: Begin streaming
- `stop()`: Stop streaming
- `push_rgb_frame(frame: np.ndarray)`: Send RGB frame (H,W,3) uint8 BGR
- `push_depth_frame(frame: np.ndarray)`: Send depth frame (H,W) uint16

### VideoReceiver

**Methods**:
- `start()`: Begin receiving
- `stop()`: Stop receiving

**Callbacks**:
- `rgb_callback(frame: np.ndarray)`: Called on RGB frame reception
- `depth_callback(frame: np.ndarray)`: Called on depth frame reception

## See Also

- [GStreamer Documentation](https://gstreamer.freedesktop.org/documentation/)
- [RTP Streaming Guide](https://gstreamer.freedesktop.org/documentation/rtp/index.html)
- [H.265/HEVC Encoding](https://trac.ffmpeg.org/wiki/Encode/H.265)
