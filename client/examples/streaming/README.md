# Streaming Client Example

This example demonstrates how to receive and visualize RGB and depth video streams from a senseSpace server.

## Features

- Receives H.265 encoded RGB video stream
- Receives H.265 lossless encoded depth stream
- Displays both streams as overlays in top-left corner (20% size)
- Optionally receives skeleton tracking data
- Hardware-accelerated decoding
- Low-latency RTP streaming

## Requirements

- GStreamer with H.265 support (see main VIDEO_STREAMING_GUIDE.md)
- PyQt5
- NumPy
- senseSpaceLib

```bash
pip install PyQt5 numpy pygobject
```

## Usage

### Basic Usage (Video Only)

```bash
python streamingClient.py --server 192.168.1.100
```

This connects to the server at `192.168.1.100` and receives RGB (port 5000) and depth (port 5001) streams.

### With Skeleton Data

```bash
python streamingClient.py --server 192.168.1.100 --skeleton-server 192.168.1.100 --skeleton-port 5555
```

This additionally receives skeleton tracking data and displays it in 3D.

### Custom Ports

```bash
python streamingClient.py --server 192.168.1.100 --rgb-port 5002 --depth-port 5003
```

### Multiple Cameras

For multi-camera setups, run multiple client instances with different ports:

```bash
# Camera 1
python streamingClient.py --server 192.168.1.100 --rgb-port 5000 --depth-port 5001

# Camera 2 (in separate terminal)
python streamingClient.py --server 192.168.1.100 --rgb-port 5002 --depth-port 5003
```

## Command Line Arguments

- `--server`: **Required**. Server IP address
- `--rgb-port`: RGB stream port (default: 5000)
- `--depth-port`: Depth stream port (default: 5001)
- `--skeleton-server`: Skeleton data server IP (default: same as --server)
- `--skeleton-port`: Skeleton data port (default: 5555)

## Keyboard Controls

- **O**: Toggle joint orientation visualization
- **P**: Toggle point cloud display
- **Q** or close window: Quit

## Display Layout

The visualization window shows:
- **3D Scene**: Skeleton data, floor grid, camera frustums (center)
- **RGB Overlay**: Top-left corner, ~20% of window size
- **Depth Overlay**: Next to RGB, top-left area

## Network Requirements

### Firewall Configuration

Ensure the client can receive UDP traffic on the streaming ports. No incoming firewall rules needed on client side for RTP, but router/NAT may need configuration.

### Bandwidth

Expected bandwidth per camera:
- RGB: 4-6 Mbps
- Depth: 2-4 Mbps
- Skeleton data: <1 Mbps
- **Total**: ~6-11 Mbps per camera

## Troubleshooting

### No Video Displayed

1. Check server is streaming: Server logs should show "Video streaming started"
2. Verify network connectivity: `ping <server_ip>`
3. Check ports are not blocked by firewall
4. Ensure GStreamer is properly installed: `gst-inspect-1.0 udpsrc`

### Video Lag/Stuttering

1. Check network quality (wired Ethernet recommended)
2. Verify GPU decoding is enabled (check logs for decoder name)
3. Reduce resolution/framerate on server side
4. Close other bandwidth-intensive applications

### Skeleton Data Not Showing

1. Verify skeleton server IP and port are correct
2. Check skeleton server is running and accepting connections
3. Look for connection errors in console output

### GStreamer Errors

If you see GStreamer errors:

1. Check GStreamer installation:
   ```bash
   gst-inspect-1.0 --version
   gst-inspect-1.0 udpsrc
   gst-inspect-1.0 rtph265depay
   ```

2. Verify Python GObject bindings:
   ```python
   import gi
   gi.require_version('Gst', '1.0')
   from gi.repository import Gst
   print(Gst.version())  # Should print GStreamer version
   ```

3. Check for missing plugins:
   ```bash
   # Linux
   sudo apt-get install gstreamer1.0-plugins-bad gstreamer1.0-libav
   
   # macOS
   brew install gst-plugins-bad gst-libav
   ```

## Advanced Usage

### Custom Video Callbacks

You can create your own callbacks to process frames:

```python
def my_rgb_callback(frame):
    """Process RGB frames"""
    print(f"RGB: {frame.shape}, dtype: {frame.dtype}")
    # Do custom processing...
    # frame is numpy array (H, W, 3) uint8 BGR

def my_depth_callback(frame):
    """Process depth frames"""
    print(f"Depth: {frame.shape}, dtype: {frame.dtype}")
    # frame is numpy array (H, W) uint16
    # Process depth data...

receiver = VideoReceiver(
    server_ip='192.168.1.100',
    rgb_port=5000,
    depth_port=5001,
    rgb_callback=my_rgb_callback,
    depth_callback=my_depth_callback
)
```

### Saving Streams to File

You can modify the receiver pipeline to save streams:

```python
# In video_streaming.py, modify pipeline to add tee and filesink
pipeline_str = (
    f"udpsrc port={self.rgb_port} ! "
    f"rtph265depay ! tee name=t "
    f"t. ! queue ! {decoder_name} ! videoconvert ! appsink "
    f"t. ! queue ! h265parse ! mp4mux ! filesink location=output.mp4"
)
```

## Performance Tips

1. **Use Wired Ethernet**: Wi-Fi adds latency and packet loss
2. **GPU Decoding**: Ensure GPU decoders are being used (check logs)
3. **Close Other Apps**: Free up bandwidth and CPU/GPU resources
4. **Adjust Window Size**: Smaller window = less rendering overhead
5. **Disable Unnecessary Features**: Turn off point cloud or skeleton if not needed

## See Also

- `../../VIDEO_STREAMING_GUIDE.md` - Complete streaming documentation
- `../../../server/README.md` - Server setup and configuration
- `../../README.md` - General client usage
