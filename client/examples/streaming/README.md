# Streaming Client Example

Real-time RGB and depth video streaming from distributed ZED cameras with hardware-accelerated H.265 encoding and lossless depth preservation.

## Table of Contents

- [Why Use Video Streaming?](#why-use-video-streaming)
  - [Benefits](#benefits)
  - [Use Cases](#use-cases)
- [How It Works](#how-it-works)
  - [Architecture](#architecture)
  - [RTP Payload Multiplexing](#rtp-payload-multiplexing)
  - [Encoding Pipeline (Server)](#encoding-pipeline-server)
  - [Decoding Pipeline (Client)](#decoding-pipeline-client)
  - [Depth Quality](#depth-quality)
  - [Platform Support](#platform-support)
- [Features](#features)
- [Requirements](#requirements)
  - [System Requirements](#system-requirements)
  - [Software Dependencies](#software-dependencies)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Basic Usage (Video Only)](#basic-usage-video-only)
  - [With Skeleton Data](#with-skeleton-data)
  - [Multi-Camera Setup](#multi-camera-setup)
- [Command Line Arguments](#command-line-arguments)
- [Keyboard Controls](#keyboard-controls)
- [Display Layout](#display-layout)
- [Network Requirements](#network-requirements)
  - [Bandwidth](#bandwidth)
  - [Firewall Configuration](#firewall-configuration)
  - [Latency Expectations](#latency-expectations)
- [Troubleshooting](#troubleshooting)
  - [No Video Displayed](#no-video-displayed)
  - [Video Lag/Stuttering](#video-lagstuttering)
  - [Skeleton Data Not Showing](#skeleton-data-not-showing)
  - [GStreamer Errors](#gstreamer-errors)
- [Performance Tips](#performance-tips)
  - [Network Optimization](#network-optimization)
  - [GPU Utilization](#gpu-utilization)
  - [Display Optimization](#display-optimization)
  - [Bandwidth Reduction](#bandwidth-reduction)
- [Advanced Usage](#advanced-usage)
  - [Verify Depth Quality](#verify-depth-quality)
  - [Custom Video Callbacks](#custom-video-callbacks)
  - [Recording Streams](#recording-streams)
  - [Monitoring & Debugging](#monitoring--debugging)
- [Technical Specifications](#technical-specifications)
  - [Video Encoding](#video-encoding)
  - [Network Protocol](#network-protocol)
  - [Hardware Acceleration](#hardware-acceleration)
  - [Depth Quality Comparison](#depth-quality-comparison)
- [See Also](#see-also)

---

## Why Use Video Streaming?

### Benefits

**1. Remote Visualization**
- View camera feeds from anywhere on the network
- No need to be physically near the cameras
- Multiple clients can connect simultaneously
- Perfect for distributed installations

**2. Network Efficiency**
- **Single UDP port** for all cameras (RGB + Depth)
- H.265 compression reduces bandwidth by ~10x vs raw
- Hardware acceleration minimizes CPU/GPU load
- ~23-37 Mbps per camera (fits easily in gigabit ethernet)

**3. Flexible Deployment**
- Separate rendering machines from camera servers
- Client runs on lower-power devices (laptops, tablets)
- Server handles heavy lifting (tracking, depth processing)
- Scale to multiple cameras without client overhead

**4. Low Latency**
- Hardware-accelerated encoding/decoding (~5-15ms)
- RTP streaming protocol optimized for real-time
- Direct UDP transmission (no TCP overhead)
- Typical end-to-end latency: 50-100ms on local network

**5. High Quality**
- RGB: 4-8 Mbps with excellent quality
- Depth: **Lossless uint16 transport** (0.1mm precision)
- 60 fps capable (hardware dependent)
- TURBO colormap for beautiful depth visualization

### Use Cases

- **Interactive installations**: Remote monitoring and control
- **Multi-camera setups**: View all cameras from single client
- **Research**: Record and analyze from remote workstation
- **Performance**: Live visualization during shows/events
- **Development**: Debug tracking without physical access to cameras

## How It Works

### Architecture

```
┌─────────────┐                 ┌─────────────┐
│   Server    │                 │   Client    │
│             │                 │             │
│  ZED Cam 0  │─┐               │             │
│  ZED Cam 1  │─┤               │  Qt Window  │
│  ZED Cam 2  │─┤               │  + 3D View  │
│             │ │               │             │
│  Tracking   │ ├──TCP 12345──→ │  Skeletons  │
│  Encoding   │ │               │             │
│             │ │               │             │
│  nvh265enc  │ └──UDP 5000───→ │  nvh265dec  │
│  (GPU)      │  RTP Multiplex  │  (GPU)      │
└─────────────┘                 └─────────────┘
```

### RTP Payload Multiplexing

**All streams on single port 5000:**

```
Server sends:
├─ Camera 0 RGB   → Payload Type 96
├─ Camera 0 Depth → Payload Type 97
├─ Camera 1 RGB   → Payload Type 98
├─ Camera 1 Depth → Payload Type 99
├─ Camera 2 RGB   → Payload Type 100
└─ Camera 2 Depth → Payload Type 101

Client receives:
UDP:5000 → rtpptdemux (by payload type)
         ├─ PT96  → RGB  decoder → Camera 0 RGB
         ├─ PT97  → Depth decoder → Camera 0 Depth
         ├─ PT98  → RGB  decoder → Camera 1 RGB
         ├─ PT99  → Depth decoder → Camera 1 Depth
         ├─ PT100 → RGB  decoder → Camera 2 RGB
         └─ PT101 → Depth decoder → Camera 2 Depth
```

### Encoding Pipeline (Server)

**RGB Stream:**
```
ZED Camera (BGR)
  ↓
appsrc (BGR → NV12 conversion)
  ↓
nvh265enc (NVIDIA GPU, ~8 Mbps, low-latency-hq)
  ↓
rtph265pay (PT=96/98/100...)
  ↓
UDP:5000
```

**Depth Stream:**
```
ZED Camera (float32 mm)
  ↓
Convert to uint16 (tenths-mm, 0.1mm precision)
  ↓
appsrc (GRAY16_LE → Y444_16LE)
  ↓
nvh265enc (lossless mode, QP=0, ~20 Mbps)
  ↓
rtph265pay (PT=97/99/101...)
  ↓
UDP:5000
```

### Decoding Pipeline (Client)

```
UDP:5000
  ↓
rtpptdemux (separates by payload type)
  ↓
├─ RGB:  rtph265depay → h265parse → nvh265dec → BGR → Qt display
└─ Depth: rtph265depay → h265parse → nvh265dec → GRAY16_LE → colormap → Qt display
```

### Depth Quality

**Precision:** 0.1mm steps (uint16 tenths-of-millimeter)
- Input: ZED float32 depth (e.g., 1234.567 mm)
- Encoded: uint16 value 12345 (tenths-mm)
- Decoded: float32 value 1234.5 mm
- **Loss: ~0.05mm** (far smaller than ZED sensor noise of ±10-30mm)

**H.265 Encoding:**
- NVIDIA: QP=0 (mathematically lossless uint16 transport)
- Apple: quality=1.0 (near-lossless)
- Software: QP 0-5 (near-lossless)

### Platform Support

**Hardware Acceleration (automatic detection):**

| Platform | Encoder | Decoder | Quality |
|----------|---------|---------|---------|
| Linux (NVIDIA) | nvh265enc | nvh265dec | Lossless |
| Linux (Intel/AMD) | vaapih265enc | vaapih265dec | Near-lossless |
| Windows (NVIDIA) | nvh265enc | nvh265dec | Lossless |
| Windows (Intel/AMD) | mfh265enc | mfh265dec | High quality |
| macOS (M1/M2/Intel) | vtenc_h265 | vtdec_h265 | Near-lossless |
| Fallback | x265enc | avdec_h265 | Near-lossless |

System automatically detects best available encoder/decoder at runtime.

## Features

- **Single-port streaming**: All cameras on UDP port 5000 (RGB + Depth multiplexed)
- **Hardware-accelerated**: H.265 encoding/decoding on GPU (NVENC/NVDEC, VideoToolbox, VAAPI, Media Foundation)
- **Lossless depth**: 0.1mm precision, QP=0 encoding (bit-perfect uint16 transport on NVIDIA)
- **Multi-camera ready**: Automatic detection and demultiplexing by payload type
- **Low latency**: 50-100ms end-to-end on local network
- **Cross-platform**: Linux, Windows, macOS with automatic encoder/decoder selection
- **Bandwidth efficient**: ~23-37 Mbps per camera (RGB + Depth)
- **Real-time visualization**: 60 fps capable with TURBO colormap depth display
- **Optional skeleton data**: Receive and display body tracking via TCP

## Requirements

### System Requirements

- **Network**: Gigabit Ethernet recommended (100 Mbps minimum)
- **GPU**: NVIDIA, Intel, AMD, or Apple Silicon for hardware acceleration
- **OS**: Linux, Windows 10/11, or macOS 10.15+

### Software Dependencies

**GStreamer with H.265 support:**

- **Linux:**
  ```bash
  sudo apt-get install python3-gi python3-gi-cairo \
      gstreamer1.0-tools gstreamer1.0-plugins-base \
      gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
      gstreamer1.0-plugins-ugly gstreamer1.0-libav
  ```

- **macOS:**
  ```bash
  brew install gstreamer gst-plugins-base gst-plugins-good \
      gst-plugins-bad gst-plugins-ugly pygobject3
  ```

- **Windows:**
  - Download from: https://gstreamer.freedesktop.org/download/
  - Install both MSVC runtime AND development packages
  - Install all plugin sets (base, good, bad, ugly)

**Python packages:**
```bash
pip install PyQt5 numpy
# Note: PyGObject (gi) should be installed via system packages (see above)
```

**Verify installation:**
```bash
# Check GStreamer
gst-inspect-1.0 --version
gst-inspect-1.0 nvh265dec  # or vtdec_h265 on macOS

# Check Python bindings
python -c "import gi; gi.require_version('Gst', '1.0'); from gi.repository import Gst; print(Gst.version())"
```

## Usage

### Quick Start

**1. Start the server** (on machine with ZED cameras):
```bash
cd server
python senseSpace_fusion_main.py --stream --viz
```

**2. Start the client** (on any machine on same network):
```bash
cd client/examples/streaming
python streamingClient.py --server <server-ip> --stream-port 5000
```

You should see:
- Server logs showing encoder detection and streaming started
- Client logs showing decoder detection and connection
- Qt window with RGB and depth overlays

### Basic Usage (Video Only)

```bash
python streamingClient.py --server 192.168.1.100 --stream-port 5000
```

This receives **all cameras** (RGB + Depth) multiplexed on port 5000. Client automatically detects number of cameras from incoming payload types.

### With Skeleton Data

```bash
python streamingClient.py --server 192.168.1.100 --stream-port 5000 --skeleton-port 12345
```

This additionally receives skeleton tracking data via TCP and displays it in 3D.

### Multi-Camera Setup

**No configuration needed!** Client automatically detects cameras:

```bash
# Server with 3 cameras
python senseSpace_fusion_main.py --stream --viz  # num_cameras=3

# Client automatically receives all 3 cameras
python streamingClient.py --server 192.168.1.100 --stream-port 5000
```

Payload type allocation:
- Camera 0: RGB=PT96, Depth=PT97
- Camera 1: RGB=PT98, Depth=PT99
- Camera 2: RGB=PT100, Depth=PT101

All on single port 5000!

## Command Line Arguments

**Required:**
- `--server`: Server IP address (e.g., `192.168.1.100` or `localhost`)

**Streaming:**
- `--stream-port`: Single port for all streams (default: 5000)

**Optional:**
- `--skeleton-port`: TCP port for skeleton data (default: 12345)
- `--num-cameras`: Expected number of cameras (auto-detected if omitted)

**Examples:**
```bash
# Basic streaming
python streamingClient.py --server 192.168.1.100 --stream-port 5000

# With skeleton tracking
python streamingClient.py --server 192.168.1.100 --stream-port 5000 --skeleton-port 12345
```

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

### Bandwidth

**Per camera (at 60 fps, 672x376):**
- RGB stream: 8-12 Mbps (H.265 compressed)
- Depth stream: 15-25 Mbps (lossless H.265)
- **Total per camera: ~23-37 Mbps**

**Multi-camera total:**
- 1 camera: ~23-37 Mbps
- 2 cameras: ~46-74 Mbps
- 3 cameras: ~69-111 Mbps

**Network recommendations:**
- **Gigabit Ethernet (1000 Mbps)**: Recommended, supports 10+ cameras
- **Fast Ethernet (100 Mbps)**: Works for 1-2 cameras
- **Wi-Fi**: Not recommended (latency, packet loss, bandwidth competition)

### Firewall Configuration

**Server (camera machine):**
- Outbound UDP 5000 (streaming)
- Outbound TCP 12345 (skeleton data)

**Client (visualization machine):**
- Inbound UDP 5000 (streaming)
- Inbound TCP 12345 (skeleton data)

**Single firewall rule needed:** Open UDP port 5000 for all cameras!

### Latency Expectations

| Network Type | Typical Latency |
|--------------|-----------------|
| Direct connection (same switch) | 50-70ms |
| Local network (1-2 hops) | 70-100ms |
| Cross-subnet | 100-150ms |
| Wi-Fi | 150-300ms+ |

Latency breakdown:
- Encoding: 5-10ms (GPU)
- Network: 1-5ms (local)
- Decoding: 5-10ms (GPU)
- Display: 16-33ms (60-30 fps)
- **Total: ~50-100ms**

## Troubleshooting

### No Video Displayed

**1. Check server is streaming:**
```
Server logs should show:
[INFO] Using hardware H.265 encoder: nvh265enc
[INFO] Depth encoding mode: lossless (QP=0)
[INFO] Created 1 RGB pipelines on port 5000
[INFO] Created 1 lossless Depth pipelines on port 5000
[INFO] RGB/Depth streaming started
```

**2. Check client connection:**
```
Client logs should show:
[INFO] Using hardware H.265 decoder: nvh265dec
[INFO] Connected to rtpptdemux pad-added signal
[INFO] RGB receiver connected: Camera 0, PT=96
[INFO] Depth receiver connected: Camera 0, PT=97
```

**3. Verify network connectivity:**
```bash
ping <server_ip>
```

**4. Test UDP port:**
```bash
# On server - check if streaming
# Linux
sudo netstat -ulnp | grep 5000
# macOS
netstat -anv | grep 5000
# Windows (PowerShell)
netstat -ano | findstr 5000

# Or use gst-launch to test (all platforms):
gst-launch-1.0 -v udpsrc port=5000 caps="application/x-rtp" ! fakesink
```

**5. Check firewall:**
```bash
# Linux (ufw)
sudo ufw allow 5000/udp

# Linux (firewalld)
sudo firewall-cmd --add-port=5000/udp --permanent
sudo firewall-cmd --reload

# macOS
# System Preferences → Security & Privacy → Firewall → Firewall Options
# Or use command line:
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /usr/local/bin/python3
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp /usr/local/bin/python3

# Windows (run PowerShell as Administrator)
New-NetFirewallRule -DisplayName "senseSpace Streaming" -Direction Inbound -Protocol UDP -LocalPort 5000 -Action Allow
# Or command prompt:
netsh advfirewall firewall add rule name="senseSpace Streaming" dir=in action=allow protocol=UDP localport=5000
```

### Video Lag/Stuttering

**Network issues:**
1. Use **wired Ethernet** (Wi-Fi adds latency and packet loss)
2. Check network load:
   - Linux: `iftop` or `nload`
   - macOS: Activity Monitor → Network tab
   - Windows: Task Manager → Performance → Ethernet
3. Verify gigabit link:
   - Linux: `ethtool eth0` (look for "Speed: 1000Mb/s")
   - macOS: Hold Option, click Wi-Fi/Network icon (shows link speed)
   - Windows: Control Panel → Network Connections → Adapter properties
4. Close bandwidth-intensive apps (downloads, streaming services)

**GPU issues:**
1. Verify **hardware decoding** is active (check logs for encoder/decoder names)
2. Update GPU drivers:
   - NVIDIA: https://www.nvidia.com/drivers
   - AMD: https://www.amd.com/support
   - Intel: https://www.intel.com/content/www/us/en/download-center/home.html
3. Close other GPU-intensive applications
4. Check GPU load:
   - NVIDIA: `nvidia-smi` or `watch -n 1 nvidia-smi`
   - AMD (Linux): `radeontop`
   - macOS: Activity Monitor → Window → GPU History
   - Windows: Task Manager → Performance → GPU

**System issues:**
1. Lower resolution on server: Edit camera settings
2. Reduce framerate: 30 fps instead of 60
3. Close unnecessary applications
4. Check CPU usage (should be <30% with GPU decoding):
   - Linux: `htop` or `top`
   - macOS: Activity Monitor → CPU tab
   - Windows: Task Manager → Performance → CPU

**Debug with GStreamer:**
```bash
# Test stream directly (bypasses client code)
gst-launch-1.0 -v udpsrc port=5000 caps="application/x-rtp" ! \
  rtpptdemux name=demux \
  demux.src_96 ! capsfilter caps="application/x-rtp,media=video,clock-rate=90000,encoding-name=H265,payload=96" ! \
  rtph265depay ! h265parse ! nvh265dec ! videoconvert ! autovideosink
```

### Skeleton Data Not Showing

1. Verify skeleton server IP and port are correct
2. Check skeleton server is running and accepting connections
3. Look for connection errors in console output

### GStreamer Errors

**"Element not found" errors:**

1. **Check GStreamer installation:**
   ```bash
   gst-inspect-1.0 --version
   gst-inspect-1.0 udpsrc
   gst-inspect-1.0 rtph265depay
   gst-inspect-1.0 rtpptdemux  # For single-port mode
   ```

2. **Check hardware decoder:**
   ```bash
   # Linux NVIDIA
   gst-inspect-1.0 nvh265dec
   
   # macOS
   gst-inspect-1.0 vtdec_h265
   
   # Software fallback
   gst-inspect-1.0 avdec_h265
   ```

3. **Install missing plugins:**
   ```bash
   # Linux (Ubuntu/Debian)
   sudo apt-get install gstreamer1.0-plugins-bad gstreamer1.0-libav gstreamer1.0-plugins-ugly
   
   # macOS
   brew install gst-plugins-bad gst-plugins-ugly gst-libav
   
   # Windows
   # Reinstall GStreamer with all plugin packs
   ```

**"Failed to negotiate caps" errors:**

This usually means format mismatch. Check:
```bash
# Linux/macOS - Enable debug output
export GST_DEBUG=3
python streamingClient.py --server <ip> --stream-port 5000

# Windows (PowerShell)
$env:GST_DEBUG="3"
python streamingClient.py --server <ip> --stream-port 5000

# Windows (cmd)
set GST_DEBUG=3
python streamingClient.py --server <ip> --stream-port 5000
```

Look for caps negotiation errors. Common fixes:
- Update GStreamer to latest version
- Ensure both server and client use same GStreamer version
- Check video format support: `gst-inspect-1.0 nvh265dec`

**"Not-negotiated" errors:**

The rtpptdemux pad connection failed. Enable debug:
```bash
# Linux/macOS
export GST_DEBUG=rtpptdemux:5,rtph265depay:5
python streamingClient.py --server <ip> --stream-port 5000

# Windows (PowerShell)
$env:GST_DEBUG="rtpptdemux:5,rtph265depay:5"
python streamingClient.py --server <ip> --stream-port 5000

# Windows (cmd)
set GST_DEBUG=rtpptdemux:5,rtph265depay:5
python streamingClient.py --server <ip> --stream-port 5000
```

Check:
- Server is sending correct payload types (96, 97, 98...)
- Capsfilter is setting proper RTP caps
- Network is delivering packets (use Wireshark)

## Advanced Usage

### Verify Depth Quality

Check if depth is truly lossless:

```python
import numpy as np

# On server side, save original depth
np.save('server_depth.npy', depth_uint16)

# On client side, save received depth
np.save('client_depth.npy', received_depth_uint16)

# Compare
server = np.load('server_depth.npy')
client = np.load('client_depth.npy')
print(f"Identical: {np.array_equal(server, client)}")
print(f"Max difference: {np.abs(server - client).max()}")
# Should be: Identical: True, Max difference: 0
```

### Custom Video Callbacks

Process frames in your own code:

```python
from libs.senseSpaceLib.senseSpace.video_streaming import VideoReceiver

def my_rgb_callback(frame):
    """
    Process RGB frames
    Args:
        frame: numpy array (H, W, 3) uint8 BGR format
    """
    print(f"RGB: {frame.shape}, dtype: {frame.dtype}")
    # Your processing here...
    # Example: OpenCV operations, ML inference, etc.

def my_depth_callback(frame):
    """
    Process depth frames
    Args:
        frame: numpy array (H, W) uint16 format
              Values in tenths-of-millimeters (divide by 10 for mm)
    """
    depth_mm = frame.astype(np.float32) / 10.0
    print(f"Depth range: {depth_mm[depth_mm>0].min():.1f} - {depth_mm.max():.1f} mm")
    # Your processing here...

# Create receiver with custom callbacks
receiver = VideoReceiver(
    server_ip='192.168.1.100',
    stream_port=5000,
    rgb_callbacks=[my_rgb_callback],  # List for multi-camera
    depth_callbacks=[my_depth_callback]
)
receiver.start()
```

### Recording Streams

**Option 1: Record at server (before encoding):**
```python
# Server-side - save raw data
import cv2
cv2.imwrite(f'frame_{timestamp}.png', rgb_frame)
np.save(f'depth_{timestamp}.npy', depth_uint16)
```

**Option 2: Record H.265 stream (network level):**
```bash
# Linux - Capture RTP packets with tcpdump
sudo tcpdump -i eth0 -w stream_recording.pcap 'udp port 5000'

# macOS - Capture RTP packets
sudo tcpdump -i en0 -w stream_recording.pcap 'udp port 5000'

# Windows - Capture with Wireshark GUI or tshark
# Install Wireshark, then use tshark command line:
# tshark -i "Ethernet" -w stream_recording.pcap -f "udp port 5000"

# All platforms - Use GStreamer to save to file
gst-launch-1.0 -e udpsrc port=5000 caps="application/x-rtp" ! \
  rtpptdemux name=demux \
  demux.src_96 ! capsfilter caps="application/x-rtp,media=video,clock-rate=90000,encoding-name=H265,payload=96" ! \
  rtph265depay ! h265parse ! mp4mux ! filesink location=rgb.mp4 \
  demux.src_97 ! capsfilter caps="application/x-rtp,media=video,clock-rate=90000,encoding-name=H265,payload=97" ! \
  rtph265depay ! h265parse ! matroskamux ! filesink location=depth.mkv
```

**Option 3: Record at client (after decoding):**
```python
def save_rgb_callback(frame):
    cv2.imwrite(f'frames/rgb_{time.time()}.jpg', frame)

def save_depth_callback(frame):
    np.save(f'frames/depth_{time.time()}.npy', frame)

receiver = VideoReceiver(
    server_ip='192.168.1.100',
    stream_port=5000,
    rgb_callbacks=[save_rgb_callback],
    depth_callbacks=[save_depth_callback]
)
```

### Monitoring & Debugging

**Enable GStreamer debug logs:**
```bash
# Levels: 0=none, 1=error, 2=warning, 3=info, 4=debug, 5=trace
# These commands work on Linux, macOS, and Windows (bash/Git Bash)

# Linux/macOS
export GST_DEBUG=3  # Info level
export GST_DEBUG=rtpptdemux:5,rtph265depay:4,nvh265dec:4  # Focus on specific components
export GST_DEBUG_FILE=gst_debug.log

# Windows (PowerShell)
$env:GST_DEBUG="3"
$env:GST_DEBUG="rtpptdemux:5,rtph265depay:4,nvh265dec:4"
$env:GST_DEBUG_FILE="gst_debug.log"

# Windows (cmd)
set GST_DEBUG=3
set GST_DEBUG=rtpptdemux:5,rtph265depay:4,nvh265dec:4
set GST_DEBUG_FILE=gst_debug.log

python streamingClient.py --server <ip> --stream-port 5000
```

**Network diagnostics:**
```bash
# Linux - Check UDP traffic
sudo tcpdump -i eth0 -n 'udp port 5000'
iftop -i eth0 -f 'port 5000'  # Measure bandwidth
netstat -su | grep 'packet receive errors'  # Check packet loss

# macOS - Network diagnostics
sudo tcpdump -i en0 -n 'udp port 5000'
nettop -m udp  # Real-time network monitoring

# Windows - Network diagnostics (PowerShell)
Get-NetUDPEndpoint | Where-Object LocalPort -eq 5000  # Check port
Get-NetAdapter | Get-NetAdapterStatistics  # Check packet loss
```

**Performance profiling:**
```bash
# Python profiling (all platforms)
python -m cProfile -o profile.stats streamingClient.py --server <ip> --stream-port 5000

# Analyze (all platforms)
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

# GPU profiling
# NVIDIA (Linux/Windows): nvprof python streamingClient.py --server <ip> --stream-port 5000
# macOS: Use Instruments.app (Xcode) for GPU profiling
```

## Performance Tips

### Network Optimization

1. **Use Gigabit Ethernet**: Wi-Fi adds 50-200ms latency
2. **Direct connection**: Same switch or direct cable for lowest latency
3. **Jumbo frames** (advanced): MTU 9000 for ~10% less CPU overhead
   ```bash
   # Linux (if network supports it)
   sudo ip link set eth0 mtu 9000
   
   # macOS
   sudo ifconfig en0 mtu 9000
   
   # Windows (PowerShell - run as Administrator)
   Set-NetAdapterAdvancedProperty -Name "Ethernet" -DisplayName "Jumbo Packet" -DisplayValue "9014"
   ```
4. **QoS/DSCP marking**: Prioritize video traffic on managed switches

### GPU Utilization

1. **Verify hardware acceleration:**
   ```
   Server logs: [INFO] Using hardware H.265 encoder: nvh265enc
   Client logs: [INFO] Using hardware H.265 decoder: nvh265dec
   ```

2. **Monitor GPU usage:**
   ```bash
   # NVIDIA (Linux/Windows)
   watch -n 1 nvidia-smi  # Linux
   nvidia-smi -l 1  # Windows/Linux alternative
   # Should show encoder/decoder utilization
   
   # macOS
   sudo powermetrics --samplers gpu_power
   # Or: Activity Monitor → Window → GPU History
   
   # Windows (without NVIDIA GPU)
   # Task Manager → Performance → GPU (shows GPU usage graph)
   ```

3. **Multiple GPUs**: Server and client can use different GPUs
   - Server: Encoding GPU (NVENC on capture machine)
   - Client: Decoding GPU (NVDEC on display machine)

### Display Optimization

1. **Window size**: Smaller window = less rendering overhead
2. **VSync**: Disable if experiencing tearing (reduces latency by 16ms)
3. **Compositor**: Disable desktop effects on Linux for lower latency
4. **Dedicated display**: Use separate GPU output for client

### Bandwidth Reduction

If network is limited:

1. **Lower resolution** (server-side):
   ```python
   camera_width=640, camera_height=360  # Instead of 672x376
   ```

2. **Reduce framerate**:
   ```python
   framerate=30  # Instead of 60
   ```

3. **Adjust RGB bitrate** (trades quality for bandwidth):
   ```python
   # In video_streaming.py, GStreamerPlatform.get_encoder()
   'bitrate': 2000  # Instead of 4000 (halves RGB bandwidth)
   ```

4. **Lossy depth** (not recommended unless desperate):
   ```python
   # Change depth encoder QP from 0 to 23 (saves ~15 Mbps but loses precision)
   ```

## Technical Specifications

### Video Encoding

**RGB Stream:**
- Codec: H.265/HEVC (Main profile)
- Format: BGR → NV12 → H.265
- Bitrate: 4-8 Mbps (configurable)
- Resolution: 672x376 (default, from ZED)
- Framerate: 30-60 fps
- Latency: 5-10ms encoding
- Container: RTP (RFC 3984)

**Depth Stream:**
- Codec: H.265/HEVC (Main 10 profile for 10-bit+)
- Format: GRAY16_LE → Y444_16LE → H.265
- Quality: Lossless (QP=0 on NVIDIA) or near-lossless
- Precision: 0.1mm (uint16 tenths-of-millimeter)
- Range: 0-6553.5mm (65535 / 10)
- Bitrate: 15-25 Mbps (lossless, variable)
- Latency: 5-10ms encoding
- Container: RTP (RFC 3984)

### Network Protocol

**RTP Multiplexing:**
- Protocol: RTP over UDP (RFC 3550)
- Port: Single UDP port (default 5000)
- Payload Types:
  - PT 96, 98, 100... = RGB streams (even)
  - PT 97, 99, 101... = Depth streams (odd)
- MTU: 1400 bytes (configurable, 8192 for jumbo frames)
- Packet loss: Handled by I-frame refresh (gop-size=1 for depth)

**Payload Type Formula:**
```python
pt_rgb = 96 + (camera_index * 2)    # Even PTs for RGB
pt_depth = 97 + (camera_index * 2)  # Odd PTs for Depth
```

**Example allocation (3 cameras):**
| Camera | RGB PT | Depth PT |
|--------|--------|----------|
| 0      | 96     | 97       |
| 1      | 98     | 99       |
| 2      | 100    | 101      |

### Hardware Acceleration

**Encoder Selection (priority order):**

Linux:
1. `nvh265enc` - NVIDIA NVENC (Kepler+)
2. `vaapih265enc` - Intel Quick Sync / AMD VCE
3. `x265enc` - Software fallback

Windows:
1. `nvh265enc` - NVIDIA NVENC
2. `mfh265enc` - Media Foundation (Intel/AMD)
3. `x265enc` - Software fallback

macOS:
1. `vtenc_h265` - VideoToolbox (M1/M2/T2)
2. `x265enc` - Software fallback

**Decoder Selection (priority order):**

Linux:
1. `nvh265dec` - NVIDIA NVDEC
2. `vaapih265dec` - VAAPI
3. `avdec_h265` - Software (libav)

Windows:
1. `nvh265dec` - NVIDIA NVDEC
2. `mfh265dec` - Media Foundation
3. `avdec_h265` - Software

macOS:
1. `vtdec_h265` - VideoToolbox
2. `avdec_h265` - Software

### Depth Quality Comparison

| Platform | Encoder | QP | Precision | Bandwidth | Lossless? |
|----------|---------|-----|-----------|-----------|-----------|
| Linux (NVIDIA) | nvh265enc | 0 | 0.1mm | 20-25 Mbps | **Yes** (bit-perfect) |
| Windows (NVIDIA) | nvh265enc | 0 | 0.1mm | 20-25 Mbps | **Yes** (bit-perfect) |
| macOS | vtenc_h265 | - | 0.1mm | 18-23 Mbps | Near-lossless |
| Software | x265enc | 0-5 | 0.1mm | 15-20 Mbps | Near-lossless |

**Note:** All platforms maintain 0.1mm precision. "Lossless" refers to uint16 transport being bit-identical after decode. The float32→uint16 conversion always introduces 0.1mm quantization.

## See Also

- [VIDEO_STREAMING_GUIDE.md](../../../VIDEO_STREAMING_GUIDE.md) - Complete streaming architecture documentation
- [Server README](../../../server/README.md) - Server setup and multi-camera configuration
- [Client README](../../README.md) - General client usage and examples
- [OPTIMIZATION_GUIDE.md](../../../OPTIMIZATION_GUIDE.md) - Performance tuning
- [GStreamer RTP documentation](https://gstreamer.freedesktop.org/documentation/rtp/index.html)
- [H.265/HEVC specification](https://en.wikipedia.org/wiki/High_Efficiency_Video_Coding)
