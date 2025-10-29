# senseSpace - Server

This directory contains the point cloud and skeleton tracking servers for senseSpace.

## Table of Contents
- [Available Servers](#available-servers)
- [Debug Screenshots](#debug-screenshots)
- [Recording and Playback](#recording-and-playback)
- [Qt Viewer Keyboard Controls](#qt-viewer-keyboard-controls)
- [Server Comparison](#server-comparison)
- [Room Calibration (Multi-Camera Fusion)](#room-calibration-multi-camera-fusion)
- [Network Ports](#network-ports)
- [Troubleshooting](#troubleshooting)

## Available Servers

### 1. `senseSpace_fusion_main.py` - Multi-Camera Fusion Server
For multi-camera setups with room calibration.

### 2. `pointCloudServer.py` - ZED SDK Segmentation Server
Standard server using ZED SDK for both skeleton tracking and person segmentation.

**Features:**
- Skeleton tracking (34 keypoints)
- Point cloud streaming (full scene or per-person)
- ZED SDK body segmentation masks
- Works with single camera or fusion mode

**Usage:**
```bash
# Single camera with per-person point clouds
python pointCloudServer.py --viz --mode per-person

# Full scene point cloud
python pointCloudServer.py --viz --mode full
```

### 3. `seg_pointcloud_server.py` - MediaPipe Segmentation Server
Advanced server using ZED SDK for skeleton tracking and **MediaPipe for high-quality person segmentation**.

**Features:**
- ZED SDK skeleton tracking (accurate person IDs)
- MediaPipe person segmentation (high-quality pixel-accurate masks)
- Per-person point clouds only
- **Superior segmentation quality** compared to ZED SDK

**Setup MediaPipe (one-time):**
```bash
cd server

# Install MediaPipe without dependencies to avoid numpy conflict
pip install --no-deps mediapipe

# Install MediaPipe dependencies (compatible with pyzed's numpy 2.x)
pip install -r requirements_segmentation.txt
```

**Why the special installation?**
MediaPipe requires numpy<2, but pyzed requires numpy>=2. Installing MediaPipe with `--no-deps` and then installing its other dependencies separately allows numpy 2.x from pyzed to remain installed. The warning about numpy incompatibility can be safely ignored - both libraries work correctly.

**Usage:**
```bash
# With visualization (default: 720p @ 60fps)
python seg_pointcloud_server.py --viz

# Without visualization
python seg_pointcloud_server.py --port 12345 --pc-port 12346

# With different quality settings
python seg_pointcloud_server.py --viz --quality 0  # 720p @ 60fps (default, best balance)
python seg_pointcloud_server.py --viz --quality 1  # 720p @ 30fps (better performance)
python seg_pointcloud_server.py --viz --quality 2  # VGA @ 60fps (faster, lower quality)
python seg_pointcloud_server.py --viz --quality 3  # VGA @ 30fps (fastest)
python seg_pointcloud_server.py --viz --quality 4  # 1080p @ 30fps (highest quality, slower)

# Recording mode - records camera streams to SVO files
python seg_pointcloud_server.py --viz
# Press 'r' to start/stop recording (saves to server/tmp/)

# Playback mode - replay recorded SVO files in endless loop
python seg_pointcloud_server.py --rec tmp/recording_12345.svo --viz  # Single camera
python seg_pointcloud_server.py --rec tmp/ --viz  # Fusion mode (auto-detects all SVO files)
```

**Quality Settings:**
- `--quality 0`: **HD720 @ 60 FPS** (1280×720) - Default, best balance
- `--quality 1`: HD720 @ 30 FPS (1280×720) - Better CPU performance
- `--quality 2`: VGA @ 60 FPS (672×376) - Faster processing
- `--quality 3`: VGA @ 30 FPS (672×376) - Fastest, lowest quality
- `--quality 4`: HD1080 @ 30 FPS (1920×1080) - Highest quality, CPU intensive

## Debug Screenshots

Both point cloud servers support debug screenshot functionality:

1. Run server with `--viz` flag
2. Press **Space** key in the Qt viewer window
3. Images are saved to `server/tmp/` folder:
   - `debug_camera_{person_id}_{timestamp}.png` - Original camera image
   - `debug_mask_*_{person_id}_{timestamp}.png` - Segmentation mask
   - `debug_overlay_{person_id}_{timestamp}.png` - Camera with mask overlay

## Recording and Playback

The `seg_pointcloud_server.py` supports recording camera streams to SVO (Stereolabs Video) files and playing them back in an endless loop.

### Recording

**Interactive Recording:**
1. Start the server with `--viz` flag
2. Press **'r'** key to start recording
3. Press **'r'** again to stop recording
4. Files are saved to `server/tmp/` directory

**File Naming:**
- Single camera: `recording_{timestamp}.svo`
- Fusion mode: `recording_cam{serial}_{timestamp}.svo` (one file per camera)

**Example:**
```bash
# Start server with visualization
python seg_pointcloud_server.py --viz

# Press 'r' to start recording
# Press 'r' again to stop recording
# Files saved to: server/tmp/recording_*.svo
```

### Playback

Playback mode allows you to replay recorded SVO files in an endless loop (when the file ends, it automatically restarts from the beginning).

**Single Camera Playback:**
```bash
python seg_pointcloud_server.py --rec tmp/recording_1234567890.svo --viz
```

**Playback:**
```bash
# Point to directory containing recording_cam*.svo files
python seg_pointcloud_server.py --rec tmp/ --viz

# The server automatically:
# 1. Discovers all SVO files in the directory
# 2. Extracts camera serial numbers from filenames
# 3. Initializes fusion mode with the recordings
# 4. **Note**: Fusion mode playback plays once and stops (looping not supported due to timestamp issues)
```

**Important Limitation**: SVO looping with fusion mode is not supported by the ZED SDK due to timestamp management in the fusion module. When SVO files restart, their timestamps reset to 0, which the fusion module interprets as old data. Fusion mode playback will play the recording once and then stop. For looping playback, use single-camera mode.

**Single Camera Playback** (supports looping):
```bash
python seg_pointcloud_server.py --rec tmp/recording_1234567890.svo --viz
```

**Use Cases:**
- Testing without physical cameras  
- Debugging specific scenarios repeatedly (**single-camera mode only**)
- Sharing reproducible test cases
- Offline development and algorithm testing

**Looping Behavior:**
- **Single camera**: Loops endlessly automatically ✓
- **Fusion mode**: Plays once then stops (ZED SDK limitation)
  - **Manual restart**: Press 'L' key to restart playback from the beginning
  - **Alternative**: Restart the server with `Ctrl+C` and run again

## Qt Viewer Keyboard Controls

When running with `--viz` flag, the Qt viewer supports these keyboard shortcuts:

- **Space**: Save debug images (camera, depth, segmentation masks)
- **'r'**: Toggle recording on/off
- **'l'**: Restart SVO playback (loop back to beginning - useful for fusion mode)
- **'p'**: Toggle point cloud display on/off
- **'c'**: Toggle camera frustum flip

## Server Comparison

| Feature | pointCloudServer.py | seg_pointcloud_server.py |
|---------|---------------------|--------------------------|
| Skeleton Tracking | ZED SDK | ZED SDK |
| Segmentation | ZED SDK | **MediaPipe** |
| Segmentation Quality | Good (coarse, ~176×142) | **Excellent (pixel-accurate)** |
| Setup Complexity | Simple | Moderate (numpy workaround) |
| Performance | Fast (~60 FPS) | Fast (~30-60 FPS) |
| Multi-camera | Yes (fusion) | **Yes (fusion with MediaPipe per-camera)** |
| Dependencies | Standard | Requires MediaPipe setup |

**Recommendation:** Use `seg_pointcloud_server.py` for best segmentation quality with single or multi-camera setups.

**How MediaPipe fusion works:**
- Runs MediaPipe segmentation on each camera independently (2D image processing)
- Extracts per-person point clouds from each camera
- Transforms point clouds to fusion coordinate space
- Merges point clouds from same person across all cameras
- Result: High-quality segmentation with proper multi-camera alignment

## Room Calibration (Multi-Camera Fusion)

The fusion server needs to be calibrated before it can be used. ZED SDK has a tool for this: ZED360.
https://www.stereolabs.com/docs/fusion/zed360

### Calibration Steps
- Setup all cameras and connect them to the computer. Be sure to use the right USB plugs, since each camera needs 5Gb/s, select the right plugs. Otherwise the camera will not work.
- Start the calibration software:
```bash
ZED360 
```
- In the application select the "Setup your Room" button.
- Set DepthMode to Neural.
- Press "Auto Discover" to get all cameras.
- Then press "Setup the Room".
- Press "Start Calibration" and walk through the space which is covered by the cameras.
- Press "Finish Calibration" when the room is covered and save the calibration JSON file into a folder. Best into the project folder of the server at "senseSpace/server/calib". Now the calibration file is set for this setup.

### Start Fusion Server
Go into the server folder and start the server (with visualization):
```bash
cd server
python3 senseSpace_fusion_main.py --viz
```
Don't forget to activate the python environment for this.

## Network Ports

Point cloud servers use two TCP ports:

1. **Skeleton Port** (default: 12345) - Skeleton tracking data
2. **Point Cloud Port** (default: 12346) - Point cloud data (compressed with zstd)

## Troubleshooting

### MediaPipe numpy conflict
```bash
# You may see: "pyzed 5.0 requires numpy<3.0,>=2.0, but you have numpy 1.26.4"
# This warning can be safely ignored - both libraries work correctly together.
```

### Port already in use
```bash
# Change the ports if defaults are taken
python seg_pointcloud_server.py --port 12350 --pc-port 12351
```

### Camera not detected
```bash
# Make sure ZED SDK is properly installed and camera is connected
# Test with: /usr/local/zed/tools/ZED_Explorer
```