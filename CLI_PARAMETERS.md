# Command-Line Parameters for SenseSpace Server

## Overview
The SenseSpace fusion server now supports runtime configuration through command-line parameters, eliminating the need to edit source code for common settings adjustments.

## New Parameters

### 1. Resolution (`--resolution`)
**Type**: Integer choice (0, 1, or 2)  
**Default**: 2 (VGA)

Controls the ZED camera resolution:
- `0` = HD720 (1280x720) - Higher quality, good for 60fps
- `1` = HD1080 (1920x1080) - Limited to 30fps
- `2` = VGA (672x376) - **Default, best performance at 60fps**

**Example**:
```bash
python senseSpace_fusion_main.py --resolution 2
```

### 2. Frame Rate (`--fps`)
**Type**: Integer choice (30 or 60)  
**Default**: 60

Controls the camera frame rate:
- `30` = 30 frames per second
- `60` = 60 frames per second - **Default for smooth tracking**

**Note**: HD1080 resolution (--resolution 1) only supports 30fps. The system will automatically adjust if you specify 60fps with HD1080.

**Example**:
```bash
python senseSpace_fusion_main.py --fps 60
```

### 3. Tracking Accuracy (`--accuracy`)
**Type**: Integer choice (0 or 1)  
**Default**: 1 (ACCURATE)

Controls the body tracking model:
- `0` = FAST - Better performance, lower quality
- `1` = ACCURATE - **Default, higher quality tracking**

**Example**:
```bash
python senseSpace_fusion_main.py --accuracy 1
```

### 4. Body Tracking Filter (`--filter`)
**Type**: Integer choice (0 or 1)  
**Default**: 0 (disabled)

Controls the duplicate body tracking filter:
- `0` = Disabled - **Default, recommended for stable tracking**
- `1` = Enabled - May cause flaky tracking behavior

**Example**:
```bash
python senseSpace_fusion_main.py --filter 0
```

### 5. Maximum Detection Range (`--max-range`)
**Type**: Float  
**Default**: 5.0

Controls the maximum distance (in meters) at which bodies will be detected and tracked. Bodies beyond this distance will be ignored.

**Range**: Typically 1.0 to 10.0 meters  
**Recommended**: 5.0 meters for most indoor scenarios

**Example**:
```bash
python senseSpace_fusion_main.py --max-range 5.0
```

**Use cases**:
- Reduce to 3.0-4.0m for close-range interactions
- Increase to 7.0-8.0m for larger spaces
- Lower values may improve performance and reduce false detections

## Validation

### Resolution/FPS Compatibility
The system automatically validates that fps is compatible with the selected resolution:
- If you specify `--resolution 1 --fps 60`, the system will warn and adjust fps to 30
- HD720 and VGA support both 30fps and 60fps

**Warning message example**:
```
[WARNING] HD1080 resolution only supports 30fps. Adjusting fps to 30.
```

## Complete Example

Run with optimal settings (default):
```bash
python server/senseSpace_fusion_main.py
```

Run with custom configuration:
```bash
python server/senseSpace_fusion_main.py --resolution 0 --fps 60 --accuracy 0 --filter 0 --max-range 5.0
```

Run in HD1080 mode (automatically adjusts to 30fps):
```bash
python server/senseSpace_fusion_main.py --resolution 1 --fps 60
# System output: [WARNING] HD1080 resolution only supports 30fps. Adjusting fps to 30.
```

Run with ACCURATE tracking model:
```bash
python server/senseSpace_fusion_main.py --accuracy 1
```

Run with shorter detection range for close interactions:
```bash
python server/senseSpace_fusion_main.py --max-range 3.0
```

## System Output

When starting the server, you'll see configuration information:
```
[INFO] Camera configuration:
[INFO]   Resolution: VGA (672x376)
[INFO]   FPS: 60
[INFO]   Tracking accuracy: ACCURATE
[INFO]   Body filter: disabled
[INFO]   Max detection range: 5.0m
```

## Existing Parameters

These new parameters work alongside existing server parameters:

### Server Configuration
- `--mode {server,viz}` - Run mode (default: server)
- `--host HOST` - Server host (default: 0.0.0.0)
- `--port PORT` - Server port (default: 12345)
- `--udp` - Use UDP instead of TCP
- `--no-cameras` - Debug mode without cameras

### Video Streaming
- `--stream` - Enable video streaming
- `--stream-host HOST` - Multicast address (default: 239.255.0.1)
- `--stream-port PORT` - Stream port (default: 5000)

### Legacy Filter Control
- `--no-filter` - Disable body filter (same as `--filter 0`)

**Note**: The new `--filter` parameter overrides `--no-filter` if both are specified.

## Recommended Settings

### For Stable Tracking (Default)
```bash
python server/senseSpace_fusion_main.py
# Equivalent to:
# --resolution 2 --fps 60 --accuracy 1 --filter 0 --max-range 5.0
```

### For High Quality (Default)
```bash
python server/senseSpace_fusion_main.py --resolution 1 --accuracy 1
# Note: HD1080 forces 30fps automatically
```

### For Maximum Performance
```bash
python server/senseSpace_fusion_main.py --accuracy 0
# Uses FAST model for better FPS
```

### For Performance Testing
```bash
python server/senseSpace_fusion_main.py --resolution 2 --fps 60 --accuracy 0 --filter 0
# Same as default - VGA already optimized for performance
```

### For Higher Quality
```bash
python server/senseSpace_fusion_main.py --resolution 0
# Uses HD720 for better image quality
```

### For Close-Range Interactions
```bash
python server/senseSpace_fusion_main.py --max-range 3.0
# Detects bodies only within 3 meters
```

### For Large Spaces
```bash
python server/senseSpace_fusion_main.py --max-range 8.0
# Extends detection to 8 meters
```

## Implementation Details

### Modified Files
1. **server/senseSpace_fusion_main.py**
   - Added argparse parameters
   - Added resolution/fps validation
   - Passes parameters to SenseSpaceServer

2. **libs/senseSpaceLib/senseSpace/server.py**
   - Updated `__init__()` to accept new parameters
   - Added runtime configuration override of class defaults
   - Added configuration logging

### Parameter Flow
```
Command Line → argparse → senseSpace_fusion_main.py → SenseSpaceServer.__init__()
→ Override class defaults → Applied during camera initialization
```

### Backward Compatibility
All parameters have sensible defaults matching the optimized settings. Existing scripts and deployment setups will continue to work without modification.
