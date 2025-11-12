# Command-Line Parameters for SenseSpace Server

## Overview
The SenseSpace fusion server now supports runtime configuration through command-line parameters, eliminating the need to edit source code for common settings adjustments.

## New Parameters

### 1. Resolution (`--resolution`)
**Type**: Integer choice (0, 1, or 2)  
**Default**: 0 (HD720)

Controls the ZED camera resolution:
- `0` = HD720 (1280x720) - **Default, recommended for 60fps**
- `1` = HD1080 (1920x1080) - Limited to 30fps
- `2` = VGA (672x376) - Lower resolution

**Example**:
```bash
python senseSpace_fusion_main.py --resolution 0
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
**Default**: 0 (FAST)

Controls the body tracking model:
- `0` = FAST - **Default, better performance, recommended for 60fps**
- `1` = ACCURATE - Higher quality but slower

**Example**:
```bash
python senseSpace_fusion_main.py --accuracy 0
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
python server/senseSpace_fusion_main.py --resolution 0 --fps 60 --accuracy 0 --filter 0
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

## System Output

When starting the server, you'll see configuration information:
```
[INFO] Camera configuration:
[INFO]   Resolution: HD720 (1280x720)
[INFO]   FPS: 60
[INFO]   Tracking accuracy: FAST
[INFO]   Body filter: disabled
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
# --resolution 0 --fps 60 --accuracy 0 --filter 0
```

### For High Quality (Slower)
```bash
python server/senseSpace_fusion_main.py --resolution 1 --accuracy 1
# Note: HD1080 forces 30fps automatically
```

### For Performance Testing
```bash
python server/senseSpace_fusion_main.py --resolution 2 --fps 60 --accuracy 0 --filter 0
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
