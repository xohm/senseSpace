# Video Recording Metadata - Future-Proof Design

## Complete Metadata Storage

The .ssrec v2.0 format stores **all necessary information** to decode video, regardless of codec or future changes.

### Header Metadata (JSON)

```json
{
  "version": "2.0",
  "record_pointcloud": true,
  "record_video": true,
  "compression": "zstd",
  "compression_level": 3,
  "start_time": 1730891234.567,
  
  "video_cameras": [
    {
      "camera_idx": 0,
      "width": 1280,
      "height": 720,
      "fps": 30,
      "codec": "h265"
    },
    {
      "camera_idx": 1,
      "width": 1280,
      "height": 720,
      "fps": 30,
      "codec": "h265"
    },
    {
      "camera_idx": 2,
      "width": 1920,
      "height": 1080,
      "fps": 60,
      "codec": "h264"  // Example: different codec per camera
    }
  ]
}
```

### Metadata Fields Explained

| Field | Purpose | Example | Future-Proof |
|-------|---------|---------|--------------|
| **version** | Format version | "2.0" | ✅ Enables format evolution |
| **codec** | Video codec name | "h265", "h264", "vp9" | ✅ Decoder selects correct parser |
| **width** | Video width | 1280 | ✅ Essential for decoding |
| **height** | Video height | 720 | ✅ Essential for decoding |
| **fps** | Frames per second | 30 | ✅ For playback timing |
| **camera_idx** | Camera identifier | 0, 1, 2 | ✅ Multi-camera support |

### What Enables Future Codec Support

#### 1. **Codec Field is Dynamic**
```python
codec = camera_info.get('codec', 'h265')  # Default to h265 for backward compatibility
```

#### 2. **Decoder is Codec-Aware**
```python
codec_map = {
    'h265': ('h265parse', 'avdec_h265'),
    'h264': ('h264parse', 'avdec_h264'),
    'vp8': ('', 'vp8dec'),
    'vp9': ('', 'vp9dec'),
    'av1': ('av1parse', 'av1dec'),
    'mpeg4': ('mpeg4videoparse', 'avdec_mpeg4'),
}

parser, decoder = codec_map[codec_lower]
```

#### 3. **Pipeline is Built Dynamically**
```python
pipeline_parts = ["appsrc name=src"]
if parser:  # Some codecs don't need parsers
    pipeline_parts.append(parser)
pipeline_parts.extend([decoder, "videoconvert", ...])
pipeline_str = " ! ".join(pipeline_parts)
```

### Example: Adding Support for New Codec (AV1)

**Recording side** (video_streaming.py):
```python
# Register camera with AV1 codec
recorder.register_video_camera(
    camera_idx=0,
    width=1920,
    height=1080,
    fps=30,
    codec='av1'  # New codec!
)
```

**Playback side** (recorder.py):
```python
# Decoder automatically uses correct GStreamer elements
codec_map = {
    'h265': ('h265parse', 'avdec_h265'),
    'h264': ('h264parse', 'avdec_h264'),
    'av1': ('av1parse', 'av1dec'),  # Just add to map!
}
```

**That's it!** The file format doesn't change, just add the codec to the map.

## Metadata Completeness Checklist

### ✅ Essential Playback Information
- [x] Video codec (h265, h264, etc.)
- [x] Resolution (width × height)
- [x] Frame rate (fps)
- [x] Camera index (for multi-camera)
- [x] Format version (for compatibility)

### ✅ Optional But Useful
- [x] Compression type (zstd, none)
- [x] Compression level
- [x] Recording start time
- [x] Point cloud recording flag

### ✅ Implicit Information
- [x] NAL unit format (stored as-is, codec-specific)
- [x] Color format (RGB stored as BGR for OpenCV compatibility)
- [x] Depth format (grayscale encoded, can be extended)

## Future Extension Points

### Easy to Add (No Format Change)

1. **New Video Codec**
   - Add to `codec_map` in decoder
   - Set `codec='newcodec'` when registering camera
   - File format unchanged!

2. **Per-Camera Settings**
   ```json
   "video_cameras": [{
     "camera_idx": 0,
     "codec": "h265",
     "bitrate": 5000000,      // Can add new fields
     "profile": "main",       // Without breaking old readers
     "colorspace": "bt709"
   }]
   ```

3. **Global Video Settings**
   ```json
   {
     "version": "2.0",
     "video_encoding_params": {  // New section
       "preset": "ultrafast",
       "tune": "zerolatency"
     }
   }
   ```

### Requires Format Change (New Version)

1. **Different Container Format**
   - Would need version 3.0
   - Could store video in separate chunks
   - Old readers would still work (skip video)

2. **Multiple Streams Per Camera**
   - Version 3.0: support 4K + 1080p from same camera
   - Backward compatible with version detection

3. **Variable Frame Rate**
   - Version 3.0: store timestamps per frame
   - Current: assumes constant fps

## Backward Compatibility Strategy

### Version Detection
```python
version = header.get('version', '1.0')
if version == '1.0':
    # No video support
elif version == '2.0':
    # Video with codec metadata
elif version == '3.0':
    # Future enhancements
```

### Graceful Degradation
```python
# If codec not recognized, try fallback
if codec_lower not in codec_map:
    print(f"[WARNING] Unsupported codec '{codec}', trying h265 as fallback")
    codec_lower = 'h265'
```

### Forward Compatibility
Old players reading new files:
- Read version "2.0" (or "3.0" in future)
- Skip unknown JSON fields
- Skip binary video sections
- Still get skeleton + point cloud data

## Real-World Examples

### Example 1: Mixed Codec Recording
```json
"video_cameras": [
  {"camera_idx": 0, "codec": "h265", "width": 1280, "height": 720},
  {"camera_idx": 1, "codec": "h264", "width": 1920, "height": 1080},
  {"camera_idx": 2, "codec": "vp9", "width": 640, "height": 480}
]
```
**Result**: Each camera decoded with appropriate codec, all in one file.

### Example 2: Future AV1 Support
```python
# Someone in 2026 adds AV1 support
codec_map['av1'] = ('av1parse', 'av1dec')

# Now files recorded with AV1 in 2026 work perfectly
# Files from 2025 with H.265 still work perfectly
```

### Example 3: Streaming Service Archives
```json
// Archive from production system
{
  "version": "2.0",
  "recording_date": "2025-11-09",
  "session_id": "prod_abc123",
  "video_cameras": [
    {
      "camera_idx": 0,
      "serial": "33253574",
      "codec": "h265",
      "hardware_encoder": "nvenc_h265",
      "bitrate": 8000000,
      "location": "front_camera"
    }
  ]
}
```
**Result**: Complete metadata for archival/debugging, all optional fields ignored by basic players.

## Codec Support Matrix

| Codec | Parser | Decoder | Status | Use Case |
|-------|--------|---------|--------|----------|
| H.265/HEVC | h265parse | avdec_h265 | ✅ Implemented | High quality, efficient |
| H.264/AVC | h264parse | avdec_h264 | ✅ Supported | Broad compatibility |
| VP8 | - | vp8dec | ✅ Supported | WebM/WebRTC |
| VP9 | - | vp9dec | ✅ Supported | Better than VP8 |
| AV1 | av1parse | av1dec | ✅ Supported | Future standard |
| MPEG-4 | mpeg4videoparse | avdec_mpeg4 | ✅ Supported | Legacy support |

## Summary: What Makes This Future-Proof?

1. ✅ **Codec stored as string** - Easy to add new codecs
2. ✅ **Resolution in metadata** - No guessing dimensions
3. ✅ **Version field** - Format can evolve
4. ✅ **Dynamic pipeline building** - Decoder adapts to codec
5. ✅ **Fallback mechanism** - Graceful degradation
6. ✅ **Extensible JSON** - Can add fields without breaking old readers
7. ✅ **Per-camera metadata** - Each camera can use different codec
8. ✅ **Codec-agnostic binary format** - NAL units stored as-is

**Bottom Line**: You can record with H.265 today, switch to AV1 tomorrow, and H.266 in 2027. All files remain playable because the decoder reads the codec from the metadata and builds the appropriate GStreamer pipeline.

The format is **fully self-describing** - everything needed for playback is in the file!
