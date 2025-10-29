# Frame Recording and Playback

Record skeleton tracking frames for offline playback and testing.

## Features

- **Compressed Storage**: Uses zstandard compression for efficient file sizes
- **Seamless Playback**: Replay recordings with original timing
- **Loop Support**: Continuous playback for testing
- **Easy Integration**: Works with all client examples

## Recording

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
- Newline-delimited JSON (NDJSON)
- Zstandard compressed
- Each line contains:
  ```json
  {
    "timestamp": 1698588622.123,
    "frame_number": 42,
    "frame": { /* Frame data */ }
  }
  ```

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
