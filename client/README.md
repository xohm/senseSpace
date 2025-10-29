# SenseSpace Client

The SenseSpace client connects to a SenseSpace server and receives body tracking data.

## Table of Contents
- [Usage](#usage)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Network Protocol](#network-protocol)

## Usage

### Command Line Mode (Default)
```bash
# Connect to localhost:12345 and print frame information
python senseSpaceClient.py

# Connect to a specific server
python senseSpaceClient.py --server 192.168.1.100 --port 12346

# Short form
python senseSpaceClient.py -s 192.168.1.100 -p 12346
```

### Visualization Mode
```bash
# Open Qt OpenGL viewer with localhost:12345
python senseSpaceClient.py --viz

# Visualization mode with custom server
python senseSpaceClient.py --viz --server 192.168.1.100

# Full form
python senseSpaceClient.py --visualization --server 192.168.1.100 --port 12346
```

## Features

### Command Line Mode
- Connects to server via TCP
- Prints frame information including:
  - Timestamp
  - Number of tracked people
  - Floor height (if detected by server)
  - Person details (ID, tracking state, confidence, joint count)

### Visualization Mode  
- 3D OpenGL viewer displaying:
  - Tracked skeleton data with anatomically correct bone connections
  - Floor grid using server-detected floor height
  - Real-time updates
  - Interactive camera controls

### Camera Controls (Visualization Mode)
- **Left mouse drag**: Orbit camera around the scene
- **Right mouse drag**: Pan camera target
- **Mouse wheel**: Zoom in/out
- **Connection indicator**: Green square (connected) / Red square (disconnected)
- **People indicators**: Green squares showing number of tracked people

## Requirements

### Command Line Mode
- Python 3.6+
- senseSpace shared library (automatically installed)

### Visualization Mode  
- All command line requirements plus:
- PyQt5
- PyOpenGL

## Installation

The client uses the shared senseSpace library. Make sure it's installed:

```bash
# From the project root
cd libs/senseSpaceLib
pip install -e .
```

For visualization mode, install additional dependencies:
```bash
pip install PyQt5 PyOpenGL
```

## Network Protocol

The client receives JSON messages from the server with the following format:

```json
{
  "type": "frame",
  "data": {
    "timestamp": 1234567.890,
    "people": [...],
    "body_model": "BODY_34",
    "floor_height": 1450.0
  }
}
```

The `floor_height` field contains the ZED SDK detected floor height in millimeters, which is used for accurate floor grid positioning in the visualization.
