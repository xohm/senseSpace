# Vector Math Examples - Floor Projection & Rectangle Interaction

## Table of Contents
- [Overview](#overview)
- [vecExample.py - Interactive Floor Rectangle](#vecexamplepy---interactive-floor-rectangle)
  - [Features](#features)
  - [Controls](#controls)
  - [Usage](#usage)
- [How It Works](#how-it-works)
  - [Ray-Plane Intersection](#ray-plane-intersection)
  - [Point-in-Rectangle Detection](#point-in-rectangle-detection)
  - [2D Coordinate Extraction](#2d-coordinate-extraction)
  - [Coordinate System](#coordinate-system)
- [Vector Math Helper Library](#vector-math-helper-library)
  - [getNormal()](#getnormal)
  - [getPlaneIntersection()](#getplaneintersection)
  - [isPointInRectangle()](#ispointinrectangle)
  - [getPointInRectangle2D()](#getpointinrectangle2d)
- [Implementation Details](#implementation-details)
  - [Floor Plane Definition](#floor-plane-definition)
  - [Rectangle Rotation](#rectangle-rotation)
  - [Visual Elements](#visual-elements)
- [Recording & Playback](#recording--playback)

---

## Overview

This example demonstrates advanced vector mathematics for spatial interaction detection in 3D environments. It showcases how to:
- Project 3D skeleton data onto a 2D floor plane
- Detect when a person enters a defined rectangular zone
- Extract precise 2D coordinates within that zone
- Visualize spatial relationships with color-coded feedback

**Key Concepts:**
- **Ray-plane intersection**: Finding where a vertical ray from the pelvis hits the floor
- **Point-in-rectangle test**: Determining if a projected point falls within a rotatable rectangle
- **2D coordinate extraction**: Converting 3D world positions to local 2D rectangle coordinates
- **Visual feedback**: Color-coded circles and crosshairs showing spatial relationships

---

## vecExample.py - Interactive Floor Rectangle

### Features

1. **Interactive Floor Rectangle**
   - Positionable 2m × 3m rectangle on the floor plane
   - Keyboard-controlled movement and rotation
   - Visual coordinate axes at rectangle origin (RGB: X=red, Y=green, Z=blue)

2. **Skeleton Tracking**
   - Tracks pelvis position of detected people
   - Projects pelvis onto floor plane using ray-plane intersection
   - Real-time detection of people standing on the rectangle

3. **Visual Feedback**
   - **Green circle**: Person is standing on the rectangle
   - **Red circle**: Person is outside the rectangle
   - **Crosshair**: Shows exact 2D position within rectangle (when on rectangle)
     - Red lines: Parallel to X-axis
     - Blue lines: Parallel to Z-axis
   - **Coordinate axes**: 50mm RGB axes at rectangle origin

4. **Recording & Playback**
   - Record sessions for debugging and analysis
   - Replay recordings to test positioning and algorithms

### Controls

| Key | Action |
|-----|--------|
| **Arrow Up** | Move rectangle forward (-Z direction) |
| **Arrow Down** | Move rectangle backward (+Z direction) |
| **Arrow Left** | Move rectangle left (-X direction) |
| **Arrow Right** | Move rectangle right (+X direction) |
| **Shift + Left** | Rotate rectangle counter-clockwise |
| **Shift + Right** | Rotate rectangle clockwise |
| **Space** | Toggle rectangle visibility |
| **R** | Toggle recording (creates timestamped .ssrec file) |

### Usage

**Live mode** (connect to server):
```bash
python vecExample.py --server localhost --port 12345
```

**Playback mode** (replay recording):
```bash
python vecExample.py --rec recordings/recording_20251030_124746.ssrec
```

**Adjust rectangle position**:
1. Run the example
2. Use arrow keys to position rectangle under person
3. Use Shift+arrows to rotate for desired orientation
4. Green circle appears when person enters rectangle
5. Crosshair shows exact 2D position (local X and Z coordinates)

---

## How It Works

### Ray-Plane Intersection

The fundamental operation is finding where a 3D point projects onto a 2D plane. This is done using **ray-plane intersection**.

**Mathematical Concept:**
```
Given:
- Ray origin: P (pelvis position in 3D)
- Ray direction: d (pointing downward: [0, -1, 0])
- Plane: defined by 3 points (p1, p2, p3)

Find: Intersection point I where ray hits plane
```

**Steps:**
1. **Calculate plane normal**: `n = (p2 - p1) × (p3 - p1)`
   - Cross product of two plane edges gives perpendicular vector
2. **Calculate plane equation**: `n · (x - p1) = 0`
   - Any point x on the plane satisfies this
3. **Substitute ray equation**: `x = P + t·d`
   - t is the distance along the ray
4. **Solve for t**: `t = n · (p1 - P) / (n · d)`
5. **Calculate intersection**: `I = P + t·d`

**In our code:**
```python
# Pelvis position (ray origin)
pelvis_vec = QVector3D(pelvis_pos.x, pelvis_pos.y, pelvis_pos.z)

# Ray pointing straight down
ray_direction = QVector3D(0, -1, 0)

# Floor plane at y=0
floor_p1 = QVector3D(0, 0, 0)
floor_p2 = QVector3D(1000, 0, 0)
floor_p3 = QVector3D(0, 0, 1000)

# Find intersection
intersection = getPlaneIntersection(
    ray_direction,
    floor_p1, floor_p2, floor_p3,
    pelvis_vec
)
```

### Point-in-Rectangle Detection

Once we have the floor intersection point, we need to determine if it falls within the rectangle boundaries.

**Approach:**
1. **Transform to rectangle space**: Convert world coordinates to rectangle-local coordinates
2. **Account for rotation**: Apply inverse rotation to align with rectangle axes
3. **Boundary check**: Test if local X and Z are within rectangle bounds

**Mathematical Steps:**
```
Given:
- Point P (world coordinates)
- Rectangle center C (world coordinates)
- Rectangle width W, depth D
- Rotation angle θ around Y-axis

Steps:
1. Translate to rectangle origin: P' = P - C
2. Inverse rotate around Y-axis:
   local_x = P'.x * cos(-θ) - P'.z * sin(-θ)
   local_z = P'.x * sin(-θ) + P'.z * cos(-θ)
3. Check bounds:
   -W/2 ≤ local_x ≤ W/2
   -D/2 ≤ local_z ≤ D/2
```

**In our code:**
```python
is_on_rectangle = isPointInRectangle(
    intersection,           # 3D point on floor
    rect_start_vec,        # Rectangle center
    rect_width,            # 2000mm (2m)
    rect_depth,            # 3000mm (3m)
    floor_normal,          # Y-axis [0,1,0]
    rect_rotation          # Degrees around Y
)
```

### 2D Coordinate Extraction

When a point is on the rectangle, we can extract its local 2D coordinates for precise positioning.

**Purpose:**
- Get exact position within rectangle (e.g., "850mm right, 1200mm forward")
- Useful for zone-based interactions, heatmaps, or position tracking
- Coordinates are in rectangle's local space (not world space)

**Returns:**
- `(local_x, local_z)` tuple
- `local_x`: Distance from center along rectangle width (-W/2 to +W/2)
- `local_z`: Distance from center along rectangle depth (-D/2 to +D/2)
- `None`: If point is not on the plane

**In our code:**
```python
coords_2d = getPointInRectangle2D(
    intersection,
    rect_start_vec,
    rect_width,
    rect_depth,
    floor_normal,
    rect_rotation
)

if coords_2d:
    local_x, local_z = coords_2d
    # local_x: -1000mm to +1000mm (2m width)
    # local_z: -1500mm to +1500mm (3m depth)
```

### Coordinate System

**World Coordinate System:**
- **X-axis** (Red): Right direction
- **Y-axis** (Green): Up direction (vertical)
- **Z-axis** (Blue): Forward direction (toward camera)
- **Origin**: Center of camera view at floor level

**Rectangle Coordinate System:**
- **Local X**: Along rectangle width (after rotation)
- **Local Z**: Along rectangle depth (after rotation)
- **Origin**: Center of rectangle (rect_start_vec)
- **Rotation**: Around world Y-axis (vertical)

**Floor Plane:**
- Defined at `y = 0` (horizontal plane through world origin)
- Normal vector: `[0, 1, 0]` (pointing up)

---

## Vector Math Helper Library

Located at: `libs/senseSpaceLib/senseSpace/vecMathHelper.py`

All functions support multiple input types:
- NumPy arrays: `np.array([x, y, z])`
- Lists/tuples: `[x, y, z]` or `(x, y, z)`
- Qt vectors: `QVector3D(x, y, z)`

### getNormal()

Calculates the surface normal of a plane defined by three points.

**Signature:**
```python
getNormal(vec1, vec2, vec3) -> QVector3D
```

**Algorithm:**
```
n = normalize((vec2 - vec1) × (vec3 - vec1))
```

**Example:**
```python
from senseSpaceLib.senseSpace.vecMathHelper import getNormal
from PyQt5.QtGui import QVector3D

# Three points on floor plane
p1 = QVector3D(0, 0, 0)
p2 = QVector3D(1000, 0, 0)
p3 = QVector3D(0, 0, 1000)

# Get normal vector (points upward: [0, 1, 0])
normal = getNormal(p1, p2, p3)
```

### getPlaneIntersection()

Finds where a ray intersects a plane defined by three points.

**Signature:**
```python
getPlaneIntersection(rayVec3, planeVec1, planeVec2, planeVec3, rayOrigin=None) -> QVector3D or None
```

**Parameters:**
- `rayVec3`: Ray direction (doesn't need to be normalized)
- `planeVec1, planeVec2, planeVec3`: Three points defining the plane
- `rayOrigin`: Starting point of ray (default: [0, 0, 0])

**Returns:**
- `QVector3D`: Intersection point
- `None`: If ray is parallel to plane (no intersection)

**Example:**
```python
# Find where pelvis projects onto floor
pelvis = QVector3D(500, 1200, -800)
ray_down = QVector3D(0, -1, 0)

floor_p1 = QVector3D(0, 0, 0)
floor_p2 = QVector3D(1000, 0, 0)
floor_p3 = QVector3D(0, 0, 1000)

intersection = getPlaneIntersection(
    ray_down,
    floor_p1, floor_p2, floor_p3,
    pelvis
)
# Result: QVector3D(500, 0, -800)
```

### isPointInRectangle()

Tests if a 3D point falls within a rotatable rectangle on a plane.

**Signature:**
```python
isPointInRectangle(point, rect_center, width, depth, plane_normal, rotation=0.0) -> bool
```

**Parameters:**
- `point`: 3D point to test
- `rect_center`: Center of rectangle
- `width`: Rectangle width (X-direction before rotation)
- `depth`: Rectangle depth (Z-direction before rotation)
- `plane_normal`: Plane normal vector
- `rotation`: Rotation in degrees around plane normal (default: 0)

**Returns:**
- `True`: Point is inside rectangle
- `False`: Point is outside rectangle or not on plane

**Example:**
```python
# Check if person is on 2m × 3m rectangle
is_inside = isPointInRectangle(
    floor_point,                    # QVector3D(100, 0, -500)
    rect_center=QVector3D(0, 0, -2000),
    width=2000,                     # 2 meters
    depth=3000,                     # 3 meters
    plane_normal=QVector3D(0, 1, 0),
    rotation=45.0                   # 45° rotation
)
```

### getPointInRectangle2D()

Extracts the local 2D coordinates of a point within a rectangle.

**Signature:**
```python
getPointInRectangle2D(point, rect_center, width, depth, plane_normal, rotation=0.0) -> tuple or None
```

**Returns:**
- `(local_x, local_z)`: 2D coordinates in rectangle space
- `None`: If point is not on the plane

**Coordinate Ranges:**
- `local_x`: `-width/2` to `+width/2`
- `local_z`: `-depth/2` to `+depth/2`
- Origin `(0, 0)` is at rectangle center

**Example:**
```python
coords = getPointInRectangle2D(
    floor_point,
    rect_center=QVector3D(0, 0, -2000),
    width=2000,
    depth=3000,
    plane_normal=QVector3D(0, 1, 0),
    rotation=0.0
)

if coords:
    local_x, local_z = coords
    print(f"Position: {local_x}mm right, {local_z}mm forward")
    # Example output: "Position: 850mm right, -1200mm forward"
```

---

## Implementation Details

### Floor Plane Definition

The floor is always defined as a horizontal plane at `y = 0`:

```python
floor_plane_points = (
    QVector3D(0, 0, 0),      # Origin
    QVector3D(1000, 0, 0),   # 1m along X-axis
    QVector3D(0, 0, 1000)    # 1m along Z-axis
)
```

This matches the floor grid visualization and ensures consistent spatial reference.

### Rectangle Rotation

Rotation is applied around the **Y-axis** (vertical) using standard 2D rotation matrices:

```python
angle_rad = math.radians(rotation_deg)
cos_a = math.cos(angle_rad)
sin_a = math.sin(angle_rad)

# Rotate point (x, z) around Y-axis
rotated_x = x * cos_a - z * sin_a
rotated_z = x * sin_a + z * cos_a
```

**Rotation Direction:**
- Positive angles: Counter-clockwise (when viewed from above)
- Negative angles: Clockwise
- Default: 0° (aligned with world X and Z axes)

### Visual Elements

**Floor Rectangle:**
- Semi-transparent blue fill: `rgba(0.2, 0.5, 0.8, 0.3)`
- Solid blue border: `rgba(0.2, 0.5, 0.8, 1.0)`, 3px width
- Projected onto floor plane using ray-plane intersection

**Pelvis Circles:**
- Green `rgba(0.0, 1.0, 0.0, 0.7)`: Person on rectangle
- Red `rgba(1.0, 0.0, 0.0, 0.7)`: Person off rectangle
- Radius: 150mm
- Polygon offset: -1.0 (drawn above floor to prevent z-fighting)

**Crosshair (when on rectangle):**
- Forms a rectangle from origin (0,0) to hit point (local_x, local_z)
- Red lines: Parallel to X-axis
- Blue lines: Parallel to Z-axis
- Line width: 5px
- Depth test disabled (always visible on top)

**Coordinate Axes:**
- Length: 50mm each
- Red: X-axis (right)
- Green: Y-axis (up)
- Blue: Z-axis (forward)
- Line width: 3px
- Located at rectangle origin (0,0) in rectangle space

---

## Recording & Playback

### Recording a Session

Press `R` during live mode to start/stop recording:

```bash
python vecExample.py --server localhost
# Press R to start recording
# [INFO] Recording started
# ... move around, adjust rectangle ...
# Press R to stop
# [INFO] Recording stopped
# File saved: recordings/recording_20251030_124746.ssrec
```

### Playing Back a Recording

```bash
python vecExample.py --rec recordings/recording_20251030_124746.ssrec
```

**Benefits:**
- Test rectangle positioning without live camera
- Debug edge cases (person entering/exiting rectangle)
- Fine-tune coordinate extraction algorithms
- Share reproducible scenarios with team

**Recording Format:**
- `.ssrec` files contain frame-by-frame skeleton data
- Includes timestamps for accurate playback
- Can be played back at original or modified speed
- All visualization features work identically in playback mode

---

## Troubleshooting

**Rectangle not visible:**
- Press Space to toggle visibility
- Check that rectangle is positioned in camera view (adjust Z coordinate)

**Circle appears in wrong location:**
- Verify floor plane height matches actual floor (y=0 default)
- Check skeleton data quality (pelvis joint detection)

**Crosshair not showing:**
- Only appears when person is ON the rectangle (green circle)
- Ensure rectangle is large enough and positioned correctly

**Line width seems thin:**
- Some OpenGL drivers limit maximum line width
- Lines use depth test disabling to ensure visibility
- Width is set to 5px but actual rendering depends on GPU

**Recording not starting:**
- Check console for error messages
- Ensure write permissions in recordings/ directory
- Verify server connection is established before recording

---

**For more vector math examples, see:**
- `basic/` - Simple skeleton visualization examples
- `../recording/` - Recording and playback examples
- `../../senseSpaceLib/senseSpace/vecMathHelper.py` - Full vector math library source
