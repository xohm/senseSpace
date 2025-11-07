# Shoulder Flipping Problem & Solutions

## The Problem

When you bend your elbow, the ZED SDK's shoulder orientation sometimes flips 180°. This is **NOT a bug** - it's a fundamental **geometric ambiguity** in skeletal tracking from 2D keypoints + depth.

### Why It Happens

The ZED body tracking algorithm:
1. Detects 2D keypoints in the image
2. Gets depth from stereo/depth sensor
3. Fits a 3D skeleton model to the points

**The Issue:**
- The solver knows the **bone direction** (shoulder → elbow vector)
- But it **doesn't know the twist** around that direction (roll is under-constrained)
- Multiple valid orientations exist for the same bone direction
- When you cross certain spatial thresholds, it "snaps" to a different solution → 180° flip

```
Same bone direction, two valid orientations:
    Shoulder
      |
      v  elbow points OUT  ←→  elbow points IN
   Elbow
```

## Solution 1: Temporal Filtering (Partial Fix)

**What:** Apply SLERP smoothing + quaternion flip correction
**Helps with:** Jitter, some rapid flips
**Doesn't fix:** Fundamental geometric ambiguity

### Implementation

```python
from senseSpaceLib.senseSpace.orientation_filter import SkeletonOrientationFilter

filter = SkeletonOrientationFilter(smoothing=0.3)
filtered_orientations = filter.filter_skeleton(local_orientations)
```

**How it works:**
1. Check dot product between consecutive quaternions
2. Flip sign if dot < 0 (ensures shortest path on 4D sphere)
3. Apply SLERP for temporal continuity

**Limitations:**
- Still flips when solver genuinely switches solutions
- Just makes the flip smoother, doesn't prevent it

## Solution 2: Geometric Reconstruction (Proper Fix)

**What:** Rebuild shoulder/elbow orientations from bone directions + stable reference
**Fixes:** Fundamental geometric ambiguity
**Used by:** Unity, Unreal (with variations)

### Implementation

```python
from senseSpaceLib.senseSpace.geometric_orientation import GeometricOrientationReconstructor

reconstructor = GeometricOrientationReconstructor(
    blend_factor=0.3,  # 30% SDK, 70% geometric
    use_filtering=True,
    filter_smoothing=0.2
)

# Reconstruct shoulders and elbows
fixed_orientations = reconstructor.reconstruct_skeleton_orientations(skeleton)
```

### Algorithm

```python
def reconstruct_shoulder(skeleton):
    # Step 1: Get bone direction
    shoulder_pos = skeleton[SHOULDER].pos
    elbow_pos = skeleton[ELBOW].pos
    bone_dir = normalize(elbow_pos - shoulder_pos)
    
    # Step 2: Get stable reference (torso up vector)
    pelvis_pos = skeleton[PELVIS].pos
    neck_pos = skeleton[NECK].pos
    up_vec = normalize(neck_pos - pelvis_pos)
    
    # Step 3: Construct rotation from direction + up
    z = bone_dir
    x = normalize(cross(up_vec, z))
    y = cross(z, x)
    rot_matrix = [x, y, z]
    
    # Step 4: Convert to quaternion
    q_geometric = matrix_to_quat(rot_matrix)
    
    # Step 5: Optionally blend with SDK orientation
    q_sdk = skeleton[SHOULDER].ori
    q_final = slerp(q_geometric, q_sdk, blend_factor)
    
    return q_final
```

### Why This Works

- **Bone direction**: Directly from 3D positions (stable)
- **Up reference**: From torso spine (consistent across frames)
- **No ambiguity**: Orientation fully constrained by geometry
- **Never flips**: Reference vector stays consistent

## Comparison

| Approach | Fixes Flipping? | Complexity | Performance |
|----------|----------------|------------|-------------|
| **None (raw SDK)** | ❌ No | Simple | Fast |
| **Temporal filtering** | ⚠️ Partial | Medium | Fast |
| **Geometric reconstruction** | ✅ Yes | Higher | Fast |
| **Blend both** | ✅ Yes (best) | Higher | Fast |

## Recommendations

### For TouchDesigner Output

**Simple scenes (person standing/sitting):**
```python
# Use SDK orientations directly with optional filtering
from senseSpaceLib.senseSpace.orientation_filter import SkeletonOrientationFilter

filter = SkeletonOrientationFilter(smoothing=0.3)
orientations = compute_bone_aligned_local_orientations(skeleton)
filtered = filter.filter_skeleton(orientations)
```

**Complex motion (arm waving, dancing):**
```python
# Use geometric reconstruction for shoulders/elbows
from senseSpaceLib.senseSpace.geometric_orientation import GeometricOrientationReconstructor

reconstructor = GeometricOrientationReconstructor(
    blend_factor=0.3,  # Adjust based on your needs
    use_filtering=True
)

# Get all orientations
all_orientations = compute_bone_aligned_local_orientations(skeleton)

# Fix problematic joints (shoulders, elbows)
fixed = reconstructor.reconstruct_skeleton_orientations(skeleton)

# Merge: use fixed for shoulders/elbows, SDK for others
for joint_idx in [7, 8, 11, 12]:  # L/R shoulder, L/R elbow
    all_orientations[joint_idx] = fixed[joint_idx]
```

### Blend Factor Guidelines

- **0.0**: Pure geometric (most stable, may lose nuance)
- **0.3**: Recommended (stable with motion detail)
- **0.5**: Balanced
- **0.7**: Mostly SDK (more responsive, may flip)
- **1.0**: Pure SDK (original behavior)

## Testing

Run the orientation example with a recording:

```bash
source .venv/bin/activate
cd client/examples/skeletonOrientation
python skeletonOrientation.py --rec recordings/recording_20251107_074044.ssrec
```

**Controls:**
- Press 'O': Toggle orientation visualization
- Press 'F': Toggle temporal filter
- Press '2': Print orientations to console

**What to look for:**
- Watch shoulder axes (joints 7, 11) while bending elbows
- With filter ON: Smoother, fewer rapid flips
- With filter OFF: Raw SDK behavior (may flip)

## References

- **Unity ZED Plugin**: Uses SDK quaternions with normalization
- **Unreal ZED Plugin**: Uses SLERP smoothing on SDK quaternions
- **This Implementation**: Geometric reconstruction + blending + filtering

## Future Improvements

1. **Adaptive smoothing**: Increase filtering when confidence is low
2. **Per-joint blend factors**: Different blend for shoulder vs elbow
3. **IMU fusion**: Use IMU data if available for better twist estimation
4. **Machine learning**: Train model to predict stable orientations
