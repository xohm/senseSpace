#!/usr/bin/env python3
"""
Quaternion Orientation Filter for Skeleton Tracking

Eliminates orientation flipping and jitter in skeleton joint rotations.
Based on spherical linear interpolation (SLERP) with flip correction.

Also includes utilities for computing T-pose delta orientations for rig control.
"""

import math
from typing import Dict, Optional, List, Tuple, Any


class QuaternionFilter:
    """
    Temporal low-pass filter for quaternions to stabilize rotation data.
    
    Attributes:
        smoothing: 0 = no filtering, 1 = heavy smoothing (0.0 - 1.0)
        prev_quat: last filtered quaternion [x, y, z, w]
    """
    
    def __init__(self, smoothing: float = 0.3):
        """
        Initialize quaternion filter.
        
        Args:
            smoothing: Smoothing factor (0-1). Higher = more smoothing.
                      0.0 = no filtering
                      0.3 = moderate filtering (recommended)
                      0.7 = heavy filtering (laggy but very stable)
        """
        self.smoothing = max(0.0, min(1.0, smoothing))
        self.prev_quat: Optional[List[float]] = None
    
    @staticmethod
    def dot(q1: List[float], q2: List[float]) -> float:
        """Compute dot product of two quaternions."""
        return q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]
    
    @staticmethod
    def normalize(q: List[float]) -> List[float]:
        """Normalize quaternion to unit length."""
        length = math.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
        if length < 0.0001:
            return [0.0, 0.0, 0.0, 1.0]  # Identity quaternion
        return [q[0]/length, q[1]/length, q[2]/length, q[3]/length]
    
    @staticmethod
    def slerp(q1: List[float], q2: List[float], t: float) -> List[float]:
        """
        Spherical linear interpolation between two quaternions.
        
        Args:
            q1: Start quaternion [x, y, z, w]
            q2: End quaternion [x, y, z, w]
            t: Interpolation parameter (0-1)
        
        Returns:
            Interpolated quaternion [x, y, z, w]
        """
        # Compute dot product
        dot = QuaternionFilter.dot(q1, q2)
        
        # Clamp dot product to avoid numerical issues
        dot = max(-1.0, min(1.0, dot))
        
        # If quaternions are very close, use linear interpolation
        if abs(dot) > 0.9995:
            result = [
                q1[0] + t * (q2[0] - q1[0]),
                q1[1] + t * (q2[1] - q1[1]),
                q1[2] + t * (q2[2] - q1[2]),
                q1[3] + t * (q2[3] - q1[3])
            ]
            return QuaternionFilter.normalize(result)
        
        # Calculate angle between quaternions
        theta = math.acos(abs(dot))
        sin_theta = math.sin(theta)
        
        # Compute interpolation weights
        w1 = math.sin((1.0 - t) * theta) / sin_theta
        w2 = math.sin(t * theta) / sin_theta
        
        # Interpolate
        result = [
            w1 * q1[0] + w2 * q2[0],
            w1 * q1[1] + w2 * q2[1],
            w1 * q1[2] + w2 * q2[2],
            w1 * q1[3] + w2 * q2[3]
        ]
        
        return QuaternionFilter.normalize(result)
    
    def _ensure_shortest_path(self, q_new: List[float]) -> List[float]:
        """
        Flip quaternion sign if dot product with previous is negative.
        This ensures interpolation takes the shortest path on the 4D sphere.
        
        Args:
            q_new: New quaternion [x, y, z, w]
        
        Returns:
            Corrected quaternion [x, y, z, w]
        """
        if self.prev_quat is not None:
            if self.dot(self.prev_quat, q_new) < 0:
                # Flip to same hemisphere
                return [-q_new[0], -q_new[1], -q_new[2], -q_new[3]]
        return q_new
    
    def filter(self, q_new: List[float]) -> List[float]:
        """
        Apply temporal smoothing to incoming quaternion.
        
        Args:
            q_new: Raw quaternion from SDK [x, y, z, w]
        
        Returns:
            Smoothed quaternion [x, y, z, w]
        """
        # Normalize input
        q_new = self.normalize(q_new)
        
        # First frame - no filtering
        if self.prev_quat is None:
            self.prev_quat = q_new
            return q_new
        
        # Ensure shortest path (fix quaternion double-cover)
        q_new = self._ensure_shortest_path(q_new)
        
        # Apply SLERP smoothing
        # amount = 1.0 - smoothing means:
        #   smoothing=0 → amount=1.0 → use new quaternion (no filtering)
        #   smoothing=1 → amount=0.0 → use old quaternion (maximum filtering)
        q_filtered = self.slerp(self.prev_quat, q_new, 1.0 - self.smoothing)
        
        # Update state
        self.prev_quat = q_filtered
        
        return q_filtered
    
    def reset(self):
        """Reset filter state (useful when tracking is lost)."""
        self.prev_quat = None


class SkeletonOrientationFilter:
    """
    Manages quaternion filters for all joints in a skeleton.
    
    Prevents orientation flipping and jitter across the entire skeleton.
    """
    
    def __init__(self, smoothing: float = 0.3, num_joints: int = 34):
        """
        Initialize skeleton orientation filter.
        
        Args:
            smoothing: Smoothing factor for all joints (0-1)
            num_joints: Number of joints in skeleton (default: 34 for BODY_34)
        """
        self.smoothing = smoothing
        self.num_joints = num_joints
        
        # Create one filter per joint
        self.joint_filters: Dict[int, QuaternionFilter] = {
            i: QuaternionFilter(smoothing) for i in range(num_joints)
        }
    
    def filter_skeleton(self, joint_orientations: Dict[int, List[float]]) -> Dict[int, List[float]]:
        """
        Filter all joint orientations in a skeleton.
        
        Args:
            joint_orientations: Dict of {joint_index: [x, y, z, w]}
        
        Returns:
            Dict of {joint_index: [x, y, z, w]} with filtered orientations
        """
        filtered = {}
        
        for joint_idx, quat in joint_orientations.items():
            if joint_idx in self.joint_filters:
                filtered[joint_idx] = self.joint_filters[joint_idx].filter(quat)
            else:
                # Joint index not in range - pass through
                filtered[joint_idx] = quat
        
        return filtered
    
    def reset(self, joint_index: Optional[int] = None):
        """
        Reset filter state.
        
        Args:
            joint_index: If specified, reset only this joint. Otherwise reset all.
        """
        if joint_index is not None:
            if joint_index in self.joint_filters:
                self.joint_filters[joint_index].reset()
        else:
            for filter_obj in self.joint_filters.values():
                filter_obj.reset()
    
    def set_smoothing(self, smoothing: float, joint_index: Optional[int] = None):
        """
        Update smoothing factor.
        
        Args:
            smoothing: New smoothing factor (0-1)
            joint_index: If specified, update only this joint. Otherwise update all.
        """
        smoothing = max(0.0, min(1.0, smoothing))
        
        if joint_index is not None:
            if joint_index in self.joint_filters:
                self.joint_filters[joint_index].smoothing = smoothing
        else:
            self.smoothing = smoothing
            for filter_obj in self.joint_filters.values():
                filter_obj.smoothing = smoothing


def get_tpose_delta_orientations(skeleton: Any, person: Optional[Any] = None, 
                                bone_lengths: Optional[Dict] = None,
                                bone_directions: Optional[Dict] = None) -> Dict[int, List[float]]:
    """
    Compute delta rotations from perfect T-pose for rig control.
    
    This function:
    1. Generates a perfect mathematical T-pose (identity quaternions)
    2. Uses forward kinematics to reconstruct bone-aligned orientations from current pose
    3. Extracts local (parent-relative) orientations
    4. Computes delta quaternions: delta = tpose_inv * current
    5. Returns delta rotations for each joint
    
    Use Case:
    - Export to TouchDesigner/Blender for character rigging
    - T-pose should show 0° rotations (identity quaternions)
    - Any pose deviation shows as rotation delta from T-pose
    - Apply these deltas to your rigged character's bones
    
    Args:
        skeleton: List of joints from ZED SDK (must have .pos and .ori attributes)
        person: Optional Person object (for global_root_orientation). If None, uses skeleton[0].ori
        bone_lengths: Optional cached bone lengths dict from first frame (for consistency)
        bone_directions: Optional cached bone directions dict from first frame (for consistency)
    
    Returns:
        Dict[int, List[float]]: {joint_idx: [x, y, z, w]} delta quaternions from T-pose
        
    Example:
        >>> from senseSpace import get_tpose_delta_orientations
        >>> deltas = get_tpose_delta_orientations(person.skeleton, person)
        >>> # deltas[5] = LEFT_SHOULDER delta rotation from T-pose
        >>> # In perfect T-pose: deltas[5] ≈ [0.0, 0.0, 0.0, 1.0]
    """
    from .visualization import reconstructSkeletonFromOrientations
    from PyQt5.QtGui import QQuaternion
    
    # Skeleton hierarchy (parent index for each joint)
    BODY34_PARENTS = [
        -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 3, 11, 12, 13, 14, 15, 15,
        0, 18, 19, 20, 0, 22, 23, 24, 3, 26, 26, 26, 26, 26, 20, 24
    ]
    
    # Generate perfect T-pose (identity quaternions for all joints)
    all_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 18, 19, 20, 22, 23, 24, 26]
    tpose_reference = {}
    identity_quat = QQuaternion()  # (0, 0, 0, 1)
    for joint_idx in all_joints:
        tpose_reference[joint_idx] = [
            identity_quat.x(), 
            identity_quat.y(), 
            identity_quat.z(), 
            identity_quat.scalar()
        ]
    
    # Use FK reconstruction to get bone-aligned world rotations
    # Use cached bone_lengths/directions if provided for consistency across frames
    _, fk_world_rotations, _, _ = reconstructSkeletonFromOrientations(
        person if person else skeleton[0], 
        skeleton, 
        bone_lengths=bone_lengths, 
        bone_directions=bone_directions
    )
    
    # Extract local (parent-relative) orientations
    local_orientations = {}
    for child_idx, parent_idx in enumerate(BODY34_PARENTS):
        if child_idx not in fk_world_rotations:
            continue
            
        child_world = fk_world_rotations[child_idx]
        
        if parent_idx < 0:
            # Root joint (pelvis) - use world orientation directly
            local_orientations[child_idx] = [
                child_world.x(),
                child_world.y(),
                child_world.z(),
                child_world.scalar()
            ]
        elif parent_idx in fk_world_rotations:
            # Compute local rotation: local = parent_world^-1 * child_world
            parent_world = fk_world_rotations[parent_idx]
            local_quat = parent_world.conjugated() * child_world
            local_orientations[child_idx] = [
                local_quat.x(),
                local_quat.y(),
                local_quat.z(),
                local_quat.scalar()
            ]
    
    # Compute delta from T-pose: delta = tpose_inv * current
    delta_orientations = {}
    for joint_idx in sorted(local_orientations.keys()):
        if joint_idx not in tpose_reference:
            continue
            
        current_quat = local_orientations[joint_idx]  # [x, y, z, w]
        tpose_quat = tpose_reference[joint_idx]       # [x, y, z, w]
        
        # Quaternion inverse (conjugate for unit quaternions): [-x, -y, -z, w]
        tpose_inv = [-tpose_quat[0], -tpose_quat[1], -tpose_quat[2], tpose_quat[3]]
        
        # Quaternion multiplication: delta = tpose_inv * current
        x1, y1, z1, w1 = tpose_inv
        x2, y2, z2, w2 = current_quat
        delta_orientations[joint_idx] = [
            w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
            w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
            w1*z2 + x1*y2 - y1*x2 + z1*w2,  # z
            w1*w2 - x1*x2 - y1*y2 - z1*z2   # w
        ]
    
    return delta_orientations


# Global cache for bone geometry (initialized on first call per person/session)
_bone_geometry_cache = {
    'bone_lengths': None,
    'bone_directions': None,
    'initialized': False
}


def get_tpose_delta_orientations_ext(person: Any, reset_cache: bool = False) -> Dict[int, List[float]]:
    """
    Simplified version: Compute T-pose delta orientations from person object only.
    
    This is a convenience wrapper around get_tpose_delta_orientations() that:
    - Automatically extracts skeleton from person
    - Manages bone geometry cache internally (initialized on first call)
    - Provides consistent results across frames (same as cached version)
    - Simple API: just pass person object!
    
    The bone geometry is cached on the FIRST call and reused for subsequent calls.
    This ensures frame-to-frame consistency. Call with reset_cache=True to reinitialize.
    
    Args:
        person: Person object with skeleton and global_root_orientation
        reset_cache: If True, clears and reinitializes the bone geometry cache
    
    Returns:
        Dict[int, List[float]]: {joint_idx: [x, y, z, w]} delta quaternions from T-pose
        
    Example:
        >>> from senseSpace import get_tpose_delta_orientations_ext
        >>> deltas = get_tpose_delta_orientations_ext(person)
        >>> # deltas[5] = LEFT_SHOULDER delta rotation from T-pose
        >>> 
        >>> # To reset cache (e.g., new person or session):
        >>> deltas = get_tpose_delta_orientations_ext(person, reset_cache=True)
    """
    global _bone_geometry_cache
    
    # Extract skeleton from person
    skeleton = person.skeleton if hasattr(person, 'skeleton') else person.get('skeleton')
    
    if not skeleton:
        raise ValueError("Person object has no skeleton data")
    
    # Reset cache if requested
    if reset_cache:
        _bone_geometry_cache['initialized'] = False
        _bone_geometry_cache['bone_lengths'] = None
        _bone_geometry_cache['bone_directions'] = None
    
    # Initialize cache on first call
    if not _bone_geometry_cache['initialized']:
        from .visualization import reconstructSkeletonFromOrientations
        
        # Compute bone geometry from first frame
        _, _, bone_lengths, bone_directions = reconstructSkeletonFromOrientations(
            person, skeleton, 
            bone_lengths=None, 
            bone_directions=None
        )
        
        _bone_geometry_cache['bone_lengths'] = bone_lengths
        _bone_geometry_cache['bone_directions'] = bone_directions
        _bone_geometry_cache['initialized'] = True
    
    # Call the full version WITH cached bone geometry
    return get_tpose_delta_orientations(
        skeleton=skeleton,
        person=person,
        bone_lengths=_bone_geometry_cache['bone_lengths'],
        bone_directions=_bone_geometry_cache['bone_directions']
    )


class AdaptiveSkeletonOrientationFilter(SkeletonOrientationFilter):
    """
    Advanced version with adaptive smoothing based on joint confidence.
    
    Lower confidence → Higher smoothing (more stable, but laggier)
    Higher confidence → Lower smoothing (more responsive)
    """
    
    def __init__(self, base_smoothing: float = 0.3, num_joints: int = 34):
        """
        Initialize adaptive skeleton orientation filter.
        
        Args:
            base_smoothing: Base smoothing factor when confidence = 1.0
            num_joints: Number of joints in skeleton
        """
        super().__init__(smoothing=base_smoothing, num_joints=num_joints)
        self.base_smoothing = base_smoothing
    
    def filter_skeleton_adaptive(
        self, 
        joint_orientations: Dict[int, List[float]],
        joint_confidences: Dict[int, float]
    ) -> Dict[int, List[float]]:
        """
        Filter skeleton with adaptive smoothing based on confidence.
        
        Args:
            joint_orientations: Dict of {joint_index: [x, y, z, w]}
            joint_confidences: Dict of {joint_index: confidence (0-100)}
        
        Returns:
            Dict of {joint_index: [x, y, z, w]} with filtered orientations
        """
        filtered = {}
        
        for joint_idx, quat in joint_orientations.items():
            if joint_idx in self.joint_filters:
                # Get confidence (normalize to 0-1 range)
                confidence = joint_confidences.get(joint_idx, 100.0) / 100.0
                confidence = max(0.0, min(1.0, confidence))
                
                # Adaptive smoothing: lower confidence → higher smoothing
                # Formula: smoothing = base_smoothing * (2.0 - confidence)
                # confidence=1.0 → smoothing = base_smoothing
                # confidence=0.5 → smoothing = 1.5 * base_smoothing
                # confidence=0.0 → smoothing = 2.0 * base_smoothing
                adaptive_smoothing = self.base_smoothing * (2.0 - confidence)
                adaptive_smoothing = max(0.0, min(1.0, adaptive_smoothing))
                
                # Temporarily update smoothing for this joint
                original_smoothing = self.joint_filters[joint_idx].smoothing
                self.joint_filters[joint_idx].smoothing = adaptive_smoothing
                
                # Filter
                filtered[joint_idx] = self.joint_filters[joint_idx].filter(quat)
                
                # Restore original smoothing
                self.joint_filters[joint_idx].smoothing = original_smoothing
            else:
                filtered[joint_idx] = quat
        
        return filtered
