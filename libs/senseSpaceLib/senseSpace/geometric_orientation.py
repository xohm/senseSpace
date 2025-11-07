#!/usr/bin/env python3
"""
Geometric Orientation Reconstruction for Skeleton Tracking

Fixes shoulder/upper-arm flipping by reconstructing orientations from bone
directions + stable reference vectors, rather than relying solely on SDK solver.

The ZED SDK's orientation solver has geometric ambiguities (especially roll around
bone axis). This module reconstructs stable orientations from 3D positions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial.transform import Rotation as R

from .orientation_filter import QuaternionFilter


# ZED Body 34 joint indices (CORRECTED)
JOINT_PELVIS = 0
JOINT_NAVAL_SPINE = 1
JOINT_CHEST_SPINE = 2
JOINT_NECK = 3
JOINT_LEFT_CLAVICLE = 4
JOINT_LEFT_SHOULDER = 5
JOINT_LEFT_ELBOW = 6
JOINT_LEFT_WRIST = 7
JOINT_LEFT_HAND = 8
JOINT_LEFT_HANDTIP = 9
JOINT_LEFT_THUMB = 10

JOINT_RIGHT_CLAVICLE = 11
JOINT_RIGHT_SHOULDER = 12
JOINT_RIGHT_ELBOW = 13
JOINT_RIGHT_WRIST = 14
JOINT_RIGHT_HAND = 15
JOINT_RIGHT_HANDTIP = 16
JOINT_RIGHT_THUMB = 17

JOINT_LEFT_HIP = 18
JOINT_LEFT_KNEE = 19
JOINT_LEFT_ANKLE = 20
JOINT_LEFT_FOOT = 21

JOINT_RIGHT_HIP = 22
JOINT_RIGHT_KNEE = 23
JOINT_RIGHT_ANKLE = 24
JOINT_RIGHT_FOOT = 25

JOINT_HEAD = 26
JOINT_NOSE = 27
JOINT_LEFT_EYE = 28
JOINT_LEFT_EAR = 29
JOINT_RIGHT_EYE = 30
JOINT_RIGHT_EAR = 31
JOINT_LEFT_HEEL = 32
JOINT_RIGHT_HEEL = 33


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize vector, returning zero vector if length is too small."""
    length = np.linalg.norm(v)
    if length < 1e-6:
        return np.zeros_like(v)
    return v / length


def make_rotation_from_direction(
    dir_vec: np.ndarray, 
    up_vec: np.ndarray,
    axis: str = 'z'
) -> np.ndarray:
    """
    Construct rotation matrix from primary direction and up reference.
    
    Args:
        dir_vec: Primary direction (bone direction), will be normalized
        up_vec: Reference up vector (e.g., torso up), will be projected perpendicular
        axis: Which axis should align with dir_vec ('x', 'y', or 'z')
    
    Returns:
        Quaternion [x, y, z, w] (scipy convention)
    """
    # Normalize primary direction
    primary = normalize_vector(dir_vec)
    if np.allclose(primary, 0):
        # Degenerate case - return identity
        return np.array([0, 0, 0, 1])
    
    # Normalize up reference
    up = normalize_vector(up_vec)
    if np.allclose(up, 0):
        # Fallback to world up
        up = np.array([0, 1, 0])
    
    # Ensure up is not parallel to primary
    if abs(np.dot(primary, up)) > 0.999:
        # Choose different up vector
        if abs(primary[1]) < 0.9:
            up = np.array([0, 1, 0])
        else:
            up = np.array([1, 0, 0])
    
    # Build orthonormal basis
    if axis == 'z':
        # Z along bone direction
        z = primary
        x = normalize_vector(np.cross(up, z))
        y = np.cross(z, x)
        rot_matrix = np.column_stack((x, y, z))
    elif axis == 'y':
        # Y along bone direction
        y = primary
        z = normalize_vector(np.cross(up, y))
        x = np.cross(y, z)
        rot_matrix = np.column_stack((x, y, z))
    elif axis == 'x':
        # X along bone direction
        x = primary
        y = normalize_vector(np.cross(x, up))
        z = np.cross(x, y)
        rot_matrix = np.column_stack((x, y, z))
    else:
        raise ValueError(f"Invalid axis '{axis}', must be 'x', 'y', or 'z'")
    
    # Convert to quaternion
    try:
        return R.from_matrix(rot_matrix).as_quat()  # [x, y, z, w]
    except:
        # Degenerate matrix - return identity
        return np.array([0, 0, 0, 1])


def get_torso_up_vector(skeleton: List) -> np.ndarray:
    """
    Compute stable up vector from torso (pelvis → spine → neck).
    
    Args:
        skeleton: List of joints with .pos attribute
    
    Returns:
        Normalized up vector [x, y, z]
    """
    # Try pelvis → neck first
    pelvis = skeleton[JOINT_PELVIS]
    neck = skeleton[JOINT_NECK]
    
    if hasattr(pelvis, 'pos'):
        pelvis_pos = np.array([pelvis.pos.x, pelvis.pos.y, pelvis.pos.z])
        neck_pos = np.array([neck.pos.x, neck.pos.y, neck.pos.z])
    else:
        pelvis_pos = np.array([pelvis['pos']['x'], pelvis['pos']['y'], pelvis['pos']['z']])
        neck_pos = np.array([neck['pos']['x'], neck['pos']['y'], neck['pos']['z']])
    
    up = neck_pos - pelvis_pos
    
    # If too short, use world up
    if np.linalg.norm(up) < 0.01:
        return np.array([0, 1, 0])
    
    return normalize_vector(up)


def get_joint_position(joint) -> np.ndarray:
    """Extract position from joint as numpy array."""
    if hasattr(joint, 'pos'):
        return np.array([joint.pos.x, joint.pos.y, joint.pos.z])
    else:
        return np.array([joint['pos']['x'], joint['pos']['y'], joint['pos']['z']])


def get_joint_orientation(joint) -> np.ndarray:
    """Extract orientation quaternion from joint as numpy array [x,y,z,w]."""
    if hasattr(joint, 'ori'):
        return np.array([joint.ori.x, joint.ori.y, joint.ori.z, joint.ori.w])
    else:
        return np.array([joint['ori']['x'], joint['ori']['y'], joint['ori']['z'], joint['ori']['w']])


class GeometricOrientationReconstructor:
    """
    Reconstructs stable joint orientations from bone directions.
    
    Particularly useful for shoulder/upper-arm to prevent flipping due to
    geometric ambiguities in the ZED solver.
    """
    
    def __init__(
        self, 
        blend_factor: float = 0.3,
        use_filtering: bool = True,
        filter_smoothing: float = 0.2
    ):
        """
        Initialize geometric orientation reconstructor.
        
        Args:
            blend_factor: How much to blend with SDK quaternion (0-1)
                         0.0 = pure geometric (most stable)
                         0.5 = 50/50 blend
                         1.0 = pure SDK (may flip)
            use_filtering: Apply temporal filtering after reconstruction
            filter_smoothing: Smoothing factor for temporal filter (0-1)
        """
        self.blend_factor = np.clip(blend_factor, 0.0, 1.0)
        self.use_filtering = use_filtering
        self.filter_smoothing = filter_smoothing
        
        # Filters for each problematic joint
        self.filters: Dict[int, QuaternionFilter] = {}
        
        # Joints that need geometric reconstruction (prone to flipping)
        self.problematic_joints = [
            JOINT_LEFT_CLAVICLE,
            JOINT_RIGHT_CLAVICLE,
            JOINT_LEFT_SHOULDER,
            JOINT_RIGHT_SHOULDER,
            JOINT_LEFT_ELBOW,
            JOINT_RIGHT_ELBOW,
        ]
        
        if use_filtering:
            for joint_idx in self.problematic_joints:
                self.filters[joint_idx] = QuaternionFilter(smoothing=filter_smoothing)
    
    def _slerp_quaternions(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """
        Spherical linear interpolation between quaternions.
        
        Args:
            q1: Start quaternion [x,y,z,w]
            q2: End quaternion [x,y,z,w]
            t: Blend factor (0-1)
        
        Returns:
            Interpolated quaternion [x,y,z,w]
        """
        # Ensure shortest path
        dot = np.dot(q1, q2)
        if dot < 0:
            q2 = -q2
            dot = -dot
        
        # Clamp dot product
        dot = np.clip(dot, -1.0, 1.0)
        
        # If very close, use linear interpolation
        if abs(dot) > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
        
        # SLERP
        theta = np.arccos(dot)
        sin_theta = np.sin(theta)
        w1 = np.sin((1.0 - t) * theta) / sin_theta
        w2 = np.sin(t * theta) / sin_theta
        
        return w1 * q1 + w2 * q2
    
    def reconstruct_clavicle(
        self,
        skeleton: List,
        side: str = 'left'
    ) -> np.ndarray:
        """
        Reconstruct clavicle orientation from geometry.
        
        The clavicle bone goes from spine to shoulder, but we need to rotate
        it 90° around Z axis to align properly with the rig's expectations.
        
        Args:
            skeleton: List of joints
            side: 'left' or 'right'
        
        Returns:
            Quaternion [x, y, z, w]
        """
        # Get joint indices
        spine_idx = JOINT_CHEST_SPINE  # Upper spine/chest (joint 2)
        if side == 'left':
            clavicle_idx = JOINT_LEFT_CLAVICLE
            shoulder_idx = JOINT_LEFT_SHOULDER
            rotation_angle = -90.0  # -90° for left side
        else:
            clavicle_idx = JOINT_RIGHT_CLAVICLE
            shoulder_idx = JOINT_RIGHT_SHOULDER
            rotation_angle = 90.0  # +90° for right side
        
        # Get positions
        spine_pos = get_joint_position(skeleton[spine_idx])
        shoulder_pos = get_joint_position(skeleton[shoulder_idx])
        
        # Bone direction (spine → shoulder, this is the clavicle bone)
        bone_dir = shoulder_pos - spine_pos
        
        # Get stable up reference from torso
        up_vec = get_torso_up_vector(skeleton)
        
        # Construct rotation (Y axis along bone)
        quat_bone = make_rotation_from_direction(bone_dir, up_vec, axis='y')
        
        # Apply 90° rotation correction around Z axis
        # Convert angle to radians
        import math
        angle_rad = math.radians(rotation_angle)
        half_angle = angle_rad / 2.0
        
        # Quaternion for rotation around Z axis: [0, 0, sin(θ/2), cos(θ/2)]
        quat_correction = np.array([0.0, 0.0, math.sin(half_angle), math.cos(half_angle)])
        
        # Multiply quaternions: quat_final = quat_bone * quat_correction
        quat_geom = self._multiply_quaternions(quat_bone, quat_correction)
        
        # Optionally blend with SDK orientation
        if self.blend_factor > 0:
            quat_sdk = get_joint_orientation(skeleton[clavicle_idx])
            quat_final = self._slerp_quaternions(quat_geom, quat_sdk, self.blend_factor)
        else:
            quat_final = quat_geom
        
        # Apply temporal filtering
        if self.use_filtering and clavicle_idx in self.filters:
            quat_final = np.array(self.filters[clavicle_idx].filter(quat_final.tolist()))
        
        return quat_final
    
    def _multiply_quaternions(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Multiply two quaternions: q_result = q1 * q2
        
        Args:
            q1, q2: Quaternions [x, y, z, w]
        
        Returns:
            Result quaternion [x, y, z, w]
        """
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])
    
    def reconstruct_upper_arm(
        self, 
        skeleton: List,
        side: str = 'left'
    ) -> np.ndarray:
        """
        Reconstruct shoulder/upper-arm orientation from geometry.
        
        Args:
            skeleton: List of joints
            side: 'left' or 'right'
        
        Returns:
            Quaternion [x, y, z, w]
        """
        # Get joint indices
        if side == 'left':
            shoulder_idx = JOINT_LEFT_SHOULDER
            elbow_idx = JOINT_LEFT_ELBOW
        else:
            shoulder_idx = JOINT_RIGHT_SHOULDER
            elbow_idx = JOINT_RIGHT_ELBOW
        
        # Get positions
        shoulder_pos = get_joint_position(skeleton[shoulder_idx])
        elbow_pos = get_joint_position(skeleton[elbow_idx])
        
        # Bone direction (shoulder → elbow)
        bone_dir = elbow_pos - shoulder_pos
        
        # Get stable up reference from torso
        up_vec = get_torso_up_vector(skeleton)
        
        # Construct rotation (Y axis along bone)
        quat_geom = make_rotation_from_direction(bone_dir, up_vec, axis='y')
        
        # Optionally blend with SDK orientation
        if self.blend_factor > 0:
            quat_sdk = get_joint_orientation(skeleton[shoulder_idx])
            quat_final = self._slerp_quaternions(quat_geom, quat_sdk, self.blend_factor)
        else:
            quat_final = quat_geom
        
        # Apply temporal filtering
        if self.use_filtering and shoulder_idx in self.filters:
            quat_final = np.array(self.filters[shoulder_idx].filter(quat_final.tolist()))
        
        return quat_final
    
    def reconstruct_forearm(
        self, 
        skeleton: List,
        side: str = 'left'
    ) -> np.ndarray:
        """
        Reconstruct elbow/forearm orientation from geometry.
        
        Args:
            skeleton: List of joints
            side: 'left' or 'right'
        
        Returns:
            Quaternion [x, y, z, w]
        """
        # Get joint indices
        if side == 'left':
            elbow_idx = JOINT_LEFT_ELBOW
            wrist_idx = JOINT_LEFT_WRIST
            shoulder_idx = JOINT_LEFT_SHOULDER
        else:
            elbow_idx = JOINT_RIGHT_ELBOW
            wrist_idx = JOINT_RIGHT_WRIST
            shoulder_idx = JOINT_RIGHT_SHOULDER
        
        # Get positions
        elbow_pos = get_joint_position(skeleton[elbow_idx])
        wrist_pos = get_joint_position(skeleton[wrist_idx])
        shoulder_pos = get_joint_position(skeleton[shoulder_idx])
        
        # Bone direction (elbow → wrist)
        bone_dir = wrist_pos - elbow_pos
        
        # Up reference: perpendicular to upper arm plane
        upper_arm_dir = elbow_pos - shoulder_pos
        up_vec = get_torso_up_vector(skeleton)
        
        # Construct rotation (Y axis along bone)
        quat_geom = make_rotation_from_direction(bone_dir, up_vec, axis='y')
        
        # Optionally blend with SDK orientation
        if self.blend_factor > 0:
            quat_sdk = get_joint_orientation(skeleton[elbow_idx])
            quat_final = self._slerp_quaternions(quat_geom, quat_sdk, self.blend_factor)
        else:
            quat_final = quat_geom
        
        # Apply temporal filtering
        if self.use_filtering and elbow_idx in self.filters:
            quat_final = np.array(self.filters[elbow_idx].filter(quat_final.tolist()))
        
        return quat_final
    
    def reconstruct_skeleton_orientations(
        self,
        skeleton: List,
        joints_to_fix: Optional[List[int]] = None
    ) -> Dict[int, np.ndarray]:
        """
        Reconstruct orientations for specified joints.
        
        Args:
            skeleton: List of joints
            joints_to_fix: List of joint indices to reconstruct, or None for defaults
        
        Returns:
            Dict of {joint_index: quaternion [x,y,z,w]}
        """
        if joints_to_fix is None:
            joints_to_fix = self.problematic_joints
        
        reconstructed = {}
        
        for joint_idx in joints_to_fix:
            if joint_idx == JOINT_LEFT_CLAVICLE:
                reconstructed[joint_idx] = self.reconstruct_clavicle(skeleton, side='left')
            elif joint_idx == JOINT_RIGHT_CLAVICLE:
                reconstructed[joint_idx] = self.reconstruct_clavicle(skeleton, side='right')
            elif joint_idx == JOINT_LEFT_SHOULDER:
                reconstructed[joint_idx] = self.reconstruct_upper_arm(skeleton, side='left')
            elif joint_idx == JOINT_RIGHT_SHOULDER:
                reconstructed[joint_idx] = self.reconstruct_upper_arm(skeleton, side='right')
            elif joint_idx == JOINT_LEFT_ELBOW:
                reconstructed[joint_idx] = self.reconstruct_forearm(skeleton, side='left')
            elif joint_idx == JOINT_RIGHT_ELBOW:
                reconstructed[joint_idx] = self.reconstruct_forearm(skeleton, side='right')
        
        return reconstructed
    
    def reset(self):
        """Reset all temporal filters."""
        for filter_obj in self.filters.values():
            filter_obj.reset()
    
    def set_blend_factor(self, blend_factor: float):
        """Update blend factor between geometric and SDK orientations."""
        self.blend_factor = np.clip(blend_factor, 0.0, 1.0)
    
    def set_filter_smoothing(self, smoothing: float):
        """Update temporal filter smoothing."""
        self.filter_smoothing = np.clip(smoothing, 0.0, 1.0)
        for filter_obj in self.filters.values():
            filter_obj.smoothing = self.filter_smoothing
