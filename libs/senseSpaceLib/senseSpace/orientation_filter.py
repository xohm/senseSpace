#!/usr/bin/env python3
"""
Quaternion Orientation Filter for Skeleton Tracking

Eliminates orientation flipping and jitter in skeleton joint rotations.
Based on spherical linear interpolation (SLERP) with flip correction.
"""

import math
from typing import Dict, Optional, List, Tuple


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
