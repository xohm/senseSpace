#!/usr/bin/env python3
"""
Vector Math Helper - Basic 3D vector math utilities

Supports both numpy arrays and PyQt5 QVector3D objects.
"""

import numpy as np
from typing import Tuple, Optional, Union

try:
    from PyQt5.QtGui import QVector3D
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False
    QVector3D = None


def _to_numpy(vec) -> np.ndarray:
    """Convert QVector3D or list to numpy array."""
    if QT_AVAILABLE and isinstance(vec, QVector3D):
        return np.array([vec.x(), vec.y(), vec.z()], dtype=np.float64)
    return np.asarray(vec, dtype=np.float64)


def _to_qvector(vec: np.ndarray):
    """Convert numpy array to QVector3D if Qt is available."""
    if QT_AVAILABLE:
        return QVector3D(float(vec[0]), float(vec[1]), float(vec[2]))
    return vec


def getNormal(vec1, vec2, vec3):
    """Calculate the normalized surface normal from three points defining a plane.
    
    The normal is calculated using the cross product of two edge vectors:
    - Edge1 = vec2 - vec1
    - Edge2 = vec3 - vec1
    - Normal = normalize(Edge1 × Edge2)
    
    Args:
        vec1: First point on the plane (numpy array, list, or QVector3D)
        vec2: Second point on the plane (numpy array, list, or QVector3D)
        vec3: Third point on the plane (numpy array, list, or QVector3D)
        
    Returns:
        Normalized surface normal vector (same type as input if QVector3D, else numpy array)
        
    Example:
        >>> # With numpy arrays
        >>> p1 = np.array([0, 0, 0])
        >>> p2 = np.array([1, 0, 0])
        >>> p3 = np.array([0, 1, 0])
        >>> normal = getNormal(p1, p2, p3)
        >>> # Returns [0, 0, 1] - pointing up in Z direction
        
        >>> # With QVector3D
        >>> p1 = QVector3D(0, 0, 0)
        >>> p2 = QVector3D(1, 0, 0)
        >>> p3 = QVector3D(0, 1, 0)
        >>> normal = getNormal(p1, p2, p3)
        >>> # Returns QVector3D(0, 0, 1)
    """
    # Remember input type
    return_qt = QT_AVAILABLE and isinstance(vec1, QVector3D)
    
    # Convert to numpy arrays
    vec1 = _to_numpy(vec1)
    vec2 = _to_numpy(vec2)
    vec3 = _to_numpy(vec3)
    
    # Calculate edge vectors
    edge1 = vec2 - vec1
    edge2 = vec3 - vec1
    
    # Calculate normal via cross product
    normal = np.cross(edge1, edge2)
    
    # Normalize the result
    length = np.linalg.norm(normal)
    if length < 1e-10:
        # Degenerate case: points are collinear
        raise ValueError("Points are collinear - cannot determine normal")
    
    normal = normal / length
    
    # Convert back to QVector3D if input was Qt
    if return_qt:
        return _to_qvector(normal)
    return normal


def getPlaneIntersection(
    rayVec3,
    planeVec1,
    planeVec2,
    planeVec3,
    rayOrigin=None
):
    """Calculate the intersection point of a ray with a plane defined by three points.
    
    Uses the ray-plane intersection formula:
    - Plane normal N = getNormal(planeVec1, planeVec2, planeVec3)
    - Ray: P(t) = rayOrigin + t * rayVec3
    - Intersection: t = dot(planeVec1 - rayOrigin, N) / dot(rayVec3, N)
    
    Args:
        rayVec3: Ray direction vector (numpy array, list, or QVector3D, will be normalized)
        planeVec1: First point defining the plane (numpy array, list, or QVector3D)
        planeVec2: Second point defining the plane (numpy array, list, or QVector3D)
        planeVec3: Third point defining the plane (numpy array, list, or QVector3D)
        rayOrigin: Origin of the ray (numpy array, list, or QVector3D, default: [0, 0, 0])
        
    Returns:
        Intersection point (same type as rayVec3 if QVector3D, else numpy array) or None if ray is parallel to plane
        
    Example:
        >>> # Ray pointing down from above (with numpy)
        >>> ray_dir = np.array([0, 0, -1])
        >>> ray_start = np.array([0.5, 0.5, 5])
        >>> # Horizontal plane at z=0
        >>> p1 = np.array([0, 0, 0])
        >>> p2 = np.array([1, 0, 0])
        >>> p3 = np.array([0, 1, 0])
        >>> hit = getPlaneIntersection(ray_dir, p1, p2, p3, ray_start)
        >>> # Returns [0.5, 0.5, 0]
        
        >>> # Same with QVector3D
        >>> ray_dir = QVector3D(0, 0, -1)
        >>> ray_start = QVector3D(0.5, 0.5, 5)
        >>> p1 = QVector3D(0, 0, 0)
        >>> p2 = QVector3D(1, 0, 0)
        >>> p3 = QVector3D(0, 1, 0)
        >>> hit = getPlaneIntersection(ray_dir, p1, p2, p3, ray_start)
        >>> # Returns QVector3D(0.5, 0.5, 0)
    """
    # Remember input type
    return_qt = QT_AVAILABLE and isinstance(rayVec3, QVector3D)
    
    # Convert to numpy arrays
    rayVec3 = _to_numpy(rayVec3)
    planeVec1 = _to_numpy(planeVec1)
    planeVec2 = _to_numpy(planeVec2)
    planeVec3 = _to_numpy(planeVec3)
    
    if rayOrigin is None:
        rayOrigin = np.array([0.0, 0.0, 0.0])
    else:
        rayOrigin = _to_numpy(rayOrigin)
    
    # Normalize ray direction
    ray_length = np.linalg.norm(rayVec3)
    if ray_length < 1e-10:
        raise ValueError("Ray direction vector is zero")
    rayVec3 = rayVec3 / ray_length
    
    # Get plane normal
    try:
        normal = getNormal(planeVec1, planeVec2, planeVec3)
    except ValueError:
        return None  # Degenerate plane
    
    # Check if ray is parallel to plane
    denominator = np.dot(rayVec3, normal)
    if abs(denominator) < 1e-10:
        # Ray is parallel to plane - no intersection
        return None
    
    # Calculate intersection parameter t
    t = np.dot(planeVec1 - rayOrigin, normal) / denominator
    
    # Calculate intersection point
    intersection = rayOrigin + t * rayVec3
    
    # Convert back to QVector3D if input was Qt
    if return_qt:
        return _to_qvector(intersection)
    return intersection


# Additional utility functions

def distance(vec1, vec2) -> float:
    """Calculate Euclidean distance between two points.
    
    Args:
        vec1: First point (numpy array, list, or QVector3D)
        vec2: Second point (numpy array, list, or QVector3D)
        
    Returns:
        Distance as float
    """
    vec1 = _to_numpy(vec1)
    vec2 = _to_numpy(vec2)
    return np.linalg.norm(vec2 - vec1)


def normalize(vec):
    """Normalize a vector to unit length.
    
    Args:
        vec: Input vector (numpy array, list, or QVector3D)
        
    Returns:
        Normalized vector (same type as input if QVector3D, else numpy array)
    """
    return_qt = QT_AVAILABLE and isinstance(vec, QVector3D)
    vec = _to_numpy(vec)
    length = np.linalg.norm(vec)
    if length < 1e-10:
        raise ValueError("Cannot normalize zero vector")
    result = vec / length
    if return_qt:
        return _to_qvector(result)
    return result


def dot(vec1, vec2) -> float:
    """Calculate dot product of two vectors.
    
    Args:
        vec1: First vector (numpy array, list, or QVector3D)
        vec2: Second vector (numpy array, list, or QVector3D)
        
    Returns:
        Dot product as float
    """
    vec1 = _to_numpy(vec1)
    vec2 = _to_numpy(vec2)
    return np.dot(vec1, vec2)


def cross(vec1, vec2):
    """Calculate cross product of two vectors.
    
    Args:
        vec1: First vector (numpy array, list, or QVector3D)
        vec2: Second vector (numpy array, list, or QVector3D)
        
    Returns:
        Cross product vector (same type as vec1 if QVector3D, else numpy array)
    """
    return_qt = QT_AVAILABLE and isinstance(vec1, QVector3D)
    vec1 = _to_numpy(vec1)
    vec2 = _to_numpy(vec2)
    result = np.cross(vec1, vec2)
    if return_qt:
        return _to_qvector(result)
    return result


def isPointInRectangle(
    point,
    rect_center,
    rect_width,
    rect_depth,
    rect_normal,
    rotation_deg=0.0
):
    """Check if a 3D point lies within a rectangle on a plane.
    
    The rectangle is defined by its center, dimensions, and orientation.
    Tests if the point projects onto the rectangle's surface area.
    
    Args:
        point: Point to test (numpy array, list, or QVector3D)
        rect_center: Center of rectangle (numpy array, list, or QVector3D)
        rect_width: Width of rectangle (X direction before rotation)
        rect_depth: Depth of rectangle (Z direction before rotation)
        rect_normal: Normal vector of the plane (numpy array, list, or QVector3D)
        rotation_deg: Rotation angle in degrees around the normal (default: 0.0)
        
    Returns:
        True if point is inside rectangle, False otherwise
        
    Example:
        >>> # Test if point is in horizontal rectangle at origin
        >>> point = QVector3D(50, 0, 100)
        >>> center = QVector3D(0, 0, 0)
        >>> normal = QVector3D(0, 1, 0)  # Horizontal plane
        >>> inside = isPointInRectangle(point, center, 200, 300, normal)
        >>> # Returns True if point is within 200x300 rectangle
    """
    # Convert to numpy
    point = _to_numpy(point)
    rect_center = _to_numpy(rect_center)
    rect_normal = _to_numpy(rect_normal)
    
    # Normalize the normal
    rect_normal = rect_normal / np.linalg.norm(rect_normal)
    
    # Vector from rectangle center to point
    to_point = point - rect_center
    
    # Check if point is on the plane (within tolerance)
    distance_to_plane = abs(np.dot(to_point, rect_normal))
    if distance_to_plane > 100.0:  # 100mm tolerance
        return False
    
    # Project point onto the plane
    projected = point - rect_normal * np.dot(to_point, rect_normal)
    local_vec = projected - rect_center
    
    # For horizontal floor (normal pointing up in Y), use world X and Z axes
    # Before rotation, rectangle is aligned with world axes
    local_x = np.array([1, 0, 0])  # World X
    local_z = np.array([0, 0, 1])  # World Z
    
    # Apply rotation around Y axis (vertical)
    if abs(rotation_deg) > 1e-6:
        angle_rad = np.radians(rotation_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Rotate X and Z axes around Y
        local_x_rot = np.array([cos_a, 0, sin_a])
        local_z_rot = np.array([-sin_a, 0, cos_a])
        local_x = local_x_rot
        local_z = local_z_rot
    
    # Project local vector onto local axes
    local_x_coord = np.dot(local_vec, local_x)
    local_z_coord = np.dot(local_vec, local_z)
    
    # Check if within rectangle bounds
    half_width = rect_width / 2.0
    half_depth = rect_depth / 2.0
    
    return (abs(local_x_coord) <= half_width and 
            abs(local_z_coord) <= half_depth)


def getPointInRectangle2D(
    point,
    rect_center,
    rect_width,
    rect_depth,
    rect_normal,
    rotation_deg=0.0
):
    """Get 2D coordinates of a 3D point projected onto a rectangle's local coordinate system.
    
    Returns the position relative to the rectangle's center in its local X,Z plane.
    The origin (0, 0) is at the rectangle center.
    
    Args:
        point: 3D point to project (numpy array, list, or QVector3D)
        rect_center: Center of rectangle (numpy array, list, or QVector3D)
        rect_width: Width of rectangle (X direction before rotation)
        rect_depth: Depth of rectangle (Z direction before rotation)
        rect_normal: Normal vector of the plane (numpy array, list, or QVector3D)
        rotation_deg: Rotation angle in degrees around the normal (default: 0.0)
        
    Returns:
        Tuple (local_x, local_z) representing 2D coordinates on the rectangle plane,
        or None if point is too far from the plane
        
    Example:
        >>> point = QVector3D(100, 0, -50)
        >>> center = QVector3D(0, 0, 0)
        >>> normal = QVector3D(0, 1, 0)
        >>> coords = getPointInRectangle2D(point, center, 2000, 3000, normal, 0)
        >>> # Returns (100.0, -50.0) - 100mm right, 50mm forward from center
    """
    # Convert to numpy
    point = _to_numpy(point)
    rect_center = _to_numpy(rect_center)
    rect_normal = _to_numpy(rect_normal)
    
    # Normalize the normal
    rect_normal = rect_normal / np.linalg.norm(rect_normal)
    
    # Vector from rectangle center to point
    to_point = point - rect_center
    
    # Check if point is on the plane (within tolerance)
    distance_to_plane = abs(np.dot(to_point, rect_normal))
    if distance_to_plane > 100.0:  # 100mm tolerance
        return None
    
    # Project point onto the plane
    projected = point - rect_normal * np.dot(to_point, rect_normal)
    local_vec = projected - rect_center
    
    # For horizontal floor (normal pointing up in Y), use world X and Z axes
    local_x_axis = np.array([1, 0, 0])  # World X
    local_z_axis = np.array([0, 0, 1])  # World Z
    
    # Apply rotation around Y axis (vertical)
    if abs(rotation_deg) > 1e-6:
        angle_rad = np.radians(rotation_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Rotate X and Z axes around Y
        local_x_axis = np.array([cos_a, 0, sin_a])
        local_z_axis = np.array([-sin_a, 0, cos_a])
    
    # Project local vector onto local axes to get 2D coordinates
    local_x = np.dot(local_vec, local_x_axis)
    local_z = np.dot(local_vec, local_z_axis)
    
    return (float(local_x), float(local_z))

