#!/usr/bin/env python3
"""
Example: Floor projection using vecMathHelper
Draws a rectangle on the floor 2m in front of camera and red circles where skeletons touch ground
"""

import argparse
import sys
import os

# Ensure local 'libs' folder is on sys.path when running from repo
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
libs_path = os.path.join(repo_root, 'libs')
if os.path.isdir(libs_path) and libs_path not in sys.path:
    sys.path.insert(0, libs_path)

# Import from shared library
from senseSpaceLib.senseSpace.vizClient import VisualizationClient
from senseSpaceLib.senseSpace.protocol import Frame, Person
from senseSpaceLib.senseSpace.vizWidget import SkeletonGLWidget
from senseSpaceLib.senseSpace.enums import UniversalJoint
from senseSpaceLib.senseSpace.vecMathHelper import (
    getNormal, 
    getPlaneIntersection, 
    isPointInRectangle,
    getPointInRectangle2D
)
from senseSpaceLib.senseSpace.visualization import draw_coordinate_axes

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QVector3D
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np


def draw_floor_rectangle(start_vec, width, depth, rotation_deg, floor_plane_points):
    """Draw a rectangle on the floor plane with position and rotation
    
    Args:
        start_vec: Center position of rectangle (QVector3D)
        width: Width of rectangle (X direction before rotation)
        depth: Depth of rectangle (Z direction before rotation)
        rotation_deg: Rotation angle in degrees around Y axis
        floor_plane_points: Tuple of (p1, p2, p3) QVector3D points defining the floor plane
    """
    # Get floor normal using vecMathHelper
    floor_p1, floor_p2, floor_p3 = floor_plane_points
    normal = getNormal(floor_p1, floor_p2, floor_p3)
    
    half_width = width / 2.0
    half_depth = depth / 2.0
    
    # Calculate corner points in local space (before rotation)
    corners_local = [
        (-half_width, -half_depth),  # Front-left
        (half_width, -half_depth),   # Front-right
        (half_width, half_depth),    # Back-right
        (-half_width, half_depth),   # Back-left
    ]
    
    # Apply rotation and translation
    angle_rad = np.radians(rotation_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    corners_3d = []
    for local_x, local_z in corners_local:
        # Rotate around Y axis (vertical)
        rotated_x = local_x * cos_a - local_z * sin_a
        rotated_z = local_x * sin_a + local_z * cos_a
        
        # Translate to world position
        world_x = start_vec.x() + rotated_x
        world_z = start_vec.z() + rotated_z
        
        # Project onto floor plane
        ray_start = QVector3D(world_x, 5000, world_z)  # Start 5m above
        ray_dir = QVector3D(0, -1, 0)  # Point down
        
        intersection = getPlaneIntersection(ray_dir, floor_p1, floor_p2, floor_p3, ray_start)
        if intersection:
            corners_3d.append(intersection)
    
    if len(corners_3d) != 4:
        return  # Failed to project all corners
    
    # Draw filled rectangle
    glBegin(GL_QUADS)
    glColor4f(0.2, 0.5, 0.8, 0.3)  # Semi-transparent blue
    for corner in corners_3d:
        glVertex3f(corner.x(), corner.y(), corner.z())
    glEnd()
    
    # Draw border
    glLineWidth(3.0)
    glBegin(GL_LINE_LOOP)
    glColor4f(0.2, 0.5, 0.8, 1.0)  # Solid blue
    for corner in corners_3d:
        glVertex3f(corner.x(), corner.y(), corner.z())
    glEnd()
    glLineWidth(2.0)  # Reset


def draw_rectangle_crosshair(
    point_3d,
    rect_center,
    rect_width,
    rect_depth,
    rect_rotation,
    floor_plane_points
):
    """Draw crosshair lines on the rectangle showing where a point projects onto it.
    
    Draws a rectangle from origin to hit point with axis-colored lines:
    - Line 1: (0,0) to (hitX, 0) - RED (X-axis)
    - Line 2: (hitX, 0) to (hitX, hitZ) - BLUE (Z-axis)
    - Line 3: (hitX, hitZ) to (0, hitZ) - RED (X-axis)
    - Line 4: (0, hitZ) to (0, 0) - BLUE (Z-axis)
    
    Args:
        point_3d: 3D point to project onto rectangle (QVector3D)
        rect_center: Center of rectangle (QVector3D)
        rect_width: Width of rectangle
        rect_depth: Depth of rectangle
        rect_rotation: Rotation in degrees
        floor_plane_points: Floor plane definition
    """
    floor_p1, floor_p2, floor_p3 = floor_plane_points
    floor_normal = getNormal(floor_p1, floor_p2, floor_p3)
    
    # Get 2D coordinates on rectangle
    coords_2d = getPointInRectangle2D(
        point_3d,
        rect_center,
        rect_width,
        rect_depth,
        floor_normal,
        rect_rotation
    )
    
    if coords_2d is None:
        return  # Point not on plane
    
    hit_x, hit_z = coords_2d
    
    # Helper function to convert 2D rectangle coords to 3D world position
    def rect_2d_to_3d(local_x, local_z):
        angle_rad = np.radians(rect_rotation)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Rotate to world space
        world_x = rect_center.x() + local_x * cos_a - local_z * sin_a
        world_z = rect_center.z() + local_x * sin_a + local_z * cos_a
        
        # Project onto floor
        ray_start = QVector3D(world_x, 5000, world_z)
        ray_dir = QVector3D(0, -1, 0)
        return getPlaneIntersection(ray_dir, floor_p1, floor_p2, floor_p3, ray_start)
    
    # Calculate all 4 corner points of the rectangle from origin to hit point
    p_origin = rect_2d_to_3d(0, 0)           # (0, 0)
    p_x_axis = rect_2d_to_3d(hit_x, 0)       # (hitX, 0)
    p_hit = rect_2d_to_3d(hit_x, hit_z)      # (hitX, hitZ)
    p_z_axis = rect_2d_to_3d(0, hit_z)       # (0, hitZ)
    
    if not all([p_origin, p_x_axis, p_hit, p_z_axis]):
        return  # Failed to project some points
    
    # Disable depth test temporarily for lines to always show on top
    glDisable(GL_DEPTH_TEST)
    
    glLineWidth(5.0)
    
    # Line 1: (0,0) to (hitX, 0) - RED (parallel to X-axis)
    glColor4f(1.0, 0.0, 0.0, 1.0)  # Red for X
    glBegin(GL_LINES)
    glVertex3f(p_origin.x(), p_origin.y(), p_origin.z())
    glVertex3f(p_x_axis.x(), p_x_axis.y(), p_x_axis.z())
    glEnd()
    
    # Line 2: (hitX, 0) to (hitX, hitZ) - BLUE (parallel to Z-axis)
    glColor4f(0.0, 0.0, 1.0, 1.0)  # Blue for Z
    glBegin(GL_LINES)
    glVertex3f(p_x_axis.x(), p_x_axis.y(), p_x_axis.z())
    glVertex3f(p_hit.x(), p_hit.y(), p_hit.z())
    glEnd()
    
    # Line 3: (hitX, hitZ) to (0, hitZ) - RED (parallel to X-axis)
    glColor4f(1.0, 0.0, 0.0, 1.0)  # Red for X
    glBegin(GL_LINES)
    glVertex3f(p_hit.x(), p_hit.y(), p_hit.z())
    glVertex3f(p_z_axis.x(), p_z_axis.y(), p_z_axis.z())
    glEnd()
    
    # Line 4: (0, hitZ) to (0, 0) - BLUE (parallel to Z-axis)
    glColor4f(0.0, 0.0, 1.0, 1.0)  # Blue for Z
    glBegin(GL_LINES)
    glVertex3f(p_z_axis.x(), p_z_axis.y(), p_z_axis.z())
    glVertex3f(p_origin.x(), p_origin.y(), p_origin.z())
    glEnd()
    
    # Re-enable depth test
    glEnable(GL_DEPTH_TEST)
    glLineWidth(2.0)  # Reset


def get_pelvis_floor_intersection(person: Person, body_model: str, floor_plane_points) -> QVector3D:
    """Calculate where pelvis ray intersects the floor using vecMathHelper
    
    Args:
        person: Person object with skeleton data
        body_model: "BODY_18" or "BODY_34"
        floor_plane_points: Tuple of (p1, p2, p3) QVector3D points defining the floor plane
        
    Returns:
        QVector3D of intersection point or None if no valid intersection
    """
    # Get pelvis position
    pelvis_data = person.get_joint(UniversalJoint.PELVIS, body_model)
    if not pelvis_data:
        return None
    
    pelvis_pos, _ = pelvis_data
    
    # Convert Position object to QVector3D
    pelvis_vec = QVector3D(pelvis_pos.x, pelvis_pos.y, pelvis_pos.z)
    
    # Unpack floor plane points
    floor_p1, floor_p2, floor_p3 = floor_plane_points
    
    # Ray pointing straight down from pelvis
    ray_direction = QVector3D(0, -1, 0)
    
    # Calculate intersection using vecMathHelper
    intersection = getPlaneIntersection(
        ray_direction,
        floor_p1,
        floor_p2,
        floor_p3,
        pelvis_vec
    )
    
    return intersection


def draw_circle_on_floor(center: QVector3D, radius: float, floor_plane_points, segments: int = 32):
    """Draw a filled circle on the floor plane
    
    Args:
        center: Center position (QVector3D) - should already be on the floor plane
        radius: Circle radius in mm
        floor_plane_points: Tuple of (p1, p2, p3) QVector3D points defining the floor plane
        segments: Number of segments for circle smoothness
    """
    # Get floor normal to orient the circle
    floor_p1, floor_p2, floor_p3 = floor_plane_points
    normal = getNormal(floor_p1, floor_p2, floor_p3)
    
    # Enable polygon offset to draw circle above the floor
    glEnable(GL_POLYGON_OFFSET_FILL)
    glPolygonOffset(-1.0, -1.0)  # Draw slightly above to prevent z-fighting
    
    # For simplicity, if floor is horizontal (y=0), draw circle directly
    # Otherwise, project circle points onto the floor plane
    glBegin(GL_TRIANGLE_FAN)
    glVertex3f(center.x(), center.y(), center.z())
    for i in range(segments + 1):
        angle = 2.0 * np.pi * i / segments
        x = center.x() + radius * np.cos(angle)
        z = center.z() + radius * np.sin(angle)
        
        # Project this XZ point onto the floor plane
        ray_start = QVector3D(x, center.y() + 100, z)  # Start slightly above center
        ray_dir = QVector3D(0, -1, 0)
        
        intersection = getPlaneIntersection(ray_dir, floor_p1, floor_p2, floor_p3, ray_start)
        if intersection:
            glVertex3f(intersection.x(), intersection.y(), intersection.z())
        else:
            # Fallback to center height if projection fails
            glVertex3f(x, center.y(), z)
    glEnd()
    
    # Disable polygon offset
    glDisable(GL_POLYGON_OFFSET_FILL)


class CustomSkeletonWidget(SkeletonGLWidget):
    """Custom visualization with floor rectangle and pelvis projection"""
    
    def onInit(self):
        """Initialize custom state"""
        self.quadric = gluNewQuadric()
        self.show_rectangle = True
        # Rectangle configuration
        self.rect_start_vec = QVector3D(0, 0, -2000)  # 2m in front of camera
        self.rect_width = 1000      # 2 meters wide
        self.rect_depth = 1500      # 3 meters deep
        self.rect_rotation = 0.0    # Rotation in degrees
    
    def onClose(self):
        """Cleanup resources"""
        if hasattr(self, 'quadric') and self.quadric:
            gluDeleteQuadric(self.quadric)
    
    def keyPressEvent(self, event):
        """Handle keyboard input"""
        move_step = 100.0  # 100mm = 10cm movement step
        rotate_step = 5.0  # 5 degrees rotation step
        
        if event.key() == Qt.Key_Space:
            # Toggle rectangle visibility
            self.show_rectangle = not self.show_rectangle
            status = "VISIBLE" if self.show_rectangle else "HIDDEN"
            print(f"[INFO] Floor rectangle: {status}")
        
        elif event.key() == Qt.Key_R:
            # Toggle recording
            if self.client:
                if self.client.is_recording():
                    self.client.stop_recording()
                    print("[INFO] Recording stopped")
                else:
                    self.client.start_recording()
                    print("[INFO] Recording started")
            else:
                print("[WARNING] Client not available")
        
        elif event.key() == Qt.Key_Up:
            # Move rectangle forward (-Z direction)
            self.rect_start_vec.setZ(self.rect_start_vec.z() - move_step)
            print(f"[INFO] Rectangle position: ({self.rect_start_vec.x():.0f}, {self.rect_start_vec.z():.0f})")
        
        elif event.key() == Qt.Key_Down:
            # Move rectangle backward (+Z direction)
            self.rect_start_vec.setZ(self.rect_start_vec.z() + move_step)
            print(f"[INFO] Rectangle position: ({self.rect_start_vec.x():.0f}, {self.rect_start_vec.z():.0f})")
        
        elif event.key() == Qt.Key_Left:
            if event.modifiers() & Qt.ShiftModifier:
                # Shift+Left: Rotate counter-clockwise
                self.rect_rotation += rotate_step
                print(f"[INFO] Rectangle rotation: {self.rect_rotation:.1f}°")
            else:
                # Left: Move left (-X direction)
                self.rect_start_vec.setX(self.rect_start_vec.x() - move_step)
                print(f"[INFO] Rectangle position: ({self.rect_start_vec.x():.0f}, {self.rect_start_vec.z():.0f})")
        
        elif event.key() == Qt.Key_Right:
            if event.modifiers() & Qt.ShiftModifier:
                # Shift+Right: Rotate clockwise
                self.rect_rotation -= rotate_step
                print(f"[INFO] Rectangle rotation: {self.rect_rotation:.1f}°")
            else:
                # Right: Move right (+X direction)
                self.rect_start_vec.setX(self.rect_start_vec.x() + move_step)
                print(f"[INFO] Rectangle position: ({self.rect_start_vec.x():.0f}, {self.rect_start_vec.z():.0f})")
        
        else:
            # Pass other keys to parent
            super().keyPressEvent(event)
       
    def draw_custom(self, frame: Frame):
        """Draw floor rectangle and pelvis ground projections"""
        if not hasattr(frame, 'people') or not frame.people:
            return
        
        # Define floor plane at y=0 (matching the floor grid)
        floor_plane_points = (
            QVector3D(0, 0, 0),
            QVector3D(1000, 0, 0),
            QVector3D(0, 0, 1000)
        )
        
        floor_p1, floor_p2, floor_p3 = floor_plane_points
        floor_normal = getNormal(floor_p1, floor_p2, floor_p3)
        
        # Draw floor rectangle
        if self.show_rectangle:
            draw_floor_rectangle(
                self.rect_start_vec,
                self.rect_width,
                self.rect_depth,
                self.rect_rotation,
                floor_plane_points
            )
            
            # Draw coordinate axes at rectangle origin (0,0) in rectangle space
            # Project origin to floor
            angle_rad = np.radians(self.rect_rotation)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            
            # Origin in world space
            origin_world_x = self.rect_start_vec.x()
            origin_world_z = self.rect_start_vec.z()
            
            # Project onto floor plane
            ray_start = QVector3D(origin_world_x, 5000, origin_world_z)
            ray_dir = QVector3D(0, -1, 0)
            origin_3d = getPlaneIntersection(ray_dir, floor_p1, floor_p2, floor_p3, ray_start)
            
            if origin_3d:
                draw_coordinate_axes(origin_3d, length=50.0, line_width=3.0)
        
        # Get body model from frame
        body_model = frame.body_model if frame.body_model else "BODY_34"
        
        # Draw circles where each person's pelvis hits the ground
        # Green if on rectangle, red if not
        for person in frame.people:
            intersection = get_pelvis_floor_intersection(person, body_model, floor_plane_points)
            if intersection:
                # Check if person is standing on the rectangle
                is_on_rectangle = isPointInRectangle(
                    intersection,
                    self.rect_start_vec,
                    self.rect_width,
                    self.rect_depth,
                    floor_normal,
                    self.rect_rotation
                )
                
                # Green if on rectangle, red otherwise
                if is_on_rectangle:
                    # Draw crosshair showing 2D position on rectangle
                    draw_rectangle_crosshair(
                        intersection,
                        self.rect_start_vec,
                        self.rect_width,
                        self.rect_depth,
                        self.rect_rotation,
                        floor_plane_points
                    )
                    
                    # Set green color for circle (after crosshair)
                    glColor4f(0.0, 1.0, 0.0, 0.7)  # Semi-transparent green
                    
                    # Get and print 2D coordinates
                    coords_2d = getPointInRectangle2D(
                        intersection,
                        self.rect_start_vec,
                        self.rect_width,
                        self.rect_depth,
                        floor_normal,
                        self.rect_rotation
                    )
                    if coords_2d:
                        local_x, local_z = coords_2d
                        # Print coordinates (optional - comment out if too verbose)
                        # print(f"[DEBUG] Person 2D coords: X={local_x:.0f}mm, Z={local_z:.0f}mm")
                else:
                    glColor4f(1.0, 0.0, 0.0, 0.7)  # Semi-transparent red
                
                draw_circle_on_floor(intersection, 150.0, floor_plane_points)  # 150mm radius circle


def main():
    parser = argparse.ArgumentParser(description="SenseSpace Floor Projection Example")
    parser.add_argument("--server", "-s", default="localhost", help="Server IP address")
    parser.add_argument("--port", "-p", type=int, default=12345, help="Server port")
    parser.add_argument("--rec", "-r", type=str, default=None, help="Path to recording file for playback")
    
    args = parser.parse_args()
    
    # Create and run visualization client
    client = VisualizationClient(
        viewer_class=CustomSkeletonWidget,
        server_ip=args.server,
        server_port=args.port,
        playback_file=args.rec,
        window_title="Floor Projection - SPACE:toggle | R:record | P:pause/play | Arrows:move | Shift+Arrows:rotate"
    )
    
    success = client.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
