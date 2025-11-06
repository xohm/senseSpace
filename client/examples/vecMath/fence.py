#!/usr/bin/env python3
"""
Example: Interactive polygon fence creation on the floor

This example demonstrates creating a custom polygon fence on the floor plane
by selecting tracked people and placing vertices at their ground positions.

HOW TO USE:
-----------
1. Start the server with skeleton tracking running
2. Run this script: python fence.py --server <SERVER_IP>
3. Make sure at least one person is being tracked

CONTROLS:
---------
Arrow Keys (Up/Down/Left/Right):
    - Cycle through all tracked people
    - Selected person is highlighted with a YELLOW SPHERE at their head

SPACE:
    - While building: Add a polygon point at the selected person's floor position
    - After closing: Clear the polygon and start a new one

BACKSPACE:
    - Remove the last polygon point (only while building)

ENTER:
    - Close the polygon (requires minimum 3 points)
    - Polygon becomes filled with semi-transparent green

R:
    - Toggle recording on/off

VISUALIZATION:
--------------
- Building polygon: Orange outline with yellow point markers
- Closed polygon: Green filled area with 50% transparency
- Person inside polygon: RED CIRCLE appears on the ground under them
- Selected person: Yellow sphere at head position

WORKFLOW:
---------
1. Press arrow keys to select a person
2. Press SPACE to place a vertex where they're standing
3. Move to another position/person and press SPACE again
4. Repeat until you have at least 3 points
5. Press ENTER to close the polygon
6. Anyone walking into the polygon will trigger a red circle on the ground
7. Press SPACE to clear and start a new polygon

TECHNICAL DETAILS:
------------------
- Uses vecMathHelper.insidePoly() for accurate 3D-to-2D projection
- Polygon vertices are placed on the floor plane (y=0)
- Detection projects pelvis position onto polygon plane
- Ray casting algorithm determines if person is inside polygon

Example use cases:
- Define restricted zones in a space
- Track when people enter/exit specific areas
- Create virtual boundaries for interactive installations
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
    insidePoly
)
from senseSpaceLib.senseSpace.visualization import draw_coordinate_axes

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QVector3D
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np


def draw_polygon_on_floor(points, floor_plane_points, filled=True, color=(0.2, 0.8, 0.5), alpha=0.5):
    """Draw a polygon on the floor plane with outline and optional fill
    
    Args:
        points: List of QVector3D points on the floor
        floor_plane_points: Tuple of (p1, p2, p3) QVector3D points defining the floor plane
        filled: Whether to fill the polygon
        color: RGB color tuple (0-1 range)
        alpha: Transparency (0-1, where 0 is fully transparent)
    """
    if len(points) < 2:
        return
    
    # Enable polygon offset to draw above the floor
    if filled and len(points) >= 3:
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(-1.0, -1.0)
        
        # Draw filled polygon
        glBegin(GL_POLYGON)
        glColor4f(color[0], color[1], color[2], alpha)
        for point in points:
            glVertex3f(point.x(), point.y(), point.z())
        glEnd()
        
        glDisable(GL_POLYGON_OFFSET_FILL)
    
    # Draw outline
    glLineWidth(4.0)
    glBegin(GL_LINE_LOOP if len(points) >= 3 else GL_LINE_STRIP)
    glColor4f(color[0], color[1], color[2], 1.0)  # Solid color for outline
    for point in points:
        glVertex3f(point.x(), point.y(), point.z())
    glEnd()
    glLineWidth(2.0)


def draw_points_on_floor(points, radius=50.0, color=(1.0, 1.0, 0.0)):
    """Draw small circles at each polygon point
    
    Args:
        points: List of QVector3D points
        radius: Radius of each point marker
        color: RGB color tuple
    """
    for point in points:
        glColor4f(color[0], color[1], color[2], 1.0)
        # Draw a small circle at each point
        segments = 16
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(point.x(), point.y(), point.z())
        for i in range(segments + 1):
            angle = 2.0 * np.pi * i / segments
            x = point.x() + radius * np.cos(angle)
            z = point.z() + radius * np.sin(angle)
            glVertex3f(x, point.y(), z)
        glEnd()


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


def get_head_position(person: Person, body_model: str) -> QVector3D:
    """Get head position for visualization
    
    Args:
        person: Person object with skeleton data
        body_model: "BODY_18" or "BODY_34"
        
    Returns:
        QVector3D of head position or None
    """
    # Use NOSE as head position
    head_data = person.get_joint(UniversalJoint.NOSE, body_model)
    
    if not head_data:
        return None
    
    head_pos, _ = head_data
    return QVector3D(head_pos.x, head_pos.y, head_pos.z)



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
    """Custom visualization with interactive polygon fence creation"""
    
    def onInit(self):
        """Initialize custom state"""
        self.quadric = gluNewQuadric()
        
        # Polygon fence state
        self.polygon_points = []  # List of QVector3D points on floor
        self.polygon_closed = False  # Whether polygon is complete
        
        # Person selection
        self.selected_person_idx = 0  # Index in frame.people list
        
        print("[INFO] === Polygon Fence Controls ===")
        print("[INFO] Arrow Keys: Select person")
        print("[INFO] SPACE: Add polygon point at selected person's position")
        print("[INFO] BACKSPACE: Remove last polygon point")
        print("[INFO] ENTER: Close polygon (finish)")
        print("[INFO] SPACE (after close): Clear and start new polygon")
        print("[INFO] R: Toggle recording")
        print("[INFO] ================================")
    
    def onClose(self):
        """Cleanup resources"""
        if hasattr(self, 'quadric') and self.quadric:
            gluDeleteQuadric(self.quadric)
    
    def keyPressEvent(self, event):
        """Handle keyboard input"""
        # Get current frame to access people list
        if not hasattr(self, 'current_frame') or not self.current_frame:
            super().keyPressEvent(event)
            return
        
        frame = self.current_frame
        if not hasattr(frame, 'people') or not frame.people:
            super().keyPressEvent(event)
            return
        
        num_people = len(frame.people)
        
        if event.key() == Qt.Key_Space:
            if self.polygon_closed:
                # Clear polygon and start new one
                self.polygon_points = []
                self.polygon_closed = False
                print("[INFO] Polygon cleared. Starting new polygon...")
            else:
                # Add point at selected person's floor position
                if num_people > 0:
                    floor_plane_points = (
                        QVector3D(0, 0, 0),
                        QVector3D(1000, 0, 0),
                        QVector3D(0, 0, 1000)
                    )
                    
                    person = frame.people[self.selected_person_idx]
                    body_model = frame.body_model if frame.body_model else "BODY_34"
                    intersection = get_pelvis_floor_intersection(person, body_model, floor_plane_points)
                    
                    if intersection:
                        self.polygon_points.append(intersection)
                        print(f"[INFO] Added point {len(self.polygon_points)}: ({intersection.x():.0f}, {intersection.z():.0f})")
                    else:
                        print("[WARNING] Could not get person's floor position")
        
        elif event.key() == Qt.Key_Backspace:
            # Remove last point
            if self.polygon_points and not self.polygon_closed:
                removed = self.polygon_points.pop()
                print(f"[INFO] Removed point. {len(self.polygon_points)} points remaining")
        
        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            # Close polygon
            if len(self.polygon_points) >= 3 and not self.polygon_closed:
                self.polygon_closed = True
                print(f"[INFO] Polygon closed with {len(self.polygon_points)} points")
            elif len(self.polygon_points) < 3:
                print("[WARNING] Need at least 3 points to close polygon")
        
        elif event.key() == Qt.Key_Up:
            # Select previous person
            if num_people > 0:
                self.selected_person_idx = (self.selected_person_idx - 1) % num_people
                person_id = frame.people[self.selected_person_idx].id
                print(f"[INFO] Selected person {self.selected_person_idx + 1}/{num_people} (ID: {person_id})")
        
        elif event.key() == Qt.Key_Down:
            # Select next person
            if num_people > 0:
                self.selected_person_idx = (self.selected_person_idx + 1) % num_people
                person_id = frame.people[self.selected_person_idx].id
                print(f"[INFO] Selected person {self.selected_person_idx + 1}/{num_people} (ID: {person_id})")
        
        elif event.key() == Qt.Key_Left:
            # Select previous person (same as up)
            if num_people > 0:
                self.selected_person_idx = (self.selected_person_idx - 1) % num_people
                person_id = frame.people[self.selected_person_idx].id
                print(f"[INFO] Selected person {self.selected_person_idx + 1}/{num_people} (ID: {person_id})")
        
        elif event.key() == Qt.Key_Right:
            # Select next person (same as down)
            if num_people > 0:
                self.selected_person_idx = (self.selected_person_idx + 1) % num_people
                person_id = frame.people[self.selected_person_idx].id
                print(f"[INFO] Selected person {self.selected_person_idx + 1}/{num_people} (ID: {person_id})")
        
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
            # Pass other keys to parent
            super().keyPressEvent(event)
    
    def draw_custom(self, frame: Frame):
        """Draw polygon fence and person indicators"""
        # Store current frame for key events
        self.current_frame = frame
        
        if not hasattr(frame, 'people') or not frame.people:
            return
        
        # Define floor plane at y=0
        floor_plane_points = (
            QVector3D(0, 0, 0),
            QVector3D(1000, 0, 0),
            QVector3D(0, 0, 1000)
        )
        
        floor_p1, floor_p2, floor_p3 = floor_plane_points
        floor_normal = getNormal(floor_p1, floor_p2, floor_p3)
        
        body_model = frame.body_model if frame.body_model else "BODY_34"
        
        # Draw polygon if we have points
        if self.polygon_points:
            # Choose color based on state
            if self.polygon_closed:
                color = (0.2, 0.8, 0.5)  # Green when closed
            else:
                color = (0.8, 0.6, 0.2)  # Orange while building
            
            draw_polygon_on_floor(
                self.polygon_points,
                floor_plane_points,
                filled=self.polygon_closed,
                color=color,
                alpha=0.5
            )
            
            # Draw point markers
            draw_points_on_floor(self.polygon_points, radius=50.0, color=(1.0, 1.0, 0.0))
        
        # Visualize selected person with sphere at head
        if len(frame.people) > 0 and self.selected_person_idx < len(frame.people):
            person = frame.people[self.selected_person_idx]
            head_pos = get_head_position(person, body_model)
            
            if head_pos:
                # Draw yellow sphere at head
                glPushMatrix()
                glTranslatef(head_pos.x(), head_pos.y(), head_pos.z())
                glColor4f(1.0, 1.0, 0.0, 0.8)  # Yellow
                gluSphere(self.quadric, 100.0, 16, 16)  # 100mm radius sphere
                glPopMatrix()
        
        # Check all people for polygon intersection and draw floor circles
        if self.polygon_closed and len(self.polygon_points) >= 3:
            for person in frame.people:
                # Get pelvis position (3D point in space)
                pelvis_data = person.get_joint(UniversalJoint.PELVIS, body_model)
                if pelvis_data:
                    pelvis_pos, _ = pelvis_data
                    pelvis_vec = QVector3D(pelvis_pos.x, pelvis_pos.y, pelvis_pos.z)
                    
                    # Check if pelvis is inside polygon using projection
                    # floor_normal points UP (0, 1, 0) - opposite of ground
                    is_inside = insidePoly(pelvis_vec, floor_normal, self.polygon_points)
                    
                    if is_inside:
                        # Get floor position for drawing the circle
                        floor_pos = get_pelvis_floor_intersection(person, body_model, floor_plane_points)
                        if floor_pos:
                            # Draw red circle on ground - slightly elevated to be visible
                            glColor4f(1.0, 0.0, 0.0, 0.8)  # Red
                            
                            # Draw filled circle
                            radius = 200.0  # 200mm radius
                            segments = 32
                            
                            # Use stronger polygon offset to ensure visibility above floor
                            glEnable(GL_POLYGON_OFFSET_FILL)
                            glPolygonOffset(-2.0, -2.0)  # Stronger offset
                            
                            # Lift circle 1mm above floor to prevent z-fighting
                            circle_y = floor_pos.y() + 1.0
                            
                            glBegin(GL_TRIANGLE_FAN)
                            glVertex3f(floor_pos.x(), circle_y, floor_pos.z())
                            for i in range(segments + 1):
                                angle = 2.0 * np.pi * i / segments
                                x = floor_pos.x() + radius * np.cos(angle)
                                z = floor_pos.z() + radius * np.sin(angle)
                                glVertex3f(x, circle_y, z)
                            glEnd()
                            
                            glDisable(GL_POLYGON_OFFSET_FILL)



def main():
    parser = argparse.ArgumentParser(description="SenseSpace Polygon Fence Example")
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
        window_title="Polygon Fence - Arrows:Select | SPACE:Add Point | BACKSPACE:Remove | ENTER:Close | R:Record"
    )
    
    success = client.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
