#!/usr/bin/env python3
"""
Qt OpenGL viewer widget for SenseSpace client
"""

import math
import time
from typing import Optional, List

from PyQt5 import QtWidgets, QtCore, QtOpenGL
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtOpenGL import QGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *

from .protocol import Frame, Camera
from .visualization import draw_skeletons_with_bones, draw_floor_grid, draw_camera


class SkeletonGLWidget(QGLWidget):
    """OpenGL widget for displaying received skeleton data from server
    
    This class is designed to be modular - override individual draw_* methods
    to customize visualization without changing the core rendering loop.
    """
    
    def __init__(self, parent=None):
        super(SkeletonGLWidget, self).__init__(parent)

        # Camera parameters (spherical coordinates) - align with server viewer defaults
        self.camera_distance = 2000.0
        self.camera_azimuth = 45.0
        self.camera_elevation = 20.0
        self.camera_target = [0.0, 700.0, 800.0]

        # Mouse interaction (Wayland-aware dragging like server viewer)
        self._dragging = False
        self._last_pos = None
        self.mouse_sensitivity = 0.5

        # Frame data
        self.current_frame: Optional[Frame] = None
        self.frame_lock = QtCore.QMutex()

        # UI update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(16)  # ~60 FPS to match server

        # Connection status
        self.last_frame_time = 0

        # Make sure widget receives mouse events and focus
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()  # ensure widget initially focused

        self.onInit()

    def onInit(self):
        """Additional initialization - override in subclass if needed"""
        pass
    
    def onClose(self):
        """Additional cleanup - override in subclass if needed"""
        pass

    
    def close(self):
        """Cleanup resources - override in subclass if needed"""
        if hasattr(self, 'update_timer'):
            self.update_timer.stop()

        self.onClose()

        super().close()

    
    def update_frame(self, frame: Frame):
        """Update the displayed frame (called from network thread)"""
        with QtCore.QMutexLocker(self.frame_lock):
            self.current_frame = frame
            self.last_frame_time = time.time()
    
    def get_current_frame(self) -> Optional[Frame]:
        """Thread-safe getter for current frame"""
        with QtCore.QMutexLocker(self.frame_lock):
            return self.current_frame
    
    # =========================================================================
    # OpenGL Initialization & Setup (override these for custom setup)
    # =========================================================================
    
    def initializeGL(self):
        """Initialize OpenGL context - override to customize GL settings"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glLineWidth(2.0)
        glPointSize(8.0)
    
    def resizeGL(self, width, height):
        """Handle window resize"""
        glViewport(0, 0, width, height)
        self.update_projection()
    
    def update_projection(self):
        """Update projection matrix - override for custom projection"""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        aspect = self.width() / max(1, self.height())
        gluPerspective(45.0, aspect, 1.0, 20000.0)
        
        glMatrixMode(GL_MODELVIEW)
    
    def setup_camera_view(self):
        """Set up camera transformation - override for custom camera behavior"""
        yaw_rad = math.radians(self.camera_azimuth)
        pitch_rad = math.radians(self.camera_elevation)
        cam_x = self.camera_target[0] + self.camera_distance * math.cos(pitch_rad) * math.sin(yaw_rad)
        cam_y = self.camera_target[1] + self.camera_distance * math.sin(pitch_rad)
        cam_z = self.camera_target[2] + self.camera_distance * math.cos(pitch_rad) * math.cos(yaw_rad)
        gluLookAt(cam_x, cam_y, cam_z, 
                  self.camera_target[0], self.camera_target[1], self.camera_target[2], 
                  0, 1, 0)
    
    # =========================================================================
    # Main Rendering (override paintGL for completely custom rendering)
    # =========================================================================
    
    def paintGL(self):
        """Main render loop - calls individual draw methods"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Set up camera
        self.setup_camera_view()

        # Get current frame
        frame = self.get_current_frame()

        if frame is None:
            self.draw_no_data()
            return

        # Draw scene components (override individual methods for custom viz)
        self.draw_floor(frame)
        self.draw_skeletons(frame)
        self.draw_cameras(frame)
        self.draw_custom(frame)  # Student custom drawing in 3D space
        self.draw_overlay(frame)  # 2D overlay always drawn last
    
    # =========================================================================
    # Individual Draw Methods (override these for custom visualizations)
    # =========================================================================
    
    def draw_no_data(self):
        """Draw when no frame data is available - override to customize"""
        draw_floor_grid(size=1000, spacing=200, height=0, color=(0.2, 0.2, 0.2))
        self.draw_connection_status(None)
    
    def draw_floor(self, frame: Frame):
        """Draw floor grid - override to customize floor visualization"""
        floor_height = getattr(frame, 'floor_height', None) or 0.0
        
        draw_floor_grid(
            size=5000,
            spacing=200,
            height=0.0,  # always draw floor at zero in client view
            color=(0.2, 0.2, 0.2),
            people_data=frame.people if hasattr(frame, 'people') else [],
            zed_floor_height=floor_height
        )
    
    def draw_skeletons(self, frame: Frame):
        """Draw skeleton data - override to customize skeleton visualization"""
        if hasattr(frame, 'people') and frame.people:
            draw_skeletons_with_bones(
                frame.people, 
                joint_color=(0.2, 0.8, 1.0), 
                bone_color=(0.8, 0.2, 0.2)
            )
    
    def draw_cameras(self, frame: Frame):
        """Draw camera frustums - override to customize camera visualization"""
        try:
            camera_list = self._parse_camera_list(frame)
            
            for i, cam_dict in enumerate(camera_list):
                try:
                    pos_tuple = self._parse_position(cam_dict.get('position'))
                    orient_dict = self._parse_orientation(cam_dict.get('orientation'))
                    
                    if pos_tuple:
                        # Draw camera position marker
                        self._draw_camera_marker(pos_tuple)
                        
                        # Draw camera frustum
                        if orient_dict:
                            self._draw_camera_frustum(pos_tuple, orient_dict)
                
                except Exception as e:
                    print(f"[WARNING] Failed to draw camera {i}: {e}")
                    continue
                    
        except Exception as e:
            print(f"[WARNING] Camera rendering failed: {e}")
    
    def draw_custom(self, frame: Frame):
        """Draw custom 3D elements - OVERRIDE THIS for student projects!
        
        This is called after all standard drawing (floor, skeletons, cameras)
        but before the 2D overlay. Perfect for adding custom visualizations.
        
        Example usage in subclass:
            def draw_custom(self, frame):
                # Draw a colored sphere at each person's position
                for person in frame.people:
                    if person.skeleton:
                        head = person.skeleton[0]  # Head joint
                        glPushMatrix()
                        glTranslatef(head.pos.x, head.pos.y, head.pos.z)
                        glColor3f(1.0, 0.0, 0.0)
                        # Draw sphere using glutSolidSphere or custom geometry
                        glPopMatrix()
        """
        pass  # Override this method to add custom drawing
    
    def draw_overlay(self, frame: Frame):
        """Draw 2D overlay (status, info, etc) - override to customize overlay"""
        self.draw_connection_status(frame)
    
    # =========================================================================
    # Helper Methods for Camera Drawing (can be overridden)
    # =========================================================================
    
    def _parse_camera_list(self, frame: Frame) -> List[dict]:
        """Parse camera data from frame into list of dicts"""
        cams = getattr(frame, 'cameras', None)
        if not cams:
            return []
        
        camera_list = []
        
        # Handle dict envelope format
        if isinstance(cams, dict):
            maybe = cams.get('cameras', None)
            if isinstance(maybe, list):
                camera_list = maybe
            elif 'serial' in cams and ('position' in cams or 'orientation' in cams):
                camera_list = [cams]
        
        # Handle list format
        elif isinstance(cams, list):
            # Guard against dict keys masquerading as list
            if all(isinstance(x, str) for x in cams) and set(cams) <= {'cameras', 'floor'}:
                camera_list = []
            else:
                camera_list = cams
        
        # Handle JSON string format
        elif isinstance(cams, str):
            import json
            try:
                parsed = json.loads(cams)
                if isinstance(parsed, dict) and 'cameras' in parsed:
                    camera_list = parsed['cameras'] or []
                elif isinstance(parsed, list):
                    camera_list = parsed
                elif isinstance(parsed, dict):
                    camera_list = [parsed]
            except Exception:
                camera_list = []
        
        # Convert Camera objects to dicts
        result = []
        for c in camera_list:
            if isinstance(c, str):
                import json
                try:
                    result.append(json.loads(c))
                except Exception:
                    continue
            elif isinstance(c, dict):
                result.append(c)
            elif hasattr(c, 'to_dict'):
                try:
                    result.append(c.to_dict())
                except Exception:
                    continue
        
        return result
    
    def _parse_position(self, pos) -> Optional[tuple]:
        """Parse position to (x, y, z) tuple"""
        if pos is None:
            return None
        if isinstance(pos, dict):
            return (float(pos.get('x', 0.0)), float(pos.get('y', 0.0)), float(pos.get('z', 0.0)))
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            return (float(pos[0]), float(pos[1]), float(pos[2]))
        return None
    
    def _parse_orientation(self, orient) -> dict:
        """Parse orientation to quaternion dict"""
        if orient is None:
            return {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
        if isinstance(orient, dict):
            return {
                'x': float(orient.get('x', 0.0)),
                'y': float(orient.get('y', 0.0)),
                'z': float(orient.get('z', 0.0)),
                'w': float(orient.get('w', 1.0))
            }
        if isinstance(orient, (list, tuple)) and len(orient) >= 4:
            return {'x': float(orient[0]), 'y': float(orient[1]), 
                    'z': float(orient[2]), 'w': float(orient[3])}
        return {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
    
    def _draw_camera_marker(self, position: tuple):
        """Draw a point marker at camera position"""
        glPointSize(8.0)
        glBegin(GL_POINTS)
        glColor3f(1.0, 0.0, 1.0)
        glVertex3f(position[0], position[1], position[2])
        glEnd()
    
    def _draw_camera_frustum(self, position: tuple, orientation: dict):
        """Draw camera frustum"""
        draw_camera(
            position=position, 
            orientation=orientation, 
            fov_deg=60.0, 
            near=50.0, 
            far=600.0, 
            color=(1.0, 1.0, 0.0), 
            scale=1.0
        )
    
    # =========================================================================
    # Connection Status Overlay
    # =========================================================================
    
    def draw_connection_status(self, frame: Optional[Frame]):
        """Draw connection and frame info as overlay"""
        # Switch to 2D overlay mode
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width(), self.height(), 0, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)
        
        # Check connection status
        current_time = time.time()
        is_connected = (current_time - self.last_frame_time) < 2.0

        # Draw status indicator
        if is_connected:
            glColor4f(0.0, 0.5, 0.0, 0.8)
            status_text = "Connected"
        else:
            glColor4f(0.5, 0.0, 0.0, 0.8)
            status_text = "Disconnected"

        # Connection indicator (top-left corner)
        glBegin(GL_QUADS)
        glVertex2f(10, 10)
        glVertex2f(30, 10)
        glVertex2f(30, 30)
        glVertex2f(10, 30)
        glEnd()

        # Draw text overlay
        try:
            people_count = len(frame.people) if (frame and frame.people) else 0
            self.renderText(35, 22, f"{status_text}")
            self.renderText(35, 40, f"People: {people_count}")
        except Exception:
            pass
        
        # Restore 3D mode
        glEnable(GL_DEPTH_TEST)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
    
    # =========================================================================
    # Mouse Interaction (override for custom camera controls)
    # =========================================================================
    
    def mousePressEvent(self, event):
        """Handle mouse press - override for custom interaction"""
        if event.button() in (Qt.LeftButton, Qt.RightButton, Qt.MiddleButton):
            self._dragging = True
            self._last_pos = event.pos()
            self.setFocus()

    def mouseReleaseEvent(self, event):
        """Handle mouse release - override for custom interaction"""
        self._dragging = False
        self._last_pos = None

    def mouseMoveEvent(self, event):
        """Handle mouse drag for camera control - override for custom controls"""
        if not self._dragging or self._last_pos is None:
            return

        dx = event.x() - self._last_pos.x()
        dy = event.y() - self._last_pos.y()
        buttons = event.buttons()

        # Left: rotate
        if buttons & Qt.LeftButton:
            self.camera_azimuth += dx * 0.5
            self.camera_elevation += -dy * 0.5
            self.camera_elevation = max(-89.9, min(89.9, self.camera_elevation))

        # Right: pan
        elif buttons & Qt.RightButton:
            pan_scale = max(0.001, self.camera_distance * 0.002)
            yaw_rad = math.radians(self.camera_azimuth)
            right_x = math.cos(yaw_rad)
            right_z = -math.sin(yaw_rad)
            self.camera_target[0] += -dx * right_x * pan_scale
            self.camera_target[1] += dy * pan_scale
            self.camera_target[2] += -dx * right_z * pan_scale

        # Middle or both buttons: zoom
        elif (buttons & Qt.MiddleButton) or ((buttons & Qt.LeftButton) and (buttons & Qt.RightButton)):
            self.camera_distance += dy * 5.0
            self.camera_distance = max(100.0, min(10000.0, self.camera_distance))

        self._last_pos = event.pos()
        self.update()

    def wheelEvent(self, event):
        """Handle mouse wheel for zoom - override for custom zoom behavior"""
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.1 if delta > 0 else 0.9
        self.camera_distance = max(100.0, min(10000.0, self.camera_distance * factor))
        self.update()

