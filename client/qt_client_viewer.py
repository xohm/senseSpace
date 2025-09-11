#!/usr/bin/env python3
"""
Qt OpenGL viewer widget for SenseSpace client
"""

import math
import time
from typing import Optional

from PyQt5 import QtWidgets, QtCore, QtOpenGL
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtOpenGL import QGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *

from senseSpaceLib.senseSpace.protocol import Frame
from senseSpaceLib.senseSpace.visualization import draw_skeletons_with_bones, draw_floor_grid, draw_camera


class ClientSkeletonGLWidget(QGLWidget):
    """OpenGL widget for displaying received skeleton data from server"""
    
    def __init__(self, parent=None):
        super(ClientSkeletonGLWidget, self).__init__(parent)

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
        self.update_timer.start(33)  # ~30 FPS

        # Connection status
        self.last_frame_time = 0
        
    def update_frame(self, frame: Frame):
        """Update the displayed frame (called from network thread)"""
        with QtCore.QMutexLocker(self.frame_lock):
            self.current_frame = frame
            self.last_frame_time = time.time()
    
    def initializeGL(self):
        """Initialize OpenGL context"""
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
        """Update projection matrix"""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        aspect = self.width() / max(1, self.height())
        gluPerspective(45.0, aspect, 1.0, 20000.0)
        
        glMatrixMode(GL_MODELVIEW)
    
    def paintGL(self):
        """Render the scene"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Set up camera
        self.setup_camera()
        
        # Get current frame data
        frame = None
        with QtCore.QMutexLocker(self.frame_lock):
            frame = self.current_frame
        
        if frame:
            # Draw floor grid: prefer ZED-detected floor height but allow
            # people-based stable fallback inside draw_floor_grid.
            floor_height = frame.floor_height
            # Match server viewer: larger 5m x 5m grid if units are mm
            draw_floor_grid(
                size=5000,
                spacing=200,
                height=None,  # let draw_floor_grid resolve priority
                color=(0.2, 0.2, 0.2),
                people_data=frame.people,
                zed_floor_height=floor_height,
            )
            
            # Draw skeletons
            if frame.people:
                # draw_skeletons_with_bones expects people_data and optional color args.
                # Previously we incorrectly passed frame.body_model as the second argument.
                draw_skeletons_with_bones(frame.people)

            # Draw cameras if present (Frame.cameras may contain protocol.Camera objects or dicts)
            try:
                cams = getattr(frame, 'cameras', None)
                if cams and draw_camera:
                    for c in cams:
                        try:
                            # Accept either Camera objects or dicts
                            if hasattr(c, 'to_dict'):
                                cd = c.to_dict()
                            else:
                                cd = c
                            pos = cd.get('position')
                            tgt = cd.get('target')
                            # position/target may be dicts {'x':..} or tuples
                            def to_tuple(v):
                                if v is None:
                                    return None
                                if isinstance(v, dict):
                                    return (float(v.get('x', 0.0)), float(v.get('y', 0.0)), float(v.get('z', 0.0)))
                                if isinstance(v, (list, tuple)):
                                    return (float(v[0]), float(v[1]), float(v[2]))
                                return None

                            pos_t = to_tuple(pos)
                            tgt_t = to_tuple(tgt)
                            if pos_t and tgt_t:
                                draw_camera(position=pos_t, target=tgt_t, fov_deg=60.0, near=50.0, far=600.0, color=(1.0, 1.0, 0.0), scale=1.0)
                        except Exception:
                            continue
            except Exception:
                pass
            
            # Draw connection status
            self.draw_connection_status(frame)
        else:
            # No frame data - draw minimal grid and connection status
            draw_floor_grid(size=1000, spacing=200, height=0, color=(0.2, 0.2, 0.2))
            self.draw_connection_status(None)
    
    def setup_camera(self):
        """Set up camera transformation using spherical coordinates"""
        # Convert spherical to cartesian coordinates
        azimuth_rad = math.radians(self.camera_azimuth)
        elevation_rad = math.radians(self.camera_elevation)
        
        camera_x = self.camera_distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
        camera_y = self.camera_distance * math.sin(elevation_rad)
        camera_z = self.camera_distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
        
        camera_pos = [
            self.camera_target[0] + camera_x,
            self.camera_target[1] + camera_y,
            self.camera_target[2] + camera_z
        ]
        
        gluLookAt(
            camera_pos[0], camera_pos[1], camera_pos[2],  # camera position
            self.camera_target[0], self.camera_target[1], self.camera_target[2],  # look at
            0, 1, 0  # up vector
        )
    
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
        is_connected = (current_time - self.last_frame_time) < 2.0  # 2 second timeout

        # Draw status indicator (small colored square)
        if is_connected:
            glColor4f(0.0, 0.5, 0.0, 0.8)  # green background
            status_text = "Connected"
        else:
            glColor4f(0.5, 0.0, 0.0, 0.8)  # red background
            status_text = "Disconnected"

        # Connection indicator (top-left corner)
        glBegin(GL_QUADS)
        glVertex2f(10, 10)
        glVertex2f(30, 10)
        glVertex2f(30, 30)
        glVertex2f(10, 30)
        glEnd()

        # Replace small green blobs with readable text overlay
        try:
            # Use QGLWidget.renderText to draw simple overlay text in pixel coords
            # Show connection status and people count
            people_count = len(frame.people) if (frame and frame.people) else 0
            self.renderText(35, 22, f"{status_text}")
            self.renderText(35, 40, f"People: {people_count}")
        except Exception:
            # renderText may not be available in some contexts; ignore if it fails
            pass
        
        # Restore 3D mode
        glEnable(GL_DEPTH_TEST)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
    
    # Mouse interaction methods
    def mousePressEvent(self, event):
        """Handle mouse press (Wayland-friendly dragging)"""
        if event.button() in (QtCore.Qt.LeftButton, QtCore.Qt.RightButton, QtCore.Qt.MiddleButton):
            self._dragging = True
            self._last_pos = event.pos()
    
    def mouseMoveEvent(self, event):
        """Handle mouse drag for camera control"""
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
            
            # Simple panning relative to current view
            yaw_rad = math.radians(self.camera_azimuth)
            
            # Right vector (perpendicular to camera direction)
            right_x = math.cos(yaw_rad)
            right_z = -math.sin(yaw_rad)
            
            # Move target: dx moves left/right, dy moves up/down
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
        """Handle mouse wheel for zoom"""
        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 0.9
        self.camera_distance *= zoom_factor
        self.camera_distance = max(500, min(10000, self.camera_distance))  # Clamp zoom

