#!/usr/bin/env python3
"""
Qt OpenGL viewer widget for SenseSpace client
"""

import math
import time
from typing import Optional

from PyQt5 import QtWidgets, QtCore, QtOpenGL
from PyQt5.QtCore import QTimer
from PyQt5.QtOpenGL import QGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *

from senseSpaceLib.senseSpace.protocol import Frame
from senseSpaceLib.senseSpace.visualization import draw_skeletons_with_bones, draw_floor_grid


class ClientSkeletonGLWidget(QGLWidget):
    """OpenGL widget for displaying received skeleton data from server"""
    
    def __init__(self, parent=None):
        super(ClientSkeletonGLWidget, self).__init__(parent)
        
        # Camera parameters (spherical coordinates)
        self.camera_distance = 3000  # mm from target
        self.camera_azimuth = 45     # degrees around Y axis
        self.camera_elevation = 20   # degrees above XZ plane
        self.camera_target = [0, 0, 0]  # look-at point
        
        # Mouse interaction
        self.last_mouse_pos = None
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
        gluPerspective(45.0, aspect, 10.0, 10000.0)
        
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
            draw_floor_grid(
                size=2000,
                spacing=100,
                height=None,  # let draw_floor_grid resolve priority
                color=(0.3, 0.3, 0.3),
                people_data=frame.people,
                zed_floor_height=floor_height,
            )
            
            # Draw skeletons
            if frame.people:
                # draw_skeletons_with_bones expects people_data and optional color args.
                # Previously we incorrectly passed frame.body_model as the second argument.
                draw_skeletons_with_bones(frame.people)
            
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
        """Handle mouse press"""
        self.last_mouse_pos = event.pos()
    
    def mouseMoveEvent(self, event):
        """Handle mouse drag for camera control"""
        if self.last_mouse_pos is None:
            self.last_mouse_pos = event.pos()
            return
        
        dx = event.x() - self.last_mouse_pos.x()
        dy = event.y() - self.last_mouse_pos.y()
        
        if event.buttons() & QtCore.Qt.LeftButton:
            # Orbit camera
            self.camera_azimuth += dx * self.mouse_sensitivity
            self.camera_elevation = max(-85, min(85, self.camera_elevation - dy * self.mouse_sensitivity))
        elif event.buttons() & QtCore.Qt.RightButton:
            # Pan camera target
            # Convert mouse movement to world coordinates
            pan_scale = self.camera_distance * 0.001
            azimuth_rad = math.radians(self.camera_azimuth)
            
            # Right vector (perpendicular to view direction in XZ plane)
            right_x = -math.sin(azimuth_rad)
            right_z = math.cos(azimuth_rad)
            
            # Up vector (always Y for now)
            up_x, up_y, up_z = 0, 1, 0
            
            self.camera_target[0] += (dx * right_x) * pan_scale
            self.camera_target[1] -= dy * pan_scale
            self.camera_target[2] += (dx * right_z) * pan_scale
        
        self.last_mouse_pos = event.pos()
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zoom"""
        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 0.9
        self.camera_distance *= zoom_factor
        self.camera_distance = max(500, min(10000, self.camera_distance))  # Clamp zoom
