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


try:
    # prefer package import, fallback to local
    from senseSpaceLib.senseSpace.protocol import Camera
except Exception:
    try:
        from ..libs.senseSpaceLib.senseSpace.protocol import Camera
    except Exception:
        from protocol import Camera


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

        # Make sure widget receives mouse events and focus
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()  # ensure widget initially focused
    
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
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Set up camera
        yaw_rad = math.radians(self.camera_azimuth)
        pitch_rad = math.radians(self.camera_elevation)
        cam_x = self.camera_target[0] + self.camera_distance * math.cos(pitch_rad) * math.sin(yaw_rad)
        cam_y = self.camera_target[1] + self.camera_distance * math.sin(pitch_rad)
        cam_z = self.camera_target[2] + self.camera_distance * math.cos(pitch_rad) * math.cos(yaw_rad)
        gluLookAt(cam_x, cam_y, cam_z, self.camera_target[0], self.camera_target[1], self.camera_target[2], 0, 1, 0)

        # Get current frame with thread safety
        frame = None
        with QtCore.QMutexLocker(self.frame_lock):
            frame = self.current_frame

        if frame is None:
            # No frame data - draw minimal grid and connection status
            draw_floor_grid(size=1000, spacing=200, height=0, color=(0.2, 0.2, 0.2))
            self.draw_connection_status(None)
            return

        # ensure floor_height is always defined
        floor_height = getattr(frame, 'floor_height', None)

        # Normalize camera list and compute centroid to auto-focus view for debugging
        cam_list = []
        raw_cams = getattr(frame, 'cameras', None)
        if raw_cams:
            # If server sent envelope dict -> extract list
            if isinstance(raw_cams, dict) and 'cameras' in raw_cams:
                cam_list = raw_cams.get('cameras') or []
            elif isinstance(raw_cams, list):
                # Guard against accidentally iterating dict keys -> ['cameras','floor']
                if all(isinstance(x, str) for x in raw_cams) and set(raw_cams) <= {'cameras', 'floor'}:
                    # invalid payload (keys only) -> ignore cameras
                    cam_list = []
                else:
                    cam_list = raw_cams
            elif isinstance(raw_cams, str):
                # try to parse JSON envelope or list
                try:
                    import json
                    parsed = json.loads(raw_cams)
                    if isinstance(parsed, dict) and 'cameras' in parsed:
                        cam_list = parsed.get('cameras') or []
                    elif isinstance(parsed, list):
                        cam_list = parsed
                except Exception:
                    cam_list = []

        # If we have valid cameras, try autofocus
        if cam_list:
            coords = []
            for c in cam_list:
                try:
                    cd = c.to_dict() if hasattr(c, 'to_dict') else (json.loads(c) if isinstance(c, str) else c)
                    pos = cd.get('position') if isinstance(cd, dict) else None
                    if isinstance(pos, dict):
                        coords.append((float(pos.get('x', 0.0)), float(pos.get('y', 0.0)), float(pos.get('z', 0.0))))
                except Exception:
                    continue
            if coords:
                cx = sum(p[0] for p in coords)/len(coords)
                cy = sum(p[1] for p in coords)/len(coords)
                cz = sum(p[2] for p in coords)/len(coords)


        # Fallback to 0.0 if no floor height found
        if floor_height is None:
            floor_height = 0.0

        # Draw floor grid with proper height (draw at Y=0 always; keep zed_floor_height for debug/logic)
        if draw_floor_grid:
            draw_floor_grid(
                size=5000,
                spacing=200,
                height=0.0,  # always draw floor at zero in client view
                color=(0.2, 0.2, 0.2),
                people_data=frame.people if hasattr(frame, 'people') else [],
                zed_floor_height=floor_height  # still pass detected floor if needed
            )

        # Draw skeletons
        if draw_skeletons_with_bones and hasattr(frame, 'people') and frame.people:
            draw_skeletons_with_bones(frame.people, joint_color=(0.2, 0.8, 1.0), bone_color=(0.8, 0.2, 0.2))

        # Draw cameras if present (Frame.cameras may contain serialized strings, dicts or Camera objects)
        try:
            cams = getattr(frame, 'cameras', None)
            if cams and draw_camera:
                camera_list = []

                # 1) If server sent the new envelope {"cameras": [...], "floor": {...}}
                if isinstance(cams, dict):
                    maybe = cams.get('cameras', None)
                    if isinstance(maybe, list):
                        camera_list = maybe
                    else:
                        # if it's a single camera dict, wrap it
                        if cams and not cams.get('floor') and isinstance(cams, dict):
                            # if cams look like a camera dict (has serial/position), use it
                            if 'serial' in cams and ('position' in cams or 'orientation' in cams):
                                camera_list = [cams]
                            else:
                                camera_list = []
                # 2) If server already set a list
                elif isinstance(cams, list):
                    camera_list = cams
                # 3) If server sent a JSON string payload
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

                # Guard: if camera_list looks like only dict keys (e.g. ['cameras','floor']) skip them
                if isinstance(camera_list, list) and all(isinstance(x, str) for x in camera_list):
                    # common symptom: a dict was iterated -> got keys
                    # ignore lists of bare tokens
                    if set(camera_list) <= {'cameras', 'floor'}:
                        camera_list = []

                # iterate and draw
                for i, c in enumerate(camera_list):
                    try:
                        cam_dict = None

                        # If string: try parse JSON
                        if isinstance(c, str):
                            import json
                            try:
                                cam_dict = json.loads(c)
                            except Exception:
                                # skip non-json strings
                                continue
                        elif isinstance(c, dict):
                            cam_dict = c
                        else:
                            # object with to_dict()
                            if hasattr(c, 'to_dict'):
                                try:
                                    cam_dict = c.to_dict()
                                except Exception:
                                    cam_dict = None

                        if not cam_dict or not isinstance(cam_dict, dict):
                            continue

                        # extract position and orientation safely
                        pos = cam_dict.get('position')
                        orient = cam_dict.get('orientation')

                        def to_tuple_pos(v):
                            if v is None:
                                return None
                            if isinstance(v, dict):
                                return (float(v.get('x', 0.0)), float(v.get('y', 0.0)), float(v.get('z', 0.0)))
                            if isinstance(v, (list, tuple)) and len(v) >= 3:
                                return (float(v[0]), float(v[1]), float(v[2]))
                            return None

                        def to_quat_dict(v):
                            if v is None:
                                return {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
                            if isinstance(v, dict):
                                return {
                                    'x': float(v.get('x', 0.0)),
                                    'y': float(v.get('y', 0.0)),
                                    'z': float(v.get('z', 0.0)),
                                    'w': float(v.get('w', 1.0))
                                }
                            if isinstance(v, (list, tuple)) and len(v) >= 4:
                                return {'x': float(v[0]), 'y': float(v[1]), 'z': float(v[2]), 'w': float(v[3])}
                            return {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}

                        pos_t = to_tuple_pos(pos)
                        orientation_dict = to_quat_dict(orient)

                        # draw visible marker for debugging
                        if pos_t is not None:
                            glPointSize(8.0)
                            glBegin(GL_POINTS)
                            glColor3f(1.0, 0.0, 1.0)
                            glVertex3f(pos_t[0], pos_t[1], pos_t[2])
                            glEnd()

                        # draw frustum if we have orientation too
                        if pos_t and orientation_dict:
                            draw_camera(position=pos_t, orientation=orientation_dict, fov_deg=60.0, near=50.0, far=800.0, color=(1.0,1.0,0.0), scale=2.0)

                    except Exception as e:
                        print(f"[WARNING] Failed to draw camera {i}: {e}")
                        continue
        except Exception as e:
            print(f"[WARNING] Camera rendering failed: {e}")
            import traceback
            traceback.print_exc()
            pass
            
        # Draw connection status
        self.draw_connection_status(frame)
    
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
        if event.button() in (Qt.LeftButton, Qt.RightButton, Qt.MiddleButton):
            self._dragging = True
            self._last_pos = event.pos()
            self.setFocus()  # ensure we receive wheel events

    def mouseReleaseEvent(self, event):
        """Handle mouse release - stop dragging"""
        self._dragging = False
        self._last_pos = None

    def mouseMoveEvent(self, event):
        """Handle mouse drag for camera control"""
        # debug early-return
        if not self._dragging or self._last_pos is None:
            # still optionally print to confirm movement without drag
            # print(f"[DEBUG] mouseMoveEvent (no-drag) pos={event.pos()}")
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
            self.camera_distance = max(100.0, min(10000.0, self.camera_distance))  # match server min 100

        self._last_pos = event.pos()
        self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return
        # Inverted smooth zooming: factor >1 for zoom in, <1 for zoom out
        factor = 1.1 if delta > 0 else 0.9
        self.camera_distance = max(100.0, min(10000.0, self.camera_distance * factor))
        self.update()

