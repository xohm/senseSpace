from OpenGL.GL import *
from OpenGL.GLU import *
import math
import sys
import time
import threading
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtOpenGL import QGLWidget
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QCursor
from typing import Optional
import socket
from PyQt5.QtWidgets import QMainWindow as _QMainWindow  # for isinstance checks
 
def _get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"

try:
    from senseSpaceLib.senseSpace.visualization import draw_skeletons_with_bones, draw_floor_grid, draw_camera
except Exception:
    try:
        from libs.senseSpaceLib.senseSpace.visualization import draw_skeletons_with_bones, draw_floor_grid, draw_camera
    except Exception:
        # best-effort fallback: allow module to run even if visualization isn't importable in this environment
        draw_skeletons_with_bones = None
        draw_floor_grid = None
        draw_camera = None

class SkeletonGLWidget(QGLWidget):
    def __init__(self, server_instance=None):
        super(SkeletonGLWidget, self).__init__()
        
        self.server = server_instance
        self.people_data = []
        self.info_text = ""
        self.zed_floor_height = None

        # camera drawing settings
        self._camera_flip = False  # when True, invert camera frustum direction

        # Camera parameters
        self.camera_distance = 2000.0
        self.camera_azimuth = 45.0   # degrees
        self.camera_elevation = 20.0  # degrees
        self.camera_target = [0.0, 700.0, 800.0]

        # Mouse state
        self._dragging = False
        self._last_pos = None

    # ------------------------
    # Data setters
    # ------------------------
    def set_people_data(self, people_data):
        self.people_data = people_data
        self.update()

    def set_info_text(self, text):
        self.info_text = text
        self.update()

    def set_floor_height(self, height):
        self.zed_floor_height = height
        self.update()

    # ------------------------
    # OpenGL setup
    # ------------------------
    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glPointSize(8.0)
        glClearColor(0.1, 0.1, 0.1, 1.0)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60.0, w / float(h or 1), 1.0, 30000.0)
        glMatrixMode(GL_MODELVIEW)

    # ------------------------
    # Mouse handling
    # ------------------------
    def mousePressEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.RightButton, Qt.MiddleButton):
            self._dragging = True
            self._last_pos = event.pos()

    def mouseMoveEvent(self, event):
        if not self._dragging or self._last_pos is None:
            return

        dx = event.x() - self._last_pos.x()
        dy = event.y() - self._last_pos.y()
        buttons = event.buttons()

        # Left: rotate (inverted directions)
        if buttons & Qt.LeftButton:
            self.camera_azimuth += -dx * 0.5  # Invert horizontal rotation
            self.camera_elevation += dy * 0.5   # Invert vertical rotation
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

        # Middle or both buttons: zoom (inverted)
        elif (buttons & Qt.MiddleButton) or ((buttons & Qt.LeftButton) and (buttons & Qt.RightButton)):
            self.camera_distance += -dy * 5.0  # Invert zoom direction
            self.camera_distance = max(100.0, min(10000.0, self.camera_distance))

        self._last_pos = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.RightButton, Qt.MiddleButton):
            self._dragging = False
            self._last_pos = None

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return
        # Inverted smooth zooming: factor >1 for zoom in, <1 for zoom out
        factor = 1.1 if delta > 0 else 0.9
        self.camera_distance = max(100.0, min(10000.0, self.camera_distance * factor))
        self.update()

    def keyPressEvent(self, event):
        # Toggle camera frustum flip with 'c'
        if event.key() == Qt.Key_C:
            self._camera_flip = not self._camera_flip
            self.update()
        else:
            super().keyPressEvent(event)

    # ------------------------
    # Rendering
    # ------------------------
    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        yaw_rad = math.radians(self.camera_azimuth)
        pitch_rad = math.radians(self.camera_elevation)

        cam_x = self.camera_target[0] + self.camera_distance * math.cos(pitch_rad) * math.sin(yaw_rad)
        cam_y = self.camera_target[1] + self.camera_distance * math.sin(pitch_rad)
        cam_z = self.camera_target[2] + self.camera_distance * math.cos(pitch_rad) * math.cos(yaw_rad)

        gluLookAt(cam_x, cam_y, cam_z,
                  self.camera_target[0], self.camera_target[1], self.camera_target[2],
                  0, 1, 0)

        # draw all configured cameras if server provides poses (fusion mode)
        try:
            server_obj = None
            # MainWindow stored server reference on construction
            parent = self.parent()
            if parent and hasattr(parent, 'server'):
                server_obj = parent.server
            poses = []
            if server_obj and getattr(server_obj, 'is_fusion_mode', False) and hasattr(server_obj, 'get_camera_poses'):
                camera_poses_data = server_obj.get_camera_poses()
                # Handle new format: {"cameras": [...], "floor": {...}}
                if isinstance(camera_poses_data, dict) and 'cameras' in camera_poses_data:
                    poses = camera_poses_data['cameras']
                else:
                    # Fallback for old format (list of poses)
                    poses = camera_poses_data if camera_poses_data else []
                    
            if poses and draw_camera:
                for p in poses:
                    try:
                        pos = p.get('position')
                        orientation = p.get('orientation')  # New: quaternion orientation
                        
                        # Convert position to tuple if it's a dict
                        if isinstance(pos, dict):
                            pos_tuple = (pos.get('x', 0.0), pos.get('y', 0.0), pos.get('z', 0.0))
                        else:
                            pos_tuple = pos
                        
                        # Handle orientation quaternion
                        if orientation and isinstance(orientation, dict):
                            orientation_dict = {
                                'x': float(orientation.get('x', 0.0)),
                                'y': float(orientation.get('y', 0.0)),
                                'z': float(orientation.get('z', 0.0)),
                                'w': float(orientation.get('w', 1.0))
                            }
                        else:
                            # Fallback to identity quaternion
                            orientation_dict = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
                        
                        if pos_tuple and orientation_dict:
                            draw_camera(
                                position=pos_tuple, 
                                orientation=orientation_dict,  # Use quaternion instead of target
                                fov_deg=60.0, 
                                near=50.0, 
                                far=600.0, 
                                color=(1.0, 1.0, 0.0), 
                                scale=1.0, 
                                flip=self._camera_flip
                            )
                    except Exception as e:
                        print(f"[WARNING] Failed to draw camera: {e}")
                        continue
            else:
                # fallback: small reference camera at origin
                if draw_camera:
                    cam_pos = (0.0, 200.0, 0.0)
                    # Use identity quaternion for fallback camera
                    cam_orientation = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
                    draw_camera(
                        position=cam_pos, 
                        orientation=cam_orientation,  # Use quaternion instead of target
                        fov_deg=60.0, 
                        near=50.0, 
                        far=600.0, 
                        color=(1.0, 1.0, 0.0), 
                        scale=1.0, 
                        flip=True
                    )
        except Exception as e:
            # keep rendering even if camera poses fail
            print(f"[WARNING] Camera rendering failed: {e}")
            pass

        # Make the floor grid larger (approx 5m x 5m if units are mm)
        draw_floor_grid(size=5000, spacing=200, height=0,
                        color=(0.2, 0.2, 0.2), people_data=self.people_data,
                        zed_floor_height=self.zed_floor_height)

        draw_skeletons_with_bones(self.people_data,
                                  joint_color=(0.2, 0.8, 1.0),
                                  bone_color=(0.8, 0.2, 0.2))

        # Draw connection status overlay
        people_count = len(self.people_data) if self.people_data else 0
        self.draw_connection_status(people_count, self.zed_floor_height)

        """
        lines = (self.info_text or '').split('\n')
        x = 10
        y = 20
        for i, line in enumerate(lines):
            self.renderText(x, y + i * 14, line)
        """

    def draw_connection_status(self, people_count: int, floor_height: Optional[float]):
        """Draw connection and system info as overlay"""
        # Save projection/modelview and switch to 2D overlay mode
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width(), self.height(), 0, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # Disable depth test and depth writes so overlay always draws on top
        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)
        # Enable alpha blending for semi-transparent background
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        try:
            # Resolve server object: widget may have been constructed with either
            # a server instance or a MainWindow reference; try several fallbacks.
            server_obj = None
            if getattr(self, "server", None) is not None:
                # if self.server looks like the actual server (has local_ip/port), use it
                if hasattr(self.server, "local_ip") or hasattr(self.server, "port") or hasattr(self.server, "clients"):
                    server_obj = self.server
                # if it is a MainWindow instance that stores .server, use that
                elif isinstance(self.server, _QMainWindow) and hasattr(self.server, "server"):
                    server_obj = getattr(self.server, "server", None)

            if server_obj is None:
                parent = self.parent()
                if parent is not None and hasattr(parent, "server"):
                    server_obj = getattr(parent, "server", None)

            # Determine displayed IP/port
            server_ip = getattr(server_obj, "local_ip", None) or getattr(self, "server_ip", None)
            server_port = getattr(server_obj, "port", None) or getattr(self, "server_port", None)
            if not server_ip:
                server_ip = _get_local_ip()
            if not server_port:
                server_port = "unknown"

            # Build lines of text
            clients = len(getattr(server_obj, "clients", [])) if server_obj else 0
            lines = [
                f"Server: {server_ip}:{server_port}",
                f"People: {people_count}",
                (f"Floor: {floor_height:.0f}mm" if floor_height is not None else "Floor: detecting..."),
                f"Clients: {clients}"
            ]

            # Background box size based on number of lines (fixed width)
            line_h = 14
            padding = 10
            
            # Draw text lines
            glColor4f(0.0, 1.0, 0.0, 1.0)
            text_x = padding + 5
            text_y0 = padding + 15
            for i, line in enumerate(lines):
                try:
                    self.renderText(text_x, text_y0 + i * line_h, line)
                except Exception:
                    # renderText may be unavailable in some GL contexts - ignore
                    pass
        finally:
            # Restore overlay GL state
            glDisable(GL_BLEND)
            glDepthMask(GL_TRUE)
            glEnable(GL_DEPTH_TEST)
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()

class MainWindow(QMainWindow):
    people_signal = pyqtSignal(object)

    def __init__(self, server=None, server_ip=None, server_port=None, get_client_count=None):
        super().__init__()
        self.setWindowTitle("senseSpace Server")
        self.setGeometry(100, 100, 800, 600)
        # Pass server object into GL widget so it can query camera poses
        self.server = server
        self.glWidget = SkeletonGLWidget(self)
        self.setCentralWidget(self.glWidget)
        self.server_ip = server_ip
        self.server_port = server_port
        self.get_client_count = get_client_count
        self.bodies_tracked = 0
        self.people_signal.connect(self.update_people)

    def update_people(self, people_data):
        self.bodies_tracked = len(people_data)
        self.glWidget.set_people_data(people_data)
        self.update_info()

    def set_floor_height(self, height):
        self.glWidget.set_floor_height(height)

    def update_info(self):
        hostport = f"{self.server_ip}:{self.server_port}" if (self.server_ip and self.server_port) else (self.server_ip or 'N/A')
        info = f"Server: {hostport}\nBodies tracked: {self.bodies_tracked}"
        if self.get_client_count:
            info += f"\nClients connected: {self.get_client_count()}"
        self.glWidget.set_info_text(info)


def start_qt_application(server):
    """Start Qt application with OpenGL viewer"""
    app = QApplication([])
    
    window = QMainWindow()
    server_ip = server.local_ip if hasattr(server, 'local_ip') else "unknown"
    window.setWindowTitle(f"SenseSpace Server - {server_ip}:{server.port}")
    window.resize(1200, 800)
    
    gl_widget = SkeletonGLWidget(server_instance=server)
    window.setCentralWidget(gl_widget)
    
    # Set up update callback so server can update the widget
    server.set_update_callback(gl_widget.update_people)
    
    return app, window


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()

    def update_loop():
        while True:
            dummy = [{"skeleton": [{"pos": {"x": 0, "y": 1, "z": 2}}]}]
            win.update_people(dummy)
            time.sleep(1 / 30)

    threading.Thread(target=update_loop, daemon=True).start()
    sys.exit(app.exec_())

