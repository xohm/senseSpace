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
    def __init__(self, parent=None):
        super().__init__(parent)
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
        gluPerspective(60.0, w / float(h or 1), 1.0, 20000.0)
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

        # Left: rotate
        if buttons & Qt.LeftButton:
            self.camera_azimuth += dx * 0.5
            self.camera_elevation += -dy * 0.5
            self.camera_elevation = max(-89.9, min(89.9, self.camera_elevation))

        # Right: pan
        elif buttons & Qt.RightButton:
            pan_scale = max(0.001, self.camera_distance * 0.002)
            azimuth_rad = math.radians(self.camera_azimuth)
            right_x = math.cos(azimuth_rad)
            right_z = -math.sin(azimuth_rad)

            self.camera_target[0] += (-dx) * right_x * pan_scale
            self.camera_target[2] += (-dx) * right_z * pan_scale
            self.camera_target[1] += (dy) * pan_scale * 0.5

        # Middle or both buttons: zoom
        elif (buttons & Qt.MiddleButton) or ((buttons & Qt.LeftButton) and (buttons & Qt.RightButton)):
            self.camera_distance += dy * 5.0
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
        # Smooth zooming: factor <1 for zoom in, >1 for zoom out
        factor = 0.9 if delta > 0 else 1.1
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
                poses = server_obj.get_camera_poses()
            if poses and draw_camera:
                for p in poses:
                    pos = p.get('position')
                    tgt = p.get('target')
                    draw_camera(position=pos, target=tgt, fov_deg=60.0, near=50.0, far=600.0, color=(1.0, 1.0, 0.0), scale=1.0, flip=self._camera_flip)
            else:
                # fallback: small reference camera at origin
                if draw_camera:
                    cam_pos = (0.0, 200.0, 0.0)
                    cam_target = (0.0, 200.0, 200.0)
                    draw_camera(position=cam_pos, target=cam_target, fov_deg=60.0, near=50.0, far=600.0, color=(1.0, 1.0, 0.0), scale=1.0, flip=True)
        except Exception:
            # keep rendering even if camera poses fail
            pass

        # Make the floor grid larger (approx 5m x 5m if units are mm)
        draw_floor_grid(size=5000, spacing=200, height=None,
                        color=(0.2, 0.2, 0.2), people_data=self.people_data,
                        zed_floor_height=self.zed_floor_height)

        draw_skeletons_with_bones(self.people_data,
                                  joint_color=(0.2, 0.8, 1.0),
                                  bone_color=(0.8, 0.2, 0.2))

        lines = (self.info_text or '').split('\n')
        x = 10
        y = 20
        for i, line in enumerate(lines):
            self.renderText(x, y + i * 14, line)

class MainWindow(QMainWindow):
    people_signal = pyqtSignal(object)

    def __init__(self, server=None, server_ip=None, server_port=None, get_client_count=None):
        super().__init__()
        self.setWindowTitle("ZED Skeleton Qt Viewer")
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

