import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtOpenGL import QGLWidget
from PyQt5.QtCore import pyqtSignal, Qt
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import sys
import os
from senseSpaceLib.senseSpace.visualization import draw_skeletons_with_bones, draw_floor_grid


class SkeletonGLWidget(QGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.people_data = []
        self.info_text = ""
        self.zed_floor_height = None  # Store detected floor height from ZED SDK
        
        # Camera parameters for 3D navigation
        self.camera_distance = 2000  # Distance from target
        self.camera_azimuth = 45     # Horizontal rotation (degrees)
        self.camera_elevation = 20   # Vertical rotation (degrees)
        self.camera_target = [0, 700, 800]  # Look at typical skeleton center

    def set_people_data(self, people_data):
        self.people_data = people_data
        self.update()

    def set_info_text(self, text):
        self.info_text = text
        self.update()

    def set_floor_height(self, height):
        """Set the floor height detected by ZED SDK."""
        self.zed_floor_height = height
        self.update()

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glPointSize(8.0)
        glClearColor(0.1, 0.1, 0.1, 1.0)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, w / float(h or 1), 10.0, 10000.0)
        glMatrixMode(GL_MODELVIEW)

    def mousePressEvent(self, event):
        self.last_mouse_pos = (event.x(), event.y())

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos:
            dx = event.x() - self.last_mouse_pos[0]
            dy = event.y() - self.last_mouse_pos[1]
            if event.buttons() & Qt.LeftButton:
                # Rotate camera
                self.camera_azimuth += dx * 0.5
                self.camera_elevation += dy * 0.5
                self.camera_elevation = max(-90, min(90, self.camera_elevation))
            elif event.buttons() & Qt.RightButton:
                # Zoom
                self.camera_distance += dy * 10
                self.camera_distance = max(100, min(5000, self.camera_distance))
            self.last_mouse_pos = (event.x(), event.y())
            self.update()

    def wheelEvent(self, event):
        # Zoom with mouse wheel
        self.camera_distance -= event.angleDelta().y() * 2
        self.camera_distance = max(100, min(5000, self.camera_distance))
        self.update()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Calculate camera position based on spherical coordinates
        yaw_rad = math.radians(self.camera_azimuth)
        pitch_rad = math.radians(self.camera_elevation)

        cam_x = self.camera_target[0] + self.camera_distance * math.cos(pitch_rad) * math.sin(yaw_rad)
        cam_y = self.camera_target[1] + self.camera_distance * math.sin(pitch_rad)
        cam_z = self.camera_target[2] + self.camera_distance * math.cos(pitch_rad) * math.cos(yaw_rad)

        # Standard up-vector where Y points up
        gluLookAt(cam_x, cam_y, cam_z,
                  self.camera_target[0], self.camera_target[1], self.camera_target[2],
                  0, 1, 0)

        # Draw coordinate axes for reference
        glBegin(GL_LINES)
        glColor3f(1, 0, 0)  # X axis - red
        glVertex3f(0, 0, 0)
        glVertex3f(200, 0, 0)
        glColor3f(0, 1, 0)  # Y axis - green
        glVertex3f(0, 0, 0)
        glVertex3f(0, 200, 0)
        glColor3f(0, 0, 1)  # Z axis - blue
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 200)
        glEnd()

        # Draw floor grid with auto-detected floor height
        draw_floor_grid(size=2000, spacing=100, height=None,
                        color=(0.2, 0.2, 0.2), people_data=self.people_data,
                        zed_floor_height=self.zed_floor_height)

        # Draw skeleton using visualization helper
        draw_skeletons_with_bones(self.people_data,
                                  joint_color=(0.2, 0.8, 1.0),  # Blue points
                                  bone_color=(0.8, 0.2, 0.2))   # Red bones

        # Draw info overlay (top left) — support multiple lines
        lines = (self.info_text or '').split('\n')
        x = 10
        y = 20
        for i, line in enumerate(lines):
            # move down by 14 pixels per line
            self.renderText(x, y + i * 14, line)


class MainWindow(QMainWindow):
    # Signal to receive people data from other threads safely
    people_signal = pyqtSignal(object)

    def __init__(self, server_ip=None, get_client_count=None):
        super().__init__()
        self.setWindowTitle("ZED Skeleton Qt Viewer")
        self.setGeometry(100, 100, 800, 600)
        self.glWidget = SkeletonGLWidget(self)
        self.setCentralWidget(self.glWidget)
        self.server_ip = server_ip
        self.get_client_count = get_client_count
        self.bodies_tracked = 0
        # connect signal to GUI update slot
        self.people_signal.connect(self.update_people)

    def update_people(self, people_data):
        self.bodies_tracked = len(people_data)
        self.glWidget.set_people_data(people_data)
        self.update_info()

    def set_floor_height(self, height):
        """Set the floor height detected by ZED SDK."""
        self.glWidget.set_floor_height(height)

    def update_info(self):
        info = f"Server IP: {self.server_ip or 'N/A'}\nBodies tracked: {self.bodies_tracked}"
        if self.get_client_count:
            info += f"\nClients connected: {self.get_client_count()}"
        self.glWidget.set_info_text(info)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    # Example: update with dummy data
    import time
    import threading
    def update_loop():
        while True:
            # Replace this with real data update
            dummy = [{"skeleton": [{"pos": {"x": 0, "y": 1, "z": 2}}]}]
            # if no real data arrives, show a synthetic T-pose like skeleton for quick verification
            win.update_people(dummy)
            time.sleep(1/30)
    threading.Thread(target=update_loop, daemon=True).start()
    sys.exit(app.exec_())
