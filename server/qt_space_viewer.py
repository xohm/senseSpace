import sys
import json
import math
from PyQt5.QtWidgets import QApplication, QMainWindow, QOpenGLWidget
from PyQt5.QtCore import Qt
from OpenGL.GL import *
from OpenGL.GLU import *
import random

def random_color(seed):
    random.seed(seed)
    return (random.random(), random.random(), random.random())

class SpaceGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.people_data = []
        self.camera_poses = []
        self.last_pos = None
        self.x_rot = 0
        self.y_rot = 0
        self.zoom = -10
        self.pan_x = 0
        self.pan_y = 0

    def set_people_data(self, people_data):
        self.people_data = people_data
        self.update()

    def set_camera_poses(self, camera_poses):
        self.camera_poses = camera_poses
        self.update()

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glPointSize(6.0)
        glClearColor(0.1, 0.1, 0.1, 1.0)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, w / float(h or 1), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(self.pan_x, self.pan_y, self.zoom)
        glRotatef(self.x_rot, 1, 0, 0)
        glRotatef(self.y_rot, 0, 1, 0)
        # Draw cameras
        for cam in self.camera_poses:
            self.draw_camera(cam)
        # Draw people
        for idx, p in enumerate(self.people_data):
            color = random_color(p.get('id', idx))
            glColor3f(*color)
            glBegin(GL_POINTS)
            for j in p["skeleton"]:
                glVertex3f(j["pos"]["x"], j["pos"]["y"], j["pos"]["z"])
            glEnd()

    def draw_camera(self, cam):
        pos = cam.get('position', [0,0,0])
        size = 0.2
        # 8 corners of the cube
        corners = [
            [size/2, size/2, size/2], [size/2, size/2, -size/2],
            [size/2, -size/2, size/2], [size/2, -size/2, -size/2],
            [-size/2, size/2, size/2], [-size/2, size/2, -size/2],
            [-size/2, -size/2, size/2], [-size/2, -size/2, -size/2]
        ]
        edges = [
            (0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7),
            (4,5), (4,6), (5,7), (6,7)
        ]
        glPushMatrix()
        glTranslatef(pos[0], pos[1], pos[2])
        glColor3f(1, 1, 0)
        glBegin(GL_LINES)
        for e in edges:
            for v in e:
                glVertex3f(*corners[v])
        glEnd()
        glPopMatrix()

    def mousePressEvent(self, event):
        self.last_pos = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            dx = event.x() - self.last_pos.x()
            dy = event.y() - self.last_pos.y()
            self.x_rot += dy
            self.y_rot += dx
        elif event.buttons() & Qt.RightButton:
            dx = event.x() - self.last_pos.x()
            dy = event.y() - self.last_pos.y()
            self.pan_x += dx * 0.01
            self.pan_y -= dy * 0.01
        self.last_pos = event.pos()
        self.update()

    def wheelEvent(self, event):
        self.zoom += event.angleDelta().y() * 0.01
        self.update()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ZED Space Qt Viewer")
        self.setGeometry(100, 100, 1000, 800)
        self.glWidget = SpaceGLWidget(self)
        self.setCentralWidget(self.glWidget)

    def update_people(self, people_data):
        self.glWidget.set_people_data(people_data)

    def update_cameras(self, camera_poses):
        self.glWidget.set_camera_poses(camera_poses)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    # Example: update with dummy data
    import time
    import threading
    def update_loop():
        while True:
            dummy_people = [{"id": 1, "skeleton": [{"pos": {"x": 0, "y": 1, "z": 2}}]}, {"id": 2, "skeleton": [{"pos": {"x": 1, "y": 1, "z": 2}}]}]
            dummy_cams = [{"position": [0,0,0]}, {"position": [2,0,0]}]
            win.update_people(dummy_people)
            win.update_cameras(dummy_cams)
            time.sleep(1/30)
    threading.Thread(target=update_loop, daemon=True).start()
    sys.exit(app.exec_())
