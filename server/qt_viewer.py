import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtOpenGL import QGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *


class SkeletonGLWidget(QGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.people_data = []
        self.info_text = ""

    def set_people_data(self, people_data):
        self.people_data = people_data
        self.update()

    def set_info_text(self, text):
        self.info_text = text
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
        gluLookAt(0, 1.5, 5, 0, 1.5, 0, 0, 1, 0)
        glColor3f(0.2, 0.8, 1.0)
        for p in self.people_data:
            glBegin(GL_POINTS)
            for j in p["skeleton"]:
                glVertex3f(j["pos"]["x"], j["pos"]["y"], j["pos"]["z"])
            glEnd()
        # Draw info overlay (top left)
        self.renderText(10, 20, self.info_text)


class MainWindow(QMainWindow):
    def __init__(self, server_ip=None, get_client_count=None):
        super().__init__()
        self.setWindowTitle("ZED Skeleton Qt Viewer")
        self.setGeometry(100, 100, 800, 600)
        self.glWidget = SkeletonGLWidget(self)
        self.setCentralWidget(self.glWidget)
        self.server_ip = server_ip
        self.get_client_count = get_client_count
        self.bodies_tracked = 0

    def update_people(self, people_data):
        self.bodies_tracked = len(people_data)
        self.glWidget.set_people_data(people_data)
        self.update_info()

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
            win.update_people(dummy)
            time.sleep(1/30)
    threading.Thread(target=update_loop, daemon=True).start()
    sys.exit(app.exec_())
