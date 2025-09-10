from OpenGL.GL import *
from OpenGL.GLU import *

class SkeletonVisualizer3D:
    """
    Utility to visualize a Person (from protocol.py) using OpenGL.
    """
    def __init__(self):
        pass

    def draw_person(self, person, color=(0.2, 0.8, 1.0)):
        """
        Draws a person skeleton using OpenGL.
        :param person: Person object (from protocol.py)
        :param color: RGB tuple for the skeleton color
        """
        glColor3f(*color)
        glPointSize(6.0)
        glBegin(GL_POINTS)
        for joint in person.skeleton:
            pos = joint.pos if hasattr(joint, 'pos') else joint["pos"]
            glVertex3f(pos["x"], pos["y"], pos["z"])
        glEnd()

    def draw_people(self, people, color=(0.2, 0.8, 1.0)):
        """
        Draws multiple people (list of Person objects)
        """
        for person in people:
            self.draw_person(person, color=color)
