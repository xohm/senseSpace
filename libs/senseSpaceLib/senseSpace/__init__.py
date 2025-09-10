"""
senseSpaceLib.senseSpace

A shared library for client and server communication, protocol handling, and visualization utilities for senseSpace.
"""


"""Lightweight package exports for senseSpaceLib.senseSpace.

This file intentionally avoids importing heavy optional dependencies (OpenGL, PyQt5, ZED SDK)
so that the package can be imported in environments where those are not installed. Import
visualization, communication, and server modules lazily from application entrypoints.
"""

from .protocol import Frame, Person, Joint
from .enums import Body18Joint, Body34Joint

__all__ = [
	'Frame', 'Person', 'Joint',
	'Body18Joint', 'Body34Joint'
]
