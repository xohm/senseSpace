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


"""SenseSpace library initialization and path setup"""

import os
import sys


def setup_paths():
    """
    Setup Python paths for SenseSpace development.
    Call this at the start of any example script.
    """
    # Get repository root (assumes calling script is in client/examples/*)
    script_dir = os.path.dirname(os.path.abspath(sys._getframe(1).f_code.co_filename))
    repo_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    
    libs_path = os.path.join(repo_root, 'libs')
    senseSpaceLib_path = os.path.join(libs_path, 'senseSpaceLib')
    client_path = os.path.join(repo_root, 'client')
    
    # Add paths if not already present
    for path in [libs_path, senseSpaceLib_path, client_path]:
        if path not in sys.path:
            sys.path.insert(0, path)


# Auto-export commonly used classes
from senseSpaceLib.senseSpace.protocol import Frame
from senseSpaceLib.senseSpace.interpretation import interpret_pose_from_angles
from senseSpaceLib.senseSpace.minimalClient import MinimalClient

__all__ = ['setup_paths', 'Frame', 'interpret_pose_from_angles', 'MinimalClient']
