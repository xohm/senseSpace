"""
senseSpaceLib.senseSpace

A shared library for client and server communication, protocol handling, and visualization utilities for senseSpace.
"""


# Import protocol and enums from local package for shared use
from .protocol import Frame, Person, Joint
from .enums import Body18Joint, Body34Joint

# Import shared communication and visualization modules
from .communication import TCPServer, TCPClient
from .visualization import draw_skeleton

# Communication utilities (TCP client/server helpers)
# Visualization utilities (for drawing skeletons, etc.)

# Add more shared logic here as needed
