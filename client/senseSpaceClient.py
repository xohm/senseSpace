#!/usr/bin/env python3
"""
SenseSpace Client - Connects to server and displays body tracking data

Usage:
    python senseSpaceClient.py --viz                                    # Visualization mode with localhost:12345
    python senseSpaceClient.py --server 192.168.1.100 --port 12346     # Command line mode with custom server
    python senseSpaceClient.py --viz --server 192.168.1.100             # Visualization mode with custom server
"""

import argparse
import sys
import os

# Ensure local 'libs' folder is on sys.path when running from repo
# repo structure: <repo_root>/client/senseSpaceClient.py and <repo_root>/libs/senseSpaceLib
# so the repo root is one level up from the client folder.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
libs_path = os.path.join(repo_root, 'libs')
if os.path.isdir(libs_path) and libs_path not in sys.path:
    sys.path.insert(0, libs_path)
elif repo_root not in sys.path:
    # Fallback: add repo root so an installed package or alternative layout can still import
    sys.path.insert(0, repo_root)

# Import from our shared library
from senseSpaceLib.senseSpace.client import CommandLineClient
from senseSpaceLib.senseSpace.vizClient import VisualizationClient
from senseSpaceLib.senseSpace.vizWidget import SkeletonGLWidget


def main():
    parser = argparse.ArgumentParser(description="SenseSpace Client - Connect to server and display body tracking data")
    parser.add_argument("--server", "-s", default="localhost", help="Server IP address (default: localhost)")
    parser.add_argument("--port", "-p", type=int, default=12345, help="Server port (default: 12345)")
    parser.add_argument("--viz", "--visualization", action="store_true", help="Enable visualization mode (Qt OpenGL)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output in command line mode")
    parser.add_argument("--rec", type=str, default=None, help="Playback mode: path to .ssrec recording file")
    
    args = parser.parse_args()
    
    # Create appropriate client type
    if args.viz:
        client = VisualizationClient(
            viewer_class=SkeletonGLWidget,  # Use default viewer
            server_ip=args.server,
            server_port=args.port,
            window_title=f"SenseSpace Client - {args.server}:{args.port}",
            playback_file=args.rec
        )
    else:
        if args.rec:
            print("[ERROR] Playback mode (--rec) requires visualization (--viz)")
            sys.exit(1)
        
        client = CommandLineClient(
            server_ip=args.server,
            server_port=args.port,
            verbose=args.verbose
        )
    
    # Run client
    success = client.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
