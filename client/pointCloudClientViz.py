#!/usr/bin/env python3
"""
Point Cloud Client with Qt Visualization for senseSpace

Receives point cloud data and skeleton data, displays using the same Qt/OpenGL viewer as the server.
"""

import sys
import os
import argparse

# Add paths
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
server_path = os.path.join(repo_root, 'server')
libs_path = os.path.join(repo_root, 'libs')
if server_path not in sys.path:
    sys.path.insert(0, server_path)
if libs_path not in sys.path:
    sys.path.insert(0, libs_path)

from pointCloudClient import PointCloudClient
from senseSpaceLib.senseSpace.client import SenseSpaceClient

try:
    from qt_viewer import MainWindow
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QTimer
    QT_AVAILABLE = True
except ImportError as e:
    print(f"[ERROR] Qt visualization not available: {e}")
    QT_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser(description="SenseSpace Point Cloud Client with Visualization")
    parser.add_argument("--server", type=str, default="localhost", help="Server IP address")
    parser.add_argument("--port", type=int, default=12345, help="Skeleton data port")
    parser.add_argument("--pc-port", type=int, default=12346, help="Point cloud server port")
    
    args = parser.parse_args()
    
    if not QT_AVAILABLE:
        print("[ERROR] Qt not available - cannot run visualization")
        return 1
    
    # Create Qt application
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.setWindowTitle(f"Point Cloud Viewer - {args.server}")
    main_window.show()
    
    # Create skeleton data client (port 12345)
    skeleton_client = SenseSpaceClient(args.server, args.port)
    
    # Setup skeleton callback
    def on_skeleton_frame(frame):
        try:
            if hasattr(main_window, 'people_signal'):
                main_window.people_signal.emit(frame.people)
            elif hasattr(main_window, 'update_people'):
                main_window.update_people(frame.people)
        except Exception as e:
            print(f"[ERROR] Skeleton update failed: {e}")
    
    skeleton_client.set_frame_callback(on_skeleton_frame)
    
    # Create point cloud client (port 12346)
    pc_client = PointCloudClient(args.server, args.pc_port)
    
    # Setup point cloud callback to update visualization
    def on_pointcloud(points, colors, timestamp):
        try:
            if hasattr(main_window, 'glWidget') and hasattr(main_window.glWidget, 'set_point_cloud'):
                main_window.glWidget.set_point_cloud(points, colors)
        except Exception as e:
            print(f"[ERROR] Point cloud update failed: {e}")
    
    def on_pc_connection(connected):
        if connected:
            print("[INFO] Point cloud connected")
        else:
            print("[WARNING] Point cloud disconnected")
    
    pc_client.on_pointcloud_received = on_pointcloud
    pc_client.on_connection_changed = on_pc_connection
    
    # Connect to servers
    print(f"[INFO] Connecting to skeleton stream at {args.server}:{args.port}...")
    if not skeleton_client.connect():
        print("[WARNING] Failed to connect to skeleton stream")
    
    print(f"[INFO] Connecting to point cloud stream at {args.server}:{args.pc_port}...")
    if not pc_client.connect():
        print("[ERROR] Failed to connect to point cloud server")
        return 1
    
    print("[INFO] Receiving data...")
    
    # Setup timer to print statistics
    stats_timer = QTimer()
    def print_stats():
        pc_stats = pc_client.get_statistics()
        if pc_stats['fps'] > 0:
            print(f"[STATS] Point Cloud FPS: {pc_stats['fps']:.1f}, "
                  f"Received: {pc_stats['bytes_received']/1024/1024:.1f} MB, "
                  f"Skeleton: {'connected' if skeleton_client.connected else 'disconnected'}")
    stats_timer.timeout.connect(print_stats)
    stats_timer.start(2000)  # Print every 2 seconds
    
    # Run Qt event loop
    try:
        return app.exec_()
    finally:
        pc_client.disconnect()
        skeleton_client.disconnect()


if __name__ == "__main__":
    sys.exit(main())
