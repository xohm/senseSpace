#!/usr/bin/env python3
"""
Simple Recording Example

Demonstrates how to programmatically control recording in senseSpace clients.
"""

import argparse
import time
import sys

from senseSpaceLib.senseSpace import setup_paths
setup_paths()

from senseSpaceLib.senseSpace.vizClient import VisualizationClient
from senseSpaceLib.senseSpace.vizWidget import SkeletonGLWidget
from senseSpaceLib.senseSpace.protocol import Frame


def main():
    parser = argparse.ArgumentParser(description="Recording Example")
    parser.add_argument("--server", default="localhost", help="Server IP")
    parser.add_argument("--port", type=int, default=12345, help="Server port")
    parser.add_argument("--rec", type=str, default=None, help="Playback mode: path to .ssrec file")
    parser.add_argument("--auto-record", type=int, default=0, 
                       help="Auto-record for N seconds (0 = manual control with 'R' key)")
    
    args = parser.parse_args()
    
    # Create visualization client
    client = VisualizationClient(
        viewer_class=SkeletonGLWidget,
        server_ip=args.server,
        server_port=args.port,
        playback_file=args.rec
    )
    
    # Auto-record mode
    if args.auto_record > 0:
        print(f"[INFO] Will auto-record for {args.auto_record} seconds after connection")
        
        def on_connection(connected):
            if connected:
                print("[INFO] Connected - starting recording...")
                client.start_recording()
                
                # Schedule stop after N seconds
                import threading
                def stop_after_delay():
                    time.sleep(args.auto_record)
                    print(f"[INFO] {args.auto_record} seconds elapsed - stopping recording")
                    client.stop_recording()
                
                threading.Thread(target=stop_after_delay, daemon=True).start()
        
        client.set_connection_callback(on_connection)
    
    else:
        print("[INFO] Manual recording mode:")
        print("  Press 'R' to start/stop recording")
        print("  Press 'P' to toggle point cloud")
        print("  Close window to exit")
    
    # Run client
    client.run()


if __name__ == "__main__":
    main()
