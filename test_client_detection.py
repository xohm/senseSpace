#!/usr/bin/env python3
"""
Test script to demonstrate automatic client detection for video streaming.

This script shows how the server automatically starts/stops encoding based on
connected clients, saving CPU/GPU resources when no clients are watching.

Usage:
    # Terminal 1 - Server
    python test_client_detection.py --mode server
    
    # Terminal 2 - Client 1
    python test_client_detection.py --mode client --server 127.0.0.1
    
    # Terminal 3 - Client 2
    python test_client_detection.py --mode client --server 127.0.0.1
"""

import sys
import argparse
import time
import numpy as np
from pathlib import Path

# Add libs to path
sys.path.insert(0, str(Path(__file__).parent / 'libs'))

from senseSpaceLib.senseSpace.video_streaming import VideoStreamer, VideoReceiver
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_server():
    """Run streaming server with client detection"""
    logger.info("=== Video Streaming Server (Client Detection Test) ===")
    logger.info("Server will automatically start streaming when clients connect")
    logger.info("and stop when all clients disconnect.\n")
    
    # Create streamer with client detection enabled (default)
    streamer = VideoStreamer(
        host='0.0.0.0',
        rgb_port=5000,
        depth_port=5001,
        width=640,
        height=480,
        framerate=15,
        enable_client_detection=True,
        client_timeout=5.0
    )
    
    logger.info("Server ready. Waiting for clients...")
    logger.info("Server is IDLE - not encoding (saving resources)\n")
    
    frame_count = 0
    last_client_count = 0
    
    try:
        while True:
            # Generate test frames (simple gradient pattern)
            # Only creates the frames, encoding happens in GStreamer
            rgb_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            depth_frame = np.random.randint(0, 5000, (480, 640), dtype=np.uint16)
            
            # Push frames (will only encode if streaming is active)
            streamer.push_rgb_frame(rgb_frame)
            streamer.push_depth_frame(depth_frame)
            
            # Show status every second
            if frame_count % 15 == 0:
                client_count = streamer.get_client_count()
                
                if client_count != last_client_count:
                    if client_count > 0:
                        logger.info(f"📹 STREAMING ACTIVE - {client_count} client(s) connected")
                    else:
                        logger.info("💤 IDLE - No clients (not encoding)")
                    last_client_count = client_count
            
            frame_count += 1
            time.sleep(1/15)  # 15 FPS
            
    except KeyboardInterrupt:
        logger.info("\nShutting down server...")
        streamer.shutdown()
        logger.info("Server stopped")


def run_client(server_ip):
    """Run streaming client"""
    logger.info(f"=== Video Streaming Client ===")
    logger.info(f"Connecting to {server_ip}:5000/5001")
    logger.info("Client will send heartbeats to keep server streaming\n")
    
    # Track frame reception
    rgb_count = [0]
    depth_count = [0]
    last_report = [time.time()]
    
    def on_rgb(frame):
        rgb_count[0] += 1
        if time.time() - last_report[0] > 2.0:
            logger.info(f"📺 Receiving: {rgb_count[0]} RGB, {depth_count[0]} depth frames")
            rgb_count[0] = 0
            depth_count[0] = 0
            last_report[0] = time.time()
    
    def on_depth(frame):
        depth_count[0] += 1
    
    # Create receiver with heartbeat enabled (default)
    receiver = VideoReceiver(
        server_ip=server_ip,
        rgb_port=5000,
        depth_port=5001,
        rgb_callback=on_rgb,
        depth_callback=on_depth,
        send_heartbeat=True,
        heartbeat_interval=2.0
    )
    
    logger.info("Starting receiver (this triggers server to start encoding)...")
    receiver.start()
    
    logger.info("✅ Client connected - server should now be streaming")
    logger.info("Press Ctrl+C to disconnect\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nDisconnecting client...")
        receiver.stop()
        logger.info("Client stopped (server will stop encoding if this was last client)")


def main():
    parser = argparse.ArgumentParser(description='Test video streaming client detection')
    parser.add_argument('--mode', choices=['server', 'client'], required=True,
                       help='Run as server or client')
    parser.add_argument('--server', type=str, default='127.0.0.1',
                       help='Server IP (for client mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'server':
        run_server()
    else:
        run_client(args.server)


if __name__ == '__main__':
    main()
