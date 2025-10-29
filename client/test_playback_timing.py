#!/usr/bin/env python3
"""Test playback timing to debug speed issues"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'libs'))

from senseSpaceLib.senseSpace import FramePlayer
import time

player = FramePlayer("./recordings/test.ssrec", loop=False, speed=1.0)

frame_times = []
last_time = None

def on_frame(frame):
    global last_time
    current = time.time()
    if last_time is not None:
        delta = (current - last_time) * 1000
        frame_times.append(delta)
        if len(frame_times) <= 20:
            print(f"Frame {len(frame_times)}: {delta:.1f}ms")
    last_time = current

player.set_frame_callback(on_frame)

if player.load_header():
    print("Starting playback...")
    start = time.time()
    player.start()
    
    while player.is_playing():
        time.sleep(0.1)
    
    duration = time.time() - start
    print(f"\nPlayback finished in {duration:.2f}s")
    if frame_times:
        print(f"Average frame time: {sum(frame_times)/len(frame_times):.1f}ms")
        print(f"Min: {min(frame_times):.1f}ms, Max: {max(frame_times):.1f}ms")
