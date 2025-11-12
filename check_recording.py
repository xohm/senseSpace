#!/usr/bin/env python3
"""
Recording Info Utility

Quickly check recording file metadata without loading the entire file.

Usage:
    python check_recording.py recording.ssrec
    python check_recording.py recordings/*.ssrec
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'libs'))

from senseSpaceLib.senseSpace import FramePlayer


def format_duration(seconds):
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def check_recording(filepath):
    """Display recording information"""
    print(f"\n{'='*70}")
    print(f"File: {filepath}")
    print(f"{'='*70}")
    
    info = FramePlayer.get_recording_info(filepath)
    
    if not info:
        print("❌ Failed to read recording file")
        return
    
    # Basic info
    print(f"Format version:  {info['version']}")
    print(f"File size:       {info['file_size_mb']} MB")
    print(f"Compression:     {info['compression']}")
    
    # Duration and frames (if available)
    if 'duration' in info:
        print(f"\n📊 Recording Statistics:")
        print(f"  Duration:      {format_duration(info['duration'])} ({info['duration']:.1f}s)")
        print(f"  Total frames:  {info['total_frames']:,}")
        print(f"  Framerate:     {info['framerate']:.1f} fps")
    
    # Content info
    print(f"\n📦 Content:")
    print(f"  Skeleton:      ✅ Always included")
    print(f"  Point clouds:  {'✅ Included' if info['has_pointcloud'] else '❌ Not included'}")
    print(f"  Video:         {'✅ Included' if info['has_video'] else '❌ Not included'}")
    
    # Video details (if available)
    if info['has_video'] and 'num_cameras' in info:
        print(f"\n🎥 Video Details:")
        print(f"  Cameras:       {info['num_cameras']}")
        print(f"  Video size:    {info.get('video_size_mb', 0)} MB")
        print(f"  Codec:         H.265 (HEVC)")
    
    print(f"{'='*70}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_recording.py <recording.ssrec> [more files...]")
        print("\nExample:")
        print("  python check_recording.py recording.ssrec")
        print("  python check_recording.py recordings/*.ssrec")
        sys.exit(1)
    
    files = sys.argv[1:]
    
    # Expand wildcards if needed
    from glob import glob
    all_files = []
    for pattern in files:
        if '*' in pattern or '?' in pattern:
            all_files.extend(glob(pattern))
        else:
            all_files.append(pattern)
    
    if not all_files:
        print("❌ No files found")
        sys.exit(1)
    
    # Check each file
    for filepath in all_files:
        if not Path(filepath).exists():
            print(f"❌ File not found: {filepath}")
            continue
        
        check_recording(filepath)
    
    # Summary if multiple files
    if len(all_files) > 1:
        print(f"\n✅ Checked {len(all_files)} recording(s)")


if __name__ == '__main__':
    main()
