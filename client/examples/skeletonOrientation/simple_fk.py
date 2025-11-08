#!/usr/bin/env python3
"""
Simple FK Example - Press SPACE to print T-pose delta orientations
"""

import argparse
import sys
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
libs_path = os.path.join(repo_root, 'libs')
if os.path.isdir(libs_path) and libs_path not in sys.path:
    sys.path.insert(0, libs_path)

from senseSpaceLib.senseSpace.vizClient import VisualizationClient
from senseSpaceLib.senseSpace.protocol import Frame
from senseSpaceLib.senseSpace.vizWidget import SkeletonGLWidget
from senseSpaceLib.senseSpace.visualization import draw_fk_skeleton_with_orientations, quaternion_to_euler
from senseSpaceLib.senseSpace import get_tpose_delta_orientations_ext

from PyQt5.QtCore import Qt


class SimpleFKWidget(SkeletonGLWidget):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.best_person = None
        self.best_person_idx = -1
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            if not self.best_person:
                print("\n[WARNING] No person detected\n")
                return
            
            print("\n" + "="*80)
            print("T-POSE DELTA ORIENTATIONS")
            print("="*80)
            
            JOINT_NAMES = {
                0: "PELVIS", 1: "NAVAL_SPINE", 2: "CHEST_SPINE", 3: "NECK",
                4: "LEFT_CLAVICLE", 5: "LEFT_SHOULDER", 6: "LEFT_ELBOW", 7: "LEFT_WRIST",
                11: "RIGHT_CLAVICLE", 12: "RIGHT_SHOULDER", 13: "RIGHT_ELBOW", 14: "RIGHT_WRIST",
                18: "LEFT_HIP", 19: "LEFT_KNEE", 20: "LEFT_ANKLE",
                22: "RIGHT_HIP", 23: "RIGHT_KNEE", 24: "RIGHT_ANKLE", 26: "HEAD"
            }
            
            try:
                delta_orientations = get_tpose_delta_orientations_ext(self.best_person)
                
                for joint_idx in sorted(delta_orientations.keys()):
                    if joint_idx not in JOINT_NAMES:
                        continue
                    
                    delta_euler = quaternion_to_euler(delta_orientations[joint_idx], order='XYZ')
                    print(f"  {joint_idx:2d} {JOINT_NAMES[joint_idx]:20s}: "
                          f"[{delta_euler[0]:7.2f}°, {delta_euler[1]:7.2f}°, {delta_euler[2]:7.2f}°]")
                
                print("="*80 + "\n")
                
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
                import traceback
                traceback.print_exc()
        else:
            super().keyPressEvent(event)
    
    def draw_custom(self, frame: Frame):
        pass
    
    def draw_skeletons(self, frame: Frame):
        if not hasattr(frame, 'people') or not frame.people:
            self.best_person = None
            self.best_person_idx = -1
            return
        
        # Find best person (highest confidence)
        max_confidence = 0
        for idx, person in enumerate(frame.people):
            confidence = person.confidence if hasattr(person, 'confidence') else person.get('confidence', 0)
            if confidence > max_confidence:
                max_confidence = confidence
                self.best_person = person
                self.best_person_idx = idx
        
        if not self.best_person:
            return
        
        # Draw FK skeleton with orientation axes - that's it!
        draw_fk_skeleton_with_orientations(self.best_person, axis_length=75.0, axis_width=15.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", "-s", default="localhost")
    parser.add_argument("--port", "-p", type=int, default=12345)
    parser.add_argument("--rec", "-r", type=str, default=None)
    args = parser.parse_args()
    
    client = VisualizationClient(
        viewer_class=SimpleFKWidget,
        server_ip=args.server,
        server_port=args.port,
        playback_file=args.rec,
        window_title="Simple FK - Press SPACE for T-pose deltas"
    )
    
    success = client.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
