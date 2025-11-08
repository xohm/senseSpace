#!/usr/bin/env python3
"""
Example: TouchDesigner Bone-Aligned Orientations with Geometric Reconstruction

For TouchDesigner rigs that interpret Euler angles as bone-aligned (Y-axis along bone),
this example uses geometric reconstruction to prevent shoulder/elbow flipping.

VISUALIZATION:
Press 'O': Toggle orientation axes on/off
Press 'R': Cycle through visualization modes:
           1) BONE-ALIGNED with geometric reconstruction (NEON - default)
              🎯 ALL joints = Bright neon colors (easy to see!)
              💫 EXTRA BRIGHT = Geometric reconstruction (stable, won't flip!)
                 Clavicles (4,11), Shoulders (5,12), Elbows (6,13)
           2) PURE SDK local orientations (accumulated through hierarchy)
              Shows what SDK provides without geometric fixes
           3) BONE-ALIGNED from positions (bright colors) - 100% Y-axis along bone
              Computed from bone directions + temporal smoothing
              ✨ Perfect Y-axis alignment + smooth motion
           4) HIERARCHICAL with branch offsets (for experimentation)
              SDK accumulated + clavicle/hip rotation offsets
              Y-axis points along bones in T-pose

PRINT TO CONSOLE:
Press SPACE: Show bone-aligned orientations as Euler angles (TouchDesigner-ready)
             Uses geometric reconstruction - shoulders/elbows won't flip!
Press 'Q': Show raw SDK quaternions (for debugging)

COMPARISON (for understanding the problem):
Press '1': Show raw SDK quaternions 
Press '2': Show bone-aligned RAW vs FILTERED comparison (temporal filter - doesn't work!)

WHY GEOMETRIC RECONSTRUCTION?
- TouchDesigner expects bone-aligned orientations (Y-axis along bone direction)
- SDK's bone-aligned orientations flip due to geometric ambiguity (roll under-constrained)
- Geometric reconstruction builds stable orientations from bone directions + torso reference
- Shoulders/elbows: 100% geometric (stable), other joints: blend SDK + geometric

ADVANCED CONTROLS:
Press 'B': Cycle blend factor (0.0=pure geometric, 1.0=pure SDK)
           Default 0.0 for shoulders/elbows (prevents flipping)
"""

import argparse
import sys
import os

# Ensure local 'libs' folder is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
libs_path = os.path.join(repo_root, 'libs')
if os.path.isdir(libs_path) and libs_path not in sys.path:
    sys.path.insert(0, libs_path)

# Import from shared library
from senseSpaceLib.senseSpace.vizClient import VisualizationClient
from senseSpaceLib.senseSpace.protocol import Frame
from senseSpaceLib.senseSpace.vizWidget import SkeletonGLWidget
from senseSpaceLib.senseSpace.visualization import (
    compute_bone_aligned_local_orientations,
    compute_stable_bone_orientations,
    quaternion_to_euler,
    calcHyrJointOrientations
)
from senseSpaceLib.senseSpace.orientation_filter import SkeletonOrientationFilter
from senseSpaceLib.senseSpace.geometric_orientation import GeometricOrientationReconstructor

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from OpenGL.GL import (
    glBegin, glEnd, glVertex3f, glColor3f, GL_LINES, glLineWidth, 
    glEnable, glDisable, GL_POLYGON_OFFSET_LINE, glPolygonOffset,
    glPushMatrix, glPopMatrix, glTranslatef, glMultMatrixf
)


class OrientationWidget(SkeletonGLWidget):
    """Minimal widget that prints orientations on SPACE key"""
    
    def __init__(self, parent=None):
        """Initialize with orientation axes visible by default"""
        super().__init__(parent)
        # Enable orientation visualization by default
        self.show_joint_orientation = True
        self.show_orientation = True  # Also set the base class flag
        # Store T-pose reference (captured with 'T' key)
        self.tpose_reference = None
        
        # Quaternion filter for bone-aligned orientations
        self.orientation_filter = SkeletonOrientationFilter(smoothing=0.3, num_joints=34)
        self.filter_enabled = False  # Disabled by default - makes things WORSE!
        
        # Geometric reconstructor for stable bone-aligned orientations
        self.geometric_reconstructor = GeometricOrientationReconstructor(blend_factor=0.0)
        
        # Visualization mode: 0=bone-aligned (neon), 1=SDK local accumulated, 2=raw SDK, 3=hierarchical with offsets
        self.vis_mode = 2  # Default to bone-aligned with neon
        
        # Cache for filtered orientations (computed each frame)
        self.filtered_orientations = {}
        
        # Store previous frame orientations for flip prevention (Mode 2)
        self.previous_raw_orientations = {}
    
    def _quat_multiply(self, q1, q2):
        """Multiply two quaternions: q1 * q2"""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return [
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ]
    
    def _quat_inverse(self, q):
        """Compute quaternion inverse (conjugate for unit quaternions)"""
        x, y, z, w = q
        return [-x, -y, -z, w]
    
    def keyPressEvent(self, event):
        """Handle keyboard input"""
        if event.key() == Qt.Key_Space:
            # TOUCHDESIGNER OUTPUT: Bone-aligned orientations as Euler angles
            if hasattr(self, 'latest_frame') and self.latest_frame and hasattr(self.latest_frame, 'people'):
                print("\n" + "="*80)
                print("TOUCHDESIGNER OUTPUT - Bone-Aligned Orientations (Euler XYZ)")
                print("="*80)
                print("\nBone-aligned = Y-axis points along bone direction")
                print("Using geometric reconstruction for clavicles/shoulders/elbows to prevent flipping")
                print()
                
                for person_idx, person in enumerate(self.latest_frame.people):
                    confidence = person.confidence if hasattr(person, 'confidence') else person.get('confidence', 0)
                    if confidence <= 50:
                        continue
                    
                    print(f"Person {person_idx} (confidence: {confidence:.1f}):")
                    skeleton = person.skeleton if hasattr(person, 'skeleton') else person.get('skeleton')
                    if not skeleton:
                        print("  No skeleton data")
                        continue
                    
                    # Get ALL bone-aligned orientations (computed from bone directions)
                    all_bone_aligned = compute_bone_aligned_local_orientations(skeleton)
                    
                    # Override clavicles/shoulders/elbows with stable geometric reconstruction
                    reconstructed = self.geometric_reconstructor.reconstruct_skeleton_orientations(skeleton)
                    for joint_idx, quat in reconstructed.items():
                        all_bone_aligned[joint_idx] = quat.tolist() if hasattr(quat, 'tolist') else quat
                    
                    # Convert to Euler angles for TouchDesigner
                    for joint_idx in sorted(all_bone_aligned.keys()):
                        quat = all_bone_aligned[joint_idx]
                        euler = quaternion_to_euler(quat, order='XYZ')
                        
                        # Format for TouchDesigner (joint index, euler angles)
                        print(f"  Joint {joint_idx:2d}: rx={euler[0]:7.2f}°, ry={euler[1]:7.2f}°, rz={euler[2]:7.2f}°")
                
                print("\n" + "="*80)
                print("TOUCHDESIGNER INTEGRATION:")
                print("- ALL joints have bone-aligned orientations (Y-axis along bone)")
                print("- Clavicles: Geometric reconstruction with ±90° rotation correction")
                print("- Shoulders/elbows: Geometric reconstruction (won't flip)")
                print("- Other joints: Bone-aligned (computed from positions)")
                print("- Send: joint_index, euler_x, euler_y, euler_z")
                print("="*80 + "\n")
            else:
                print("[WARNING] No skeleton data available")
        
        elif event.key() == Qt.Key_Q:
            # Press 'Q': Show raw SDK quaternions (local_orientation_per_joint)
            if hasattr(self, 'latest_frame') and self.latest_frame and hasattr(self.latest_frame, 'people'):
                print("\n" + "="*80)
                print("RAW ZED SDK LOCAL ORIENTATIONS (Quaternions [x, y, z, w])")
                print("="*80)
                print("Direct output from local_orientation_per_joint")
                print("These are LOCAL orientations (relative to parent joint)")
                print("Only showing persons with confidence > 50")
                print()
                
                for person_idx, person in enumerate(self.latest_frame.people):
                    # Check confidence
                    confidence = person.confidence if hasattr(person, 'confidence') else person.get('confidence', 0)
                    if confidence <= 50:
                        continue
                    
                    print(f"Person {person_idx} (confidence: {confidence:.1f}):")
                    skeleton = person.skeleton if hasattr(person, 'skeleton') else person.get('skeleton')
                    if not skeleton:
                        print("  No skeleton data")
                        continue
                    
                    for i, joint in enumerate(skeleton):
                        ori = joint.ori if hasattr(joint, 'ori') else joint.get('ori')
                        if not ori:
                            print(f"  Joint {i:2d}: No orientation data")
                            continue
                        
                        if hasattr(ori, 'x'):
                            quat = [ori.x, ori.y, ori.z, ori.w]
                        else:
                            quat = [ori["x"], ori["y"], ori["z"], ori["w"]]
                        
                        # Show raw quaternion values
                        print(f"  Joint {i:2d}: [{quat[0]:8.5f}, {quat[1]:8.5f}, {quat[2]:8.5f}, {quat[3]:8.5f}]")
                    print()
                
                print("="*80)
                print("This is the raw SDK data - no filtering, no processing")
                print("="*80 + "\n")
            else:
                print("[WARNING] No skeleton data available")
        
        elif event.key() == Qt.Key_1:
            # Option 1: Show raw SDK quaternions
            if hasattr(self, 'latest_frame') and self.latest_frame and hasattr(self.latest_frame, 'people'):
                print("\n" + "="*80)
                print("RAW SDK LOCAL ORIENTATIONS (Quaternions)")
                print("="*80)
                
                for person_idx, person in enumerate(self.latest_frame.people):
                    print(f"\nPerson {person_idx}:")
                    skeleton = person.skeleton if hasattr(person, 'skeleton') else person.get('skeleton')
                    if not skeleton:
                        print("  No skeleton data")
                        continue
                    
                    for i, joint in enumerate(skeleton):
                        ori = joint.ori if hasattr(joint, 'ori') else joint.get('ori')
                        if not ori:
                            continue
                        
                        if hasattr(ori, 'x'):
                            quat = [ori.x, ori.y, ori.z, ori.w]
                        else:
                            quat = [ori["x"], ori["y"], ori["z"], ori["w"]]
                        
                        print(f"  Joint {i:2d}: quat=[{quat[0]:7.4f}, {quat[1]:7.4f}, {quat[2]:7.4f}, {quat[3]:7.4f}]")
                
                print("="*80 + "\n")
            else:
                print("[WARNING] No skeleton data available")
        
        elif event.key() == Qt.Key_2:
            # Option 2: Show bone-aligned local orientations WITH FILTERING
            if hasattr(self, 'latest_frame') and self.latest_frame and hasattr(self.latest_frame, 'people'):
                print("\n" + "="*80)
                filter_status = "ENABLED" if self.filter_enabled else "DISABLED"
                print(f"OPTION 2: BONE-ALIGNED LOCAL ORIENTATIONS")
                print(f"  Temporal Filter: {filter_status}")
                print("="*80)
                print("(Press 'F' to toggle filtering | Press '+/-' to adjust smoothing)")
                print()
                
                for person_idx, person in enumerate(self.latest_frame.people):
                    print(f"\nPerson {person_idx}:")
                    skeleton = person.skeleton if hasattr(person, 'skeleton') else person.get('skeleton')
                    if not skeleton:
                        print("  No skeleton data")
                        continue
                    
                    # Compute bone-aligned orientations (UNFILTERED)
                    local_orientations_raw = compute_bone_aligned_local_orientations(skeleton)
                    
                    # Compute FILTERED version
                    local_orientations_filtered = self.orientation_filter.filter_skeleton(local_orientations_raw.copy())
                    
                    # Show both for shoulders/elbows
                    problem_joints = [5, 6, 12, 13]  # L/R shoulder, L/R elbow (CORRECTED)
                    
                    for joint_idx in sorted(local_orientations_raw.keys()):
                        quat_raw = local_orientations_raw[joint_idx]
                        euler_raw = quaternion_to_euler(quat_raw, order='XYZ')
                        
                        if joint_idx in problem_joints:
                            quat_filtered = local_orientations_filtered[joint_idx]
                            euler_filtered = quaternion_to_euler(quat_filtered, order='XYZ')
                            
                            print(f"  Joint {joint_idx:2d} (shoulder/elbow):")
                            print(f"    RAW:      [{euler_raw[0]:7.2f}°, {euler_raw[1]:7.2f}°, {euler_raw[2]:7.2f}°]")
                            print(f"    FILTERED: [{euler_filtered[0]:7.2f}°, {euler_filtered[1]:7.2f}°, {euler_filtered[2]:7.2f}°]")
                        else:
                            print(f"  Joint {joint_idx:2d}: Euler = [{euler_raw[0]:7.2f}°, {euler_raw[1]:7.2f}°, {euler_raw[2]:7.2f}°]")
                
                print("="*80 + "\n")
            else:
                print("[WARNING] No skeleton data available")
        
        elif event.key() == Qt.Key_F:
            # Toggle filter on/off
            self.filter_enabled = not self.filter_enabled
            status = "ENABLED" if self.filter_enabled else "DISABLED"
            print(f"\n[INFO] Quaternion filter {status}")
            print(f"       Smoothing factor: {self.orientation_filter.smoothing:.2f}")
            if not self.filter_enabled:
                print("       Resetting filter state...")
                self.orientation_filter.reset()
        
        elif event.key() == Qt.Key_R:
            # Cycle through visualization modes
            self.vis_mode = (self.vis_mode + 1) % 4  # Now 4 modes
            
            mode_names = [
                "BONE-ALIGNED WITH GEOMETRIC FIX (NEON)",
                "PURE SDK LOCAL (ACCUMULATED)",
                "RAW SDK ORIENTATIONS",
                "HIERARCHICAL WITH BRANCH OFFSETS"
            ]
            
            print(f"\n[INFO] Visualization mode: {mode_names[self.vis_mode]}")
            
            if self.vis_mode == 0:
                print(f"       BONE-ALIGNED orientations (Y-axis along bone)")
                print(f"       🎯 ALL JOINTS = NEON colors (bright and visible!)")
                print(f"       💫 EXTRA BRIGHT = Geometric reconstruction (stable, won't flip!)")
                print(f"          Joints 4,5,6,11,12,13 = Clavicles/Shoulders/Elbows")
            elif self.vis_mode == 1:
                print(f"       Pure SDK local orientations (relative to parent)")
                print(f"       Accumulated through skeleton hierarchy")
                print(f"       NO geometric reconstruction applied")
            elif self.vis_mode == 2:
                print(f"       BONE-ALIGNED from positions + temporal smoothing")
                print(f"       Computed from bone vectors → 100% Y-axis alignment")
                print(f"       Smoothed for stable motion")
            else:
                print(f"       Hierarchical SDK orientations WITH branch offsets")
                print(f"       Accumulated through hierarchy + clavicle/hip rotations")
                print(f"       Shows Y-axis pointing along bones in T-pose")
        
        elif event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            # Increase smoothing
            new_smoothing = min(1.0, self.orientation_filter.smoothing + 0.1)
            self.orientation_filter.set_smoothing(new_smoothing)
            print(f"\n[INFO] Smoothing increased to {new_smoothing:.2f}")
        
        elif event.key() == Qt.Key_Minus or event.key() == Qt.Key_Underscore:
            # Decrease smoothing
            new_smoothing = max(0.0, self.orientation_filter.smoothing - 0.1)
            self.orientation_filter.set_smoothing(new_smoothing)
            print(f"\n[INFO] Smoothing decreased to {new_smoothing:.2f}")
        
        elif event.key() == Qt.Key_B:
            # Cycle blend factor for geometric reconstruction
            blend_factors = [0.0, 0.3, 0.5, 0.7, 1.0]
            current = self.geometric_reconstructor.blend_factor
            
            # Find next blend factor
            try:
                idx = blend_factors.index(current)
                next_idx = (idx + 1) % len(blend_factors)
            except ValueError:
                next_idx = 0
            
            new_blend = blend_factors[next_idx]
            self.geometric_reconstructor.blend_factor = new_blend
            
            print(f"\n[INFO] Geometric reconstruction blend factor: {new_blend:.1f}")
            print(f"       0.0 = Pure geometric (stable, no flip)")
            print(f"       1.0 = Pure SDK (may flip)")
            print(f"       Shoulders/elbows use 0.0 regardless (always stable)")
        
        else:
            # Pass other keys to parent
            super().keyPressEvent(event)
    
    def draw_custom(self, frame: Frame):
        """Store latest frame and compute local orientations for visualization"""
        self.latest_frame = frame
        
        # Draw mode text overlay
        self._draw_mode_text()
        
        # Compute local orientations cache for visualization
        if hasattr(frame, 'people') and frame.people:
            self.filtered_orientations = {}
            
            for person_idx, person in enumerate(frame.people):
                skeleton = person.skeleton if hasattr(person, 'skeleton') else person.get('skeleton')
                if not skeleton:
                    continue
                
                # Get ALL local orientations (PURE SDK - no geometric reconstruction)
                local_orientations = {}
                for joint_idx, joint in enumerate(skeleton):
                    ori = joint.ori if hasattr(joint, 'ori') else joint.get('ori')
                    if ori:
                        if hasattr(ori, 'x'):
                            local_orientations[joint_idx] = [ori.x, ori.y, ori.z, ori.w]
                        else:
                            local_orientations[joint_idx] = [ori['x'], ori['y'], ori['z'], ori['w']]
                
                # Store PURE SDK local orientations for visualization
                # (no geometric reconstruction applied)
                self.filtered_orientations[person_idx] = local_orientations
    
    def draw_skeletons(self, frame: Frame):
        """Override to draw skeletons with our custom orientations"""
        if not hasattr(frame, 'people') or not frame.people:
            return
        
        from senseSpaceLib.senseSpace.visualization import draw_skeletons_with_bones
        
        # Filter people by confidence > 50
        high_confidence_people = []
        for person in frame.people:
            confidence = person.confidence if hasattr(person, 'confidence') else person.get('confidence', 0)
            if confidence > 50:
                high_confidence_people.append(person)
        
        if not high_confidence_people:
            return
        
        # Draw skeleton (joints and bones) only for high confidence people
        draw_skeletons_with_bones(
            high_confidence_people, 
            joint_color=(0.2, 0.8, 1.0), 
            bone_color=(0.8, 0.2, 0.2),
            show_orientation=False  # We'll draw custom orientations
        )
        
        # Draw orientation axes based on mode
        if self.show_orientation and hasattr(frame, 'people'):
            for person_idx, person in enumerate(frame.people):
                # Check confidence
                confidence = person.confidence if hasattr(person, 'confidence') else person.get('confidence', 0)
                if confidence <= 50:
                    continue
                
                skeleton = person.skeleton if hasattr(person, 'skeleton') else person.get('skeleton')
                if not skeleton:
                    continue
                
                if self.vis_mode == 0:
                    # Mode 0: BONE-ALIGNED orientations with geometric reconstruction (NEON)
                    self._draw_bone_aligned_orientations(skeleton, axis_length=150.0)
                elif self.vis_mode == 1:
                    # Mode 1: PURE SDK local orientations accumulated through hierarchy
                    if person_idx in self.filtered_orientations:
                        self._draw_filtered_joint_orientations(
                            person,
                            skeleton, 
                            self.filtered_orientations[person_idx], 
                            axis_length=150.0
                        )
                elif self.vis_mode == 2:
                    # Mode 2: BONE-ALIGNED orientations (from positions) WITH TEMPORAL SMOOTHING
                    # Compute orientations directly from bone directions for 100% Y-axis alignment
                    from senseSpaceLib.senseSpace.visualization import compute_bone_aligned_local_orientations
                    
                    # Compute bone-aligned orientations from positions
                    bone_aligned_locals = compute_bone_aligned_local_orientations(skeleton)
                    
                    # Apply temporal smoothing to prevent jitter
                    self._draw_smoothed_bone_aligned_orientations(
                        person, skeleton, bone_aligned_locals, axis_length=150.0
                    )
                else:
                    # Mode 3: HIERARCHICAL with branch offsets
                    self._draw_hierarchical_orientations(person, skeleton, axis_length=100.0)
    
    def _draw_filtered_joint_orientations(self, person, skeleton, local_orientations, axis_length=150.0):
        """Draw RGB axes showing the filtered/reconstructed orientations
        
        Args:
            person: Person object with global_root_orientation
            skeleton: List of joints with local orientations
            local_orientations: Dict of local orientation quaternions
            axis_length: Length of visualization axes
        """
        
        # Define BODY_34 skeleton hierarchy (CORRECTED)
        BODY34_PARENTS = [
            -1,  # 0: PELVIS (root)
            0,   # 1: NAVAL_SPINE
            1,   # 2: CHEST_SPINE
            2,   # 3: NECK
            2,   # 4: LEFT_CLAVICLE
            4,   # 5: LEFT_SHOULDER
            5,   # 6: LEFT_ELBOW
            6,   # 7: LEFT_WRIST
            7,   # 8: LEFT_HAND
            8,   # 9: LEFT_HANDTIP
            8,   # 10: LEFT_THUMB
            2,   # 11: RIGHT_CLAVICLE
            11,  # 12: RIGHT_SHOULDER
            12,  # 13: RIGHT_ELBOW
            13,  # 14: RIGHT_WRIST
            14,  # 15: RIGHT_HAND
            15,  # 16: RIGHT_HANDTIP
            15,  # 17: RIGHT_THUMB
            0,   # 18: LEFT_HIP
            18,  # 19: LEFT_KNEE
            19,  # 20: LEFT_ANKLE
            20,  # 21: LEFT_FOOT
            0,   # 22: RIGHT_HIP
            22,  # 23: RIGHT_KNEE
            23,  # 24: RIGHT_ANKLE
            24,  # 25: RIGHT_FOOT
            3,   # 26: HEAD
            26,  # 27: NOSE
            26,  # 28: LEFT_EYE
            26,  # 29: LEFT_EAR
            26,  # 30: RIGHT_EYE
            26,  # 31: RIGHT_EAR
            20,  # 32: LEFT_HEEL
            24,  # 33: RIGHT_HEEL
        ]
        
        def quat_to_rotation_matrix(q):
            """Convert quaternion [x,y,z,w] to rotation matrix"""
            x, y, z, w = q[0], q[1], q[2], q[3]
            return [
                [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
            ]
        
        def quat_multiply(q1, q2):
            """Multiply two quaternions"""
            x1, y1, z1, w1 = q1
            x2, y2, z2, w2 = q2
            return [
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2,
                w1*w2 - x1*x2 - y1*y2 - z1*z2
            ]
        
        def euler_to_quat(roll, pitch, yaw):
            """Convert Euler angles (degrees) to quaternion [x,y,z,w]"""
            from math import sin, cos, radians
            roll = radians(roll)
            pitch = radians(pitch)
            yaw = radians(yaw)
            
            cy = cos(yaw * 0.5)
            sy = sin(yaw * 0.5)
            cp = cos(pitch * 0.5)
            sp = sin(pitch * 0.5)
            cr = cos(roll * 0.5)
            sr = sin(roll * 0.5)
            
            return [
                sr * cp * cy - cr * sp * sy,  # x
                cr * sp * cy + sr * cp * sy,  # y
                cr * cp * sy - sr * sp * cy,  # z
                cr * cp * cy + sr * sp * sy   # w
            ]
        
        # Bind pose rotations for branching bones (chest->clavicle->shoulder)
        # These make Y-axis point along the bone in T-pose
        bind_pose_rotations = {
            4: euler_to_quat(0, 0, -90),   # LEFT_CLAVICLE: rotate -90° Z
            11: euler_to_quat(0, 0, 90),   # RIGHT_CLAVICLE: rotate +90° Z
            18: euler_to_quat(0, 0, 180),  # LEFT_HIP: rotate 180° Z (point down)
            22: euler_to_quat(0, 0, 180),  # RIGHT_HIP: rotate 180° Z (point down)
        }
        
        # Compute world orientations from local orientations
        world_orientations = {}
        
        # Get global root orientation from person if available
        global_root_quat = None
        if hasattr(person, 'global_root_orientation') and person.global_root_orientation is not None:
            gro = person.global_root_orientation
            if hasattr(gro, 'x'):
                global_root_quat = [gro.x, gro.y, gro.z, gro.w]
            else:
                global_root_quat = [gro['x'], gro['y'], gro['z'], gro['w']]
        
        for joint_idx in sorted(local_orientations.keys()):
            parent_idx = BODY34_PARENTS[joint_idx]
            
            if parent_idx < 0:
                # Root joint (pelvis) - use global_root_orientation if available
                if global_root_quat is not None:
                    world_orientations[joint_idx] = global_root_quat
                else:
                    # Fallback: assume local orientation is world (will be wrong if person is rotated!)
                    world_orientations[joint_idx] = local_orientations[joint_idx]
                    print("[WARNING] No global_root_orientation available - pelvis orientation will be incorrect!")
            elif parent_idx in world_orientations:
                # local_world = parent_world × local
                parent_world = world_orientations[parent_idx]
                local_quat = local_orientations[joint_idx]
                world_quat = quat_multiply(parent_world, local_quat)
                
                # Apply bind pose rotation if this joint branches out
                if joint_idx in bind_pose_rotations:
                    bind_rot = bind_pose_rotations[joint_idx]
                    world_quat = quat_multiply(world_quat, bind_rot)
                
                world_orientations[joint_idx] = world_quat
        
        # Draw axes for each joint
        glLineWidth(2.0)
        
        for joint_idx, quat in world_orientations.items():
            if joint_idx >= len(skeleton):
                continue
            
            joint = skeleton[joint_idx]
            pos = joint.pos if hasattr(joint, 'pos') else joint.get('pos')
            if pos is None:
                continue
            
            # Get position
            if hasattr(pos, 'x'):
                px, py, pz = pos.x, pos.y, pos.z
            else:
                px, py, pz = pos['x'], pos['y'], pos['z']
            
            # Convert quaternion to rotation matrix
            rot_matrix = quat_to_rotation_matrix(quat)
            
            # Extract axis vectors (column vectors of rotation matrix)
            x_axis = [rot_matrix[0][0], rot_matrix[1][0], rot_matrix[2][0]]
            y_axis = [rot_matrix[0][1], rot_matrix[1][1], rot_matrix[2][1]]
            z_axis = [rot_matrix[0][2], rot_matrix[1][2], rot_matrix[2][2]]
            
            # Highlight shoulders and elbows (prone to flipping) only when filtering is enabled
            is_shoulder_elbow = joint_idx in [5, 6, 12, 13]  # L/R shoulder, L/R elbow (CORRECTED)
            
            # Use brighter colors for shoulders/elbows when filter is ON to show what's being filtered
            if is_shoulder_elbow and self.filter_enabled and self.show_filtered:
                # Bright colors = filtered joints
                x_color = (1.0, 0.3, 0.3)  # Bright red
                y_color = (0.3, 1.0, 0.3)  # Bright green
                z_color = (0.3, 0.3, 1.0)  # Bright blue
                length = axis_length * 1.3
            else:
                # Normal colors
                x_color = (1.0, 0.0, 0.0)  # Red
                y_color = (0.0, 1.0, 0.0)  # Green
                z_color = (0.0, 0.0, 1.0)  # Blue
                length = axis_length
            
            # Draw X axis
            glBegin(GL_LINES)
            glColor3f(*x_color)
            glVertex3f(px, py, pz)
            glVertex3f(px + x_axis[0]*length, py + x_axis[1]*length, pz + x_axis[2]*length)
            glEnd()
            
            # Draw Y axis
            glBegin(GL_LINES)
            glColor3f(*y_color)
            glVertex3f(px, py, pz)
            glVertex3f(px + y_axis[0]*length, py + y_axis[1]*length, pz + y_axis[2]*length)
            glEnd()
            
            # Draw Z axis
            glBegin(GL_LINES)
            glColor3f(*z_color)
            glVertex3f(px, py, pz)
            glVertex3f(px + z_axis[0]*length, py + z_axis[1]*length, pz + z_axis[2]*length)
            glEnd()
        
        glLineWidth(1.0)
    
    def _draw_bone_aligned_orientations(self, skeleton, axis_length=150.0):
        """Draw bone-aligned orientations with geometric reconstruction (what goes to TouchDesigner)"""
        
        # Reconstruct ALL joints geometrically (not just problematic ones)
        all_bone_aligned = self.geometric_reconstructor.reconstruct_skeleton_orientations(
            skeleton, 
            reconstruct_all=True
        )
        
        def quat_to_rotation_matrix(q):
            """Convert quaternion [x,y,z,w] to rotation matrix"""
            x, y, z, w = q[0], q[1], q[2], q[3]
            return [
                [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
            ]
        
        # Reduce axis length by 50%
        axis_length = axis_length * 0.5
        
        # Enable polygon offset to make lines visible on top of skeleton
        glEnable(GL_POLYGON_OFFSET_LINE)
        glPolygonOffset(-1.0, -1.0)
        
        glLineWidth(4.0)  # Thicker neon lines
        
        # Joints that use geometric reconstruction (highlighted)
        # ALL joints are now geometrically reconstructed!
        
        for joint_idx, quat in all_bone_aligned.items():
            if joint_idx >= len(skeleton):
                continue
            
            joint = skeleton[joint_idx]
            pos = joint.pos if hasattr(joint, 'pos') else joint.get('pos')
            if pos is None:
                continue
            
            # Get position
            if hasattr(pos, 'x'):
                px, py, pz = pos.x, pos.y, pos.z
            else:
                px, py, pz = pos['x'], pos['y'], pos['z']
            
            # Convert quaternion to rotation matrix
            rot_matrix = quat_to_rotation_matrix(quat)
            
            # Extract axis vectors (column vectors of rotation matrix)
            x_axis = [rot_matrix[0][0], rot_matrix[1][0], rot_matrix[2][0]]
            y_axis = [rot_matrix[0][1], rot_matrix[1][1], rot_matrix[2][1]]
            z_axis = [rot_matrix[0][2], rot_matrix[1][2], rot_matrix[2][2]]
            
            # ALL joints get NEON colors (all geometrically reconstructed!)
            # Bright magenta/green/cyan for visibility
            x_color = (1.0, 0.0, 0.6)  # Bright magenta (X-axis)
            y_color = (0.0, 1.0, 0.0)  # Bright green (Y-axis - bone direction!)
            z_color = (0.0, 0.6, 1.0)  # Bright cyan (Z-axis)
            length = axis_length
            line_width = 4.0  # Thick lines for visibility
            
            glLineWidth(line_width)
            
            # Draw X axis
            glBegin(GL_LINES)
            glColor3f(*x_color)
            glVertex3f(px, py, pz)
            glVertex3f(px + x_axis[0]*length, py + x_axis[1]*length, pz + x_axis[2]*length)
            glEnd()
            
            # Draw Y axis (BONE DIRECTION - most important!)
            glBegin(GL_LINES)
            glColor3f(*y_color)
            glVertex3f(px, py, pz)
            glVertex3f(px + y_axis[0]*length, py + y_axis[1]*length, pz + y_axis[2]*length)
            glEnd()
            
            # Draw Z axis
            glBegin(GL_LINES)
            glColor3f(*z_color)
            glVertex3f(px, py, pz)
            glVertex3f(px + z_axis[0]*length, py + z_axis[1]*length, pz + z_axis[2]*length)
            glEnd()
        
        glLineWidth(1.0)
        glDisable(GL_POLYGON_OFFSET_LINE)
    
    def _draw_smoothed_bone_aligned_orientations(self, person, skeleton, bone_aligned_locals, axis_length=150.0):
        """Draw bone-aligned orientations with temporal smoothing (Mode 2)
        
        Takes bone-aligned LOCAL orientations (computed from positions) and applies
        temporal smoothing, then accumulates through hierarchy for visualization.
        
        Args:
            person: Person object with global_root_orientation
            skeleton: List of joints
            bone_aligned_locals: Dict {joint_idx: local quaternion} from compute_bone_aligned_local_orientations
            axis_length: Length of visualization axes
        """
        from PyQt5.QtGui import QQuaternion, QVector3D, QMatrix4x4
        from senseSpaceLib.senseSpace.protocol import Body34Joint
        
        # Define BODY_34 skeleton hierarchy
        BODY34_PARENTS = [
            -1,  # 0: PELVIS (root)
            0,   # 1: NAVAL_SPINE
            1,   # 2: CHEST_SPINE
            2,   # 3: NECK
            2,   # 4: LEFT_CLAVICLE
            4,   # 5: LEFT_SHOULDER
            5,   # 6: LEFT_ELBOW
            6,   # 7: LEFT_WRIST
            2,   # 8: RIGHT_CLAVICLE
            8,   # 9: RIGHT_SHOULDER
            9,   # 10: RIGHT_ELBOW
            10,  # 11: RIGHT_WRIST
            0,   # 12: LEFT_HIP
            12,  # 13: LEFT_KNEE
            13,  # 14: LEFT_ANKLE
            14,  # 15: LEFT_FOOT
            0,   # 16: RIGHT_HIP
            16,  # 17: RIGHT_KNEE
            17,  # 18: RIGHT_ANKLE
            18,  # 19: RIGHT_FOOT
            3,   # 20: HEAD
            7,   # 21: LEFT_HAND_THUMB_1
            21,  # 22: LEFT_HAND_THUMB_2
            22,  # 23: LEFT_HAND_THUMB_3
            23,  # 24: LEFT_HAND_THUMB_4
            7,   # 25: LEFT_HAND_INDEX_1
            7,   # 26: LEFT_HAND_MIDDLE_1
            7,   # 27: LEFT_HAND_PINKY_1
            7,   # 28: LEFT_HAND_RING_1
            11,  # 29: RIGHT_HAND_THUMB_1
            29,  # 30: RIGHT_HAND_THUMB_2
            30,  # 31: RIGHT_HAND_THUMB_3
            31,  # 32: RIGHT_HAND_THUMB_4
            11,  # 33: RIGHT_HAND_INDEX_1
        ]
        
        # Apply temporal smoothing (SLERP)
        smoothing = 0.5  # 50% blend
        
        def quat_slerp(q1, q2, t):
            """Spherical linear interpolation"""
            import math
            # Ensure q1 and q2 are QQuaternions
            if not isinstance(q1, QQuaternion):
                q1 = QQuaternion(q1[3], q1[0], q1[1], q1[2]) if isinstance(q1, list) else q1
            if not isinstance(q2, QQuaternion):
                q2 = QQuaternion(q2[3], q2[0], q2[1], q2[2]) if isinstance(q2, list) else q2
            
            dot = QQuaternion.dotProduct(q1, q2)
            
            # If dot < 0, negate q2 to take shorter path
            if dot < 0:
                q2 = QQuaternion(-q2.scalar(), -q2.x(), -q2.y(), -q2.z())
                dot = -dot
            
            # Use QQuaternion's built-in slerp
            return QQuaternion.slerp(q1, q2, t)
        
        # Smooth orientations
        smoothed_locals = {}
        for joint_idx, quat in bone_aligned_locals.items():
            # Convert to QQuaternion if needed
            if isinstance(quat, list):
                current_quat = QQuaternion(quat[3], quat[0], quat[1], quat[2])
            elif hasattr(quat, 'scalar'):
                current_quat = quat
            else:
                current_quat = QQuaternion(quat['w'], quat['x'], quat['y'], quat['z'])
            
            # Apply smoothing if we have previous data
            if hasattr(self, 'previous_bone_orientations') and joint_idx in self.previous_bone_orientations:
                prev_quat = self.previous_bone_orientations[joint_idx]
                smoothed_locals[joint_idx] = quat_slerp(prev_quat, current_quat, 1.0 - smoothing)
            else:
                smoothed_locals[joint_idx] = current_quat
        
        # Store for next frame
        if not hasattr(self, 'previous_bone_orientations'):
            self.previous_bone_orientations = {}
        self.previous_bone_orientations = smoothed_locals.copy()
        
        # Get pelvis world orientation
        if hasattr(person, 'global_root_orientation') and person.global_root_orientation:
            gro = person.global_root_orientation
            if hasattr(gro, 'x'):
                pelvis_world = QQuaternion(gro.w, gro.x, gro.y, gro.z).normalized()
            else:
                pelvis_world = QQuaternion(gro['w'], gro['x'], gro['y'], gro['z']).normalized()
        else:
            pelvis_world = QQuaternion()  # Identity fallback
        
        # Accumulate through hierarchy
        world_orientations = {}
        world_orientations[0] = pelvis_world  # PELVIS
        
        for joint_idx in range(1, len(skeleton)):
            parent_idx = BODY34_PARENTS[joint_idx]
            if parent_idx == -1:
                continue
            
            if parent_idx in world_orientations and joint_idx in smoothed_locals:
                parent_world = world_orientations[parent_idx]
                local_quat = smoothed_locals[joint_idx]
                world_orientations[joint_idx] = parent_world * local_quat
        
        # Draw world orientations
        glEnable(GL_POLYGON_OFFSET_LINE)
        glPolygonOffset(-1.0, -1.0)
        glLineWidth(3.0)
        
        for joint_idx, world_quat in world_orientations.items():
            if joint_idx >= len(skeleton):
                continue
            
            joint = skeleton[joint_idx]
            pos = joint.pos if hasattr(joint, 'pos') else joint.get('pos')
            if pos is None:
                continue
            
            # Get position
            if hasattr(pos, 'x'):
                px, py, pz = pos.x, pos.y, pos.z
            else:
                px, py, pz = pos['x'], pos['y'], pos['z']
            
            # Create 4x4 rotation matrix
            m4 = QMatrix4x4()
            m4.rotate(world_quat)
            
            # Bright colors for bone-aligned smoothed mode
            x_color = (1.0, 0.3, 0.3)  # Bright red
            y_color = (0.3, 1.0, 0.3)  # Bright green (Y along bone!)
            z_color = (0.3, 0.3, 1.0)  # Bright blue
            
            glPushMatrix()
            glTranslatef(px, py, pz)
            glMultMatrixf(m4.data())
            
            # Draw X axis
            glBegin(GL_LINES)
            glColor3f(*x_color)
            glVertex3f(0, 0, 0)
            glVertex3f(axis_length, 0, 0)
            glEnd()
            
            # Draw Y axis (bone direction!)
            glBegin(GL_LINES)
            glColor3f(*y_color)
            glVertex3f(0, 0, 0)
            glVertex3f(0, axis_length, 0)
            glEnd()
            
            # Draw Z axis
            glBegin(GL_LINES)
            glColor3f(*z_color)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 0, axis_length)
            glEnd()
            
            glPopMatrix()
        
        glLineWidth(1.0)
        glDisable(GL_POLYGON_OFFSET_LINE)
    
    def _draw_accumulated_stabilized_orientations(self, person, skeleton, stabilized_locals, axis_length=150.0):
        """Draw RGB axes showing stabilized SDK orientations accumulated through hierarchy (Mode 2)
        
        Takes stabilized LOCAL orientations (with flip prevention) and accumulates them
        through the skeleton hierarchy to produce WORLD orientations where Y-axis is bone-aligned.
        
        Args:
            person: Person object with global_root_orientation
            skeleton: List of joints
            stabilized_locals: Dict {joint_idx: local quaternion [x,y,z,w]} - stabilized
            axis_length: Length of visualization axes
        """
        from PyQt5.QtGui import QQuaternion, QVector3D, QMatrix4x4
        from senseSpaceLib.senseSpace.protocol import Body34Joint
        
        # Define BODY_34 skeleton hierarchy
        BODY34_PARENTS = [
            -1,  # 0: PELVIS (root)
            0,   # 1: NAVAL_SPINE
            1,   # 2: CHEST_SPINE
            2,   # 3: NECK
            2,   # 4: LEFT_CLAVICLE
            4,   # 5: LEFT_SHOULDER
            5,   # 6: LEFT_ELBOW
            6,   # 7: LEFT_WRIST
            2,   # 8: RIGHT_CLAVICLE (parent is CHEST_SPINE)
            8,   # 9: RIGHT_SHOULDER
            9,   # 10: RIGHT_ELBOW
            10,  # 11: RIGHT_WRIST
            0,   # 12: LEFT_HIP
            12,  # 13: LEFT_KNEE
            13,  # 14: LEFT_ANKLE
            14,  # 15: LEFT_FOOT
            0,   # 16: RIGHT_HIP
            16,  # 17: RIGHT_KNEE
            17,  # 18: RIGHT_ANKLE
            18,  # 19: RIGHT_FOOT
            3,   # 20: HEAD
            7,   # 21: LEFT_HAND_THUMB_1
            21,  # 22: LEFT_HAND_THUMB_2
            22,  # 23: LEFT_HAND_THUMB_3
            23,  # 24: LEFT_HAND_THUMB_4
            7,   # 25: LEFT_HAND_INDEX_1
            7,   # 26: LEFT_HAND_MIDDLE_1
            7,   # 27: LEFT_HAND_PINKY_1
            7,   # 28: LEFT_HAND_RING_1
            11,  # 29: RIGHT_HAND_THUMB_1
            29,  # 30: RIGHT_HAND_THUMB_2
            30,  # 31: RIGHT_HAND_THUMB_3
            31,  # 32: RIGHT_HAND_THUMB_4
            11,  # 33: RIGHT_HAND_INDEX_1
        ]
        
        # Helper to get stabilized local quaternion as QQuaternion
        def get_stabilized_local(joint_idx):
            if joint_idx in stabilized_locals:
                q = stabilized_locals[joint_idx]
                return QQuaternion(q[3], q[0], q[1], q[2]).normalized()
            else:
                # Fallback to skeleton data if not in stabilized dict
                joint = skeleton[joint_idx]
                ori = joint.ori if hasattr(joint, 'ori') else joint.get('ori')
                if ori:
                    if hasattr(ori, 'x'):
                        return QQuaternion(ori.w, ori.x, ori.y, ori.z).normalized()
                    else:
                        return QQuaternion(ori['w'], ori['x'], ori['y'], ori['z']).normalized()
            return QQuaternion()  # Identity
        
        # Get pelvis world orientation
        if hasattr(person, 'global_root_orientation') and person.global_root_orientation:
            gro = person.global_root_orientation
            if hasattr(gro, 'x'):
                pelvis_world = QQuaternion(gro.w, gro.x, gro.y, gro.z).normalized()
            else:
                pelvis_world = QQuaternion(gro['w'], gro['x'], gro['y'], gro['z']).normalized()
        else:
            pelvis_world = QQuaternion()  # Identity fallback
        
        # Accumulate through hierarchy: world[child] = world[parent] * local[child]
        world_orientations = {}
        world_orientations[0] = pelvis_world  # PELVIS
        
        for joint_idx in range(1, len(skeleton)):
            parent_idx = BODY34_PARENTS[joint_idx]
            if parent_idx == -1:
                continue
            
            if parent_idx in world_orientations:
                parent_world = world_orientations[parent_idx]
                local_quat = get_stabilized_local(joint_idx)
                world_orientations[joint_idx] = parent_world * local_quat
        
        # Draw world orientations
        glEnable(GL_POLYGON_OFFSET_LINE)
        glPolygonOffset(-1.0, -1.0)
        glLineWidth(3.0)
        
        for joint_idx, world_quat in world_orientations.items():
            if joint_idx >= len(skeleton):
                continue
            
            joint = skeleton[joint_idx]
            pos = joint.pos if hasattr(joint, 'pos') else joint.get('pos')
            if pos is None:
                continue
            
            # Get position
            if hasattr(pos, 'x'):
                px, py, pz = pos.x, pos.y, pos.z
            else:
                px, py, pz = pos['x'], pos['y'], pos['z']
            
            # Create 4x4 rotation matrix
            m4 = QMatrix4x4()
            m4.rotate(world_quat)
            
            # Bright colors for stabilized accumulated mode
            x_color = (1.0, 0.3, 0.3)  # Bright red
            y_color = (0.3, 1.0, 0.3)  # Bright green
            z_color = (0.3, 0.3, 1.0)  # Bright blue
            
            glPushMatrix()
            glTranslatef(px, py, pz)
            glMultMatrixf(m4.data())
            
            # Draw X axis
            glBegin(GL_LINES)
            glColor3f(*x_color)
            glVertex3f(0, 0, 0)
            glVertex3f(axis_length, 0, 0)
            glEnd()
            
            # Draw Y axis
            glBegin(GL_LINES)
            glColor3f(*y_color)
            glVertex3f(0, 0, 0)
            glVertex3f(0, axis_length, 0)
            glEnd()
            
            # Draw Z axis
            glBegin(GL_LINES)
            glColor3f(*z_color)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 0, axis_length)
            glEnd()
            
            glPopMatrix()
        
        glLineWidth(1.0)
        glDisable(GL_POLYGON_OFFSET_LINE)
    
    def _draw_hierarchical_orientations(self, person, skeleton, axis_length=100.0):
        """Draw hierarchical accumulated orientations with branch offsets (Mode 3)
        
        Args:
            person: Person object with global_root_orientation
            skeleton: List of joints
            axis_length: Length of visualization axes
        """
        from senseSpaceLib.senseSpace.visualization import calcHyrJointOrientations
        from PyQt5.QtGui import QQuaternion, QVector3D, QMatrix4x4
        
        # Calculate hierarchical orientations with branch offsets
        # Pass person object so it can access global_root_orientation
        world_orientations = calcHyrJointOrientations(person, skeleton)
        
        # Enable polygon offset to make lines render on top
        glEnable(GL_POLYGON_OFFSET_LINE)
        glPolygonOffset(1.0, 5.0)
        
        glLineWidth(5.0)
        
        for joint_key, quat in world_orientations.items():
            # Convert enum to int if needed
            joint_idx = joint_key.value if hasattr(joint_key, 'value') else joint_key
            
            if joint_idx >= len(skeleton):
                continue
            
            joint = skeleton[joint_idx]
            pos = joint.pos if hasattr(joint, 'pos') else joint.get('pos')
            if pos is None:
                continue
            
            # Get position
            if hasattr(pos, 'x'):
                px, py, pz = pos.x, pos.y, pos.z
            else:
                px, py, pz = pos['x'], pos['y'], pos['z']
            
            # Convert quaternion [x,y,z,w] to QQuaternion (scalar, x, y, z)
            q = QQuaternion(quat[3], quat[0], quat[1], quat[2])
            
            # Create 4x4 rotation matrix from quaternion
            m4 = QMatrix4x4()
            m4.rotate(q)
            
            # Standard RGB colors for hierarchy mode
            x_color = (1.0, 0.0, 0.0)  # Red (X-axis)
            y_color = (0.0, 1.0, 0.0)  # Green (Y-axis)
            z_color = (0.0, 0.0, 1.0)  # Blue (Z-axis)

            glPushMatrix()
            glTranslatef(px, py, pz)
            glMultMatrixf(m4.data())

            # Draw X axis
            glBegin(GL_LINES)
            glColor3f(*x_color)
            glVertex3f(0, 0, 0)
            glVertex3f(axis_length, 0, 0)
            glEnd()
            
            # Draw Y axis
            glBegin(GL_LINES)
            glColor3f(*y_color)
            glVertex3f(0, 0, 0)
            glVertex3f(0, axis_length, 0)
            glEnd()
            
            # Draw Z axis
            glBegin(GL_LINES)
            glColor3f(*z_color)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 0, axis_length)
            glEnd()

            glPopMatrix()
        
        glLineWidth(1.0)
        glDisable(GL_POLYGON_OFFSET_LINE)

            
        glLineWidth(1.0)
    
    def _draw_mode_text(self):
        """Draw current visualization mode as text overlay in the 3D view"""
        mode_names = [
            "Mode 0: BONE-ALIGNED (NEON)",
            "Mode 1: SDK ACCUMULATED",
            "Mode 2: BONE-ALIGNED + SMOOTHED",
            "Mode 3: HIERARCHICAL + OFFSETS"
        ]
        
        # Use Qt's renderText for OpenGL rendering
        self.qglColor(QColor(255, 255, 0))  # Bright yellow
        self.renderText(10, 80, mode_names[self.vis_mode])


def main():
    parser = argparse.ArgumentParser(description="SenseSpace Orientation Example")
    parser.add_argument("--server", "-s", default="localhost", help="Server IP address")
    parser.add_argument("--port", "-p", type=int, default=12345, help="Server port")
    parser.add_argument("--rec", "-r", type=str, default=None, help="Path to recording file")
    
    args = parser.parse_args()
    
    # Create and run visualization client
    client = VisualizationClient(
        viewer_class=OrientationWidget,
        server_ip=args.server,
        server_port=args.port,
        playback_file=args.rec,
        window_title="Skeleton Orientations - Press SPACE to print orientations"
    )
    
    success = client.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
