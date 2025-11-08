from OpenGL.GL import *
from OpenGL.GLU import *
from .enums import Body34Joint, Body18Joint
from PyQt5.QtGui import QQuaternion
from PyQt5.QtGui import QVector3D

# Global variables for floor height stabilization
_floor_height_history = []
_stable_floor_height = None
_floor_height_samples = 30  # Number of samples to keep for moving average

def _person_to_dict(person):
    """Normalize a person entry to a dict with 'skeleton' being a list of joint dicts.
    Accepts either a dataclass `Person` (with .skeleton list of Joint objects) or a dict
    (already serialized). Returns a dict view without modifying the original.
    """
    # If it's already a dict-like object
    if isinstance(person, dict):
        return person

    # Dataclass-style: has attribute 'skeleton'
    if hasattr(person, 'skeleton'):
        sk = []
        for j in person.skeleton:
            # joint may be dataclass with .pos attribute or a dict
            if hasattr(j, 'pos'):
                sk.append({
                    'i': getattr(j, 'i', None),
                    'pos': getattr(j, 'pos'),
                    'ori': getattr(j, 'ori', None) if hasattr(j, 'ori') else None,
                    'conf': getattr(j, 'conf', 0.0)
                })
            else:
                sk.append(j)
        return {'skeleton': sk, 'id': getattr(person, 'id', None), 'confidence': getattr(person, 'confidence', None)}

    # Fallback: try to treat as mapping
    try:
        return dict(person)
    except Exception:
        return {'skeleton': []}


class SkeletonVisualizer3D:
    """
    Utility to visualize a Person (from protocol.py) using OpenGL.
    """
    def __init__(self):
        pass

    def draw_person(self, person, color=(0.2, 0.8, 1.0)):
        """
        Draws a person skeleton using OpenGL.
        :param person: Person object (from protocol.py)
        :param color: RGB tuple for the skeleton color
        """
        glColor3f(*color)
        glPointSize(6.0)
        glBegin(GL_POINTS)
        p = _person_to_dict(person)
        for joint in p.get('skeleton', []):
            pos = joint['pos'] if isinstance(joint, dict) else getattr(joint, 'pos', None)
            if pos is None:
                continue
            glVertex3f(pos["x"], pos["y"], pos["z"])
        glEnd()

    def draw_people(self, people, color=(0.2, 0.8, 1.0)):
        """
        Draws multiple people (list of Person objects)
        """
        for person in people:
            self.draw_person(person, color=color)

    def _normalize_confidence(conf):
        """Map various confidence ranges into 0.0..1.0"""
        try:
            if conf is None:
                return 1.0
            val = float(conf)
            # if value appears to be a percentage e.g. 0..100
            if val > 1.0:
                val = val / 100.0
            # clamp
            return max(0.0, min(1.0, val))
        except Exception:
            return 1.0


    def draw_skeleton_with_bones(self, people_data, joint_color=(0.2, 0.8, 1.0), bone_color=(0.8, 0.2, 0.2), 
                                 show_orientation=False, orientation_length=100.0):
        """
        Draws skeleton with bone connections for BODY_34 model.
        :param people_data: List of people data (serialized format)
        :param joint_color: RGB tuple for joint points
        :param bone_color: RGB tuple for bone lines
        :param show_orientation: If True, draw RGB coordinate axes at each joint showing orientation (default False)
        :param orientation_length: Length of orientation axes in mm (default 100mm)
        """

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        for p in people_data:
            p = _person_to_dict(p)
            skeleton = p.get("skeleton", [])
            confidence = p.get("confidence", 1.0)
            alpha = type(self)._normalize_confidence(confidence)
            
            # Draw skeleton bones (lines between joints) for BODY_34
            glColor3f(*bone_color)
            glColor4f(bone_color[0], bone_color[1], bone_color[2], alpha)

            glBegin(GL_LINES)
            
            # BODY_34 bone connections using enum values for clarity
            J = Body34Joint  # Shorthand for readability
            bones = [
                # Spine
                (J.PELVIS.value, J.NAVAL_SPINE.value),
                (J.NAVAL_SPINE.value, J.CHEST_SPINE.value),
                (J.CHEST_SPINE.value, J.NECK.value),
                (J.NECK.value, J.HEAD.value),
                
                # Left arm
                (J.NECK.value, J.LEFT_CLAVICLE.value),
                (J.LEFT_CLAVICLE.value, J.LEFT_SHOULDER.value),
                (J.LEFT_SHOULDER.value, J.LEFT_ELBOW.value),
                (J.LEFT_ELBOW.value, J.LEFT_WRIST.value),
                (J.LEFT_WRIST.value, J.LEFT_HAND.value),
                (J.LEFT_HAND.value, J.LEFT_HANDTIP.value),
                (J.LEFT_HAND.value, J.LEFT_THUMB.value),
                
                # Right arm  
                (J.NECK.value, J.RIGHT_CLAVICLE.value),
                (J.RIGHT_CLAVICLE.value, J.RIGHT_SHOULDER.value),
                (J.RIGHT_SHOULDER.value, J.RIGHT_ELBOW.value),
                (J.RIGHT_ELBOW.value, J.RIGHT_WRIST.value),
                (J.RIGHT_WRIST.value, J.RIGHT_HAND.value),
                (J.RIGHT_HAND.value, J.RIGHT_HANDTIP.value),
                (J.RIGHT_HAND.value, J.RIGHT_THUMB.value),
                
                # Left leg
                (J.PELVIS.value, J.LEFT_HIP.value),
                (J.LEFT_HIP.value, J.LEFT_KNEE.value),
                (J.LEFT_KNEE.value, J.LEFT_ANKLE.value),
                (J.LEFT_ANKLE.value, J.LEFT_FOOT.value),
                (J.LEFT_ANKLE.value, J.LEFT_HEEL.value),
                
                # Right leg
                (J.PELVIS.value, J.RIGHT_HIP.value),
                (J.RIGHT_HIP.value, J.RIGHT_KNEE.value),
                (J.RIGHT_KNEE.value, J.RIGHT_ANKLE.value),
                (J.RIGHT_ANKLE.value, J.RIGHT_FOOT.value),
                (J.RIGHT_ANKLE.value, J.RIGHT_HEEL.value),
                
                # Face
                (J.HEAD.value, J.NOSE.value),
                (J.HEAD.value, J.LEFT_EYE.value),
                (J.HEAD.value, J.RIGHT_EYE.value),
                (J.LEFT_EYE.value, J.LEFT_EAR.value),
                (J.RIGHT_EYE.value, J.RIGHT_EAR.value),
            ]
            
            for bone in bones:
                if bone[0] < len(skeleton) and bone[1] < len(skeleton):
                    j1 = skeleton[bone[0]]["pos"]
                    j2 = skeleton[bone[1]]["pos"]
                    # Handle both Position object and dict
                    if hasattr(j1, 'x'):
                        glVertex3f(j1.x, j1.y, j1.z)
                    else:
                        glVertex3f(j1["x"], j1["y"], j1["z"])
                    if hasattr(j2, 'x'):
                        glVertex3f(j2.x, j2.y, j2.z)
                    else:
                        glVertex3f(j2["x"], j2["y"], j2["z"])
            
            glEnd()
            
            # Draw joint points on top of bones
            glColor3f(*joint_color)
            glPointSize(8.0)
            glBegin(GL_POINTS)
            for j in skeleton:
                pos = j['pos'] if isinstance(j, dict) else getattr(j, 'pos', None)
                if pos is None:
                    continue
                # Handle both Position object and dict
                if hasattr(pos, 'x'):
                    glVertex3f(pos.x, pos.y, pos.z)
                else:
                    glVertex3f(pos["x"], pos["y"], pos["z"])
            glEnd()
            
            # Draw orientation axes if requested
            if show_orientation:
                self._draw_joint_orientations(skeleton, orientation_length)

        glDisable(GL_BLEND)
    
    def _draw_joint_orientations(self, skeleton, axis_length=100.0):
        """
        Draw RGB coordinate axes showing bone-aligned orientation of each joint.
        
        Computes a bone-aligned coordinate frame from the SDK's local orientations:
        - Y-axis = bone direction (parent → child) from geometry
        - X and Z axes = derived from SDK orientation's twist around the bone
        
        This gives you anatomically meaningful frames where Y points along bones,
        while preserving the rotational information from the SDK.
        
        :param skeleton: List of joint data (dict or objects with pos and ori)
        :param axis_length: Length of each axis in mm
        """
        import math
        
        # Define BODY_34 skeleton hierarchy (parent index for each joint)
        BODY34_PARENTS = [
            -1,  # 0: PELVIS (root, no parent)
            0,   # 1: SPINE_NAVAL
            1,   # 2: SPINE_CHEST
            2,   # 3: NECK
            3,   # 4: LEFT_CLAVICLE
            4,   # 5: LEFT_SHOULDER
            5,   # 6: LEFT_ELBOW
            6,   # 7: LEFT_WRIST
            7,   # 8: LEFT_HAND
            8,   # 9: LEFT_HANDTIP
            8,   # 10: LEFT_THUMB
            3,   # 11: RIGHT_CLAVICLE
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
        
        def normalize(v):
            """Normalize a 3D vector"""
            length = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
            if length < 0.0001:
                return [0, 1, 0]
            return [v[0]/length, v[1]/length, v[2]/length]
        
        def cross(a, b):
            """Cross product of two 3D vectors"""
            return [
                a[1]*b[2] - a[2]*b[1],
                a[2]*b[0] - a[0]*b[2],
                a[0]*b[1] - a[1]*b[0]
            ]
        
        def dot(a, b):
            """Dot product of two 3D vectors"""
            return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
        
        def normalize_quat(q):
            """Normalize quaternion and ensure positive w for consistency"""
            x, y, z, w = q
            length = math.sqrt(x*x + y*y + z*z + w*w)
            if length < 0.0001:
                return [0, 0, 0, 1]
            
            x, y, z, w = x/length, y/length, z/length, w/length
            
            # Ensure w is positive (avoids double-cover ambiguity)
            if w < 0:
                return [-x, -y, -z, -w]
            return [x, y, z, w]
        
        def quat_multiply(q1, q2):
            """Multiply two quaternions [x, y, z, w]"""
            x1, y1, z1, w1 = q1
            x2, y2, z2, w2 = q2
            
            return [
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2,
                w1*w2 - x1*x2 - y1*y2 - z1*z2
            ]
        
        def quat_to_rotation_matrix(q):
            """Convert quaternion [x, y, z, w] to 3x3 rotation matrix"""
            x, y, z, w = q[0], q[1], q[2], q[3]
            
            r00 = 1 - 2*(y*y + z*z)
            r01 = 2*(x*y - w*z)
            r02 = 2*(x*z + w*y)
            
            r10 = 2*(x*y + w*z)
            r11 = 1 - 2*(x*x + z*z)
            r12 = 2*(y*z - w*x)
            
            r20 = 2*(x*z - w*y)
            r21 = 2*(y*z + w*x)
            r22 = 1 - 2*(x*x + y*y)
            
            return [
                [r00, r01, r02],
                [r10, r11, r12],
                [r20, r21, r22]
            ]
        
        def rotation_from_vector_to_vector(from_vec, to_vec):
            """
            Compute quaternion that rotates from_vec to align with to_vec.
            Both vectors should be normalized.
            Returns quaternion [x, y, z, w]
            """
            # Normalize inputs
            from_len = math.sqrt(from_vec[0]**2 + from_vec[1]**2 + from_vec[2]**2)
            to_len = math.sqrt(to_vec[0]**2 + to_vec[1]**2 + to_vec[2]**2)
            
            if from_len < 0.0001 or to_len < 0.0001:
                return [0, 0, 0, 1]  # Identity
            
            from_norm = [from_vec[0]/from_len, from_vec[1]/from_len, from_vec[2]/from_len]
            to_norm = [to_vec[0]/to_len, to_vec[1]/to_len, to_vec[2]/to_len]
            
            # Dot product
            dot = from_norm[0]*to_norm[0] + from_norm[1]*to_norm[1] + from_norm[2]*to_norm[2]
            
            # Check if vectors are already aligned
            if dot > 0.9999:
                return [0, 0, 0, 1]  # Identity
            
            # Check if vectors are opposite (180° rotation)
            if dot < -0.9999:
                # Find an orthogonal axis
                # Try X axis first
                axis = [1, 0, 0]
                test = abs(from_norm[0])
                if test > 0.9:  # from_vec is too aligned with X
                    axis = [0, 1, 0]  # Use Y instead
                
                # Cross product to get perpendicular axis
                axis = [
                    axis[1]*from_norm[2] - axis[2]*from_norm[1],
                    axis[2]*from_norm[0] - axis[0]*from_norm[2],
                    axis[0]*from_norm[1] - axis[1]*from_norm[0]
                ]
                axis_len = math.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)
                if axis_len > 0.0001:
                    axis = [axis[0]/axis_len, axis[1]/axis_len, axis[2]/axis_len]
                return [axis[0], axis[1], axis[2], 0]  # 180° rotation
            
            # Normal case: compute rotation axis and angle
            # Axis = from × to (cross product)
            axis = [
                from_norm[1]*to_norm[2] - from_norm[2]*to_norm[1],
                from_norm[2]*to_norm[0] - from_norm[0]*to_norm[2],
                from_norm[0]*to_norm[1] - from_norm[1]*to_norm[0]
            ]
            
            # Quaternion: [axis * sin(angle/2), cos(angle/2)]
            # Use half-angle formula: w = sqrt((1 + dot) / 2)
            w = math.sqrt((1.0 + dot) / 2.0)
            # xyz = axis / (2 * w)
            if w > 0.0001:
                xyz_scale = 0.5 / w
                quat = [axis[0]*xyz_scale, axis[1]*xyz_scale, axis[2]*xyz_scale, w]
            else:
                # Fallback for very small w
                quat = [axis[0], axis[1], axis[2], 0]
            
            return normalize_quat(quat)
        
        def rotation_matrix_to_quat(m):
            """Convert 3x3 rotation matrix to quaternion [x, y, z, w]"""
            # m[row][col] format
            trace = m[0][0] + m[1][1] + m[2][2]
            
            if trace > 0:
                s = 0.5 / math.sqrt(trace + 1.0)
                w = 0.25 / s
                x = (m[2][1] - m[1][2]) * s
                y = (m[0][2] - m[2][0]) * s
                z = (m[1][0] - m[0][1]) * s
            elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
                s = 2.0 * math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2])
                w = (m[2][1] - m[1][2]) / s
                x = 0.25 * s
                y = (m[0][1] + m[1][0]) / s
                z = (m[0][2] + m[2][0]) / s
            elif m[1][1] > m[2][2]:
                s = 2.0 * math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2])
                w = (m[0][2] - m[2][0]) / s
                x = (m[0][1] + m[1][0]) / s
                y = 0.25 * s
                z = (m[1][2] + m[2][1]) / s
            else:
                s = 2.0 * math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1])
                w = (m[1][0] - m[0][1]) / s
                x = (m[0][2] + m[2][0]) / s
                y = (m[1][2] + m[2][1]) / s
                z = 0.25 * s
            
            return normalize_quat([x, y, z, w])
        
        def quat_slerp(q1, q2, t):
            """Spherical linear interpolation between two quaternions.
            t=0 returns q1, t=1 returns q2, t=0.5 is halfway between."""
            # Normalize inputs
            q1 = normalize_quat(q1)
            q2 = normalize_quat(q2)
            
            # Compute dot product
            dot = q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]
            
            # If dot is negative, negate q2 to take shorter path
            if dot < 0.0:
                q2 = [-q2[0], -q2[1], -q2[2], -q2[3]]
                dot = -dot
            
            # Clamp dot to avoid numerical issues
            dot = max(-1.0, min(1.0, dot))
            
            # If quaternions are very close, use linear interpolation
            if dot > 0.9995:
                result = [
                    q1[0] + t * (q2[0] - q1[0]),
                    q1[1] + t * (q2[1] - q1[1]),
                    q1[2] + t * (q2[2] - q1[2]),
                    q1[3] + t * (q2[3] - q1[3])
                ]
                return normalize_quat(result)
            
            # Calculate interpolation angle
            theta = math.acos(dot)
            sin_theta = math.sin(theta)
            
            # Calculate interpolation weights
            w1 = math.sin((1.0 - t) * theta) / sin_theta
            w2 = math.sin(t * theta) / sin_theta
            
            # Interpolate
            result = [
                w1 * q1[0] + w2 * q2[0],
                w1 * q1[1] + w2 * q2[1],
                w1 * q1[2] + w2 * q2[2],
                w1 * q1[3] + w2 * q2[3]
            ]
            
            return normalize_quat(result)
        
        # Get orientations from server and chain them to get world orientations
        # Server sends local_orientation_per_joint (parent-relative)
        # We must chain them: global(child) = global(parent) × local(child)
        
        world_orientations = {}
        positions = {}
        
        for i, joint in enumerate(skeleton):
            # Get position
            pos = joint['pos'] if isinstance(joint, dict) else getattr(joint, 'pos', None)
            if pos is not None:
                if hasattr(pos, 'x'):
                    positions[i] = [pos.x, pos.y, pos.z]
                else:
                    positions[i] = [pos["x"], pos["y"], pos["z"]]
            
            # Get LOCAL orientation from server
            ori = joint['ori'] if isinstance(joint, dict) else getattr(joint, 'ori', None)
            if ori is None:
                continue
            
            # Extract quaternion [x, y, z, w]
            if hasattr(ori, 'x'):
                local_quat = [ori.x, ori.y, ori.z, ori.w]
            else:
                local_quat = [ori["x"], ori["y"], ori["z"], ori["w"]]
            
            # Normalize
            local_quat = normalize_quat(local_quat)
            
            parent_idx = BODY34_PARENTS[i]
            
            # Chain quaternions to get global orientation
            if parent_idx < 0:
                # Root - local IS global
                world_orientations[i] = local_quat
            elif parent_idx in world_orientations:
                # Child - multiply parent's global × this joint's local
                parent_global = world_orientations[parent_idx]
                child_global = quat_multiply(parent_global, local_quat)
                world_orientations[i] = normalize_quat(child_global)
        
        # Draw orientations by applying fixed rotation offsets per joint type
        # First, let's compute bone directions to see what rotation we actually need
        glDisable(GL_DEPTH_TEST)
        glLineWidth(2.0)
                
        # For bone-aligned visualization, we need to:
        # 1. Calculate current bone direction (child - parent position)
        # 2. Extract SDK's Z-axis (twist direction) 
        # 3. Build new orientation: Y = bone_dir, Z = SDK_Z projected perpendicular to bone_dir
        # 4. Use temporal smoothing on ALL joints to prevent flipping
        
        # Initialize temporal smoothing storage
        if not hasattr(self, '_prev_orientations'):
            self._prev_orientations = {}
        
        # Joints to align with their bones
        bone_aligned_joints = {5, 6, 7, 8, 9, 10, 4,  # Right arm
                               11, 12, 13, 14, 15, 16, 17,  # Left arm
                               1, 2, 3,  # Spine/neck
                               18, 19, 20, 21,  # Left leg
                               22, 23, 24, 25}  # Right leg
        
        for i in range(len(skeleton)):
            if i not in world_orientations or i not in positions:
                continue
            
            px, py, pz = positions[i]
            parent_idx = BODY34_PARENTS[i]
            
            # Get SDK world orientation
            world_quat = world_orientations[i]
            
            # Apply bone alignment for specified joints
            if i in bone_aligned_joints and parent_idx >= 0 and parent_idx in positions:
                # Calculate actual bone direction from current positions
                ppx, ppy, ppz = positions[parent_idx]
                bone_vec = [px - ppx, py - ppy, pz - ppz]
                bone_len = math.sqrt(bone_vec[0]**2 + bone_vec[1]**2 + bone_vec[2]**2)
                
                if bone_len > 0.001:  # Valid bone
                    # Normalize bone direction - this will be our new Y-axis
                    new_y = [bone_vec[0]/bone_len, bone_vec[1]/bone_len, bone_vec[2]/bone_len]
                    
                    # Use SDK Z-axis as reference (contains twist info)
                    rot_matrix = quat_to_rotation_matrix(world_quat)
                    sdk_z = [rot_matrix[0][2], rot_matrix[1][2], rot_matrix[2][2]]
                    
                    # Project SDK Z onto plane perpendicular to bone
                    dot_zy = sdk_z[0]*new_y[0] + sdk_z[1]*new_y[1] + sdk_z[2]*new_y[2]
                    sdk_z_perp = [
                        sdk_z[0] - dot_zy*new_y[0],
                        sdk_z[1] - dot_zy*new_y[1],
                        sdk_z[2] - dot_zy*new_y[2]
                    ]
                    sdk_z_len = math.sqrt(sdk_z_perp[0]**2 + sdk_z_perp[1]**2 + sdk_z_perp[2]**2)
                    
                    # Build reference Z-axis
                    if sdk_z_len > 0.1:
                        new_z = [sdk_z_perp[0]/sdk_z_len, sdk_z_perp[1]/sdk_z_len, sdk_z_perp[2]/sdk_z_len]
                    else:
                        # SDK Z parallel to bone - use fallback
                        if abs(new_y[1]) < 0.9:
                            ref = [0, 1, 0]
                        else:
                            ref = [1, 0, 0]
                        
                        new_z = [
                            ref[1]*new_y[2] - ref[2]*new_y[1],
                            ref[2]*new_y[0] - ref[0]*new_y[2],
                            ref[0]*new_y[1] - ref[1]*new_y[0]
                        ]
                        z_len = math.sqrt(new_z[0]**2 + new_z[1]**2 + new_z[2]**2)
                        if z_len > 0.001:
                            new_z = [new_z[0]/z_len, new_z[1]/z_len, new_z[2]/z_len]
                    
                    # TEMPORAL SMOOTHING: Check if current Z flipped relative to previous frame
                    if i in self._prev_orientations:
                        prev_z = self._prev_orientations[i]
                        
                        # Dot product tells us if vectors point in same direction
                        dot_z = new_z[0]*prev_z[0] + new_z[1]*prev_z[1] + new_z[2]*prev_z[2]
                        
                        # If dot < 0, vectors point opposite directions (180° flip)
                        if dot_z < 0:
                            # Negate to maintain continuity with previous frame
                            new_z = [-new_z[0], -new_z[1], -new_z[2]]
                    
                    # Store current Z for next frame comparison
                    self._prev_orientations[i] = new_z
                    
                    # New X = Y × Z
                    new_x = [
                        new_y[1]*new_z[2] - new_y[2]*new_z[1],
                        new_y[2]*new_z[0] - new_y[0]*new_z[2],
                        new_y[0]*new_z[1] - new_y[1]*new_z[0]
                    ]
                    
                    # Build rotation matrix
                    aligned_matrix = [
                        [new_x[0], new_y[0], new_z[0]],
                        [new_x[1], new_y[1], new_z[1]],
                        [new_x[2], new_y[2], new_z[2]]
                    ]
                    
                    world_quat = rotation_matrix_to_quat(aligned_matrix)
            
            # Convert to rotation matrix
            rot_matrix = quat_to_rotation_matrix(world_quat)
            x_axis = [rot_matrix[0][0], rot_matrix[1][0], rot_matrix[2][0]]
            y_axis = [rot_matrix[0][1], rot_matrix[1][1], rot_matrix[2][1]]
            z_axis = [rot_matrix[0][2], rot_matrix[1][2], rot_matrix[2][2]]
            
            # Draw at parent position for bones, joint position for root
            if parent_idx >= 0 and parent_idx in positions:
                ppx, ppy, ppz = positions[parent_idx]
                draw_x, draw_y, draw_z = ppx, ppy, ppz
            else:
                draw_x, draw_y, draw_z = px, py, pz
            
            # Scale axes
            x_vec = [x_axis[0]*axis_length, x_axis[1]*axis_length, x_axis[2]*axis_length]
            y_vec = [y_axis[0]*axis_length, y_axis[1]*axis_length, y_axis[2]*axis_length]
            z_vec = [z_axis[0]*axis_length, z_axis[1]*axis_length, z_axis[2]*axis_length]
            
            # Draw axes at the drawing position (parent joint for bones, joint itself for root)
            glBegin(GL_LINES)
            
            # X-axis (red)
            glColor3f(1.0, 0.0, 0.0)
            glVertex3f(draw_x, draw_y, draw_z)
            glVertex3f(draw_x + x_vec[0], draw_y + x_vec[1], draw_z + x_vec[2])
            
            # Y-axis (green) - bone direction
            glColor3f(0.0, 1.0, 0.0)
            glVertex3f(draw_x, draw_y, draw_z)
            glVertex3f(draw_x + y_vec[0], draw_y + y_vec[1], draw_z + y_vec[2])
            
            # Z-axis (blue)
            glColor3f(0.0, 0.0, 1.0)
            glVertex3f(draw_x, draw_y, draw_z)
            glVertex3f(draw_x + z_vec[0], draw_y + z_vec[1], draw_z + z_vec[2])
            
            glEnd()
        
        # Re-enable depth test
        glEnable(GL_DEPTH_TEST)
        glLineWidth(1.0)


def compute_bone_aligned_local_orientations(skeleton):
    """
    Compute bone-aligned local orientations for all joints in a skeleton.
    
    Returns local orientations (as quaternions) where:
    - Y-axis points along the bone direction (parent → child)
    - X and Z axes preserve the SDK's twist/rotation information
    - Orientations are LOCAL (relative to parent), ready for rigging
    
    :param skeleton: List of joint data (dict or objects with pos and ori)
    :return: Dictionary {joint_index: [x, y, z, w]} of local quaternions
    """
    import math
    
    # Define BODY_34 skeleton hierarchy
    BODY34_PARENTS = [
        -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 3, 11, 12, 13, 14, 15, 15,
        0, 18, 19, 20, 0, 22, 23, 24, 3, 26, 26, 26, 26, 26, 20, 24
    ]
    
    def normalize(v):
        length = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
        if length < 0.0001:
            return [0, 1, 0]
        return [v[0]/length, v[1]/length, v[2]/length]
    
    def cross(a, b):
        return [
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]
        ]
    
    def dot(a, b):
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
    
    def normalize_quat(q):
        """Normalize quaternion and ensure positive w for consistency"""
        x, y, z, w = q
        length = math.sqrt(x*x + y*y + z*z + w*w)
        if length < 0.0001:
            return [0, 0, 0, 1]
        
        x, y, z, w = x/length, y/length, z/length, w/length
        
        # Ensure w is positive (avoids double-cover ambiguity)
        if w < 0:
            return [-x, -y, -z, -w]
        return [x, y, z, w]
    
    def quat_multiply(q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return [
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ]
    
    def quat_inverse(q):
        """Compute quaternion inverse (conjugate for unit quaternions)"""
        x, y, z, w = q
        return [-x, -y, -z, w]
    
    def quat_to_rotation_matrix(q):
        x, y, z, w = q[0], q[1], q[2], q[3]
        r00 = 1 - 2*(y*y + z*z)
        r01 = 2*(x*y - w*z)
        r02 = 2*(x*z + w*y)
        r10 = 2*(x*y + w*z)
        r11 = 1 - 2*(x*x + z*z)
        r12 = 2*(y*z - w*x)
        r20 = 2*(x*z - w*y)
        r21 = 2*(y*z + w*x)
        r22 = 1 - 2*(x*x + y*y)
        return [
            [r00, r01, r02],
            [r10, r11, r12],
            [r20, r21, r22]
        ]
    
    def rotation_matrix_to_quat(m):
        """Convert 3x3 rotation matrix to quaternion [x, y, z, w]"""
        trace = m[0][0] + m[1][1] + m[2][2]
        
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (m[2][1] - m[1][2]) * s
            y = (m[0][2] - m[2][0]) * s
            z = (m[1][0] - m[0][1]) * s
        elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
            s = 2.0 * math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2])
            w = (m[2][1] - m[1][2]) / s
            x = 0.25 * s
            y = (m[0][1] + m[1][0]) / s
            z = (m[0][2] + m[2][0]) / s
        elif m[1][1] > m[2][2]:
            s = 2.0 * math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2])
            w = (m[0][2] - m[2][0]) / s
            x = (m[0][1] + m[1][0]) / s
            y = 0.25 * s
            z = (m[1][2] + m[2][1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1])
            w = (m[1][0] - m[0][1]) / s
            x = (m[0][2] + m[2][0]) / s
            y = (m[1][2] + m[2][1]) / s
            z = 0.25 * s
        
        return [x, y, z, w]
    
    # Pass 1: Get SDK world orientations and positions
    sdk_world_orientations = {}
    positions = {}
    
    for i, joint in enumerate(skeleton):
        # Get position
        pos = joint['pos'] if isinstance(joint, dict) else getattr(joint, 'pos', None)
        if pos is not None:
            if hasattr(pos, 'x'):
                positions[i] = [pos.x, pos.y, pos.z]
            else:
                positions[i] = [pos["x"], pos["y"], pos["z"]]
        
        # Get SDK orientation and accumulate
        ori = joint['ori'] if isinstance(joint, dict) else getattr(joint, 'ori', None)
        if ori is None:
            continue
        
        if hasattr(ori, 'x'):
            local_quat = [ori.x, ori.y, ori.z, ori.w]
        else:
            local_quat = [ori["x"], ori["y"], ori["z"], ori["w"]]
        
        # Normalize input quaternion
        local_quat = normalize_quat(local_quat)
        
        parent_idx = BODY34_PARENTS[i]
        
        if parent_idx < 0:
            sdk_world_orientations[i] = local_quat
        elif parent_idx in sdk_world_orientations:
            parent_world = sdk_world_orientations[parent_idx]
            child_world = quat_multiply(parent_world, local_quat)
            sdk_world_orientations[i] = normalize_quat(child_world)
    
    # Pass 2: Compute bone-aligned world orientations
    bone_aligned_world_orientations = {}
    
    for i in range(len(skeleton)):
        if i not in sdk_world_orientations or i not in positions:
            continue
        
        px, py, pz = positions[i]
        parent_idx = BODY34_PARENTS[i]
        
        # Get SDK orientation
        sdk_quat = sdk_world_orientations[i]
        rot_matrix = quat_to_rotation_matrix(sdk_quat)
        sdk_x = [rot_matrix[0][0], rot_matrix[1][0], rot_matrix[2][0]]
        sdk_y = [rot_matrix[0][1], rot_matrix[1][1], rot_matrix[2][1]]
        sdk_z = [rot_matrix[0][2], rot_matrix[1][2], rot_matrix[2][2]]
        
        # Compute bone-aligned frame
        if parent_idx >= 0 and parent_idx in positions:
            ppx, ppy, ppz = positions[parent_idx]
            bone_vec = [px - ppx, py - ppy, pz - ppz]
            bone_len = math.sqrt(bone_vec[0]**2 + bone_vec[1]**2 + bone_vec[2]**2)
            
            if bone_len > 1.0:
                # Y-axis = bone direction
                y_axis = normalize(bone_vec)
                
                # Create stable base frame using world reference
                world_up = [0, 1, 0]
                world_forward = [0, 0, 1]
                world_right = [1, 0, 0]
                
                # Pick world axis most perpendicular to bone
                dot_up = abs(dot(y_axis, world_up))
                dot_forward = abs(dot(y_axis, world_forward))
                dot_right = abs(dot(y_axis, world_right))
                
                if dot_up < dot_forward and dot_up < dot_right:
                    reference = world_up
                elif dot_forward < dot_right:
                    reference = world_forward
                else:
                    reference = world_right
                
                # Create base perpendicular vectors
                base_x = normalize(cross(y_axis, reference))
                base_z = normalize(cross(base_x, y_axis))
                
                # Measure twist from SDK orientation
                sdk_x_perp = [sdk_x[0] - dot(sdk_x, y_axis) * y_axis[0],
                              sdk_x[1] - dot(sdk_x, y_axis) * y_axis[1],
                              sdk_x[2] - dot(sdk_x, y_axis) * y_axis[2]]
                
                sdk_x_perp_len = math.sqrt(sdk_x_perp[0]**2 + sdk_x_perp[1]**2 + sdk_x_perp[2]**2)
                
                if sdk_x_perp_len > 0.1:
                    # Compute twist angle
                    sdk_x_norm = normalize(sdk_x_perp)
                    
                    cos_twist = dot(base_x, sdk_x_norm)
                    sin_twist = dot(base_z, sdk_x_norm)
                    
                    # Apply twist to base frame
                    x_axis = [cos_twist * base_x[0] + sin_twist * base_z[0],
                             cos_twist * base_x[1] + sin_twist * base_z[1],
                             cos_twist * base_x[2] + sin_twist * base_z[2]]
                    z_axis = [-sin_twist * base_x[0] + cos_twist * base_z[0],
                             -sin_twist * base_x[1] + cos_twist * base_z[1],
                             -sin_twist * base_x[2] + cos_twist * base_z[2]]
                else:
                    # SDK X parallel to bone, use base frame
                    x_axis = base_x
                    z_axis = base_z
                
                # Build rotation matrix from axes (column vectors)
                bone_aligned_matrix = [
                    [x_axis[0], y_axis[0], z_axis[0]],
                    [x_axis[1], y_axis[1], z_axis[1]],
                    [x_axis[2], y_axis[2], z_axis[2]]
                ]
                
                bone_aligned_world_orientations[i] = rotation_matrix_to_quat(bone_aligned_matrix)
            else:
                # Too short, use SDK
                bone_aligned_world_orientations[i] = sdk_quat
        else:
            # Root joint
            bone_aligned_world_orientations[i] = sdk_quat
    
    # Pass 3: Convert world orientations to local orientations
    bone_aligned_local_orientations = {}
    
    for i in range(len(skeleton)):
        if i not in bone_aligned_world_orientations:
            continue
        
        parent_idx = BODY34_PARENTS[i]
        
        if parent_idx < 0:
            # Root joint - world IS local
            bone_aligned_local_orientations[i] = bone_aligned_world_orientations[i]
        elif parent_idx in bone_aligned_world_orientations:
            # local = inverse(parent_world) × child_world
            parent_world = bone_aligned_world_orientations[parent_idx]
            child_world = bone_aligned_world_orientations[i]
            
            parent_inv = quat_inverse(parent_world)
            local_orientation = quat_multiply(parent_inv, child_world)
            bone_aligned_local_orientations[i] = normalize_quat(local_orientation)
    
    return bone_aligned_local_orientations


def compute_stable_bone_orientations(skeleton, reconstructor=None):
    """
    Compute bone-aligned orientations with geometric reconstruction to fix shoulder flipping.
    
    This function combines:
    1. Bone-aligned local orientations (from compute_bone_aligned_local_orientations)
    2. Geometric reconstruction for problematic joints (shoulders, elbows)
    3. Temporal filtering to prevent jitter
    
    The result is stable orientations that don't flip during arm movement.
    
    :param skeleton: List of joint data (dict or objects with pos and ori)
    :param reconstructor: Optional GeometricOrientationReconstructor instance.
                         If None, a new one is created with default settings.
    :return: Dictionary {joint_index: [x, y, z, w]} of local quaternions
    """
    # Import here to avoid circular dependency
    try:
        from .geometric_orientation import GeometricOrientationReconstructor
    except ImportError:
        # Fallback to regular bone-aligned if scipy not available
        import warnings
        warnings.warn("GeometricOrientationReconstructor not available (scipy missing?). "
                     "Using regular bone-aligned orientations without geometric fix.")
        return compute_bone_aligned_local_orientations(skeleton)
    
    # Get base bone-aligned orientations
    bone_aligned = compute_bone_aligned_local_orientations(skeleton)
    
    # Create reconstructor if not provided
    if reconstructor is None:
        reconstructor = GeometricOrientationReconstructor(
            blend_factor=0.3,  # 30% SDK, 70% geometric
            use_filtering=True,
            filter_smoothing=0.2
        )
    
    # Reconstruct problematic joints (shoulders, elbows)
    try:
        reconstructed = reconstructor.reconstruct_skeleton_orientations(skeleton)
        
        # Merge: use reconstructed for problematic joints, bone-aligned for others
        result = bone_aligned.copy()
        for joint_idx, quat in reconstructed.items():
            # Convert numpy array to list if needed
            if hasattr(quat, 'tolist'):
                result[joint_idx] = quat.tolist()
            else:
                result[joint_idx] = quat
        
        return result
    except Exception as e:
        import warnings
        warnings.warn(f"Geometric reconstruction failed: {e}. Using bone-aligned orientations.")
        return bone_aligned


def quaternion_to_euler(q, order='XYZ'):
    """
    Convert quaternion [x, y, z, w] to Euler angles in degrees.
    
    :param q: Quaternion [x, y, z, w]
    :param order: Rotation order ('XYZ', 'ZYX', etc.)
    :return: [rx, ry, rz] in degrees
    """
    import math
    
    x, y, z, w = q
    
    if order == 'XYZ':
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]
    
    # Add other rotation orders as needed
    else:
        raise NotImplementedError(f"Rotation order {order} not implemented")


def draw_skeleton(person, color=(0.2, 0.8, 1.0)):
    """
    Backwards-compatible helper that draws a single person's skeleton using the module-level visualizer.
    """
    v = SkeletonVisualizer3D()
    v.draw_person(person, color=color)


def draw_skeletons_with_bones(people_data, joint_color=(0.2, 0.8, 1.0), bone_color=(0.8, 0.2, 0.2), 
                              show_orientation=False, orientation_length=100.0):
    """
    Helper function to draw skeletons with bone connections.
    :param people_data: List of people data
    :param joint_color: RGB tuple for joint points
    :param bone_color: RGB tuple for bone lines
    :param show_orientation: If True, draw RGB coordinate axes at each joint (default False)
    :param orientation_length: Length of orientation axes in mm (default 100mm)
    """
    v = SkeletonVisualizer3D()
    v.draw_skeleton_with_bones(people_data, joint_color=joint_color, bone_color=bone_color,
                                show_orientation=show_orientation, orientation_length=orientation_length)


def estimate_floor_height(people_data):
    """
    Estimate floor height based on foot/ankle positions of tracked people.
    Returns the estimated floor Y coordinate, or None if no data available.
    """
    if not people_data:
        return None
    
    # Collect foot/ankle Y positions
    foot_heights = []
    
    for person in people_data:
        p = _person_to_dict(person)
        skeleton = p.get("skeleton", [])
        if len(skeleton) > 32:  # Ensure we have BODY_34 data with heel joints
            # Get foot/ankle joint indices (BODY_34 model) - using correct indices
            ankle_joints = [20, 24]  # LEFT_ANKLE, RIGHT_ANKLE
            foot_joints = [21, 25]   # LEFT_FOOT, RIGHT_FOOT
            heel_joints = [32, 33]   # LEFT_HEEL, RIGHT_HEEL
            
            for joint_idx in ankle_joints + foot_joints + heel_joints:
                if joint_idx < len(skeleton):
                    joint = skeleton[joint_idx]
                    y_pos = joint["pos"]["y"]
                    foot_heights.append(y_pos)
    
    if foot_heights:
        # Use the minimum Y value (lowest point) as floor estimate
        # Add a small offset as feet aren't exactly on the floor
        estimated_floor = min(foot_heights) - 50  # 50mm below lowest foot point
        return estimated_floor
    
    return None


def get_stable_floor_height(people_data):
    """
    Get a stable floor height that doesn't jump around frame by frame.
    Uses a moving average of floor height estimates.
    """
    global _floor_height_history, _stable_floor_height
    
    # Get current frame estimate
    current_estimate = estimate_floor_height(people_data)
    
    if current_estimate is not None:
        # Add to history
        _floor_height_history.append(current_estimate)
        
        # Keep only recent samples
        if len(_floor_height_history) > _floor_height_samples:
            _floor_height_history.pop(0)
        
        # Calculate stable height using median (more robust than mean)
        if len(_floor_height_history) >= 5:  # Need at least 5 samples
            sorted_heights = sorted(_floor_height_history)
            median_height = sorted_heights[len(sorted_heights) // 2]
            
            # Only update if the change is significant (> 50mm) to avoid drift
            if _stable_floor_height is None or abs(median_height - _stable_floor_height) > 50:
                _stable_floor_height = median_height
    
    return _stable_floor_height


def draw_floor_grid(size=2000, spacing=100, height=0, color=(0.3, 0.3, 0.3), people_data=None, zed_floor_height=None):
    """
    Draw a floor grid for spatial reference.
    :param size: Total size of the grid in mm (ZED coordinates)
    :param spacing: Spacing between grid lines in mm
    :param height: Y-coordinate height of the floor (None for auto-detection)
    :param color: RGB tuple for grid line color
    :param people_data: People data for floor height estimation (if height=None)
    :param zed_floor_height: Floor height detected by ZED SDK findFloorPlane (preferred)
    """
    # Priority order for floor height detection:
    # 1. Explicit height parameter
    # 2. ZED SDK detected floor height
    # 3. Stable foot-based estimation from people data
    # 4. Fallback to origin (0)
    
    glColor3f(*color)
    glLineWidth(1.0)
    glBegin(GL_LINES)
    
    # Calculate grid bounds
    half_size = size // 2
    
    # Draw lines parallel to X-axis (running along Z)
    for z in range(-half_size, half_size + 1, spacing):
        glVertex3f(-half_size, height, z)
        glVertex3f(half_size, height, z)
    
    # Draw lines parallel to Z-axis (running along X)
    for x in range(-half_size, half_size + 1, spacing):
        glVertex3f(x, height, -half_size)
        glVertex3f(x, height, half_size)
    
    glEnd()
    
    # Draw coordinate axes for reference
    glLineWidth(3.0)
    glBegin(GL_LINES)
    
    # X-axis (red)
    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(0, height, 0)
    glVertex3f(200, height, 0)
    
    # Z-axis (blue) 
    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(0, height, 0)
    glVertex3f(0, height, 200)
    
    # Y-axis (green) - pointing up from floor
    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(0, height, 0)
    glVertex3f(0, height + 200, 0)
    
    glEnd()


def draw_coordinate_axes(origin, length=50.0, line_width=3.0):
    """
    Draw RGB coordinate axes at a given origin point.
    :param origin: Origin point as QVector3D, tuple (x,y,z), or dict with x,y,z keys
    :param length: Length of each axis in mm (default 50mm)
    :param line_width: Width of the axis lines (default 3.0)
    """
    # Convert origin to tuple
    if hasattr(origin, 'x'):  # QVector3D
        ox, oy, oz = origin.x(), origin.y(), origin.z()
    elif isinstance(origin, dict):
        ox, oy, oz = origin['x'], origin['y'], origin['z']
    else:
        ox, oy, oz = origin
    
    # Disable depth test to ensure axes are always visible
    glDisable(GL_DEPTH_TEST)
    
    glLineWidth(line_width)
    glBegin(GL_LINES)
    
    # X-axis (red)
    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(ox, oy, oz)
    glVertex3f(ox + length, oy, oz)
    
    # Y-axis (green)
    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(ox, oy, oz)
    glVertex3f(ox, oy + length, oz)
    
    # Z-axis (blue)
    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(ox, oy, oz)
    glVertex3f(ox, oy, oz + length)
    
    glEnd()
    
    # Re-enable depth test
    glEnable(GL_DEPTH_TEST)


def draw_camera(position, orientation, up=(0.0, 1.0, 0.0), fov_deg=45.0, aspect=16.0/9.0, near=200.0, far=800.0, color=(1.0, 1.0, 0.0), scale=1.0, flip=False):
    """
    Draw a simple camera frustum and axes using lines.
    :param position: (x,y,z) or {'x':..,'y':..,'z':..} camera position in world coordinates
    :param orientation: {'x':..,'y':..,'z':..,'w':..} quaternion representing camera rotation
    :param up: up vector (used as fallback)
    :param fov_deg: vertical field of view in degrees
    :param aspect: aspect ratio (width/height)
    :param near: near plane distance from camera
    :param far: far plane distance from camera
    :param color: RGB tuple for camera lines
    :param scale: scale multiplier for visualization size
    :param flip: flip the forward direction
    """
    # Convert position to tuple if it's a dict or Position object
    if isinstance(position, dict):
        pos = (position['x'], position['y'], position['z'])
    elif hasattr(position, 'x'):
        pos = (position.x, position.y, position.z)
    else:
        pos = position
    
    # Convert quaternion to rotation matrix and extract basis vectors
    def quaternion_to_matrix(quat):
        """Convert quaternion {'x':..,'y':..,'z':..,'w':..} or Quaternion object to 3x3 rotation matrix"""
        if isinstance(quat, dict):
            x, y, z, w = quat['x'], quat['y'], quat['z'], quat['w']
        elif hasattr(quat, 'x'):
            x, y, z, w = quat.x, quat.y, quat.z, quat.w
        else:
            x, y, z, w = quat[0], quat[1], quat[2], quat[3]
        
        # Normalize quaternion
        import math
        norm = math.sqrt(x*x + y*y + z*z + w*w)
        if norm == 0:
            return [[1,0,0], [0,1,0], [0,0,1]]  # Identity matrix
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
        
        # Convert to rotation matrix
        return [
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
        ]
    
    # small vector helpers
    def sub(a, b):
        return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
    def add(a, b):
        return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
    def mul(a, s):
        return (a[0]*s, a[1]*s, a[2]*s)
    def norm(v):
        import math
        l = math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])
        if l == 0:
            return (0.0, 0.0, 0.0)
        return (v[0]/l, v[1]/l, v[2]/l)

    # Get rotation matrix from quaternion
    rot_matrix = quaternion_to_matrix(orientation)
    
    # Extract camera basis vectors from rotation matrix
    right = (rot_matrix[0][0], rot_matrix[1][0], rot_matrix[2][0])    # X-axis
    cam_up = (rot_matrix[0][1], rot_matrix[1][1], rot_matrix[2][1])   # Y-axis
    forward = (-rot_matrix[0][2], -rot_matrix[1][2], -rot_matrix[2][2])  # -Z-axis (camera forward)
    
    if flip:
        forward = (-forward[0], -forward[1], -forward[2])

    import math
    fov_rad = math.radians(fov_deg)
    nh = math.tan(fov_rad / 2.0) * near
    nw = nh * aspect
    fh = math.tan(fov_rad / 2.0) * far
    fw = fh * aspect

    # near/far centers
    nc = add(pos, mul(forward, near*scale))
    fc = add(pos, mul(forward, far*scale))

    # near plane corners
    ntl = add(add(nc, mul(cam_up, nh*scale)), mul(right, -nw*scale))
    ntr = add(add(nc, mul(cam_up, nh*scale)), mul(right, nw*scale))
    nbl = add(add(nc, mul(cam_up, -nh*scale)), mul(right, -nw*scale))
    nbr = add(add(nc, mul(cam_up, -nh*scale)), mul(right, nw*scale))

    # far plane corners
    ftl = add(add(fc, mul(cam_up, fh*scale)), mul(right, -fw*scale))
    ftr = add(add(fc, mul(cam_up, fh*scale)), mul(right, fw*scale))
    fbl = add(add(fc, mul(cam_up, -fh*scale)), mul(right, -fw*scale))
    fbr = add(add(fc, mul(cam_up, -fh*scale)), mul(right, fw*scale))

    glColor3f(*color)
    glLineWidth(1.5)

    # draw frustum lines
    glBegin(GL_LINES)
    # near plane outline
    for a, b in ((ntl, ntr), (ntr, nbr), (nbr, nbl), (nbl, ntl)):
        glVertex3f(a[0], a[1], a[2])
        glVertex3f(b[0], b[1], b[2])

    # far plane outline
    for a, b in ((ftl, ftr), (ftr, fbr), (fbr, fbl), (fbl, ftl)):
        glVertex3f(a[0], a[1], a[2])
        glVertex3f(b[0], b[1], b[2])

    # connect near to far
    for a, b in ((ntl, ftl), (ntr, ftr), (nbl, fbl), (nbr, fbr)):
        glVertex3f(a[0], a[1], a[2])
        glVertex3f(b[0], b[1], b[2])

    # lines from camera position to near plane corners (visualizes camera pyramid)
    for corner in (ntl, ntr, nbl, nbr):
        glVertex3f(pos[0], pos[1], pos[2])
        glVertex3f(corner[0], corner[1], corner[2])

    glEnd()

    # draw small local axes at camera position
    glLineWidth(2.0)
    glBegin(GL_LINES)
    # X (red) - camera right
    glColor3f(1.0, 0.0, 0.0)
    rx = add(pos, mul(right, 50.0*scale))
    glVertex3f(pos[0], pos[1], pos[2])
    glVertex3f(rx[0], rx[1], rx[2])
    # Y (green) - camera up
    glColor3f(0.0, 1.0, 0.0)
    uy = add(pos, mul(cam_up, 50.0*scale))
    glVertex3f(pos[0], pos[1], pos[2])
    glVertex3f(uy[0], uy[1], uy[2])
    # Z (blue) - camera forward
    glColor3f(0.0, 0.0, 1.0)
    fz = add(pos, mul(forward, 100.0*scale))
    glVertex3f(pos[0], pos[1], pos[2])
    glVertex3f(fz[0], fz[1], fz[2])
    glEnd()


def stabilize_sdk_orientations(skeleton, previous_orientations=None):
    """
    Stabilize raw SDK orientations to prevent flipping.
    
    This addresses the fundamental issue with ZED SDK orientations:
    - Positions are RELIABLE (directly triangulated from depth)
    - Bone directions are RELIABLE (derived from positions)
    - Roll/twist is UNRELIABLE (underconstrained, causes flips)
    
    Strategy:
    1. Keep SDK orientations as-is (they're the best estimate we have)
    2. Fix quaternion sign flips (q and -q are the same rotation)
    3. Apply minimal temporal smoothing to prevent jitter
    
    Args:
        skeleton: List of joints with 'ori' attributes
        previous_orientations: Dict of previous frame orientations for temporal consistency
    
    Returns:
        dict: {joint_idx: quaternion [x,y,z,w]} - stabilized orientations
    """
    import math
    
    def quat_dot(q1, q2):
        """Dot product of two quaternions"""
        return q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]
    
    def quat_negate(q):
        """Negate quaternion (same rotation, opposite sign)"""
        return [-q[0], -q[1], -q[2], -q[3]]
    
    def quat_slerp(q1, q2, t):
        """Spherical linear interpolation between quaternions"""
        dot = quat_dot(q1, q2)
        
        # If dot < 0, negate q2 to take shorter path
        if dot < 0:
            q2 = quat_negate(q2)
            dot = -dot
        
        # Clamp dot to avoid numerical issues
        dot = max(-1.0, min(1.0, dot))
        
        # If quaternions are very close, use linear interpolation
        if dot > 0.9995:
            result = [
                q1[0] + t * (q2[0] - q1[0]),
                q1[1] + t * (q2[1] - q1[1]),
                q1[2] + t * (q2[2] - q1[2]),
                q1[3] + t * (q2[3] - q1[3])
            ]
            # Normalize
            length = math.sqrt(sum(x*x for x in result))
            if length > 0:
                result = [x/length for x in result]
            return result
        
        # Slerp
        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        w1 = math.sin((1 - t) * theta) / sin_theta
        w2 = math.sin(t * theta) / sin_theta
        
        return [
            w1 * q1[0] + w2 * q2[0],
            w1 * q1[1] + w2 * q2[1],
            w1 * q1[2] + w2 * q2[2],
            w1 * q1[3] + w2 * q2[3]
        ]
    
    stabilized = {}
    smoothing_factor = 0.5  # Blend 50% previous, 50% current (balanced smoothing)
    
    for joint_idx, joint in enumerate(skeleton):
        ori = joint.ori if hasattr(joint, 'ori') else joint.get('ori')
        if not ori:
            continue
        
        # Extract current quaternion
        if hasattr(ori, 'x'):
            current_quat = [ori.x, ori.y, ori.z, ori.w]
        else:
            current_quat = [ori['x'], ori['y'], ori['z'], ori['w']]
        
        # If we have previous frame data, ensure consistency
        if previous_orientations and joint_idx in previous_orientations:
            prev_quat = previous_orientations[joint_idx]
            
            # Fix sign flip: if dot product is negative, we're on opposite hemisphere
            dot = quat_dot(current_quat, prev_quat)
            if dot < 0:
                current_quat = quat_negate(current_quat)
            
            # Apply temporal smoothing (slerp)
            stabilized[joint_idx] = quat_slerp(prev_quat, current_quat, 1.0 - smoothing_factor)
        else:
            # No previous data, use current as-is
            stabilized[joint_idx] = current_quat
    
    return stabilized


def calcHyrJointOrientations(person, skeleton):
    """
    Calculate bone-aligned orientations where Y-axis points along each bone.
    
    Uses joint positions to compute bone directions, then creates orientations
    where the Y-axis aligns with the bone from parent to child.
    
    Args:
        person: Person object with global_root_orientation attribute
        skeleton: List of joints with 'pos' and 'ori' attributes
    
    Returns:
        dict: {joint_idx: quaternion [x,y,z,w]} - bone-aligned world orientations
    """
    
    # Define parent-child relationships
    BODY34_CHILDREN = {
        0: 1,    # PELVIS -> NAVAL_SPINE
        1: 2,    # NAVAL_SPINE -> CHEST_SPINE
        2: 3,    # CHEST_SPINE -> NECK
        3: 20,   # NECK -> HEAD
        2: 4,    # CHEST_SPINE -> LEFT_CLAVICLE (override)
        4: 5,    # LEFT_CLAVICLE -> LEFT_SHOULDER
        5: 6,    # LEFT_SHOULDER -> LEFT_ELBOW
        6: 7,    # LEFT_ELBOW -> LEFT_WRIST
        7: 8,    # LEFT_WRIST -> LEFT_HAND
        2: 11,   # CHEST_SPINE -> RIGHT_CLAVICLE (override)
        11: 12,  # RIGHT_CLAVICLE -> RIGHT_SHOULDER
        12: 13,  # RIGHT_SHOULDER -> RIGHT_ELBOW
        13: 14,  # RIGHT_ELBOW -> RIGHT_WRIST
        14: 15,  # RIGHT_WRIST -> RIGHT_HAND
        0: 18,   # PELVIS -> LEFT_HIP (override)
        18: 19,  # LEFT_HIP -> LEFT_KNEE
        19: 20,  # LEFT_KNEE -> LEFT_ANKLE
        20: 21,  # LEFT_ANKLE -> LEFT_FOOT
        0: 22,   # PELVIS -> RIGHT_HIP (override)
        22: 23,  # RIGHT_HIP -> RIGHT_KNEE
        23: 24,  # RIGHT_KNEE -> RIGHT_ANKLE
        24: 25,  # RIGHT_ANKLE -> RIGHT_FOOT
    }
    
    def get_position(joint_idx):
        """Get position as QVector3D"""
        if joint_idx >= len(skeleton):
            return None
        joint = skeleton[joint_idx]
        pos = joint.pos if hasattr(joint, 'pos') else joint.get('pos')
        if pos is None:
            return None
        if hasattr(pos, 'x'):
            return QVector3D(pos.x, pos.y, pos.z)
        else:
            return QVector3D(pos['x'], pos['y'], pos['z'])
    
    def get_local_orientation(joint_idx):
        """Get SDK local orientation as QQuaternion"""
        if joint_idx >= len(skeleton):
            return QQuaternion()  # Identity
        joint = skeleton[joint_idx]
        ori = joint.ori if hasattr(joint, 'ori') else joint.get('ori')
        if not ori:
            return QQuaternion()  # Identity
        if hasattr(ori, 'x'):
            return QQuaternion(ori.w, ori.x, ori.y, ori.z).normalized()
        else:
            return QQuaternion(ori['w'], ori['x'], ori['y'], ori['z']).normalized()
    
    def create_bone_aligned_orientation(bone_direction, local_ori):
        """
        Create quaternion where Y-axis points along bone_direction,
        but use local_ori to determine the proper perpendicular plane (twist).
        
        This gives stable orientations that respect the SDK's twist/rotation
        while ensuring Y aligns with the bone.
        """
        # Normalize bone direction - this will be our Y-axis
        y_axis = bone_direction.normalized()
        
        # Get the Z-axis from the local orientation (forward direction)
        # This preserves the twist/rotation from the SDK
        local_z = local_ori.rotatedVector(QVector3D(0, 0, 1))
        
        # Project local_z onto the plane perpendicular to y_axis
        # This removes any component parallel to the bone
        dot = QVector3D.dotProduct(local_z, y_axis)
        z_axis_projected = local_z - y_axis * dot
        
        # If projection is too small, use a fallback
        if z_axis_projected.length() < 0.01:
            # Local Z is parallel to bone - use local X instead
            local_x = local_ori.rotatedVector(QVector3D(1, 0, 0))
            dot = QVector3D.dotProduct(local_x, y_axis)
            z_axis_projected = local_x - y_axis * dot
        
        z_axis = z_axis_projected.normalized()
        
        # X-axis = Y × Z (right-hand rule)
        x_axis = QVector3D.crossProduct(y_axis, z_axis).normalized()
        
        # Re-orthogonalize Z = X × Y to ensure perfect orthogonality
        z_axis = QVector3D.crossProduct(x_axis, y_axis).normalized()
        
        # Build quaternion from rotation matrix
        from PyQt5.QtGui import QMatrix3x3
        m = QMatrix3x3([
            x_axis.x(), y_axis.x(), z_axis.x(),
            x_axis.y(), y_axis.y(), z_axis.y(),
            x_axis.z(), y_axis.z(), z_axis.z()
        ])
        
        return QQuaternion.fromRotationMatrix(m)
    
    # Compute bone-aligned orientations
    world = {}
    
    # Get pelvis world orientation
    if hasattr(person, 'global_root_orientation') and person.global_root_orientation:
        gro = person.global_root_orientation
        if hasattr(gro, 'x'):
            world[Body34Joint.PELVIS] = QQuaternion(gro.w, gro.x, gro.y, gro.z).normalized()
        else:
            world[Body34Joint.PELVIS] = QQuaternion(gro['w'], gro['x'], gro['y'], gro['z']).normalized()
    else:
        world[Body34Joint.PELVIS] = QQuaternion()
    
    # Define full hierarchy - each joint accumulates from its hierarchical parent
    HIERARCHICAL_PARENT = {
        # Spine chain
        Body34Joint.NAVAL_SPINE: Body34Joint.PELVIS,
        Body34Joint.CHEST_SPINE: Body34Joint.NAVAL_SPINE,
        Body34Joint.NECK: Body34Joint.CHEST_SPINE,
        Body34Joint.HEAD: Body34Joint.NECK,
        # Left arm chain
        Body34Joint.LEFT_CLAVICLE: Body34Joint.CHEST_SPINE,
        Body34Joint.LEFT_SHOULDER: Body34Joint.LEFT_CLAVICLE,
        Body34Joint.LEFT_ELBOW: Body34Joint.LEFT_SHOULDER,
        Body34Joint.LEFT_WRIST: Body34Joint.LEFT_ELBOW,
        Body34Joint.LEFT_HAND: Body34Joint.LEFT_WRIST,
        # Right arm chain
        Body34Joint.RIGHT_CLAVICLE: Body34Joint.CHEST_SPINE,
        Body34Joint.RIGHT_SHOULDER: Body34Joint.RIGHT_CLAVICLE,
        Body34Joint.RIGHT_ELBOW: Body34Joint.RIGHT_SHOULDER,
        Body34Joint.RIGHT_WRIST: Body34Joint.RIGHT_ELBOW,
        Body34Joint.RIGHT_HAND: Body34Joint.RIGHT_WRIST,
        # Left leg chain
        Body34Joint.LEFT_HIP: Body34Joint.PELVIS,
        Body34Joint.LEFT_KNEE: Body34Joint.LEFT_HIP,
        Body34Joint.LEFT_ANKLE: Body34Joint.LEFT_KNEE,
        Body34Joint.LEFT_FOOT: Body34Joint.LEFT_ANKLE,
        # Right leg chain
        Body34Joint.RIGHT_HIP: Body34Joint.PELVIS,
        Body34Joint.RIGHT_KNEE: Body34Joint.RIGHT_HIP,
        Body34Joint.RIGHT_ANKLE: Body34Joint.RIGHT_KNEE,
        Body34Joint.RIGHT_FOOT: Body34Joint.RIGHT_ANKLE,
    }
    
    # For each joint, align Y-axis to point toward child
    for parent_idx, child_idx in BODY34_CHILDREN.items():
        parent_pos = get_position(parent_idx)
        child_pos = get_position(child_idx)
        local_ori = get_local_orientation(parent_idx)
        
        if parent_pos and child_pos:
            bone_direction = child_pos - parent_pos
            if bone_direction.length() > 0.001:  # Avoid zero-length bones
                # Check if this joint has a hierarchical parent
                if parent_idx in HIERARCHICAL_PARENT:
                    hierarchical_parent = HIERARCHICAL_PARENT[parent_idx]
                    parent_world = world.get(hierarchical_parent, QQuaternion())
                    
                    # Transform bone direction from world to parent's local space
                    parent_world_inv = parent_world.conjugated()
                    bone_direction_local = parent_world_inv.rotatedVector(bone_direction)
                    
                    # Create bone-aligned orientation in parent's local space
                    bone_aligned_local = create_bone_aligned_orientation(bone_direction_local, local_ori)
                    
                    # Transform to world space: world = parent_world * local
                    world[parent_idx] = parent_world * bone_aligned_local
                else:
                    # Root joints: use world-space bone direction directly
                    world[parent_idx] = create_bone_aligned_orientation(bone_direction, local_ori)
    
    # Convert to [x,y,z,w] format
    result = {}
    for joint_idx, quat in world.items():
        result[joint_idx] = [quat.x(), quat.y(), quat.z(), quat.scalar()]
    
    return result


def reconstructSkeletonFromOrientations(person, skeleton, bone_lengths=None, bone_directions=None):
    """
    Reconstruct skeleton using forward kinematics from local orientations only.
    
    This shows what the skeleton looks like if we only use the SDK's local rotations
    and fixed bone lengths (measured from the first frame). Useful for:
    - Understanding how orientations drive the skeleton
    - Exporting to animation systems that use FK
    - Validating rotation data quality
    
    Args:
        person: Person object with global_root_orientation and local_position_per_joint
        skeleton: List of joints with 'pos' and 'ori' attributes
        bone_lengths: Dict {child_idx: length} - if None, computed from current frame
        bone_directions: Dict {child_idx: QVector3D} - bind pose directions
    
    Returns:
        tuple: (world_positions, world_rotations, bone_lengths, bone_directions)
               world_positions: {joint_idx: QVector3D}
               world_rotations: {joint_idx: QQuaternion}
               bone_lengths: {child_idx: float} - for caching
               bone_directions: {child_idx: QVector3D} - for caching
    """
    
    # Skeleton hierarchy (parent index for each joint)
    BODY34_PARENTS = [
        -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 3, 11, 12, 13, 14, 15, 15,
        0, 18, 19, 20, 0, 22, 23, 24, 3, 26, 26, 26, 26, 26, 20, 24
    ]
    
    def get_position(joint_idx):
        """Get position as QVector3D"""
        if joint_idx >= len(skeleton):
            return None
        joint = skeleton[joint_idx]
        pos = joint.pos if hasattr(joint, 'pos') else joint.get('pos')
        if pos is None:
            return None
        if hasattr(pos, 'x'):
            return QVector3D(pos.x, pos.y, pos.z)
        else:
            return QVector3D(pos['x'], pos['y'], pos['z'])
    
    def get_local_orientation(joint_idx):
        """Get SDK local orientation as QQuaternion"""
        if joint_idx >= len(skeleton):
            return QQuaternion()
        joint = skeleton[joint_idx]
        ori = joint.ori if hasattr(joint, 'ori') else joint.get('ori')
        if not ori:
            return QQuaternion()
        if hasattr(ori, 'x'):
            return QQuaternion(ori.w, ori.x, ori.y, ori.z).normalized()
        else:
            return QQuaternion(ori['w'], ori['x'], ori['y'], ori['z']).normalized()
    
    # Initialize bone lengths and directions from first frame if not provided
    if bone_lengths is None or bone_directions is None:
        bone_lengths = {}
        bone_directions = {}
        
        # First pass: compute world orientations for the bind pose
        bind_world_orientations = {}
        
        # Get pelvis orientation for bind pose
        if hasattr(person, 'global_root_orientation') and person.global_root_orientation:
            gro = person.global_root_orientation
            if hasattr(gro, 'x'):
                bind_world_orientations[0] = QQuaternion(gro.w, gro.x, gro.y, gro.z).normalized()
            else:
                bind_world_orientations[0] = QQuaternion(gro['w'], gro['x'], gro['y'], gro['z']).normalized()
        else:
            bind_world_orientations[0] = get_local_orientation(0)
        
        # Accumulate bind orientations through hierarchy
        for child_idx, parent_idx in enumerate(BODY34_PARENTS):
            if parent_idx >= 0 and parent_idx in bind_world_orientations:
                local_ori = get_local_orientation(child_idx)
                bind_world_orientations[child_idx] = bind_world_orientations[parent_idx] * local_ori
        
        # Second pass: compute bone directions in parent's LOCAL space
        for child_idx, parent_idx in enumerate(BODY34_PARENTS):
            if parent_idx < 0:
                continue
            
            parent_pos = get_position(parent_idx)
            child_pos = get_position(child_idx)
            
            if parent_pos and child_pos:
                # Bone vector in world space
                bone_vec_world = child_pos - parent_pos
                length = bone_vec_world.length()
                
                if length > 0.001:
                    bone_lengths[child_idx] = length
                    
                    # Transform bone direction to parent's LOCAL space
                    # This is the bind pose direction
                    if parent_idx in bind_world_orientations:
                        parent_world_rot = bind_world_orientations[parent_idx]
                        parent_world_inv = parent_world_rot.conjugated()
                        bone_vec_local = parent_world_inv.rotatedVector(bone_vec_world)
                        bone_directions[child_idx] = bone_vec_local.normalized()
                    else:
                        # Fallback: use world space direction
                        bone_directions[child_idx] = bone_vec_world.normalized()
                else:
                    # Fallback for zero-length bones
                    bone_lengths[child_idx] = 0.0
                    bone_directions[child_idx] = QVector3D(0, 1, 0)
    
    # Get pelvis world transform
    world_positions = {}
    world_rotations = {}
    bone_aligned_rotations = {}  # Separate dict for bone-aligned orientations
    
    # Pelvis world orientation - use global_root_orientation which represents the pelvis world rotation
    if hasattr(person, 'global_root_orientation') and person.global_root_orientation:
        gro = person.global_root_orientation
        if hasattr(gro, 'x'):
            world_rotations[0] = QQuaternion(gro.w, gro.x, gro.y, gro.z).normalized()
        else:
            world_rotations[0] = QQuaternion(gro['w'], gro['x'], gro['y'], gro['z']).normalized()
    else:
        # Fallback: use pelvis local orientation
        pelvis_local = get_local_orientation(0)
        if pelvis_local:
            world_rotations[0] = pelvis_local
        else:
            world_rotations[0] = QQuaternion()
    
    bone_aligned_rotations[0] = world_rotations[0]  # Pelvis same for both
    
    # Pelvis position - use SDK tracked position
    pelvis_pos = get_position(0)
    if pelvis_pos:
        world_positions[0] = pelvis_pos
    else:
        world_positions[0] = QVector3D(0, 0, 0)
    
    # Forward kinematics recursion
    def fk_recurse(parent_idx):
        """Recursively compute child transforms using proper FK"""
        for child_idx, parent in enumerate(BODY34_PARENTS):
            if parent != parent_idx:
                continue
            
            # Get local orientation from SDK
            local_ori = get_local_orientation(child_idx)
            
            # FK Step 1: Compute child's world rotation
            # world_rotation = parent_world_rotation × local_rotation
            parent_world_rot = world_rotations[parent_idx]
            world_rotations[child_idx] = parent_world_rot * local_ori
            
            # FK Step 2: Compute child's world position
            # The bone direction is determined by the bone's bind-pose direction
            # rotated by the PARENT's world rotation (not the child's!)
            if child_idx in bone_directions and child_idx in bone_lengths:
                bone_len = bone_lengths[child_idx]
                # Get bind-pose bone direction in parent's local space
                bone_dir_local = bone_directions[child_idx]
                # Rotate by parent's world rotation to get world-space bone direction
                bone_dir_world = parent_world_rot.rotatedVector(bone_dir_local)
                # Position = parent_pos + bone_direction * length
                world_positions[child_idx] = world_positions[parent_idx] + bone_dir_world * bone_len
                
                # Create bone-aligned orientation at PARENT joint
                # Y-axis points along the bone from parent to child
                bone_direction_world = bone_dir_world.normalized()
                
                # Use parent's rotation to get perpendicular axes
                sdk_z = parent_world_rot.rotatedVector(QVector3D(0, 0, 1))
                
                # Project SDK Z onto plane perpendicular to bone
                dot = QVector3D.dotProduct(sdk_z, bone_direction_world)
                z_projected = sdk_z - bone_direction_world * dot
                
                # If projection too small, use SDK X instead
                if z_projected.length() < 0.01:
                    sdk_x = parent_world_rot.rotatedVector(QVector3D(1, 0, 0))
                    dot = QVector3D.dotProduct(sdk_x, bone_direction_world)
                    z_projected = sdk_x - bone_direction_world * dot
                
                if z_projected.length() < 0.01:
                    # Still too small, use a perpendicular vector
                    if abs(bone_direction_world.y()) < 0.9:
                        z_projected = QVector3D(0, 1, 0) - bone_direction_world * bone_direction_world.y()
                    else:
                        z_projected = QVector3D(1, 0, 0) - bone_direction_world * bone_direction_world.x()
                
                z_axis = z_projected.normalized()
                y_axis = bone_direction_world
                x_axis = QVector3D.crossProduct(y_axis, z_axis).normalized()
                z_axis = QVector3D.crossProduct(x_axis, y_axis).normalized()
                
                # Build bone-aligned quaternion for PARENT joint
                from PyQt5.QtGui import QMatrix3x3
                m = QMatrix3x3([
                    x_axis.x(), y_axis.x(), z_axis.x(),
                    x_axis.y(), y_axis.y(), z_axis.y(),
                    x_axis.z(), y_axis.z(), z_axis.z()
                ])
                bone_aligned_rotations[parent_idx] = QQuaternion.fromRotationMatrix(m)
            else:
                # No bone length data - use SDK position
                child_pos = get_position(child_idx)
                if child_pos:
                    world_positions[child_idx] = child_pos
                else:
                    world_positions[child_idx] = world_positions[parent_idx]
                
                # For joints without children, use their SDK orientation
                if child_idx not in bone_aligned_rotations:
                    bone_aligned_rotations[child_idx] = world_rotations[child_idx]
            
            # Recurse to children
            fk_recurse(child_idx)
    
    # Start recursion from pelvis
    fk_recurse(0)
    
    return world_positions, bone_aligned_rotations, bone_lengths, bone_directions
