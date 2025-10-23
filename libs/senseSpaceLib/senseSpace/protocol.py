from dataclasses import dataclass
from typing import List, Dict, Union
from .enums import Body18Joint, Body34Joint, SkeletonAngle, UniversalJoint
from typing import Optional
import numpy as np


# Lightweight data structures for network efficiency
@dataclass
class Position:
    """3D position with reduced memory footprint via __slots__"""
    __slots__ = ("x", "y", "z")
    x: float
    y: float
    z: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dict for JSON serialization"""
        return {"x": float(self.x), "y": float(self.y), "z": float(self.z)}
    
    def to_list(self) -> List[float]:
        """Convert to list for compact JSON serialization"""
        return [float(self.x), float(self.y), float(self.z)]

    @staticmethod
    def from_dict(d: Union[Dict[str, float], List[float], tuple]) -> "Position":
        """Create from dict or list/tuple"""
        if isinstance(d, (list, tuple)):
            return Position(x=float(d[0]), y=float(d[1]), z=float(d[2]))
        return Position(x=float(d.get("x", 0.0)), y=float(d.get("y", 0.0)), z=float(d.get("z", 0.0)))


@dataclass
class Quaternion:
    """Quaternion orientation with reduced memory footprint via __slots__"""
    __slots__ = ("x", "y", "z", "w")
    x: float
    y: float
    z: float
    w: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dict for JSON serialization"""
        return {"x": float(self.x), "y": float(self.y), "z": float(self.z), "w": float(self.w)}
    
    def to_list(self) -> List[float]:
        """Convert to list for compact JSON serialization"""
        return [float(self.x), float(self.y), float(self.z), float(self.w)]

    @staticmethod
    def from_dict(d: Union[Dict[str, float], List[float], tuple]) -> "Quaternion":
        """Create from dict or list/tuple"""
        if isinstance(d, (list, tuple)):
            return Quaternion(x=float(d[0]), y=float(d[1]), z=float(d[2]), w=float(d[3]))
        return Quaternion(
            x=float(d.get("x", 0.0)),
            y=float(d.get("y", 0.0)),
            z=float(d.get("z", 0.0)),
            w=float(d.get("w", 1.0))
        )


@dataclass
class Joint:
    i: int                        # raw index from server (0..17 or 0..33)
    pos: Position                 # 3D position (was dict, now lightweight dataclass)
    ori: Quaternion               # quaternion orientation (was dict, now lightweight dataclass)
    conf: float                   # confidence 0..1

    def as_enum(self, body_model: str) -> Union[Body18Joint, Body34Joint]:
        """Return this joint as the appropriate enum"""
        if body_model == "BODY_18":
            return Body18Joint(self.i)
        elif body_model == "BODY_34":
            return Body34Joint(self.i)
        else:
            return self.i  # fallback: raw index

    def to_dict(self):
        return {
            "i": self.i,
            "pos": self.pos.to_dict(),
            "ori": self.ori.to_dict(),
            "conf": self.conf
        }

    @staticmethod
    def from_dict(d):
        return Joint(
            i=d["i"],
            pos=Position.from_dict(d["pos"]),
            ori=Quaternion.from_dict(d["ori"]),
            conf=d["conf"]
        )


@dataclass
class Person:
    id: int
    tracking_state: str
    confidence: float
    skeleton: List[Joint]

    def get_joint(self, joint_id: UniversalJoint, body_model: str = "BODY_34") -> Optional[tuple[Position, Quaternion]]:
        """Get joint position and orientation by universal joint ID
        
        Args:
            joint_id: UniversalJoint enum value
            body_model: "BODY_18" or "BODY_34" (default: "BODY_34")
            
        Returns:
            tuple of (Position, Quaternion) if joint exists, None otherwise
            
        Example:
            pos, quat = person.get_joint(UniversalJoint.LEFT_WRIST, frame.body_model)
            if pos and quat:
                print(f"Left wrist at ({pos.x}, {pos.y}, {pos.z})")
        """
        # Map universal joint to format-specific index
        if body_model == "BODY_18":
            index = joint_id.to_body18_index()
        elif body_model == "BODY_34":
            index = joint_id.to_body34_index()
        else:
            return None
        
        if index is None:
            return None  # Joint not available in this format
        
        # Find joint in skeleton
        for joint in self.skeleton:
            if joint.i == index:
                return (joint.pos, joint.ori)
        
        return None  # Joint not found in skeleton data
    
    def to_dict(self):
        return {
            "id": self.id,
            "tracking_state": self.tracking_state,
            "confidence": self.confidence,
            "skeleton": [j.to_dict() for j in self.skeleton]
        }

    @staticmethod
    def from_dict(d):
        return Person(
            id=d["id"],
            tracking_state=d["tracking_state"],
            confidence=d["confidence"],
            skeleton=[Joint.from_dict(j) for j in d["skeleton"]]
        )

    def get_skeletal_angles(self) -> dict:
        """
        Calculate all skeletal angles for this person
        
        Returns:
            Dictionary containing skeletal measurements using SkeletonAngle enum keys
        """
        angles = {}
        
        # -----------------------------------------------------------
        # Helper functions
        # -----------------------------------------------------------
        def get_pos(joint: Joint) -> Optional[np.ndarray]:
            """Extract [x, y, z] from joint (mm → m)."""
            if not joint or not joint.pos:
                return None
            pos = joint.pos
            # Support both Position object and dict
            if hasattr(pos, 'x'):
                return np.array([pos.x, pos.y, pos.z], dtype=float) / 1000.0
            else:
                return np.array([pos['x'], pos['y'], pos['z']], dtype=float) / 1000.0

        def get_joint_by_enum(joint_enum: Union[Body18Joint, Body34Joint]) -> Optional[np.ndarray]:
            """Get joint position by enum."""
            for joint in self.skeleton:
                if joint.i == joint_enum.value:
                    return get_pos(joint)
            return None

        def get_joint_object_by_enum(joint_enum: Union[Body18Joint, Body34Joint]) -> Optional[Joint]:
            """Get full joint object by enum."""
            for joint in self.skeleton:
                if joint.i == joint_enum.value:
                    return joint
            return None

        def compute_joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[float]:
            """
            Compute angle at point B formed by points A-B-C using law of cosines.
            Returns angle in degrees (0-180°).
            """
            if a is None or b is None or c is None:
                return None
            
            a, b, c = np.array(a), np.array(b), np.array(c)
            ab = np.linalg.norm(a - b)
            bc = np.linalg.norm(b - c)
            ac = np.linalg.norm(a - c)
            
            denominator = 2 * ab * bc
            if denominator < 1e-8:
                return 180.0
            
            cos_angle = (ab**2 + bc**2 - ac**2) / denominator
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            return float(np.degrees(np.arccos(cos_angle)))

        def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> Optional[float]:
            """Calculate angle between two vectors in degrees."""
            if v1 is None or v2 is None:
                return None
            v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
            v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
            cos_angle = np.dot(v1_norm, v2_norm)
            return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

        def vector_to_angles(v: np.ndarray) -> tuple:
            """
            Convert a 3D vector to azimuth and elevation angles.
            
            Args:
                v: normalized 3D vector in body space [x, y, z]
            
            Returns:
                (azimuth, elevation) in degrees
                - azimuth: left(+)/right(-) angle around Y axis (yaw)
                - elevation: up(+)/down(-) angle from horizontal (pitch)
            """
            x, y, z = v
            azimuth = np.degrees(np.arctan2(x, z))  # Keep original, body frame is now correct
            elevation = np.degrees(np.arcsin(np.clip(y, -1.0, 1.0)))  # up-down
            return float(azimuth), float(elevation)

        def quaternion_to_forward_vector(quat: Dict[str, float]) -> np.ndarray:
            """
            Convert a quaternion to a forward-facing direction vector.
            
            Args:
                quat: quaternion dict with keys 'x', 'y', 'z', 'w'
            
            Returns:
                Forward direction vector [x, y, z] in world space
            """
            x, y, z, w = quat['x'], quat['y'], quat['z'], quat['w']
            
            # Forward vector in local space is typically [0, 0, 1]
            # Rotate it by the quaternion to get world space direction
            # Using quaternion rotation formula: v' = q * v * q^(-1)
            
            # For forward vector [0, 0, 1], the rotation simplifies to:
            forward_x = 2 * (x*z + w*y)
            forward_y = 2 * (y*z - w*x)
            forward_z = 1 - 2 * (x*x + y*y)
            
            forward = np.array([forward_x, forward_y, forward_z])
            return forward / (np.linalg.norm(forward) + 1e-8)

        # Alternative: try different quaternion axes
        def quaternion_to_vectors(quat: Dict[str, float]) -> tuple:
            """Get forward, right, and up vectors from quaternion."""
            x, y, z, w = quat['x'], quat['y'], quat['z'], quat['w']
            
            # Forward (Z-axis)
            forward = np.array([
                2 * (x*z + w*y),
                2 * (y*z - w*x),
                1 - 2 * (x*x + y*y)
            ])
            
            # Right (X-axis)
            right = np.array([
                1 - 2 * (y*y + z*z),
                2 * (x*y + w*z),
                2 * (x*z - w*y)
            ])
            
            # Up (Y-axis)
            up = np.array([
                2 * (x*y - w*z),
                1 - 2 * (x*x + z*z),
                2 * (y*z + w*x)
            ])
            
            return forward, right, up

        def quaternion_to_euler_angles(quat: Dict[str, float]) -> tuple:
            """
            Convert quaternion to Euler angles (yaw, pitch, roll).
            
            Args:
                quat: quaternion dict with keys 'x', 'y', 'z', 'w'
            
            Returns:
                (yaw, pitch, roll) in degrees
                - yaw: rotation around Y axis (left/right)
                - pitch: rotation around X axis (up/down)
                - roll: rotation around Z axis (tilt)
            """
            x, y, z, w = quat['x'], quat['y'], quat['z'], quat['w']
            
            # Roll (x-axis rotation)
            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x * x + y * y)
            roll = np.arctan2(sinr_cosp, cosr_cosp)
            
            # Pitch (y-axis rotation)
            sinp = 2 * (w * y - z * x)
            if abs(sinp) >= 1:
                pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
            else:
                pitch = np.arcsin(sinp)
            
            # Yaw (z-axis rotation)
            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)
            
            return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

        # -----------------------------------------------------------
        # Extract all joints (BODY_34 format)
        # -----------------------------------------------------------
        pelvis = get_joint_by_enum(Body34Joint.PELVIS)
        naval = get_joint_by_enum(Body34Joint.NAVAL_SPINE)
        chest = get_joint_by_enum(Body34Joint.CHEST_SPINE)
        neck = get_joint_by_enum(Body34Joint.NECK)
        head = get_joint_by_enum(Body34Joint.HEAD)
        
        left_clavicle = get_joint_by_enum(Body34Joint.LEFT_CLAVICLE)
        right_clavicle = get_joint_by_enum(Body34Joint.RIGHT_CLAVICLE)
        
        left_shoulder = get_joint_by_enum(Body34Joint.LEFT_SHOULDER)
        left_elbow = get_joint_by_enum(Body34Joint.LEFT_ELBOW)
        left_wrist = get_joint_by_enum(Body34Joint.LEFT_WRIST)
        
        right_shoulder = get_joint_by_enum(Body34Joint.RIGHT_SHOULDER)
        right_elbow = get_joint_by_enum(Body34Joint.RIGHT_ELBOW)
        right_wrist = get_joint_by_enum(Body34Joint.RIGHT_WRIST)
        
        left_hip = get_joint_by_enum(Body34Joint.LEFT_HIP)
        left_knee = get_joint_by_enum(Body34Joint.LEFT_KNEE)
        left_ankle = get_joint_by_enum(Body34Joint.LEFT_ANKLE)
        
        right_hip = get_joint_by_enum(Body34Joint.RIGHT_HIP)
        right_knee = get_joint_by_enum(Body34Joint.RIGHT_KNEE)
        right_ankle = get_joint_by_enum(Body34Joint.RIGHT_ANKLE)

        # Extract eye and nose joints
        left_eye = get_joint_by_enum(Body34Joint.LEFT_EYE)
        right_eye = get_joint_by_enum(Body34Joint.RIGHT_EYE)
        nose = get_joint_by_enum(Body34Joint.NOSE)

        # -----------------------------------------------------------
        # Body coordinate system (local frame)
        # -----------------------------------------------------------
        body_frame_valid = False
        x_body = None
        y_body = None
        z_body = None
        
        if left_shoulder is not None and right_shoulder is not None:
            # X-axis: right direction (right_shoulder - left_shoulder)
            x_body = right_shoulder - left_shoulder
            x_body = x_body / (np.linalg.norm(x_body) + 1e-8)
            
            # Z-axis: forward direction (perpendicular to shoulders, in XZ plane)
            up_world = np.array([0, 1, 0])
            z_body = np.cross(up_world, x_body)  # FIX: Swap order to get forward, not backward
            z_body = z_body / (np.linalg.norm(z_body) + 1e-8)
            
            # Y-axis: up direction (perpendicular to X and Z)
            y_body = np.cross(z_body, x_body)
            # Ensure Y points up (positive Y in world space)
            if y_body[1] < 0:  # If Y component is negative, flip it
                y_body = -y_body
            y_body = y_body / (np.linalg.norm(y_body) + 1e-8)
            
            # Store body axes
            angles[SkeletonAngle.BODY_RIGHT_DIRECTION] = x_body
            angles[SkeletonAngle.BODY_FORWARD_DIRECTION] = z_body
            angles[SkeletonAngle.BODY_UP_DIRECTION] = y_body
            body_frame_valid = True

        def to_body_space(v: np.ndarray) -> np.ndarray:
            """Transform a world vector into body-local coordinates."""
            if not body_frame_valid:
                return v  # Return as-is if body frame not established
            return np.array([
                np.dot(v, x_body),
                np.dot(v, y_body),
                np.dot(v, z_body)
            ])

        # -----------------------------------------------------------
        # Head yaw (left/right) relative to body using eyes
        # -----------------------------------------------------------
        if left_shoulder is not None and right_shoulder is not None and left_eye is not None and right_eye is not None:
            # 1. Build body reference frame from shoulders
            shoulder_vec = right_shoulder - left_shoulder
            shoulder_vec = shoulder_vec / (np.linalg.norm(shoulder_vec) + 1e-8)

            up_world = np.array([0.0, 1.0, 0.0])
            # FIX: Reverse cross product order to get correct body forward
            body_forward = np.cross(up_world, shoulder_vec)  # Changed from (shoulder_vec, up_world)
            body_forward = body_forward / (np.linalg.norm(body_forward) + 1e-8)

            # Optional: blend with torso direction for stability
            if naval is not None and chest is not None:
                torso_forward = chest - naval
                torso_forward = torso_forward / (np.linalg.norm(torso_forward) + 1e-8)
                body_forward = 0.5 * body_forward + 0.5 * torso_forward
                body_forward = body_forward / (np.linalg.norm(body_forward) + 1e-8)

            # 2. Face direction from eyes
            # Eye line goes from left to right eye
            eye_line = right_eye - left_eye
            eye_line = eye_line / (np.linalg.norm(eye_line) + 1e-8)
            
            # FIX: Reverse cross product order to get correct face forward
            face_forward = np.cross(eye_line, up_world)  # Changed from (up_world, eye_line)
            face_forward = face_forward / (np.linalg.norm(face_forward) + 1e-8)
            
            # Verify direction with nose if available
            if nose is not None:
                eye_center = (left_eye + right_eye) / 2.0
                nose_vec = nose - eye_center
                # If nose points backward relative to face_forward, flip it
                if np.dot(face_forward, nose_vec) < 0:
                    face_forward = -face_forward

            # 3. Project both onto horizontal plane (ignore vertical component)
            body_proj = np.array([body_forward[0], 0, body_forward[2]])
            face_proj = np.array([face_forward[0], 0, face_forward[2]])
            body_proj = body_proj / (np.linalg.norm(body_proj) + 1e-8)
            face_proj = face_proj / (np.linalg.norm(face_proj) + 1e-8)

            # 4. Signed angle in horizontal plane
            # Positive = turn to the right, Negative = turn to the left
            cross_y = face_proj[0]*body_proj[2] - face_proj[2]*body_proj[0]
            dot = np.dot(body_proj, face_proj)
            yaw_deg = np.degrees(np.arctan2(cross_y, dot))

            # 5. Optional noise suppression
            if abs(yaw_deg) < 3.0:
                yaw_deg = 0.0

            angles[SkeletonAngle.HEAD_ROTATION] = yaw_deg

        # -----------------------------------------------------------
        # Head pitch (look up/down) - simple elevation angle
        # -----------------------------------------------------------
        if neck is not None and head is not None:
            head_vec = head - neck
            head_vec_norm = head_vec / (np.linalg.norm(head_vec) + 1e-8)
            
            # Pitch is simply the elevation angle
            pitch = np.degrees(np.arcsin(np.clip(head_vec_norm[1], -1.0, 1.0)))
            
            # Compensate for natural upward tilt (calibrate this value)
            NATURAL_HEAD_TILT = 45.0  # Adjust based on your skeleton
            pitch = pitch - NATURAL_HEAD_TILT
            
            angles[SkeletonAngle.HEAD_LOOK_UP_DOWN] = pitch
        
        # -----------------------------------------------------------
        # Head tilt/roll (ear to shoulder) - using shoulder plane
        # -----------------------------------------------------------
        if neck is not None and head is not None and left_shoulder is not None and right_shoulder is not None:
            head_vec = head - neck
            shoulder_vec = right_shoulder - left_shoulder
            # Project head vector onto shoulder plane
            shoulder_norm = shoulder_vec / (np.linalg.norm(shoulder_vec) + 1e-8)
            head_tilt_component = np.dot(head_vec, shoulder_norm)
            # Angle from vertical in shoulder plane
            angles[SkeletonAngle.NECK_TILT] = float(np.degrees(np.arctan2(head_tilt_component, head_vec[1])))

        # -----------------------------------------------------------
        # Arm angles (clavicle -> shoulder -> elbow)
        # -----------------------------------------------------------
        if left_clavicle is not None and left_shoulder is not None and left_elbow is not None:
            angles[SkeletonAngle.LEFT_ARM_ANGLE] = compute_joint_angle(left_clavicle, left_shoulder, left_elbow)
        
        if right_clavicle is not None and right_shoulder is not None and right_elbow is not None:
            angles[SkeletonAngle.RIGHT_ARM_ANGLE] = compute_joint_angle(right_clavicle, right_shoulder, right_elbow)

        # -----------------------------------------------------------
        # Elbow angles (shoulder -> elbow -> wrist)
        # -----------------------------------------------------------
        if left_shoulder is not None and left_elbow is not None and left_wrist is not None:
            angles[SkeletonAngle.LEFT_ELBOW_ANGLE] = compute_joint_angle(left_shoulder, left_elbow, left_wrist)
        
        if right_shoulder is not None and right_elbow is not None and right_wrist is not None:
            angles[SkeletonAngle.RIGHT_ELBOW_ANGLE] = compute_joint_angle(right_shoulder, right_elbow, right_wrist)

        # -----------------------------------------------------------
        # Arm 3D orientation (azimuth/elevation in body space)
        # -----------------------------------------------------------
        if body_frame_valid:
            # Left arm
            if left_shoulder is not None and left_wrist is not None:
                left_arm_dir = left_wrist - left_shoulder
                left_arm_dir = left_arm_dir / (np.linalg.norm(left_arm_dir) + 1e-8)
                left_arm_local = to_body_space(left_arm_dir)
                angles[SkeletonAngle.LEFT_ARM_DIRECTION_VECTOR] = left_arm_local
                l_az, l_el = vector_to_angles(left_arm_local)
                angles[SkeletonAngle.LEFT_ARM_AZIMUTH] = l_az
                angles[SkeletonAngle.LEFT_ARM_ELEVATION] = l_el
            
            # Right arm
            if right_shoulder is not None and right_wrist is not None:
                right_arm_dir = right_wrist - right_shoulder
                right_arm_dir = right_arm_dir / (np.linalg.norm(right_arm_dir) + 1e-8)
                right_arm_local = to_body_space(right_arm_dir)
                angles[SkeletonAngle.RIGHT_ARM_DIRECTION_VECTOR] = right_arm_local
                r_az, r_el = vector_to_angles(right_arm_local)
                angles[SkeletonAngle.RIGHT_ARM_AZIMUTH] = r_az
                angles[SkeletonAngle.RIGHT_ARM_ELEVATION] = r_el

        # -----------------------------------------------------------
        # Hip angles (naval -> hip -> knee)
        # -----------------------------------------------------------
        if naval is not None and left_hip is not None and left_knee is not None:
            angles[SkeletonAngle.LEFT_HIP_ANGLE] = compute_joint_angle(naval, left_hip, left_knee)
        
        if naval is not None and right_hip is not None and right_knee is not None:
            angles[SkeletonAngle.RIGHT_HIP_ANGLE] = compute_joint_angle(naval, right_hip, right_knee)

        # -----------------------------------------------------------
        # Knee angles (hip -> knee -> ankle)
        # -----------------------------------------------------------
        if left_hip is not None and left_knee is not None and left_ankle is not None:
            angles[SkeletonAngle.LEFT_KNEE_ANGLE] = compute_joint_angle(left_hip, left_knee, left_ankle)
        
        if right_hip is not None and right_knee is not None and right_ankle is not None:
            angles[SkeletonAngle.RIGHT_KNEE_ANGLE] = compute_joint_angle(right_hip, right_knee, right_ankle)

        # -----------------------------------------------------------
        # Leg 3D orientation (azimuth/elevation in body space)
        # -----------------------------------------------------------
        if body_frame_valid:
            # Left leg (thigh direction: hip -> knee)
            if left_hip is not None and left_knee is not None:
                left_leg_dir = left_knee - left_hip
                left_leg_dir = left_leg_dir / (np.linalg.norm(left_leg_dir) + 1e-8)
                left_leg_local = to_body_space(left_leg_dir)
                angles[SkeletonAngle.LEFT_LEG_DIRECTION_VECTOR] = left_leg_local
                l_leg_az, l_leg_el = vector_to_angles(left_leg_local)
                angles[SkeletonAngle.LEFT_LEG_AZIMUTH] = l_leg_az
                angles[SkeletonAngle.LEFT_LEG_ELEVATION] = l_leg_el
            
            # Right leg
            if right_hip is not None and right_knee is not None:
                right_leg_dir = right_knee - right_hip
                right_leg_dir = right_leg_dir / (np.linalg.norm(right_leg_dir) + 1e-8)
                right_leg_local = to_body_space(right_leg_dir)
                angles[SkeletonAngle.RIGHT_LEG_DIRECTION_VECTOR] = right_leg_local
                r_leg_az, r_leg_el = vector_to_angles(right_leg_local)
                angles[SkeletonAngle.RIGHT_LEG_AZIMUTH] = r_leg_az
                angles[SkeletonAngle.RIGHT_LEG_ELEVATION] = r_leg_el

        # -----------------------------------------------------------
        # Legs - DELETE THIS COMPLETE SECTION (lines 467-520)
        # -----------------------------------------------------------
        
        # ... rest of existing code for hip/knee angles, body position ...
        
        return angles


@dataclass
class Camera:
    serial: str
    position: Position             # 3D position (was dict, now lightweight dataclass)
    orientation: Quaternion        # quaternion orientation (was dict, now lightweight dataclass)

    def to_dict(self):
        return {
            'serial': self.serial,
            'position': self.position.to_dict(),
            'orientation': self.orientation.to_dict()
        }

    @staticmethod
    def from_dict(d):
        return Camera(
            serial=d.get('serial', ''),
            position=Position.from_dict(d.get('position', {'x': 0, 'y': 0, 'z': 0})),
            orientation=Quaternion.from_dict(d.get('orientation', {'x': 0, 'y': 0, 'z': 0, 'w': 1}))
        )


@dataclass
class Frame:
    timestamp: float
    people: List[Person]
    body_model: str = ""   # set from config
    floor_height: float = None  # ZED SDK detected floor height in mm
    cameras: Optional[List[Dict]] = None  # optional list of camera pose dicts or Camera objects

    def to_dict(self):
        data = {
            "timestamp": self.timestamp,
            "people": [p.to_dict() for p in self.people],
            "body_model": self.body_model,
        }
        # Only include floor_height when present
        if self.floor_height is not None:
            data["floor_height"] = self.floor_height
        # Include cameras if provided
        if self.cameras is not None:
            # If Camera objects, convert to dicts; otherwise assume list of dicts
            out_cams = []
            for c in self.cameras:
                try:
                    out_cams.append(c.to_dict())
                except Exception:
                    out_cams.append(c)
            data["cameras"] = out_cams
        return data

    @staticmethod
    def from_dict(d):
        cams = d.get("cameras", None)
        if cams is not None:
            cam_objs = []
            for c in cams:
                try:
                    cam_objs.append(Camera.from_dict(c))
                except Exception:
                    cam_objs.append(c)
        else:
            cam_objs = None

        return Frame(
            timestamp=d["timestamp"],
            people=[Person.from_dict(p) for p in d["people"]],
            body_model=d.get("body_model", ""),
            floor_height=d.get("floor_height", None),
            cameras=cam_objs
        )
