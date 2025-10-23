from enum import Enum
from typing import Optional

class Body18Joint(Enum):
    """ZED SDK BODY_18 joint indices"""
    PELVIS = 0
    NAVAL_SPINE = 1
    CHEST_SPINE = 2
    NECK = 3
    LEFT_CLAVICLE = 4
    LEFT_SHOULDER = 5
    LEFT_ELBOW = 6
    LEFT_WRIST = 7
    LEFT_HAND = 8
    LEFT_HANDTIP = 9
    LEFT_THUMB = 10
    RIGHT_CLAVICLE = 11
    RIGHT_SHOULDER = 12
    RIGHT_ELBOW = 13
    RIGHT_WRIST = 14
    RIGHT_HAND = 15
    RIGHT_HANDTIP = 16
    RIGHT_THUMB = 17

class Body34Joint(Enum):
    """ZED SDK BODY_34 joint indices"""
    PELVIS = 0
    NAVAL_SPINE = 1
    CHEST_SPINE = 2
    NECK = 3
    LEFT_CLAVICLE = 4
    LEFT_SHOULDER = 5
    LEFT_ELBOW = 6
    LEFT_WRIST = 7
    LEFT_HAND = 8
    LEFT_HANDTIP = 9
    LEFT_THUMB = 10
    RIGHT_CLAVICLE = 11
    RIGHT_SHOULDER = 12
    RIGHT_ELBOW = 13
    RIGHT_WRIST = 14
    RIGHT_HAND = 15
    RIGHT_HANDTIP = 16
    RIGHT_THUMB = 17
    LEFT_HIP = 18
    LEFT_KNEE = 19
    LEFT_ANKLE = 20
    LEFT_FOOT = 21
    RIGHT_HIP = 22
    RIGHT_KNEE = 23
    RIGHT_ANKLE = 24
    RIGHT_FOOT = 25
    HEAD = 26
    NOSE = 27
    LEFT_EYE = 28
    LEFT_EAR = 29
    RIGHT_EYE = 30
    RIGHT_EAR = 31
    LEFT_HEEL = 32
    RIGHT_HEEL = 33

class SkeletonAngle(Enum):
    """Enum for all skeleton angles"""
    # Body orientation
    BODY_FORWARD_DIRECTION = "body_forward_direction"
    BODY_RIGHT_DIRECTION = "body_right_direction"
    BODY_UP_DIRECTION = "body_up_direction"
    
    # Head angles (relative to body orientation)
    HEAD_YAW = "head_yaw"  # Left/right rotation relative to body forward (0° = forward)
    HEAD_PITCH = "head_pitch"  # Up/down angle (0° = horizontal, +up, -down)
    HEAD_ROLL = "head_roll"  # Left/right tilt (ear to shoulder, 0° = level)
    
    # Keep old names for backward compatibility
    HEAD_ROTATION = "head_yaw"  # Alias
    HEAD_LOOK_UP_DOWN = "head_pitch"  # Alias
    NECK_TILT = "head_roll"  # Alias
    
    # Arm angles - joint angles
    LEFT_ARM_ANGLE = "left_arm"  # Clavicle -> Shoulder -> Elbow
    RIGHT_ARM_ANGLE = "right_arm"  # Clavicle -> Shoulder -> Elbow
    LEFT_ELBOW_ANGLE = "left_elbow"  # Shoulder -> Elbow -> Wrist
    RIGHT_ELBOW_ANGLE = "right_elbow"  # Shoulder -> Elbow -> Wrist
    
    # Arm 3D orientation (azimuth/elevation)
    LEFT_ARM_AZIMUTH = "left_arm_azimuth"  # Left/right angle relative to body forward
    LEFT_ARM_ELEVATION = "left_arm_elevation"  # Up/down angle from horizontal
    RIGHT_ARM_AZIMUTH = "right_arm_azimuth"
    RIGHT_ARM_ELEVATION = "right_arm_elevation"
    LEFT_ARM_DIRECTION_VECTOR = "left_arm_direction"  # 3D direction in body space
    RIGHT_ARM_DIRECTION_VECTOR = "right_arm_direction"
    
    # Leg angles - joint angles
    LEFT_HIP_ANGLE = "left_hip"  # Naval -> Hip -> Knee
    RIGHT_HIP_ANGLE = "right_hip"  # Naval -> Hip -> Knee
    LEFT_KNEE_ANGLE = "left_knee"  # Hip -> Knee -> Ankle
    RIGHT_KNEE_ANGLE = "right_knee"  # Hip -> Knee -> Ankle
    
    # Leg 3D orientation (azimuth/elevation)
    LEFT_LEG_AZIMUTH = "left_leg_azimuth"  # Left/right angle relative to body forward
    LEFT_LEG_ELEVATION = "left_leg_elevation"  # Up/down angle from vertical
    RIGHT_LEG_AZIMUTH = "right_leg_azimuth"
    RIGHT_LEG_ELEVATION = "right_leg_elevation"
    LEFT_LEG_DIRECTION_VECTOR = "left_leg_direction"  # 3D direction in body space
    RIGHT_LEG_DIRECTION_VECTOR = "right_leg_direction"

class UniversalJoint(Enum):
    """Universal joint identifiers that work with both BODY_18 and BODY_34 formats
    
    These map to the common joints available in both skeleton formats.
    For BODY_34-specific joints, use Body34Joint directly.
    """
    # Head and face
    NOSE = 0
    NECK = 1
    RIGHT_EYE = 2
    LEFT_EYE = 3
    RIGHT_EAR = 4
    LEFT_EAR = 5
    
    # Upper body
    RIGHT_SHOULDER = 11
    LEFT_SHOULDER = 12
    RIGHT_ELBOW = 13
    LEFT_ELBOW = 14
    RIGHT_WRIST = 15
    LEFT_WRIST = 16
    
    # Lower body
    RIGHT_HIP = 22
    LEFT_HIP = 23
    RIGHT_KNEE = 24
    LEFT_KNEE = 25
    RIGHT_ANKLE = 26
    LEFT_ANKLE = 27
    
    # BODY_34 specific joints (will return None for BODY_18)
    PELVIS = 28
    NAVAL_SPINE = 29
    CHEST_SPINE = 30
    LEFT_CLAVICLE = 31
    RIGHT_CLAVICLE = 32
    LEFT_HEEL = 33
    RIGHT_HEEL = 34
    
    def to_body18_index(self) -> Optional[int]:
        """Map to BODY_18 joint index (None if not available)"""
        mapping = {
            UniversalJoint.NOSE: 0,
            UniversalJoint.NECK: 1,
            UniversalJoint.RIGHT_SHOULDER: 2,
            UniversalJoint.RIGHT_ELBOW: 3,
            UniversalJoint.RIGHT_WRIST: 4,
            UniversalJoint.LEFT_SHOULDER: 5,
            UniversalJoint.LEFT_ELBOW: 6,
            UniversalJoint.LEFT_WRIST: 7,
            UniversalJoint.RIGHT_HIP: 8,
            UniversalJoint.RIGHT_KNEE: 9,
            UniversalJoint.RIGHT_ANKLE: 10,
            UniversalJoint.LEFT_HIP: 11,
            UniversalJoint.LEFT_KNEE: 12,
            UniversalJoint.LEFT_ANKLE: 13,
            UniversalJoint.RIGHT_EYE: 14,
            UniversalJoint.LEFT_EYE: 15,
            UniversalJoint.RIGHT_EAR: 16,
            UniversalJoint.LEFT_EAR: 17,
        }
        return mapping.get(self, None)
    
    def to_body34_index(self) -> Optional[int]:
        """Map to BODY_34 joint index (None if not available)"""
        mapping = {
            UniversalJoint.PELVIS: 0,
            UniversalJoint.NAVAL_SPINE: 1,
            UniversalJoint.CHEST_SPINE: 2,
            UniversalJoint.NECK: 3,
            UniversalJoint.LEFT_CLAVICLE: 4,
            UniversalJoint.LEFT_SHOULDER: 5,
            UniversalJoint.LEFT_ELBOW: 6,
            UniversalJoint.LEFT_WRIST: 7,
            UniversalJoint.RIGHT_CLAVICLE: 11,
            UniversalJoint.RIGHT_SHOULDER: 12,
            UniversalJoint.RIGHT_ELBOW: 13,
            UniversalJoint.RIGHT_WRIST: 14,
            UniversalJoint.LEFT_HIP: 18,
            UniversalJoint.LEFT_KNEE: 19,
            UniversalJoint.LEFT_ANKLE: 20,
            UniversalJoint.LEFT_HEEL: 21,
            UniversalJoint.RIGHT_HIP: 22,
            UniversalJoint.RIGHT_KNEE: 23,
            UniversalJoint.RIGHT_ANKLE: 24,
            UniversalJoint.RIGHT_HEEL: 25,
            UniversalJoint.NOSE: 26,
            UniversalJoint.LEFT_EYE: 27,
            UniversalJoint.RIGHT_EYE: 28,
            UniversalJoint.LEFT_EAR: 29,
            UniversalJoint.RIGHT_EAR: 30,
        }
        return mapping.get(self, None)

