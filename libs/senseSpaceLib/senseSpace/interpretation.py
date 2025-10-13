"""
Pose interpretation utilities for SenseSpace.

This module provides human-readable interpretations of skeleton angles,
accounting for tracking limitations and providing semantic descriptions.
"""

from .enums import SkeletonAngle
import numpy as np


def interpret_pose_from_angles(angles: dict) -> str:
    """
    Interpret skeletal angles into a human-readable pose description
    
    Args:
        angles: Dictionary of skeletal angles from get_skeletal_angles()
        
    Returns:
        Formatted string describing the detected pose
    """
    descriptions = []
    numeric_measurements = []
    
    # Body orientation
    body_facing = angles.get(SkeletonAngle.BODY_FORWARD_DIRECTION)
    if body_facing is not None:
        numeric_measurements.append(f"• Body facing: [{body_facing[0]:.2f}, {body_facing[1]:.2f}, {body_facing[2]:.2f}]")
    
    # Head orientation - FIX THE THRESHOLD
    head_rotation = angles.get(SkeletonAngle.HEAD_ROTATION)
    if head_rotation is not None:
        # Use 10° threshold for better detection
        if head_rotation > 10:
            direction = "left"
        elif head_rotation < -10:
            direction = "right"
        else:
            direction = "forward"
        descriptions.append(f"Head facing {direction}")
        numeric_measurements.append(f"• Head rotation: {head_rotation:.0f}° ({direction})")
    
    # Neck tilt - DON'T add to descriptions, only to numeric
    neck_tilt = angles.get(SkeletonAngle.NECK_TILT)
    if neck_tilt is not None:
        numeric_measurements.append(f"• Neck tilt: {neck_tilt:.0f}°")
    
    # Head pitch (looking up/down) - USE ROTATION, NOT TILT
    head_pitch = angles.get(SkeletonAngle.HEAD_LOOK_UP_DOWN)
    if head_pitch is not None:
        numeric_measurements.append(f"• Head pitch: {head_pitch:.0f}°")
        # Determine looking direction based on ROTATION first, then pitch
        if head_rotation is not None and abs(head_rotation) > 15:
            # Head is turned left or right
            if head_rotation > 15:
                descriptions.append("Looking left")
            else:
                descriptions.append("Looking right")
        else:
            # Head is facing forward, check pitch for up/down
            if head_pitch > 15:
                descriptions.append("Looking up")
            elif head_pitch < -15:
                descriptions.append("Looking down")
            else:
                descriptions.append("Looking straight ahead")
    
    # Arms
    for side in ['LEFT', 'RIGHT']:
        arm_elevation = angles.get(SkeletonAngle[f'{side}_ARM_ELEVATION'])
        arm_azimuth = angles.get(SkeletonAngle[f'{side}_ARM_AZIMUTH'])
        elbow_angle = angles.get(SkeletonAngle[f'{side}_ELBOW_ANGLE'])
        
        if arm_elevation is not None:
            arm_desc = _describe_arm_position(arm_elevation, arm_azimuth, side.lower())
            descriptions.append(arm_desc)
            numeric_measurements.append(f"• {side.capitalize()} arm: elevation={arm_elevation:.0f}°, azimuth={arm_azimuth:.0f}°")
        
        if elbow_angle is not None:
            elbow_state = _describe_elbow(elbow_angle)
            if elbow_state:
                descriptions.append(f"{side.capitalize()} elbow {elbow_state} ({elbow_angle:.0f}°)")
                numeric_measurements.append(f"• {side.capitalize()} elbow: {elbow_angle:.0f}°")
    
    # Body position classification - REWRITE THIS SECTION
    left_hip = angles.get(SkeletonAngle.LEFT_HIP_ANGLE)
    right_hip = angles.get(SkeletonAngle.RIGHT_HIP_ANGLE)
    left_knee = angles.get(SkeletonAngle.LEFT_KNEE_ANGLE)
    right_knee = angles.get(SkeletonAngle.RIGHT_KNEE_ANGLE)
    
    avg_hip = (left_hip + right_hip) / 2 if left_hip and right_hip else None
    avg_knee = (left_knee + right_knee) / 2 if left_knee and right_knee else None
    
    body_position = "UNKNOWN"
    if avg_hip and avg_knee:
        # Standing: knees > 140° (even if hips slightly bent from posture)
        if avg_knee > 140:
            body_position = "STANDING"
        # Sitting: hips < 140° AND knees < 120° (both significantly bent)
        elif avg_hip < 140 and avg_knee < 120:
            body_position = "SITTING"
        # Laying: hips < 100° AND knees < 90° (deeply bent, horizontal)
        elif avg_hip < 100 and avg_knee < 90:
            body_position = "LAYING"
        else:
            # In-between = standing with bent knees
            body_position = "STANDING"
    
    descriptions.append(f"Body position: {body_position}")
    
    # Legs - ADD LEG DESCRIPTIONS HERE
    for side in ['LEFT', 'RIGHT']:
        leg_elevation = angles.get(SkeletonAngle[f'{side}_LEG_ELEVATION'])
        leg_azimuth = angles.get(SkeletonAngle[f'{side}_LEG_AZIMUTH'])
        hip_angle = angles.get(SkeletonAngle[f'{side}_HIP_ANGLE'])
        knee_angle = angles.get(SkeletonAngle[f'{side}_KNEE_ANGLE'])
        
        # Add leg orientation to descriptions if available
        if leg_elevation is not None and leg_azimuth is not None:
            leg_desc = _describe_leg_position(leg_elevation, leg_azimuth, side.lower())
            if leg_desc:
                descriptions.append(leg_desc)
            numeric_measurements.append(f"• {side.capitalize()} leg: elevation={leg_elevation:.0f}°, azimuth={leg_azimuth:.0f}°")
        
        if hip_angle is not None:
            hip_state = _describe_hip(hip_angle)
            if hip_state:
                descriptions.append(f"{side.capitalize()} hip {hip_state} ({hip_angle:.0f}°)")
                numeric_measurements.append(f"• {side.capitalize()} hip: {hip_angle:.0f}°")
        
        if knee_angle is not None:
            knee_state = _describe_knee(knee_angle)
            if knee_state:
                descriptions.append(f"{side.capitalize()} knee {knee_state} ({knee_angle:.0f}°)")
                numeric_measurements.append(f"• {side.capitalize()} knee: {knee_angle:.0f}°")
    
    # Hip and knee averages
    if avg_hip:
        numeric_measurements.append(f"• Avg hip angle: {avg_hip:.0f}°")
    if avg_knee:
        numeric_measurements.append(f"• Avg knee angle: {avg_knee:.0f}°")
    
    # Build final output
    output = "Detected pose features:\n\n"
    for desc in descriptions:
        output += f"• {desc}\n"
    
    if numeric_measurements:
        output += "\nNumeric measurements:\n"
        for measurement in numeric_measurements:
            output += f"{measurement}\n"
    
    return output.strip()


def _describe_leg_position(elevation: float, azimuth: float, side: str) -> str:
    """Describe leg position based on elevation and azimuth"""
    desc_parts = [f"{side.capitalize()} leg"]
    
    # Elevation description (note: elevation is negative when leg points down)
    if elevation > -30:  # Leg raised or horizontal
        desc_parts.append("raised")
    elif elevation > -60:  # Slightly lowered
        desc_parts.append("slightly lowered")
    else:  # Normal standing/sitting position
        return None  # Don't describe normal leg position
    
    # Azimuth description (lateral position)
    if abs(azimuth) > 20:
        if side == 'left':
            if azimuth < 0:
                desc_parts.append("to the left")
            else:
                desc_parts.append("crossing right")
        else:  # right leg
            if azimuth > 0:
                desc_parts.append("to the right")
            else:
                desc_parts.append("crossing left")
    
    # Only return if there's something interesting to say
    if len(desc_parts) > 2:
        return " ".join(desc_parts)
    return None


def _describe_arm_position(elevation: float, azimuth: float, side: str) -> str:
    """Describe arm position based on elevation and azimuth"""
    desc_parts = [f"{side.capitalize()} arm"]
    
    # Elevation description (vertical angle)
    if elevation > 60:
        desc_parts.append("raised overhead")
    elif elevation > 20:
        desc_parts.append("raised")
    elif elevation > -20:
        desc_parts.append("extended horizontally")
    elif elevation > -60:
        desc_parts.append("lowered")
    else:
        desc_parts.append("down")
    
    # Azimuth description (horizontal direction)
    # For left arm: negative azimuth = to the left, positive = to the right
    # For right arm: positive azimuth = to the right, negative = to the left
    if side == 'left':
        if azimuth < -45:
            desc_parts.append("to the left")
        elif azimuth > 45:
            desc_parts.append("to the right")
    else:  # right arm
        if azimuth > 45:
            desc_parts.append("to the right")
        elif azimuth < -45:
            desc_parts.append("to the left")
    
    return " ".join(desc_parts)


def _describe_elbow(angle: float) -> str:
    """Describe elbow bend state"""
    if angle < 90:
        return "deeply bent"
    elif angle < 140:
        return "bent"
    elif angle > 170:
        return None  # Straight, no need to mention
    else:
        return "slightly bent"


def _describe_hip(angle: float) -> str:
    """Describe hip bend state"""
    if angle < 100:
        return "deeply bent"
    elif angle < 140:
        return "bent"
    elif angle > 170:
        return None  # Straight
    else:
        return "slightly bent"


def _describe_knee(angle: float) -> str:
    """Describe knee bend state"""
    if angle < 90:
        return "deeply bent"
    elif angle < 140:
        return "bent"
    elif angle > 170:
        return None  # Straight
    else:
        return "slightly bent"