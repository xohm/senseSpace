"""
Pose interpretation utilities for SenseSpace.

This module provides human-readable interpretations of skeleton angles,
accounting for tracking limitations and providing semantic descriptions.
"""

import numpy as np
from typing import Dict, Union, List
from .enums import SkeletonAngle


def interpret_pose_from_angles(angles: Dict[SkeletonAngle, Union[float, np.ndarray, None]]) -> str:
    """
    Interpret pose features from skeleton angles into human-readable descriptions.
    
    This function accounts for skeleton tracking limitations (e.g., limited head rotation range)
    and provides semantic descriptions suitable for LLM processing or human understanding.
    
    Args:
        angles: dict from Person.get_skeletal_angles()
        
    Returns:
        str: Human-readable pose description with features and measurements
    """
    if not angles:
        return "No skeleton data available"

    desc = []
    features = []

    # -----------------------------------------------------------
    # Body orientation
    # -----------------------------------------------------------
    if SkeletonAngle.BODY_FORWARD_DIRECTION in angles:
        forward = angles[SkeletonAngle.BODY_FORWARD_DIRECTION]
        features.append(f"Body facing: [{forward[0]:.2f}, {forward[1]:.2f}, {forward[2]:.2f}]")

    # -----------------------------------------------------------
    # Head orientation
    # NOTE: Skeleton tracking has limited head rotation range (~±45° max)
    # Convention: Positive = LEFT (counterclockwise from above), Negative = RIGHT (clockwise)
    # -----------------------------------------------------------
    head_rotation = angles.get(SkeletonAngle.HEAD_ROTATION)
    if head_rotation is not None:
        # Skeleton tracking typically maxes out around ±30-45°
        # We scale interpretation accordingly
        if abs(head_rotation) < 10:
            desc.append("Head facing forward")
            direction = "forward"
        elif head_rotation > 30:
            # At tracking limit - likely turning further left
            desc.append(f"Head turned strongly left ({head_rotation:.0f}°, likely 60-90°)")
            direction = "strongly left (at tracking limit)"
        elif head_rotation > 15:
            desc.append(f"Head turned left ({head_rotation:.0f}°)")
            direction = "left"
        elif head_rotation < -30:
            # At tracking limit - likely turning further right
            desc.append(f"Head turned strongly right ({abs(head_rotation):.0f}°, likely 60-90°)")
            direction = "strongly right (at tracking limit)"
        elif head_rotation < -15:
            desc.append(f"Head turned right ({abs(head_rotation):.0f}°)")
            direction = "right"
        else:
            desc.append("Head facing forward")
            direction = "forward"
        
        features.append(f"Head rotation: {head_rotation:.0f}° ({direction})")

    neck_tilt = angles.get(SkeletonAngle.NECK_TILT)
    if neck_tilt is not None:
        if abs(neck_tilt) < 10:
            desc.append("Head level")
        else:
            side = "left" if neck_tilt < 0 else "right"
            desc.append(f"Head tilted {side} ({abs(neck_tilt):.0f}°)")
        features.append(f"Neck tilt: {neck_tilt:.0f}°")

    head_updown = angles.get(SkeletonAngle.HEAD_LOOK_UP_DOWN)
    if head_updown is not None:
        if head_updown < -15:
            desc.append("Looking down")
        elif head_updown > 15:
            desc.append("Looking up")
        else:
            desc.append("Looking straight ahead")
        features.append(f"Head pitch: {head_updown:.0f}°")

    # -----------------------------------------------------------
    # Arms - 3D orientation
    # -----------------------------------------------------------
    for side, az_key, el_key, elbow_key in [
        ("Left", SkeletonAngle.LEFT_ARM_AZIMUTH, SkeletonAngle.LEFT_ARM_ELEVATION, SkeletonAngle.LEFT_ELBOW_ANGLE),
        ("Right", SkeletonAngle.RIGHT_ARM_AZIMUTH, SkeletonAngle.RIGHT_ARM_ELEVATION, SkeletonAngle.RIGHT_ELBOW_ANGLE)
    ]:
        azimuth = angles.get(az_key)
        elevation = angles.get(el_key)
        elbow_angle = angles.get(elbow_key)
        
        if azimuth is not None and elevation is not None:
            # Describe direction
            if elevation > 60:
                elev_desc = "raised overhead"
            elif elevation > 20:
                elev_desc = "raised"
            elif elevation > -20:
                elev_desc = "extended horizontally"
            elif elevation > -60:
                elev_desc = "lowered"
            else:
                elev_desc = "down"
            
            if abs(azimuth) < 20:
                az_desc = "forward"
            elif azimuth > 60:
                az_desc = "to the right"
            elif azimuth > 20:
                az_desc = "diagonally right"
            elif azimuth < -60:
                az_desc = "to the left"
            elif azimuth < -20:
                az_desc = "diagonally left"
            else:
                az_desc = ""
            
            desc.append(f"{side} arm {elev_desc} {az_desc}".strip())
            features.append(f"{side} arm: elevation={elevation:.0f}°, azimuth={azimuth:.0f}°")
        
        if elbow_angle is not None:
            if elbow_angle < 90:
                desc.append(f"{side} elbow tightly bent ({elbow_angle:.0f}°)")
            elif elbow_angle < 140:
                desc.append(f"{side} elbow bent ({elbow_angle:.0f}°)")
            features.append(f"{side} elbow: {elbow_angle:.0f}°")

    # -----------------------------------------------------------
    # Legs and body position
    # -----------------------------------------------------------
    left_hip = angles.get(SkeletonAngle.LEFT_HIP_ANGLE)
    right_hip = angles.get(SkeletonAngle.RIGHT_HIP_ANGLE)
    left_knee = angles.get(SkeletonAngle.LEFT_KNEE_ANGLE)
    right_knee = angles.get(SkeletonAngle.RIGHT_KNEE_ANGLE)

    hip_angles = [a for a in [left_hip, right_hip] if a is not None]
    knee_angles = [a for a in [left_knee, right_knee] if a is not None]

    if hip_angles and knee_angles:
        avg_hip = np.mean(hip_angles)
        avg_knee = np.mean(knee_angles)
        
        # Body position classification
        if avg_hip < 120 and avg_knee < 130:
            desc.append("Body position: SITTING")
        elif avg_hip < 140:
            desc.append("Body position: CROUCHING")
        elif avg_knee < 140:
            desc.append("Body position: STANDING with bent knees")
        else:
            desc.append("Body position: STANDING")
        
        features.append(f"Avg hip angle: {avg_hip:.0f}°")
        features.append(f"Avg knee angle: {avg_knee:.0f}°")

    # Individual leg descriptions
    for side, hip_key, knee_key in [
        ("Left", SkeletonAngle.LEFT_HIP_ANGLE, SkeletonAngle.LEFT_KNEE_ANGLE),
        ("Right", SkeletonAngle.RIGHT_HIP_ANGLE, SkeletonAngle.RIGHT_KNEE_ANGLE)
    ]:
        hip_angle = angles.get(hip_key)
        knee_angle = angles.get(knee_key)
        
        if hip_angle is not None:
            if hip_angle < 100:
                desc.append(f"{side} hip deeply bent ({hip_angle:.0f}°)")
            elif hip_angle < 140:
                desc.append(f"{side} hip bent ({hip_angle:.0f}°)")
            features.append(f"{side} hip: {hip_angle:.0f}°")
            
        if knee_angle is not None:
            if knee_angle < 100:
                desc.append(f"{side} knee deeply bent ({knee_angle:.0f}°)")
            elif knee_angle < 140:
                desc.append(f"{side} knee bent ({knee_angle:.0f}°)")
            features.append(f"{side} knee: {knee_angle:.0f}°")

    # -----------------------------------------------------------
    # Format output
    # -----------------------------------------------------------
    result = []
    result.append("Detected pose features:\n")
    result += ["• " + d for d in desc]
    result.append("\nNumeric measurements:")
    result += ["• " + f for f in features]
    return "\n".join(result)