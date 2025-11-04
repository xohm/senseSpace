#!/usr/bin/env python3
"""
Body Tracking Duplicate Filter for ZED SDK

Solves the problem where ZED Body Tracking outputs duplicate persons (same physical person
with multiple IDs) when tracking confidence drops. This happens because:
1. The tracker loses confidence and marks person as SEARCHING/OFF
2. The detector (DNN) re-detects the same person as a new ID
3. Both IDs exist temporarily → duplication

This filter merges/suppresses duplicates by:
- Comparing spatial proximity (position)
- Checking skeleton scale/height similarity
- Preferring older, more stable IDs
- Maintaining short-term memory for ID reassignment
"""

import time
import numpy as np
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
import pyzed.sl as sl


@dataclass
class PersonTrack:
    """Track history for a person"""
    person_id: int
    first_seen: float
    last_seen: float
    position_history: List[np.ndarray] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    tracking_state_history: List[sl.OBJECT_TRACKING_STATE] = field(default_factory=list)
    height: Optional[float] = None
    
    def update(self, position: np.ndarray, confidence: float, tracking_state: sl.OBJECT_TRACKING_STATE, height: float):
        """Update track with new observation"""
        self.last_seen = time.time()
        self.position_history.append(position)
        self.confidence_history.append(confidence)
        self.tracking_state_history.append(tracking_state)
        self.height = height
        
        # Keep only last 30 observations (~1 second at 30fps)
        max_history = 30
        if len(self.position_history) > max_history:
            self.position_history = self.position_history[-max_history:]
            self.confidence_history = self.confidence_history[-max_history:]
            self.tracking_state_history = self.tracking_state_history[-max_history:]
    
    @property
    def avg_confidence(self) -> float:
        """Average confidence over recent history"""
        if not self.confidence_history:
            return 0.0
        return sum(self.confidence_history[-10:]) / min(10, len(self.confidence_history))
    
    @property
    def is_stable(self) -> bool:
        """Check if track is stable (OK state, good confidence)"""
        if not self.tracking_state_history:
            return False
        recent_states = self.tracking_state_history[-5:]
        ok_count = sum(1 for s in recent_states if s == sl.OBJECT_TRACKING_STATE.OK)
        return ok_count >= 3 and self.avg_confidence > 50


class BodyTrackingFilter:
    """
    Filter for removing duplicate person detections from ZED Body Tracking.
    
    Detects when the same physical person has multiple IDs and merges them,
    keeping the older, more stable ID.
    """
    
    def __init__(self, 
                 duplicate_distance_threshold: float = 0.4,  # meters
                 height_similarity_threshold: float = 0.15,   # 15% difference
                 memory_duration: float = 2.0,                # seconds
                 confidence_diff_threshold: float = 30.0):
        """
        Initialize body tracking filter.
        
        Args:
            duplicate_distance_threshold: Max distance (m) to consider persons as duplicates
            height_similarity_threshold: Max relative height difference to merge (0.15 = 15%)
            memory_duration: How long to remember lost tracks for ID reassignment (seconds)
            confidence_diff_threshold: Min confidence difference to prefer one track over another
        """
        self.duplicate_distance_threshold = duplicate_distance_threshold
        self.height_similarity_threshold = height_similarity_threshold
        self.memory_duration = memory_duration
        self.confidence_diff_threshold = confidence_diff_threshold
        
        # Active tracks: currently visible persons
        self.active_tracks: Dict[int, PersonTrack] = {}
        
        # Lost tracks: recently lost, kept for ID reassignment
        self.lost_tracks: Dict[int, PersonTrack] = {}
        
        # Merged ID mapping: new_id -> old_id
        self.merged_ids: Dict[int, int] = {}
    
    def filter_bodies(self, bodies: sl.Bodies) -> sl.Bodies:
        """
        Filter body tracking results to remove duplicates.
        
        Args:
            bodies: Raw body tracking results from ZED SDK
            
        Returns:
            Filtered body tracking results with duplicates removed
        """
        current_time = time.time()
        
        # If no bodies, return as-is
        if not bodies.body_list:
            return bodies
        
        # Extract current detections
        current_persons = {}
        for body in bodies.body_list:
            person_id = body.id
            position = body.position  # sl.float3
            pos_array = np.array([position[0], position[1], position[2]])
            confidence = body.confidence
            tracking_state = body.tracking_state
            
            # Get height from keypoints if available
            height = self._calculate_height(body)
            
            current_persons[person_id] = {
                'body': body,
                'position': pos_array,
                'confidence': confidence,
                'tracking_state': tracking_state,
                'height': height
            }
        
        # Find and resolve duplicates
        ids_to_remove = set()
        id_remapping = {}  # new_id -> keep_id
        
        person_ids = list(current_persons.keys())
        
        # FIRST: Check if any current persons match recently lost tracks
        # This handles the case where ZED assigns a new ID to the same person
        for person_id in person_ids:
            if person_id in ids_to_remove:
                continue
            
            person = current_persons[person_id]
            
            # Check against recently lost tracks
            for lost_id, lost_track in self.lost_tracks.items():
                if lost_id == person_id:
                    continue  # Same ID
                
                # Calculate distance to lost track's last known position
                if lost_track.position_history:
                    last_pos = lost_track.position_history[-1]
                    distance = np.linalg.norm(person['position'] - last_pos)
                    
                    # If very close to where lost track was last seen
                    if distance < self.duplicate_distance_threshold:
                        # Height check if available
                        height_matches = True
                        if person['height'] and lost_track.height:
                            height_diff = abs(person['height'] - lost_track.height) / max(person['height'], lost_track.height)
                            height_matches = height_diff < self.height_similarity_threshold
                        
                        if height_matches:
                            # This new person is likely the lost track with a new ID
                            # Keep the OLD ID (lost_id), remove the NEW ID (person_id)
                            ids_to_remove.add(person_id)
                            id_remapping[person_id] = lost_id
                            print(f"[FILTER] Reassigned new ID {person_id} -> old ID {lost_id} "
                                  f"(distance: {distance:.2f}m, lost {current_time - lost_track.last_seen:.1f}s ago)")
                            break
        
        # SECOND: Check current persons against each other for duplicates
        for i, id1 in enumerate(person_ids):
            if id1 in ids_to_remove:
                continue
                
            p1 = current_persons[id1]
            
            for id2 in person_ids[i+1:]:
                if id2 in ids_to_remove:
                    continue
                    
                p2 = current_persons[id2]
                
                # Check if these are duplicates
                if self._are_duplicates(p1, p2):
                    # Determine which ID to keep
                    keep_id, remove_id = self._choose_better_id(
                        id1, p1, id2, p2
                    )
                    
                    ids_to_remove.add(remove_id)
                    id_remapping[remove_id] = keep_id
                    
                    print(f"[FILTER] Merged duplicate persons: {remove_id} -> {keep_id} "
                          f"(distance: {np.linalg.norm(p1['position'] - p2['position']):.2f}m)")
        
        # Update active tracks
        new_active_tracks = {}
        for person_id, person in current_persons.items():
            if person_id in ids_to_remove:
                continue
                
            # Check if this is a remapped ID
            actual_id = id_remapping.get(person_id, person_id)
            
            # Update or create track
            if actual_id in self.active_tracks:
                track = self.active_tracks[actual_id]
            elif actual_id in self.lost_tracks:
                # Reassign from lost tracks
                track = self.lost_tracks[actual_id]
                print(f"[FILTER] Reassigned lost track ID {actual_id}")
            else:
                # New person
                track = PersonTrack(
                    person_id=actual_id,
                    first_seen=current_time,
                    last_seen=current_time
                )
                print(f"[FILTER] New person tracked: ID {actual_id}")
            
            track.update(
                person['position'],
                person['confidence'],
                person['tracking_state'],
                person['height']
            )
            new_active_tracks[actual_id] = track
        
        # Move lost tracks
        for person_id, track in self.active_tracks.items():
            if person_id not in new_active_tracks:
                # Track lost
                self.lost_tracks[person_id] = track
                print(f"[FILTER] Lost track: ID {person_id}")
        
        # Clean up old lost tracks
        self.lost_tracks = {
            pid: track for pid, track in self.lost_tracks.items()
            if current_time - track.last_seen < self.memory_duration
        }
        
        self.active_tracks = new_active_tracks
        
        # If no duplicates found, return original bodies object
        if not ids_to_remove:
            return bodies
        
        # Create filtered bodies object only if we found duplicates
        filtered_bodies = sl.Bodies()
        filtered_bodies.is_new = bodies.is_new
        filtered_bodies.is_tracked = bodies.is_tracked
        
        # Add non-duplicate bodies to filtered list
        for person_id, person in current_persons.items():
            if person_id not in ids_to_remove:
                filtered_bodies.body_list.append(person['body'])
        
        print(f"[FILTER] Removed {len(ids_to_remove)} duplicate(s): {len(current_persons)} -> {len(filtered_bodies.body_list)} bodies")
        
        return filtered_bodies
    
    def _are_duplicates(self, p1: dict, p2: dict) -> bool:
        """Check if two persons are likely duplicates of the same physical person"""
        # Distance check
        distance = np.linalg.norm(p1['position'] - p2['position'])
        if distance > self.duplicate_distance_threshold:
            return False
        
        # Height similarity check (if available)
        if p1['height'] is not None and p2['height'] is not None:
            height_diff = abs(p1['height'] - p2['height']) / max(p1['height'], p2['height'])
            if height_diff > self.height_similarity_threshold:
                return False
        
        # At least one should have low confidence or non-OK tracking state
        both_stable = (
            p1['tracking_state'] == sl.OBJECT_TRACKING_STATE.OK and 
            p2['tracking_state'] == sl.OBJECT_TRACKING_STATE.OK and
            p1['confidence'] > 70 and p2['confidence'] > 70
        )
        
        if both_stable:
            # If both are very confident, require very close proximity
            return distance < 0.2
        
        return True
    
    def _choose_better_id(self, id1: int, p1: dict, id2: int, p2: dict) -> Tuple[int, int]:
        """
        Choose which person ID to keep when merging duplicates.
        
        Returns:
            (keep_id, remove_id)
        """
        # Prefer older track (in active tracks)
        track1_age = self.active_tracks.get(id1)
        track2_age = self.active_tracks.get(id2)
        
        if track1_age and track2_age:
            # Both are known - prefer older
            if track1_age.first_seen < track2_age.first_seen:
                return (id1, id2)
            else:
                return (id2, id1)
        elif track1_age:
            return (id1, id2)
        elif track2_age:
            return (id2, id1)
        
        # Prefer better tracking state
        state_priority = {
            sl.OBJECT_TRACKING_STATE.OK: 3,
            sl.OBJECT_TRACKING_STATE.SEARCHING: 2,
            sl.OBJECT_TRACKING_STATE.OFF: 1,
        }
        
        priority1 = state_priority.get(p1['tracking_state'], 0)
        priority2 = state_priority.get(p2['tracking_state'], 0)
        
        if priority1 > priority2:
            return (id1, id2)
        elif priority2 > priority1:
            return (id2, id1)
        
        # Prefer higher confidence
        if p1['confidence'] > p2['confidence'] + self.confidence_diff_threshold:
            return (id1, id2)
        elif p2['confidence'] > p1['confidence'] + self.confidence_diff_threshold:
            return (id2, id1)
        
        # Default: keep lower ID (probably older)
        return (id1, id2) if id1 < id2 else (id2, id1)
    
    def _calculate_height(self, body: sl.BodyData) -> Optional[float]:
        """Calculate person height from skeleton keypoints"""
        try:
            keypoints = body.keypoint
            
            # Get head and foot positions
            head_idx = sl.BODY_PARTS.HEAD.value
            left_ankle_idx = sl.BODY_PARTS.LEFT_ANKLE.value
            right_ankle_idx = sl.BODY_PARTS.RIGHT_ANKLE.value
            
            head_pos = keypoints[head_idx]
            left_ankle = keypoints[left_ankle_idx]
            right_ankle = keypoints[right_ankle_idx]
            
            # Use average of both ankles
            ankle_y = (left_ankle[1] + right_ankle[1]) / 2
            head_y = head_pos[1]
            
            height = abs(head_y - ankle_y)
            
            # Sanity check (typical human height: 1.5 - 2.1m)
            if 1.3 < height < 2.3:
                return height
            
        except Exception:
            pass
        
        return None
    
    def get_stats(self) -> dict:
        """Get filter statistics"""
        return {
            'active_tracks': len(self.active_tracks),
            'lost_tracks': len(self.lost_tracks),
            'total_tracks': len(self.active_tracks) + len(self.lost_tracks)
        }
    
    def reset(self):
        """Reset filter state"""
        self.active_tracks.clear()
        self.lost_tracks.clear()
        self.merged_ids.clear()
        print("[FILTER] Reset body tracking filter")
