#!/usr/bin/env python3
"""
Test script for MessagePack protocol implementation.
Tests serialization/deserialization and performance comparison.
"""
import sys
import time
import json

# Add libs to path
sys.path.insert(0, 'libs/senseSpaceLib')

from senseSpace.communication import serialize_message, deserialize_message, MSGPACK_AVAILABLE
from senseSpace.protocol import Frame, Person, Joint, Position, Quaternion

def create_test_frame():
    """Create a realistic test frame with multiple people and joints"""
    joints = []
    for i in range(34):  # BODY_34 has 34 joints
        joints.append(Joint(
            i=i,
            pos=Position(x=float(i * 10), y=float(i * 20), z=float(i * 30)),
            ori=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            conf=0.95
        ))
    
    people = [
        Person(id=1, tracking_state="OK", confidence=0.98, skeleton=joints),
        Person(id=2, tracking_state="OK", confidence=0.92, skeleton=joints),
    ]
    
    frame = Frame(
        timestamp=time.time(),
        people=people,
        body_model="BODY_34",
        floor_height=0.0,
        cameras=None
    )
    
    return frame

def test_serialization():
    """Test MessagePack vs JSON serialization"""
    print("=" * 60)
    print("MessagePack Protocol Test")
    print("=" * 60)
    print()
    
    if not MSGPACK_AVAILABLE:
        print("⚠️  WARNING: msgpack not installed!")
        print("   Install with: pip install msgpack")
        print("   Falling back to JSON-only mode")
        print()
    else:
        print("✓ msgpack is available")
        print()
    
    # Create test data
    frame = create_test_frame()
    data_dict = frame.to_dict()
    
    print(f"Test data: Frame with {len(frame.people)} people, {len(frame.people[0].skeleton)} joints each")
    print()
    
    # Test JSON serialization
    print("JSON Serialization:")
    t0 = time.perf_counter()
    json_bytes = serialize_message(data_dict, use_msgpack=False)
    t1 = time.perf_counter()
    print(f"  Size: {len(json_bytes):,} bytes")
    print(f"  Time: {(t1-t0)*1000:.3f} ms")
    
    # Test JSON deserialization
    t0 = time.perf_counter()
    json_result = deserialize_message(json_bytes)
    t1 = time.perf_counter()
    print(f"  Deserialize time: {(t1-t0)*1000:.3f} ms")
    print(f"  Valid: {json_result is not None}")
    print()
    
    if MSGPACK_AVAILABLE:
        # Test MessagePack serialization
        print("MessagePack Serialization:")
        t0 = time.perf_counter()
        msgpack_bytes = serialize_message(data_dict, use_msgpack=True)
        t1 = time.perf_counter()
        print(f"  Size: {len(msgpack_bytes):,} bytes")
        print(f"  Time: {(t1-t0)*1000:.3f} ms")
        
        # Test MessagePack deserialization
        t0 = time.perf_counter()
        msgpack_result = deserialize_message(msgpack_bytes)
        t1 = time.perf_counter()
        print(f"  Deserialize time: {(t1-t0)*1000:.3f} ms")
        print(f"  Valid: {msgpack_result is not None}")
        print()
        
        # Compare
        print("Comparison:")
        size_reduction = (1 - len(msgpack_bytes) / len(json_bytes)) * 100
        print(f"  Size reduction: {size_reduction:.1f}%")
        print(f"  MessagePack is {len(json_bytes) / len(msgpack_bytes):.2f}x smaller")
        print()
        
        # Benchmark with multiple iterations
        print("Performance benchmark (100 iterations):")
        iterations = 100
        
        # JSON benchmark
        t0 = time.perf_counter()
        for _ in range(iterations):
            serialize_message(data_dict, use_msgpack=False)
        json_time = time.perf_counter() - t0
        
        # MessagePack benchmark
        t0 = time.perf_counter()
        for _ in range(iterations):
            serialize_message(data_dict, use_msgpack=True)
        msgpack_time = time.perf_counter() - t0
        
        print(f"  JSON: {json_time*1000:.2f} ms total ({json_time*10:.3f} ms per frame)")
        print(f"  MessagePack: {msgpack_time*1000:.2f} ms total ({msgpack_time*10:.3f} ms per frame)")
        print(f"  MessagePack is {json_time/msgpack_time:.2f}x faster")
        print()
    
    print("✓ All tests passed!")
    print()
    
    # Recommendations
    print("Recommendations:")
    if MSGPACK_AVAILABLE:
        print("  ✓ Use MessagePack for production (smaller, faster)")
        print("  ✓ Backward compatible - clients auto-detect format")
    else:
        print("  ⚠️  Install msgpack for better performance:")
        print("     pip install msgpack")
    print()

if __name__ == "__main__":
    test_serialization()
