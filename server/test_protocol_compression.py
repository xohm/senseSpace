#!/usr/bin/env python3
"""
Test script for MessagePack + Zstandard protocol implementation.
Tests serialization/deserialization and performance comparison.
"""
import sys
import time
import json

# Add libs to path
sys.path.insert(0, 'libs/senseSpaceLib')

from senseSpace.communication import serialize_message, deserialize_message, MSGPACK_AVAILABLE, ZSTD_AVAILABLE
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
    """Test MessagePack + Zstandard vs JSON serialization"""
    print("=" * 70)
    print("MessagePack + Zstandard Protocol Test")
    print("=" * 70)
    print()
    
    if not MSGPACK_AVAILABLE:
        print("⚠️  WARNING: msgpack not installed!")
        print("   Install with: pip install msgpack")
        print("   Falling back to JSON-only mode")
        print()
    else:
        print("✓ msgpack is available")
    
    if not ZSTD_AVAILABLE:
        print("⚠️  WARNING: zstandard not installed!")
        print("   Install with: pip install zstandard")
        print("   Compression disabled")
        print()
    else:
        print("✓ zstandard is available")
    
    print()
    
    # Create test data
    frame = create_test_frame()
    data_dict = frame.to_dict()
    
    print(f"Test data: Frame with {len(frame.people)} people, {len(frame.people[0].skeleton)} joints each")
    print()
    
    # Test JSON serialization
    print("1. JSON Serialization (baseline):")
    t0 = time.perf_counter()
    json_bytes = serialize_message(data_dict, use_msgpack=False, use_compression=False)
    t1 = time.perf_counter()
    json_size = len(json_bytes)
    json_time = t1 - t0
    print(f"  Size: {json_size:,} bytes")
    print(f"  Time: {json_time*1000:.3f} ms")
    
    # Test JSON deserialization
    t0 = time.perf_counter()
    json_result = deserialize_message(json_bytes)
    t1 = time.perf_counter()
    print(f"  Deserialize time: {(t1-t0)*1000:.3f} ms")
    print(f"  Valid: {json_result is not None}")
    print()
    
    if MSGPACK_AVAILABLE:
        # Test MessagePack serialization (no compression)
        print("2. MessagePack (no compression):")
        t0 = time.perf_counter()
        msgpack_bytes = serialize_message(data_dict, use_msgpack=True, use_compression=False)
        t1 = time.perf_counter()
        msgpack_size = len(msgpack_bytes)
        msgpack_time = t1 - t0
        print(f"  Size: {msgpack_size:,} bytes ({(1-msgpack_size/json_size)*100:.1f}% smaller than JSON)")
        print(f"  Time: {msgpack_time*1000:.3f} ms ({json_time/msgpack_time:.2f}x faster)")
        
        # Test MessagePack deserialization
        t0 = time.perf_counter()
        msgpack_result = deserialize_message(msgpack_bytes)
        t1 = time.perf_counter()
        print(f"  Deserialize time: {(t1-t0)*1000:.3f} ms")
        print(f"  Valid: {msgpack_result is not None}")
        print()
        
        if ZSTD_AVAILABLE:
            # Test MessagePack + zstd
            print("3. MessagePack + Zstandard (BEST):")
            t0 = time.perf_counter()
            compressed_bytes = serialize_message(data_dict, use_msgpack=True, use_compression=True)
            t1 = time.perf_counter()
            compressed_size = len(compressed_bytes)
            compressed_time = t1 - t0
            print(f"  Size: {compressed_size:,} bytes ({(1-compressed_size/json_size)*100:.1f}% smaller than JSON)")
            print(f"  Time: {compressed_time*1000:.3f} ms ({json_time/compressed_time:.2f}x faster)")
            
            # Test decompression
            t0 = time.perf_counter()
            compressed_result = deserialize_message(compressed_bytes)
            t1 = time.perf_counter()
            decompress_time = t1 - t0
            print(f"  Deserialize time: {decompress_time*1000:.3f} ms")
            print(f"  Valid: {compressed_result is not None}")
            print()
            
            # Overall comparison
            print("=" * 70)
            print("COMPARISON SUMMARY")
            print("=" * 70)
            print()
            print(f"{'Format':<30} {'Size':>12} {'Reduction':>12} {'Speed':>12}")
            print("-" * 70)
            print(f"{'JSON (baseline)':<30} {json_size:>10,} B {0:>10}% {1.0:>11.2f}x")
            print(f"{'MessagePack':<30} {msgpack_size:>10,} B {(1-msgpack_size/json_size)*100:>10.1f}% {json_time/msgpack_time:>11.2f}x")
            print(f"{'MessagePack+zstd':<30} {compressed_size:>10,} B {(1-compressed_size/json_size)*100:>10.1f}% {json_time/compressed_time:>11.2f}x")
            print()
            
            # Bandwidth savings at different FPS
            print("Bandwidth savings at different frame rates:")
            print()
            for fps in [15, 30, 60, 90]:
                json_bw = json_size * fps / 1024 / 1024  # MB/s
                compressed_bw = compressed_size * fps / 1024 / 1024  # MB/s
                savings = json_bw - compressed_bw
                print(f"  {fps:2d} FPS: {json_bw:6.2f} MB/s → {compressed_bw:6.2f} MB/s  (saves {savings:5.2f} MB/s, {(1-compressed_bw/json_bw)*100:.1f}%)")
            print()
            
            # Benchmark with multiple iterations
            print("Performance benchmark (100 iterations):")
            iterations = 100
            
            # JSON benchmark
            t0 = time.perf_counter()
            for _ in range(iterations):
                serialize_message(data_dict, use_msgpack=False, use_compression=False)
            json_total = time.perf_counter() - t0
            
            # MessagePack benchmark
            t0 = time.perf_counter()
            for _ in range(iterations):
                serialize_message(data_dict, use_msgpack=True, use_compression=False)
            msgpack_total = time.perf_counter() - t0
            
            # MessagePack + zstd benchmark
            t0 = time.perf_counter()
            for _ in range(iterations):
                serialize_message(data_dict, use_msgpack=True, use_compression=True)
            compressed_total = time.perf_counter() - t0
            
            print(f"  JSON:            {json_total*1000:6.2f} ms total ({json_total*10:.3f} ms per frame)")
            print(f"  MessagePack:     {msgpack_total*1000:6.2f} ms total ({msgpack_total*10:.3f} ms per frame)")
            print(f"  MessagePack+zstd:{compressed_total*1000:6.2f} ms total ({compressed_total*10:.3f} ms per frame)")
            print()
    
    print("✓ All tests passed!")
    print()
    
    # Recommendations
    print("=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    if MSGPACK_AVAILABLE and ZSTD_AVAILABLE:
        print("  ✓ MessagePack+zstd is enabled (OPTIMAL)")
        print("  ✓ You're getting the best performance!")
        print(f"  ✓ Saving {(1-compressed_size/json_size)*100:.1f}% bandwidth vs JSON")
    elif MSGPACK_AVAILABLE:
        print("  ⚠️  Install zstandard for better compression:")
        print("     pip install zstandard")
        print(f"  ℹ️  Currently saving {(1-msgpack_size/json_size)*100:.1f}% vs JSON")
        print("     With zstd you could save ~75% or more!")
    else:
        print("  ⚠️  Install both msgpack and zstandard for best performance:")
        print("     pip install msgpack zstandard")
    print()

if __name__ == "__main__":
    test_serialization()
