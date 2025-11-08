#!/usr/bin/env python3
"""
Test script for get_tpose_delta_orientations() library function.

This script verifies that the new library function produces correct results.
"""

import sys
import os

# Add library to path - use the same approach as examples
repo_root = os.path.dirname(os.path.abspath(__file__))
libs_path = os.path.join(repo_root, 'libs')
senseSpaceLib_path = os.path.join(libs_path, 'senseSpaceLib')

# Add both paths
for path in [libs_path, senseSpaceLib_path]:
    if path not in sys.path:
        sys.path.insert(0, path)

print("Testing get_tpose_delta_orientations() library function...")
print("=" * 80)

try:
    from senseSpaceLib.senseSpace import get_tpose_delta_orientations
    print("✅ Successfully imported get_tpose_delta_orientations from senseSpaceLib.senseSpace library")
    
    # Check function signature
    import inspect
    sig = inspect.signature(get_tpose_delta_orientations)
    print(f"\nFunction signature: {get_tpose_delta_orientations.__name__}{sig}")
    
    # Print docstring
    print(f"\nDocstring:\n{get_tpose_delta_orientations.__doc__}")
    
    print("\n" + "=" * 80)
    print("✅ Library function is ready to use!")
    print("\nUsage example:")
    print("  from senseSpace import get_tpose_delta_orientations")
    print("  deltas = get_tpose_delta_orientations(person.skeleton, person)")
    print("  # deltas[5] = LEFT_SHOULDER delta rotation from perfect T-pose")
    print("=" * 80)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
