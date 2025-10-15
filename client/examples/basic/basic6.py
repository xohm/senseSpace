# -----------------------------------------------------------------------------
# Sense Space
# -----------------------------------------------------------------------------
# Example: Python Basics - Main Function and Program Structure
# -----------------------------------------------------------------------------
# IAD, Zurich University of the Arts / zhdk.ch
# Max Rheiner
# -----------------------------------------------------------------------------

# ============================================================================
# IMPORT MODULES
# ============================================================================

import sys
import time

# ============================================================================
# FUNCTIONS
# ============================================================================

def greet_user(name):
    """Greet the user"""
    print(f"Hello, {name}!")
    print("Welcome to the program.\n")

def calculate_sum(numbers):
    """Calculate sum of numbers"""
    total = 0
    for num in numbers:
        total += num
    return total

def calculate_average(numbers):
    """Calculate average of numbers"""
    if len(numbers) == 0:
        return 0
    return calculate_sum(numbers) / len(numbers)

def process_data(data):
    """Process some data"""
    print("Processing data...")
    
    # Calculate statistics
    total = calculate_sum(data)
    avg = calculate_average(data)
    minimum = min(data)
    maximum = max(data)
    
    # Print results
    print(f"  Total: {total}")
    print(f"  Average: {avg:.2f}")
    print(f"  Min: {minimum}")
    print(f"  Max: {maximum}")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main program entry point"""
    
    # Print header
    print("=" * 60)
    print("Python Program Example")
    print("=" * 60)
    print()
    
    # Greet user
    greet_user("Student")
    
    # Create some data
    numbers = [10, 25, 15, 30, 20]
    print(f"Numbers: {numbers}\n")
    
    # Process the data
    process_data(numbers)
    
    print()
    print("=" * 60)
    print("Program finished!")
    print("=" * 60)

# ============================================================================
# PROGRAM ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # This code only runs when the file is executed directly
    # (not when imported as a module)
    main()