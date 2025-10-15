# -----------------------------------------------------------------------------
# Sense Space
# -----------------------------------------------------------------------------
# Example: Python Basics - Variables and Functions
# -----------------------------------------------------------------------------
# IAD, Zurich University of the Arts / zhdk.ch
# Max Rheiner
# -----------------------------------------------------------------------------

# ============================================================================
# VARIABLES AND DATA TYPES
# ============================================================================

# Numbers
age = 25
temperature = 36.5

# Strings (text)
name = "Alice"
message = "Hello!"

# Boolean (True/False)
is_student = True

print(name, "is", age, "years old")
print("Student?", is_student)

# ============================================================================
# MATH OPERATIONS
# ============================================================================

a = 10
b = 3

print("\nMath with", a, "and", b)
print("Addition:", a + b)
print("Subtraction:", a - b)
print("Multiplication:", a * b)
print("Division:", a / b)

# ============================================================================
# FUNCTIONS
# ============================================================================

def greet(name):
    """Say hello to someone"""
    print("Hello,", name + "!")

def calculate_area(width, height):
    """Calculate rectangle area"""
    return width * height

# Call functions
greet("Bob")
area = calculate_area(5, 3)
print("Area:", area)

# ============================================================================
# CONDITIONALS
# ============================================================================

score = 85

if score >= 90:
    print("Grade: A")
elif score >= 80:
    print("Grade: B")
else:
    print("Grade: C")
