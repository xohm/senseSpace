# -----------------------------------------------------------------------------
# Sense Space
# -----------------------------------------------------------------------------
# Example: Python Basics - Lists and Loops
# -----------------------------------------------------------------------------
# IAD, Zurich University of the Arts / zhdk.ch
# Max Rheiner
# -----------------------------------------------------------------------------

# ============================================================================
# LISTS
# ============================================================================

# Create a list
fruits = ["apple", "banana", "orange"]
print("Fruits:", fruits)

# Access items (index starts at 0)
print("First:", fruits[0])
print("Last:", fruits[-1])

# Add items
fruits.append("grape")
print("Updated:", fruits)

# ============================================================================
# FOR LOOPS
# ============================================================================

# Loop through list
print("\nEach fruit:")
for fruit in fruits:
    print("  -", fruit)

# Loop with numbers
print("\nCount to 5:")
for i in range(1, 6):
    print(i)

# ============================================================================
# DICTIONARIES
# ============================================================================

person = {
    "name": "Max",
    "age": 30,
    "city": "Zurich"
}

print("\nPerson:")
print("Name:", person["name"])
print("Age:", person["age"])

# ============================================================================
# COMBINING CONCEPTS
# ============================================================================

students = [
    {"name": "Anna", "score": 92, "friend": ["Tom", "Lisa", "i"]},
    {"name": "Tom", "score": 88},
    {"name": "Lisa", "score": 76}
]

print("\nStudent grades:")
for student in students:
    name = student["name"]
    score = student["score"]
    
    if score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    else:
        grade = "C"
    
    print(f"{name}: {score} = {grade}")
