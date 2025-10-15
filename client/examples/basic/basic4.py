# -----------------------------------------------------------------------------
# Sense Space
# -----------------------------------------------------------------------------
# Example: Python Basics - Functions
# -----------------------------------------------------------------------------
# IAD, Zurich University of the Arts / zhdk.ch
# Max Rheiner
# -----------------------------------------------------------------------------

# ============================================================================
# SIMPLE FUNCTIONS
# ============================================================================

def greet(name):
    """Say hello to someone"""
    print("Hello,", name + "!")

def add_numbers(a, b):
    """Add two numbers and return result"""
    return a + b

# Call functions
greet("Alice")
result = add_numbers(5, 3)
print("5 + 3 =", result)

# ============================================================================
# FUNCTIONS WITH DEFAULT ARGUMENTS
# ============================================================================

def greet_with_title(name, title="Mr."):
    """Greet with optional title"""
    print(f"Hello, {title} {name}!")

greet_with_title("Smith")           # Uses default "Mr."
greet_with_title("Jones", "Dr.")    # Custom title

# ============================================================================
# FUNCTIONS WITH MULTIPLE RETURN VALUES
# ============================================================================

def calculate_rectangle(width, height):
    """Calculate area and perimeter"""
    area = width * height
    perimeter = 2 * (width + height)
    return area, perimeter

# Unpack multiple return values
a, p = calculate_rectangle(5, 3)
print(f"Rectangle: area={a}, perimeter={p}")

# ============================================================================
# FUNCTIONS THAT MODIFY LISTS
# ============================================================================

def add_item(items, new_item):
    """Add item to list"""
    items.append(new_item)
    print(f"Added '{new_item}' to list")

shopping_list = ["milk", "bread"]
print("Before:", shopping_list)
add_item(shopping_list, "eggs")
print("After:", shopping_list)

# ============================================================================
# FUNCTIONS WITH VARIABLE ARGUMENTS
# ============================================================================

def print_all(*args):
    """Print all arguments"""
    print("Items:")
    for item in args:
        print("  -", item)

print_all("apple", "banana", "orange")

def create_person(**kwargs):
    """Create person from keyword arguments"""
    return kwargs

person = create_person(name="Max", age=30, city="Zurich")
print("Person:", person)

# ============================================================================
# LAMBDA FUNCTIONS (Anonymous Functions)
# ============================================================================

# Short one-line functions
square = lambda x: x * x
print("Square of 5:", square(5))

# Used with lists
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x * x, numbers))
print("Squared:", squared)
