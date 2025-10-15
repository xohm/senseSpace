# -----------------------------------------------------------------------------
# Sense Space
# -----------------------------------------------------------------------------
# Example: Python Basics - Classes and Objects
# -----------------------------------------------------------------------------
# IAD, Zurich University of the Arts / zhdk.ch
# Max Rheiner
# -----------------------------------------------------------------------------

# ============================================================================
# SIMPLE CLASS
# ============================================================================

class Person:
    """A simple person class"""
    
    def __init__(self, name, age):
        """Constructor - called when creating new person"""
        self.name = name
        self.age = age
    
    def greet(self):
        """Method to greet"""
        print(f"Hello, I'm {self.name} and I'm {self.age} years old")

# Create objects (instances)
person1 = Person("Alice", 25)
person2 = Person("Bob", 30)

person1.greet()
person2.greet()

# ============================================================================
# CLASS WITH METHODS
# ============================================================================

class Rectangle:
    """Rectangle with width and height"""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        """Calculate area"""
        return self.width * self.height
    
    def perimeter(self):
        """Calculate perimeter"""
        return 2 * (self.width + self.height)
    
    def describe(self):
        """Print description"""
        print(f"Rectangle {self.width}x{self.height}")
        print(f"  Area: {self.area()}")
        print(f"  Perimeter: {self.perimeter()}")

# Create and use rectangle
rect = Rectangle(5, 3)
rect.describe()

# ============================================================================
# CLASS WITH PRIVATE ATTRIBUTES
# ============================================================================

class BankAccount:
    """Bank account with balance"""
    
    def __init__(self, owner, balance=0):
        self.owner = owner
        self._balance = balance  # "_" indicates private
    
    def deposit(self, amount):
        """Add money"""
        self._balance += amount
        print(f"Deposited {amount}. New balance: {self._balance}")
    
    def withdraw(self, amount):
        """Remove money"""
        if amount <= self._balance:
            self._balance -= amount
            print(f"Withdrew {amount}. New balance: {self._balance}")
        else:
            print("Insufficient funds!")
    
    def get_balance(self):
        """Get current balance"""
        return self._balance

# Create account
account = BankAccount("Max", 100)
account.deposit(50)
account.withdraw(30)
print(f"Final balance: {account.get_balance()}")

# ============================================================================
# INHERITANCE
# ============================================================================

class Animal:
    """Base animal class"""
    
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        """Animals make sounds"""
        print("Some sound")

class Dog(Animal):
    """Dog inherits from Animal"""
    
    def speak(self):
        """Override speak method"""
        print(f"{self.name} says: Woof!")

class Cat(Animal):
    """Cat inherits from Animal"""
    
    def speak(self):
        """Override speak method"""
        print(f"{self.name} says: Meow!")

# Create animals
dog = Dog("Buddy")
cat = Cat("Whiskers")

dog.speak()
cat.speak()

# ============================================================================
# CLASS WITH CLASS VARIABLES
# ============================================================================

class Counter:
    """Counter with class variable to track total instances"""
    
    count = 0  # Class variable (shared by all instances)
    
    def __init__(self, name):
        self.name = name
        Counter.count += 1  # Increment class variable
    
    @classmethod
    def get_count(cls):
        """Class method to get count"""
        return cls.count

# Create counters
c1 = Counter("first")
c2 = Counter("second")
c3 = Counter("third")

print(f"Total counters created: {Counter.get_count()}")