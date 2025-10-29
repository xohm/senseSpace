# Python Basics Examples

A step-by-step introduction to Python programming.

## Table of Contents
- [Installing Python](#installing-python)
- [Setting Up a Virtual Environment (venv)](#setting-up-a-virtual-environment-venv)
- [Running the Examples](#running-the-examples)
- [Example Files](#example-files)

---

## Installing Python

### 🐧 Linux (Ubuntu/Debian)

Python 3 is usually pre-installed. Check version:

```bash
python3 --version
```

If not installed:

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

### 🪟 Windows

1. Download Python from [python.org/downloads](https://www.python.org/downloads/)
2. Run installer
3. ✅ **Important:** Check "Add Python to PATH"
4. Click "Install Now"

Verify installation:

```cmd
python --version
```

### 🍎 macOS

**Option 1: Official installer**
1. Download from [python.org/downloads](https://www.python.org/downloads/)
2. Run the `.pkg` installer

**Option 2: Homebrew** (recommended)
```bash
brew install python3
```

Verify installation:

```bash
python3 --version
```

---

## Setting Up a Virtual Environment (venv)

A virtual environment keeps your project dependencies separate and organized. **Highly recommended!**

### Why Use a Virtual Environment?

- ✅ Isolate project dependencies
- ✅ Avoid conflicts between different projects
- ✅ Easy to recreate environment on another machine
- ✅ Keep your system Python clean

### Creating a Virtual Environment

**Important:** We create ONE venv at the project root to use for all examples!

**Linux/macOS:**
```bash
# Navigate to the senseSpace root directory
cd ~/Documents/development/own/senseSpace

# Create virtual environment at project root
python3 -m venv .venv

# Activate it
source .venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
# Navigate to the senseSpace root directory
cd C:\Users\YourName\Documents\senseSpace

# Create virtual environment at project root
python -m venv .venv

# Activate it
.venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
# If you get execution policy error, run this first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate:
.venv\Scripts\Activate.ps1
```

### Working with Virtual Environment

When activated, you'll see `(.venv)` in your terminal:

```bash
(.venv) user@computer:~/senseSpace$
```

Now you can install packages that stay in this environment only:

```bash
# Install a package (example)
pip install numpy

# See installed packages
pip list

# Save dependencies to file (from project root)
pip freeze > requirements.txt
```

### Deactivating Virtual Environment

When you're done:

```bash
deactivate
```

### Recreating Environment on Another Machine

If you have a `requirements.txt` file:

```bash
# Navigate to project root
cd ~/Documents/development/own/senseSpace  # Linux/macOS
cd C:\Users\YourName\Documents\senseSpace  # Windows

# Create and activate venv (as shown above)
python3 -m venv .venv              # Linux/macOS
python -m venv .venv               # Windows

source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate             # Windows

# Install all dependencies
pip install -r requirements.txt
```

---

## Running the Examples

**First time setup:**

```bash
# 1. Navigate to project root
cd ~/Documents/development/own/senseSpace  # Linux/macOS
cd C:\Users\YourName\Documents\senseSpace  # Windows

# 2. Create and activate virtual environment (see above)
source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate             # Windows

# 3. No additional packages needed for basic examples!
```

**Run any example:**

```bash
# Make sure venv is activated (you see "(.venv)" in prompt)
# Navigate to examples folder
cd client/examples/basic

# Run examples
python basic1.py
python basic2.py
# etc.
```

💡 **Note:** The basic examples (basic1-6) don't require any extra packages. For more advanced senseSpace examples, you'll need to install dependencies from the project root.

---

## Example Files Overview

### 📄 `basic1.py` - Hello World
- Print statements
- Simple variables
- Basic math
- First function

**Concepts:** Output, variables, functions

---

### 📄 `basic2.py` - Variables and Functions
- Data types (numbers, strings, booleans)
- Math operations
- Defining functions
- Function parameters and return values
- Conditionals (if/elif/else)

**Concepts:** Variables, functions, logic

---

### 📄 `basic3.py` - Lists and Loops
- Creating and accessing lists
- List operations (append, length)
- For loops
- Dictionaries (key-value pairs)
- Combining concepts (loop through list of dictionaries)

**Concepts:** Collections, iteration

---

### 📄 `basic4.py` - Functions
- Simple functions
- Default arguments
- Multiple return values
- Functions that modify lists
- Variable arguments (`*args`, `**kwargs`)
- Lambda functions (anonymous functions)

**Concepts:** Advanced function features

---

### 📄 `basic5.py` - Classes and Objects
- Simple classes with `__init__` constructor
- Class methods
- Private attributes (using `_` prefix)
- Inheritance (parent/child classes)
- Class variables (shared across instances)
- Class methods (`@classmethod`)

**Concepts:** Object-oriented programming

---

### 📄 `basic6.py` - Program Structure
- Import modules
- Helper functions
- Main function
- `if __name__ == "__main__"` pattern
- Proper program organization

**Concepts:** Code structure, best practices

---

## Learning Path

Follow the examples in order:

1. **basic1.py** → Get started with Python
2. **basic2.py** → Learn variables and functions
3. **basic3.py** → Work with lists and loops
4. **basic4.py** → Master functions
5. **basic5.py** → Understand classes
6. **basic6.py** → Structure larger programs

---

## Next Steps

After completing these examples:
- Try modifying the code
- Combine concepts from different files
- Create your own small programs
- Explore the other senseSpace examples (requires additional setup)

### For Advanced senseSpace Examples

Many senseSpace examples require additional packages:

```bash
# 1. Make sure you're in project root with venv activated
cd ~/Documents/development/own/senseSpace  # Linux/macOS
cd C:\Users\YourName\Documents\senseSpace  # Windows

source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate             # Windows

# 2. Install senseSpace dependencies
pip install -r requirements.txt

# 3. Now you can run advanced examples
cd client/examples/speech
python speechClient.py --viz
```

---

## Troubleshooting

**Q: `venv` command not found**  
A: Install `python3-venv` package (Linux) or reinstall Python with all components (Windows/macOS)

**Q: Permission denied when activating (Windows)**  
A: Run PowerShell as Administrator and execute: `Set-ExecutionPolicy RemoteSigned`

**Q: How do I know if venv is activated?**  
A: You'll see `(.venv)` at the start of your command prompt

**Q: Can I delete the .venv folder?**  
A: Yes! Just recreate it when needed. Your code is safe.

**Q: Should I create separate venvs for different examples?**  
A: No! Use the single `.venv` at project root for all senseSpace examples.

---

## Resources

- [Python Official Tutorial](https://docs.python.org/3/tutorial/)
- [Python Documentation](https://docs.python.org/3/)
- [Real Python Tutorials](https://realpython.com/)
- [Python Virtual Environments Guide](https://docs.python.org/3/tutorial/venv.html)

---

**IAD, Zurich University of the Arts / zhdk.ch**  
Max Rheiner