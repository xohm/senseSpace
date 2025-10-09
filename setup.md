# Setup IDE (Integrated Development Environment)

This guide explains how to set up **Visual Studio Code** with **Python** and **GitHub Copilot** on both **Windows** and **macOS**.

---

## Windows Setup

### 1. Install Visual Studio Code

1. Go to [https://code.visualstudio.com/](https://code.visualstudio.com/).
2. Click **Download for Windows**.
3. Run the installer and accept the defaults.
4. When prompted, **check the box** to:

   * “Add `code` to PATH” (so you can open VS Code from the terminal).

After installation, open the terminal and verify:

```bash
code --version
```

You should see a version number like `1.xx.x`.

---

### 2. Open VS Code and Install the Python Extension

1. Launch **Visual Studio Code**.
2. Go to the **Extensions** view (`Ctrl+Shift+X`).
3. Search for **Python** (from Microsoft).
4. Click **Install**.

---

### 3. Install GitHub Copilot

1. In VS Code, open the **Extensions** view again.
2. Search for **GitHub Copilot**.
3. Click **Install**.
4. When prompted, **sign in** with your GitHub account.
5. Once logged in, Copilot should activate automatically.

---

## 🍏 macOS Setup

### 1. Install Visual Studio Code

You can install either from the website or via Homebrew.

#### Option A — from website

1. Visit [https://code.visualstudio.com/](https://code.visualstudio.com/).
2. Click **Download for macOS (Intel or Apple Silicon)**.
3. Drag **Visual Studio Code.app** into your **Applications** folder.

#### Option B — via Homebrew

If you use [Homebrew](https://brew.sh/):

```bash
brew install --cask visual-studio-code
```

Verify installation:

```bash
code --version
```

### 2. Install the Python Extension in VS Code

1. Launch **VS Code**.
2. Press `Cmd+Shift+X` to open the **Extensions** view.
3. Search for **Python**.
4. Click **Install**.

---

### 3. Install GitHub Copilot

1. In VS Code, open **Extensions** again.
2. Search for **GitHub Copilot**.
3. Click **Install**.
4. Sign in with your GitHub account when prompted.
5. You’re done — Copilot will start suggesting code automatically.

---

## ✅ Verify Everything Works

To test that Python and Copilot work together:

1. Open a new file in VS Code named `test.py`.
2. Type:

```python
def greet(name):
    # Copilot will suggest a line like:
    # return f"Hello, {name}!"
```

3. If you see Copilot suggestions appear — everything is set up correctly 🎉

---

## 💡 Tips

* Use `Ctrl+`` (Windows) or `Cmd+`` (macOS) to open the built-in terminal in VS Code.
* To update extensions: open the **Extensions** view and click the **Update** button.
* To manage Copilot settings, go to
  **File → Preferences → Settings → Extensions → GitHub Copilot**.

---

## 🧯 Troubleshooting

| Problem                      | Possible Fix                                                                              |
| ---------------------------- | ----------------------------------------------------------------------------------------- |
| **`python` not found**       | Reinstall Python and check **“Add Python to PATH”** during installation.                  |
| **`code` not found**         | Reinstall VS Code and check **“Add to PATH”** option.                                     |
| **Copilot not suggesting**   | Make sure you are signed in to GitHub and have an active Copilot subscription.            |
| **Multiple Python versions** | In VS Code, press `Ctrl+Shift+P` → “Python: Select Interpreter” → choose the correct one. |

---

🎉 **Done!**
You now have a full setup with **VS Code**, **Python**, and **GitHub Copilot** ready to use on Windows and macOS.
