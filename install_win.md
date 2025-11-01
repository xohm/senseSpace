# Installation Win64

## Table of Contents
- [Setup Python](#setup-python)
- [Download senseSpace](#download-sensespace)
- [Setup the Environment](#setup-the-environment)
- [Install Libraries](#install-libraries)
  - [GPU Acceleration](#gpu-acceleration)
- [Install Streaming Libraries (Optional)](#install-streaming-libraries-optional)
- [Install Examples](#install-examples)
  - [Speech](#speech)
- [Install Ollama (Optional)](#install-ollama-optional)
  - [Download and Install Ollama](#download-and-install-ollama)
  - [Verify Installation](#verify-installation)
  - [Download a Model](#download-a-model)
  - [Test Ollama](#test-ollama)
  - [GPU Acceleration](#gpu-acceleration-1)
  - [Configure Ollama for senseSpace](#configure-ollama-for-sensespace)
  - [Install Python Dependencies](#install-python-dependencies)
- [Reactivating the Environment](#reactivating-the-environment)

## Setup Python
We use Python 3.11 because Pyo supports this version.
Download [Python 3.11](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe) and install it with activated PATH in the installer.

If you have multiple python version you can set your desired version to the active one. You can also just define this in the virtual environment, but here the information in case you want to set it.
- Open the command line, see the python version you have:
```
py --list
```
- Select 3.11:
```
py -3.11
```

## Download senseSpace
- Go to the folder where you like to install senseSpace
- Open the command line and type:
```
git clone https://github.com/xohm/senseSpace.git
```

## Setup the Environment
- Go into the senseSpace folder
- Setup the Python virtual environment:

**Command Prompt:**
```
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
```

**PowerShell:**
```
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Check Version**
```
python --verison
```
- this needs to output the 3.11 version

## Install Libraries
- Be sure you are in the senseSpace root directory:

**Command Prompt + PowerShell:**
```
pip install -e .
```

- Install the senseSpace library, go to "senseSpace\libs\senseSpaceLib"

**Command Prompt + PowerShell:**
```
cd .\libs\senseSpaceLib
pip install -e .
```

### GPU Acceleration
In case you have a Nvidia GPU, setup the Cuda supported path, this accelerates all the AI responses.
- Setup the Cuda PyTorch version, newer GPU should use Cuda 12.1:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
- For older Nvidia GPU:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

To verify CUDA is working:
```
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

This should output `CUDA available: True` if properly installed.

## Install Streaming Libraries (Optional)

The video streaming feature allows real-time video transmission from ZED cameras to remote clients using GStreamer. This is optional and only needed if you want to use the `--stream` flag with the server.

### System Dependencies

**Important:** Setting up GStreamer on Windows requires several steps.

#### 1. Install GStreamer Runtime

Download and install both the runtime and development installers from the [GStreamer website](https://gstreamer.freedesktop.org/download/):

- **GStreamer 1.22.x runtime** (MSVC 64-bit): [Download](https://gstreamer.freedesktop.org/data/pkg/windows/)
- **GStreamer 1.22.x development** (MSVC 64-bit): [Download](https://gstreamer.freedesktop.org/data/pkg/windows/)

During installation:
- Choose **Complete** installation (not Typical)
- Default install path is `C:\gstreamer\1.0\msvc_x86_64\`
- The installer should automatically add GStreamer to your PATH

#### 2. Install PyGObject for Windows

Download and install the PyGObject all-in-one installer:
- [PyGObject for Windows](https://github.com/wingtk/gvsbuild/releases)
- Download the latest `PyGObject-3.xx.x-MSVC-py3.11-x64.exe` (match your Python version)
- Run the installer and follow the wizard

#### 3. Verify PATH Configuration

Ensure these paths are in your system PATH (adjust version numbers as needed):
```
C:\gstreamer\1.0\msvc_x86_64\bin
C:\Program Files\GTK3-Runtime Win64\bin
```

To check your PATH in PowerShell:
```powershell
$env:PATH -split ';' | Select-String gstreamer
```

Or in Command Prompt:
```cmd
echo %PATH% | findstr gstreamer
```

### Install Python Streaming Package

Install the senseSpace library with streaming support:

**Command Prompt + PowerShell:**
```
pip install -e ".[streaming]"
```

### Verify Installation

Test if GStreamer is working:

**Command Prompt + PowerShell:**
```
python -c "import gi; gi.require_version('Gst', '1.0'); from gi.repository import Gst; Gst.init(None); print('GStreamer OK')"
```

Test if streaming module loads:
```
python -c "from senseSpace.video_streaming import VideoStreamer; print('Streaming module OK')"
```

### Troubleshooting

If you get import errors:

1. **DLL load failed**: Ensure GStreamer bin directory is in PATH and restart your terminal
2. **Module not found**: Reinstall PyGObject installer matching your Python version
3. **Version mismatch**: Make sure GStreamer runtime and development versions match

### Usage

Start the server with streaming enabled:

**Command Prompt + PowerShell:**
```
cd server
python senseSpace_fusion_main.py --viz --stream
```

Connect a client:
```
cd client\examples\streaming
python streamingClient.py --server localhost
```

**Note:** Streaming only activates when a client is connected, so there's zero overhead when no clients are active.

## Install Examples

### Speech

- Install "Visual Studio Build Tools for C++": https://visualstudio.microsoft.com/visual-cpp-build-tools/
- go to "senseSpace\client\examples\speech"
```
cd .\client\examples\speech
pip install -r requirements.txt
```
- This examples should download all needed data, so you only have to start it. But you have to check if your microphone or speaker device is set, since the defaults might not work. You can check which devices you have very simple, just start the example and check the output:
```
python speech_audio_io.py
```
- The output should be something like this
```
======================================================================
🎤 INPUT devices:
  [ 0] Microsoft Sound Mapper - Input
  [ 1] Microphone (2- HD Pro Webcam C9
  [ 2] Microphone (Realtek(R) Audio)
  [ 3] Headset Microphone (Oculus Virt
  [ 4] Echo Cancelling Speakerphone (J
  [10] Primary Sound Capture Driver

🔊 OUTPUT devices:
  [ 5] Microsoft Sound Mapper - Output
  [ 6] Lautsprecher (Realtek(R) Audio)
  [ 7] DELL U4021QW (HD Audio Driver f
  [ 8] Echo Cancelling Speakerphone (J
  [ 9] Headphones (Oculus Virtual Audi
  [15] Primary Sound Driver
  [16] Lautsprecher (Realtek(R) Audio)
  [17] DELL U4021QW (HD Audio Driver for Display Audio)
  [18] Echo Cancelling Speakerphone (Jabra SPEAK 510 USB)
  [19] Headphones (Oculus Virtual Audio Device)
```
- Select your input and output accordingly:
```
python speech_audio_io.py  --mic 0 --speaker 5
```
- Be carefull with setting the devices, you might get errors if you set this wrong:
```
[AUDIO] Using specified output device 5: Microsoft Sound Mapper - Output
❌ Error in VAD recording: Unanticipated host error [PaErrorCode -9999]: 'There is no driver installed on your system.' [MME error 6]
```

## Install Ollama (Optional)
Ollama allows you to run Large Language Models locally on your machine. This is useful for the speech examples and AI-powered features.

### Download and Install Ollama
- Download Ollama for Windows from [https://ollama.com/download/windows](https://ollama.com/download/windows)
- Run the installer and follow the installation wizard
- Ollama will install as a Windows service and start automatically

### Verify Installation
Open a new command prompt or PowerShell window and check if Ollama is running:

**Command Prompt + PowerShell:**
```
ollama --version
```

### Download a Model
To use Ollama with senseSpace, download a language model. For example, to download the Llama 3.2 model:

**Command Prompt + PowerShell:**
```
ollama pull llama3.2:1b
```

Other recommended models:
- `ollama pull llama3.2:1b` - Smaller, faster model (1B parameters)
- `ollama pull llama3.2:3b` - Medium model (3B parameters)
- `ollama pull mistral` - Alternative high-quality model

### Test Ollama
Test if Ollama is working correctly:

**Command Prompt + PowerShell:**
```
ollama run llama3.2:1b
```

Type a message and press Enter. Type `/bye` to exit.

### GPU Acceleration
If you have an NVIDIA GPU and installed CUDA (see GPU Acceleration section above), Ollama will automatically use your GPU for faster inference.

To verify GPU usage, check the Ollama logs or use:
```
ollama ps
```

This will show running models and whether they're using GPU acceleration.

### Configure Ollama for senseSpace
The speech examples are configured to connect to Ollama on `localhost:11434` by default. No additional configuration is needed if Ollama is running on the same machine.

### Install Python Dependencies
```
cd C:\path\to\senseSpace\client\examples\llm\
pip install -r requirements.txt
```

## Reactivating the Environment
After closing the command line or restarting your computer, you need to reactivate the virtual environment:

**Command Prompt:**
```
cd C:\path\to\senseSpace
.venv\Scripts\activate.bat
```

**PowerShell:**
```
cd C:\path\to\senseSpace
.venv\Scripts\Activate.ps1
```