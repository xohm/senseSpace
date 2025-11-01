# Installation Linux

## Table of Contents
- [Setup Python](#setup-python)
- [Download senseSpace](#download-sensespace)
- [Setup the Environment](#setup-the-environment)
- [Install Libraries](#install-libraries)
  - [GPU Acceleration (NVIDIA)](#gpu-acceleration-nvidia)
- [Install Streaming Libraries (Optional)](#install-streaming-libraries-optional)
- [Install Examples](#install-examples)
  - [Speech](#speech)
- [Install Ollama (Optional)](#install-ollama-optional)
  - [Download and Install Ollama](#download-and-install-ollama)
  - [Verify Installation](#verify-installation)
  - [Download a Model](#download-a-model)
  - [Test Ollama](#test-ollama)
  - [GPU Acceleration](#gpu-acceleration)
  - [Configure Ollama for senseSpace](#configure-ollama-for-sensespace)
- [Reactivating the Environment](#reactivating-the-environment)

## Setup Python
We use Python 3.12.

**Ubuntu/Debian:**
```
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev
```

**Fedora/RHEL:**
```
sudo dnf install python3.12 python3.12-devel
```

**Arch Linux:**
```
sudo pacman -S python
```

If you have multiple Python versions, you can verify which versions are installed:
```
python3.12 --version
```

## Download senseSpace
- Navigate to the folder where you want to install senseSpace
- Open Terminal and type:
```
git clone https://github.com/xohm/senseSpace.git
```

## Setup the Environment
- Go into the senseSpace folder
- Setup the Python virtual environment:

```
cd senseSpace
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

**Check Version:**
```
python --version
```
- This needs to output the 3.12 version

## Install Libraries
- Be sure you are in the senseSpace root directory:

```
pip install -e .
```

- Install the senseSpace library, go to "senseSpace/libs/senseSpaceLib"

```
cd libs/senseSpaceLib
pip install -e .
cd ../..
```

### GPU Acceleration (NVIDIA)
For NVIDIA GPUs with CUDA support, install PyTorch with CUDA. PyTorch is installed as part of the speech example requirements.

To verify CUDA is available:
```
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

This should output `CUDA available: True` if properly installed.

**For CUDA 12.1 (newer GPUs):**
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8 (older GPUs):**
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Note:** AMD GPUs with ROCm support may work but are not officially tested.

## Install Streaming Libraries (Optional)

The video streaming feature allows real-time video transmission from ZED cameras to remote clients using GStreamer. This is optional and only needed if you want to use the `--stream` flag with the server.

### System Dependencies

Install GStreamer and PyGObject system packages:

**Ubuntu/Debian/KDE Neon:**
```
sudo apt-get install -y \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gstreamer-1.0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libcairo2-dev \
    libgirepository1.0-dev
```

**Fedora/RHEL:**
```
sudo dnf install -y \
    python3-gobject \
    python3-cairo \
    gstreamer1 \
    gstreamer1-plugins-base \
    gstreamer1-plugins-good \
    gstreamer1-plugins-bad-free \
    gstreamer1-plugins-ugly-free \
    gstreamer1-libav \
    cairo-devel \
    gobject-introspection-devel
```

**Arch Linux:**
```
sudo pacman -S \
    python-gobject \
    python-cairo \
    gstreamer \
    gst-plugins-base \
    gst-plugins-good \
    gst-plugins-bad \
    gst-plugins-ugly \
    gst-libav \
    cairo \
    gobject-introspection
```

### Virtual Environment Setup

**Important:** PyGObject must use the system-installed version. If you're using a virtual environment, create symlinks:

```bash
# Adjust Python version (3.12 in this example) to match yours
ln -s /usr/lib/python3/dist-packages/gi .venv/lib/python3.12/site-packages/gi
ln -s /usr/lib/python3/dist-packages/cairo .venv/lib/python3.12/site-packages/cairo
```

### Install Python Streaming Package

Install the senseSpace library with streaming support:
```
pip install -e ".[streaming]"
```

### Verify Installation

Test if GStreamer is working:
```
python -c "import gi; gi.require_version('Gst', '1.0'); from gi.repository import Gst; Gst.init(None); print('GStreamer OK')"
```

Test if streaming module loads:
```
python -c "from senseSpace.video_streaming import VideoStreamer; print('Streaming module OK')"
```

### Usage

Start the server with streaming enabled:
```
cd server
python senseSpace_fusion_main.py --viz --stream
```

Connect a client:
```
cd client/examples/streaming
python streamingClient.py --server localhost
```

**Note:** Streaming only activates when a client is connected, so there's zero overhead when no clients are active.

## Install Examples

### Speech

#### Linux-Specific Prerequisites

Before installing the speech example, you need to install audio system dependencies:

**Ubuntu/Debian/KDE Neon:**
```
sudo apt update
sudo apt install -y \
    build-essential \
    python3-dev \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libjack-jackd2-dev \
    libasound-dev \
    liblo-dev \
    libsndfile1-dev \
    libfftw3-dev \
    libpulse-dev \
    libsamplerate0-dev \
    pkg-config
```

**Fedora/RHEL:**
```
sudo dnf install -y \
    gcc gcc-c++ \
    python3-devel \
    portaudio-devel \
    jack-audio-connection-kit-devel \
    alsa-lib-devel \
    liblo-devel \
    libsndfile-devel \
    fftw-devel \
    pulseaudio-libs-devel \
    libsamplerate-devel \
    pkgconfig
```

**Arch Linux:**
```
sudo pacman -S \
    base-devel \
    python \
    portaudio \
    jack2 \
    alsa-lib \
    liblo \
    libsndfile \
    fftw \
    libpulse \
    libsamplerate
```

**Note:** These libraries ensure all audio backends (PulseAudio/PipeWire, ALSA, JACK) are available for Pyo.

#### Install Speech Dependencies

- Go to "senseSpace/client/examples/speech"
```
cd client/examples/speech
pip install -r requirements.txt
```

**Note:** Linux may require additional audio permissions. You may need to add your user to the `audio` group:
```
sudo usermod -a -G audio $USER
```
Then log out and back in for the changes to take effect.

#### Verify Pyo Installation

Test if Pyo is working correctly:
```
python -c "from pyo import Server; s = Server().boot(); print('Pyo working!')"
```

If you encounter audio backend errors, you can specify the audio backend:
```
python -c "from pyo import Server; s = Server(audio='portaudio').boot(); print('Pyo working!')"
```

#### Running the Speech Example

- This example should download all needed data, so you only have to start it. But you have to check if your microphone or speaker device is set, since the defaults might not work. You can check which devices you have very simply, just start the example and check the output:
```
python speech_audio_io.py
```

- The output should be something like this:
```
======================================================================
🎤 INPUT devices:
  [ 0] default
  [ 1] USB Audio Device
  [ 2] Built-in Audio Analog Stereo

🔊 OUTPUT devices:
  [ 3] default
  [ 4] USB Audio Device
  [ 5] Built-in Audio Analog Stereo
======================================================================
```

- Select your input and output accordingly:
```
python speech_audio_io.py --mic 1 --speaker 4
```

- Be careful with setting the devices, you might get errors if you set this wrong.

## Install Ollama (Optional)
Ollama allows you to run Large Language Models locally on your machine. This is useful for the speech examples and AI-powered features.

### Download and Install Ollama

**Using the install script (recommended):**
```
curl -fsSL https://ollama.com/install.sh | sh
```

This will install Ollama and set it up as a systemd service.

**Manual installation:**
Download from [https://ollama.com/download/linux](https://ollama.com/download/linux) and follow the instructions.

### Verify Installation
Check if Ollama is running:
```
ollama --version
```

Check if the service is running:
```
systemctl status ollama
```

### Download a Model
To use Ollama with senseSpace, download a language model. For example, to download the Llama 3.2 model:
```
ollama pull llama3.2
```

Other recommended models:
- `ollama pull llama3.2:1b` - Smaller, faster model (1B parameters)
- `ollama pull llama3.2:3b` - Medium model (3B parameters)
- `ollama pull mistral` - Alternative high-quality model

### Test Ollama
Test if Ollama is working correctly:
```
ollama run llama3.2
```

Type a message and press Enter. Type `/bye` to exit.

### GPU Acceleration
If you have an NVIDIA GPU and installed CUDA (see GPU Acceleration section above), Ollama will automatically use your GPU for faster inference.

To verify GPU usage:
```
ollama ps
```

This will show running models and whether they're using GPU acceleration.

You can also check GPU usage with:
```
nvidia-smi
```

### Configure Ollama for senseSpace
The speech examples are configured to connect to Ollama on `localhost:11434` by default. No additional configuration is needed if Ollama is running on the same machine.

If Ollama is not starting automatically, you can start it manually:
```
sudo systemctl start ollama
```

To enable it to start on boot:
```
sudo systemctl enable ollama
```

## Reactivating the Environment
After closing Terminal or restarting your computer, you need to reactivate the virtual environment:
```
cd /path/to/senseSpace
source .venv/bin/activate
```