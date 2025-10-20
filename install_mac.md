# Installation macOS

## Table of Contents
- [Setup Python](#setup-python)
- [Download senseSpace](#download-sensespace)
- [Setup the Environment](#setup-the-environment)
- [Install Libraries](#install-libraries)
  - [GPU Acceleration (Apple Silicon)](#gpu-acceleration-apple-silicon)
- [Install Examples](#install-examples)
  - [Speech](#speech)
- [Install Ollama (Optional)](#install-ollama-optional)
  - [Install Python Dependencies](#install-python-dependencies)
- [Reactivating the Environment](#reactivating-the-environment)

## Setup Python
We use Python 3.11 because Pyo supports this version.

**Using Homebrew (Recommended):**
```
brew update
brew install python@3.11
```

If you have multiple Python versions, you can verify which versions are installed:
```
python3.11 --version
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
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

**Check Version:**
```
python --version
```
- This needs to output the 3.11 version

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

### GPU Acceleration (Apple Silicon)
For Apple Silicon Macs (M1/M2/M3/M4), PyTorch will automatically use the Metal Performance Shaders (MPS) backend for GPU acceleration when available. PyTorch is installed as part of the speech example requirements.

To verify if MPS (Metal) is available and working:
```
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}'); print(f'MPS built: {torch.backends.mps.is_built()}')"
```

This should output:
```
MPS available: True
MPS built: True
```

**Note:** Intel Macs will use CPU only, as they don't support Metal acceleration.

## Install Examples

### Speech

#### macOS-Specific Prerequisites

Before installing the speech example, you need to install some system dependencies:

**1. Install Xcode Command Line Tools:**
```
xcode-select --install
```

**2. Install build dependencies via Homebrew:**
```
brew install autoconf automake libtool portaudio
```

**3. Build liblo (Required for pyo):**

On macOS, you need to manually build `liblo` version 0.29 before installing `pyo`:

```
# Download and build liblo 0.29
cd /tmp
curl -L https://github.com/radarsat1/liblo/releases/download/0.29/liblo-0.29.tar.gz -o liblo-0.29.tar.gz
tar -xzf liblo-0.29.tar.gz
cd liblo-0.29

# Configure and build
./configure
make
sudo make install

# Update library cache
sudo update_dyld_shared_cache
```

#### Install Speech Dependencies

- Go to "senseSpace/client/examples/speech"
```
cd client/examples/speech
pip install -r requirements.txt
```

If you encounter issues with `pyo` installation after building liblo, try:
```
pip install --no-cache-dir pyo
```

**Note:** macOS may prompt for microphone permissions on first run.

#### Running the Speech Example

- This example should download all needed data, so you only have to start it. But you have to check if your microphone or speaker device is set, since the defaults might not work. You can check which devices you have very simply, just start the example and check the output:
```
python speech_audio_io.py
```

- The output should be something like this:
```
======================================================================
🎤 INPUT devices:
  [ 0] Built-in Microphone
  [ 1] External Microphone
  [ 2] USB Audio Device

🔊 OUTPUT devices:
  [ 3] Built-in Output
  [ 4] External Speakers
  [ 5] Headphones
======================================================================
```

- Select your input and output accordingly:
```
python speech_audio_io.py --mic 0 --speaker 3
```

- Be careful with setting the devices, you might get errors if you set this wrong.

## Install Ollama (Optional)
Ollama allows you to run Large Language Models locally on your machine. This is useful for the speech examples and AI-powered features.

### Download and Install Ollama
- Download Ollama for macOS from [https://ollama.com/download/mac](https://ollama.com/download/mac)
- Open the downloaded .dmg file and drag Ollama to your Applications folder
- Launch Ollama from your Applications folder
- Ollama will run in the menu bar and start automatically

Alternatively, install via Homebrew:
```
brew install ollama
```

Then start the Ollama service:
```
ollama serve
```

### Verify Installation
Open a new Terminal window and check if Ollama is running:
```
ollama --version
```

### Download a Model
To use Ollama with senseSpace, download a language model. For example, to download the Llama 3.2:1b model:
```
ollama pull llama3.2:1b
```

Other recommended models:
- `ollama pull llama3.2:1b` - Smaller, faster model (1B parameters)
- `ollama pull llama3.2:3b` - Medium model (3B parameters)
- `ollama pull mistral` - Alternative high-quality model

### Test Ollama
Test if Ollama is working correctly:
```
ollama run llama3.2:1b
```

Type a message and press Enter. Type `/bye` to exit.

### GPU Acceleration
**Apple Silicon (M1/M2/M3/M4):** Ollama will automatically use the Metal Performance Shaders (MPS) for GPU acceleration on Apple Silicon Macs.

**Intel Macs:** Ollama will use CPU only.

To verify a model is running:
```
ollama ps
```

This will show running models and their resource usage.

### Configure Ollama for senseSpace
The speech examples are configured to connect to Ollama on `localhost:11434` by default. No additional configuration is needed if Ollama is running on the same machine.

### Install Python Dependencies
```
cd ./senseSpace/client/examples/llm/
pip install -r requirements.txt
```

## Reactivating the Environment
After closing Terminal or restarting your computer, you need to reactivate the virtual environment:
```
cd /path/to/senseSpace
source .venv/bin/activate
```