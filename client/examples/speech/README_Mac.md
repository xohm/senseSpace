# Speech-Enabled LLM Examples

## Overview
These examples use Large Language Models with voice interaction capabilities for pose analysis. Combine skeleton tracking with speech recognition (Whisper) and text-to-speech (Piper) for natural voice-controlled pose analysis.

## Installation

### macOS-Specific Setup

#### 1. Install Xcode Command Line Tools

Required for building native libraries:

```bash
xcode-select --install
```

If already installed, you'll see a message saying so.

#### 2. Building liblo (Required for pyo)

On macOS, you may need to manually build `liblo` version 0.29 before installing `pyo`:

```bash
# Install build dependencies
brew install autoconf automake libtool

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

After building liblo, proceed with Python dependencies.

### Python Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

If you encounter issues with `pyo` installation after building liblo, try:
```bash
pip install --no-cache-dir pyo
```

## Additional macOS Notes

- **Piper TTS**: The `piper-tts` Python package should install without issues on macOS
- **Microphone Access**: macOS may prompt for microphone permissions on first run
- **PortAudio**: Install via Homebrew if needed: `brew install portaudio`

## Running Examples

After installation, run the speech client:
```bash
python speechClient.py --viz
```

---

**IAD, Zurich University of the Arts / zhdk.ch**  
Max Rheiner

