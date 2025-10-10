# Speech-Enabled LLM Examples

## Overview
These examples use Large Language Models with voice interaction capabilities for pose analysis. Combine skeleton tracking with speech recognition (Whisper) and text-to-speech (Piper) for natural voice-controlled pose analysis.

## Installation

### Python Dependencies
Install required packages:
```bash
pip install -r requirements.txt
```

The requirements.txt includes:
```txt
# Core dependencies
requests>=2.31.0
numpy>=1.24.0
python-dotenv>=1.0.0

# LLM clients
openai>=1.0.0

# Speech dependencies
openai-whisper>=20231117
pyo>=1.0.5
```

### Speech Dependencies

#### 1. Whisper (Speech Recognition)
Already installed via `openai-whisper` package above.

**Models** (downloaded automatically on first use):
- `tiny` - Fastest, lowest quality (~39 MB)
- `base` - Good balance (default, ~74 MB)
- `small` - Better accuracy (~244 MB)
- `medium` - High accuracy (~769 MB)

#### 2. Piper (Text-to-Speech)

**Linux (Ubuntu/Debian):**
```bash
# Download latest release
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz
tar -xzf piper_amd64.tar.gz
sudo mv piper/piper /usr/local/bin/
```

**macOS:**
```bash
brew install piper-tts
```

**Windows:**
Download installer from https://github.com/rhasspy/piper/releases

**Download voice model:**
```bash
# Download a voice (e.g., US English - Lessac)
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json

# Move to piper models directory
mkdir -p ~/.local/share/piper/voices
mv en_US-lessac-medium.onnx* ~/.local/share/piper/voices/
```

#### 3. Pyo (Audio Engine)

**Why Pyo?**
- Professional-grade audio engine
- Low latency recording and playback
- Cross-platform support
- No additional audio backend needed

**Linux:**
```bash
# Install system dependencies first
sudo apt-get install libjack-jackd2-dev libportmidi-dev portaudio19-dev liblo-dev

# Then install pyo
pip install pyo
```

**macOS:**
```bash
brew install portaudio portmidi liblo
pip install pyo
```

**Windows:**
```bash
pip install pyo
# May require Microsoft Visual C++ Build Tools
```

#### 4. Verify Installation
```bash
# Check Whisper
whisper --help

# Check Piper
piper --help

# Check Pyo
python -c "from pyo import *; s = Server(); s.boot(); s.start(); print('Pyo version:', version); s.stop(); s.shutdown()"
```

---

## Examples

### speech.py - Voice-Interactive Pose Analysis
Combines Ollama LLM with Whisper (speech recognition) and Piper (text-to-speech) using **Pyo audio engine** for professional audio quality.

**Features:**
- 🎤 **Voice commands** - Control with natural speech
- 🔊 **Spoken responses** - LLM answers read aloud
- ⌨️ **Keyboard control** - Traditional keyboard shortcuts
- 🎵 **Professional audio** - Pyo engine for low-latency, high-quality audio
- 🔒 **100% Local** - No cloud dependencies
- 🎚️ **Real-time processing** - Direct audio I/O without buffering

**Setup:**

1. **Install Ollama:**
   
   **Linux:**
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```
   
   **macOS:**
   ```bash
   brew install ollama
   ```
   
   **Windows:**
   Download installer from https://ollama.com/download

2. **Pull a model:**
   ```bash
   ollama pull llama3.2        # Recommended: Fast & good quality (3B params)
   ```

3. **Install speech dependencies** (see Installation section above)

4. **Run with speech enabled:**
   ```bash
   python speech.py --server localhost --viz --speech
   ```

**Keyboard shortcuts:**
- `SPACE` - Describe the current pose (text only)
- `A` - Analyze pose emotions/mood (text only)
- `B` - Suggest exercise or activity (text only)
- `V` - Toggle voice recognition on/off

**Voice commands:**
- Say **"describe"** or **"what"** - Describe the pose (with spoken response)
- Say **"analyze"** or **"emotion"** - Analyze emotions (with spoken response)
- Say **"suggest"** or **"exercise"** - Suggest activity (with spoken response)

**Command-line options:**
```bash
# Basic usage
python speech.py --server localhost --viz --speech

# Custom model
python speech.py --server localhost --viz --speech --model llama3.1

# Remote server
python speech.py --server 192.168.1.100 --viz --speech

# Without visualization (headless)
python speech.py --server localhost --speech
```

---

### llm.py - OpenAI API
Uses the OpenAI API to inquire pose analysis data.

**Setup:**
1. Get an API key from https://platform.openai.com/api-keys
2. Create a `.env` file in this directory:
   ```env
   OPENAI_API_KEY=sk-your-key-here
   OPENAI_MODEL=gpt-4o-mini
   ```
3. Run:
   ```bash
   python llm.py --server localhost --viz
   ```

**Keyboard shortcuts:**
- `SPACE` - Describe the current pose
- `A` - Analyze pose emotions/mood
- `B` - Suggest exercise or activity

---

### llm_ollama.py - Local Ollama LLM
Uses a local Ollama LLM for better latency, privacy, and no internet dependency.

**Setup:**

1. **Install Ollama** (see speech.py setup above)

2. **Pull a model:**
   ```bash
   ollama pull llama3.2        # Recommended
   ```

3. **Run:**
   ```bash
   python llm_ollama.py --server localhost --viz
   ```

**Keyboard shortcuts:**
- `SPACE` - Describe the current pose
- `A` - Analyze pose emotions/mood
- `B` - Suggest exercise or activity

---

## Comparison

| Feature | OpenAI (llm.py) | Ollama (llm_ollama.py) | Speech (speech.py) |
|---------|----------------|------------------------|-------------------|
| **Cost** | ~$0.15/1M tokens | Free (local) | Free (local) |
| **Speed** | ~1-2s | ~2-5s | ~3-6s (+ TTS) |
| **Privacy** | Cloud | 100% local | 100% local |
| **Internet** | Required | Not required | Not required |
| **Voice Control** | ❌ | ❌ | ✅ |
| **Spoken Output** | ❌ | ❌ | ✅ |
| **Audio Engine** | - | - | Pyo (professional) |
| **Dependencies** | API key | Ollama + Model | Ollama + Whisper + Piper + Pyo |
| **Setup Time** | < 1 min | ~5 min (download) | ~15 min (all deps) |
| **Disk Space** | Minimal | ~2-4 GB | ~3-5 GB |

---

## Speech System Architecture

```
┌─────────────────┐
│  Microphone     │
└────────┬────────┘
         │ (Audio input via Pyo Server)
         ▼
┌─────────────────┐
│  Pyo Recording  │ ◄── Low-latency audio capture
│  (NewTable)     │
└────────┬────────┘
         │ (WAV file)
         ▼
┌─────────────────┐
│    Whisper      │ ◄── Speech-to-Text (local)
└────────┬────────┘
         │ (Text command)
         ▼
┌─────────────────┐
│  Command Parser │ ◄── "describe", "analyze", "suggest"
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Ollama LLM     │ ◄── Pose analysis (local)
└────────┬────────┘
         │ (Text response)
         ▼
┌─────────────────┐
│     Piper       │ ◄── Text-to-Speech (local)
└────────┬────────┘
         │ (WAV file)
         ▼
┌─────────────────┐
│  Pyo Playback   │ ◄── Professional audio output
│  (SndTable/Osc) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Speakers     │
└─────────────────┘
```

---

## Troubleshooting

### Speech System

**"Whisper not found":**
```bash
pip install openai-whisper
whisper --help  # Verify installation
```

**"Piper not found":**
```bash
# Check installation
piper --help

# Download voice model if missing
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
```

**"Pyo failed to initialize":**
```bash
# Linux: Install audio libraries
sudo apt-get install libjack-jackd2-dev portaudio19-dev libportmidi-dev liblo-dev

# macOS: Install audio libraries
brew install portaudio portmidi liblo

# Test Pyo installation
python -c "from pyo import *; s = Server(); s.boot(); s.start(); print('SUCCESS: Pyo initialized'); s.stop(); s.shutdown()"

# List available audio devices
python -c "from pyo import *; print(pa_get_devices_infos())"
```

**Pyo audio device errors:**
```bash
# If you get "PortAudio error" or device conflicts:

# 1. Check which audio backend Pyo is using
python -c "from pyo import *; s = Server(); s.boot(); print('Backend:', pa_get_default_output())"

# 2. Try selecting a specific device
# Edit speech.py and modify SpeechEngine.__init__():
# self.server.setOutputDevice(1)  # Try different device numbers
# self.server.setInputDevice(1)

# 3. Use jackd backend (Linux, professional audio)
sudo apt-get install jackd2
jackd -d alsa &  # Start JACK server
# Then run speech.py
```

**Voice recognition not working:**
- Check microphone permissions
- Verify microphone input: `arecord -l` (Linux) or System Preferences (macOS)
- Test Pyo recording:
  ```python
  from pyo import *
  s = Server(duplex=1).boot().start()
  rec = Input(chnl=0)
  rec.out()  # Should hear yourself if mic is working
  time.sleep(3)
  s.stop()
  ```
- Reduce background noise
- Speak clearly, 1-2 seconds after "[SPEECH] Listening..." appears

**No audio output:**
- Check speaker/headphone connection
- Test Pyo playback:
  ```python
  from pyo import *
  s = Server().boot().start()
  sine = Sine(freq=440, mul=0.3).out()  # 440 Hz test tone
  time.sleep(2)
  s.stop()
  ```
- Try different output device (see "Pyo audio device errors" above)

**Slow transcription:**
- Use smaller Whisper model: modify `whisper_model="tiny"` in `SpeechEngine.__init__()`
- Reduce recording duration: change `RECORD_SECONDS = 3` to `2` in `_listen_loop_pyo()`
- Enable GPU acceleration: `pip install torch torchaudio` (NVIDIA GPU required)

**Audio quality issues:**
- Check sample rate: Pyo defaults to system rate (usually 44100 or 48000 Hz)
- Reduce background noise: Use directional microphone or noise gate
- Adjust volume: Modify `mul=0.5` in `_play_wav_pyo()` for louder/quieter output

### Ollama

**"Cannot connect to Ollama":**
```bash
# Check if running
ollama list

# Start manually
ollama serve
```

**"Model not found":**
```bash
ollama pull llama3.2
```

**Slow responses:**
```bash
# Use smaller model
ollama pull llama3.2:1b

# Check GPU usage (NVIDIA)
nvidia-smi

# Run with specific model
python speech.py --server localhost --viz --speech --model llama3.2:1b
```

### OpenAI

**"API key not found":**
- Ensure `.env` file exists with valid `OPENAI_API_KEY`

**Rate limits:**
- Upgrade OpenAI plan or add delays between requests

---

## Performance Tips

### Audio (Pyo)
- **Lower latency**: Use JACK backend on Linux (`jackd -d alsa`)
- **Better quality**: Use external USB audio interface
- **Reduce CPU**: Decrease buffer size in `Server(buffersize=256)`
- **Professional setup**: Use dedicated audio hardware with ASIO (Windows) or CoreAudio (macOS)

### Speech Recognition (Whisper)
- **Faster transcription**: Use `whisper_model="tiny"` or `"base"`
- **Better accuracy**: Use `whisper_model="small"` or `"medium"`
- **GPU acceleration**: Install PyTorch with CUDA for 5-10x speedup
  ```bash
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
- **Reduce latency**: Decrease `RECORD_SECONDS` from 3 to 2 seconds

### LLM Responses
- **Faster**: Use smaller models (`llama3.2:1b`, `qwen2.5:1.5b`)
- **Better quality**: Use larger models (`llama3.1:8b`)
- **GPU acceleration**: Ensure NVIDIA/AMD drivers installed (Ollama auto-detects)
- **Reduce response time**: Decrease `num_predict` in LLM payload (currently 150 tokens)

### Overall System
- **Prioritize speed**: tiny Whisper + llama3.2:1b + reduce recording time
- **Prioritize quality**: small Whisper + llama3.1:8b + professional microphone
- **Balance**: base Whisper + llama3.2:3b (default configuration)

---

## Recommended Models

### Whisper (Speech Recognition)
| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| `tiny` | 39 MB | ⚡⚡⚡ | Fair | Testing, low-end hardware |
| `base` | 74 MB | ⚡⚡ | Good | **Recommended** - balanced |
| `small` | 244 MB | ⚡ | Very good | Better accuracy needed |
| `medium` | 769 MB | Slow | Excellent | Maximum accuracy, GPU required |

### Piper (Text-to-Speech)
| Voice | Quality | Speed | Use Case |
|-------|---------|-------|----------|
| `en_US-lessac-low` | Good | Fast | Quick responses |
| `en_US-lessac-medium` | Very good | Medium | **Recommended** - balanced |
| `en_US-lessac-high` | Excellent | Slow | Best quality |

More voices: https://rhasspy.github.io/piper-samples/

### Ollama (LLM)
| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| `llama3.2:1b` | 1.3 GB | ⚡⚡⚡ | Good | CPU-only, fast responses |
| `llama3.2:3b` | 2.0 GB | ⚡⚡ | Very good | **Recommended** - balanced |
| `llama3.1:8b` | 4.7 GB | ⚡ | Excellent | GPU required, best quality |
| `qwen2.5:7b` | 4.7 GB | ⚡ | Excellent | Alternative to llama3.1 |

---

## Example Session

```bash
$ python speech.py --server localhost --viz --speech

[INIT] Connected to server
[PYO] Audio server initialized
[SPEECH] Voice interface enabled (using pyo)
[INIT] Ollama server connected at http://localhost:11434
[INIT] Available models: llama3.2:latest
[INIT] Using model: llama3.2
[INFO] Keyboard shortcuts:
  SPACE - Describe the current pose
  A     - Analyze pose emotions/mood
  B     - Suggest exercise or activity
  V     - Toggle voice recognition
[INFO] Voice commands: 'describe', 'analyze', 'suggest'
[SPEECH] Voice recognition started. Say 'analyze', 'describe', or 'suggest'

[SPEECH] Listening...
[WHISPER] Transcribing audio...
[WHISPER] Transcribed: 'describe my pose'
[VOICE] Command: describe pose
[LLM] Analyzing person 1 (confidence: 95.2)
[LLM] Requesting analysis for person 1...
[RESPONSE] You are standing upright with arms relaxed at your sides, facing forward in a neutral, balanced stance.
[PIPER] Speaking: 'You are standing upright with arms relaxed at your sides...'
```

---

## Advanced Configuration

### Customizing Pyo Audio Settings

Edit `speech.py` and modify the `SpeechEngine.__init__()` method:

```python
# Lower latency (requires more CPU)
self.server = Server(duplex=1, buffersize=128)

# Specific audio devices
self.server.setOutputDevice(2)  # Use device #2 for output
self.server.setInputDevice(1)   # Use device #1 for microphone

# Higher sample rate (better quality, more CPU)
self.server.setSamplingRate(48000)
```

### Customizing Voice Recognition

Edit `_listen_loop_pyo()` in `speech.py`:

```python
# Record for 2 seconds instead of 3
RECORD_SECONDS = 2

# Use different sample rate
SAMPLE_RATE = 16000  # Whisper expects 16kHz

# Add noise gate (only record if volume above threshold)
mic = Input(chnl=0)
gate = Gate(mic, thresh=-40, risetime=0.01, falltime=0.1)
recorder = TableRec(gate, table=recording_table)
```

### Adding Audio Effects

You can add real-time audio effects using Pyo:

```python
# Add reverb to TTS output
from pyo import *

def _play_wav_pyo(self, wav_file):
    snd_table = SndTable(wav_file)
    osc = Osc(table=snd_table, freq=snd_table.getRate())
    
    # Add reverb effect
    rev = Freeverb(osc, size=0.8, damp=0.7, bal=0.3, mul=0.5)
    rev.out()
    
    duration = snd_table.getDur()
    time.sleep(duration + 0.5)  # Extra time for reverb tail
    
    rev.stop()
```

---

## License
See main repository LICENSE file.



