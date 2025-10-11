#!/usr/bin/env python3
"""
Speech I/O Helper class
- Records with VAD (Voice Activity Detection)
- Transcribes with Whisper
- Synthesizes speech with Piper
- Plays audio with pyo
- Cross-platform: Linux, macOS, Windows
"""

import sounddevice as sd
import numpy as np
import whisper
import torch
import webrtcvad
import subprocess
import tempfile
import os
import sys
import platform
import threading
import queue
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

# Suppress pyo warnings
os.environ['PYO_GUI_WX'] = '0'
from pyo import Server, DataTable, TableRead

# Default paths for models (OS-agnostic)
SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "models"

# Model download URLs
MODEL_URLS = {
    "en_US-lessac-medium": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
    },
    "en_US-amy-medium": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json"
    },
    "de_DE-thorsten_emotional-medium": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten_emotional/medium/de_DE-thorsten_emotional-medium.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten_emotional/medium/de_DE-thorsten_emotional-medium.onnx.json"
    }
}


def download_model(model_name: str, force: bool = False) -> bool:
    """
    Download Piper model files to models directory
    
    Args:
        model_name: Model name (e.g., 'en_US-amy-medium')
        force: If True, re-download even if exists
    
    Returns:
        True if model is available, False otherwise
    """
    if model_name not in MODEL_URLS:
        print(f"❌ Unknown model '{model_name}'")
        return False
    
    MODELS_DIR.mkdir(exist_ok=True)
    
    onnx_path = MODELS_DIR / f"{model_name}.onnx"
    json_path = MODELS_DIR / f"{model_name}.onnx.json"
    
    # Check if already exists
    if not force and onnx_path.exists() and json_path.exists():
        return True
    
    print(f"📥 Downloading {model_name}...")
    
    try:
        # Download .onnx file
        if force or not onnx_path.exists():
            print(f"   Downloading {onnx_path.name}...")
            urllib.request.urlretrieve(MODEL_URLS[model_name]["onnx"], onnx_path)
        
        # Download .onnx.json file
        if force or not json_path.exists():
            print(f"   Downloading {json_path.name}...")
            urllib.request.urlretrieve(MODEL_URLS[model_name]["json"], json_path)
        
        print(f"✅ Downloaded {model_name}")
        return True
    
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        # Clean up partial downloads
        if onnx_path.exists():
            onnx_path.unlink()
        if json_path.exists():
            json_path.unlink()
        return False


def find_piper_model(model_name: str, auto_download: bool = True) -> Optional[Path]:
    """
    Find Piper model in models directory, download if missing
    
    Args:
        model_name: Model name without .onnx extension
        auto_download: If True, download model if not found
    
    Returns:
        Path to .onnx file or None if not found
    """
    model_file = f"{model_name}.onnx"
    local_path = MODELS_DIR / model_file
    
    if local_path.exists():
        return local_path
    
    # Try to download
    if auto_download:
        if download_model(model_name):
            return local_path
    
    return None


def find_piper_executable() -> Optional[str]:
    """
    Find Piper executable across platforms
    
    Returns:
        Path to piper executable or None if not found
    """
    # Check if piper is in PATH
    from shutil import which
    piper_cmd = which("piper")
    if piper_cmd:
        return piper_cmd
    
    # Windows-specific: check common install locations
    if platform.system() == "Windows":
        common_paths = [
            Path.home() / "AppData/Local/Programs/piper/piper.exe",
            Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "piper/piper.exe",
            Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")) / "piper/piper.exe",
        ]
        for p in common_paths:
            if p.exists():
                return str(p)
    
    return None


class SpeechAudioIO:
    """
    Simple speech I/O: record → transcribe → synthesize → playback
    Cross-platform: Linux, macOS, Windows
    """
    
    def __init__(
        self,
        mic_index: Optional[int] = None,
        speaker_index: Optional[int] = None,
        whisper_model: str = "tiny",
        whisper_device: Optional[str] = None,  # None = auto-detect, "cpu" = force CPU, "cuda" = force CUDA
        whisper_language: Optional[str] = None,  # Force specific language (e.g., "en"), None = auto-detect
        piper_model_en: Optional[str] = "en_US-amy-medium",  # English voice
        piper_model_de: Optional[str] = "de_DE-thorsten_emotional-medium",  # German voice
        piper_model_fallback: Optional[str] = "en_US-lessac-medium",  # Fallback for unknown languages
        auto_download_models: bool = True,
        silence_duration: float = 1.2,
        vad_aggressiveness: int = 2,
        min_speech_duration: float = 0.8,
        min_rms_threshold: float = 0.015
    ):
        self.mic_index = mic_index
        self.speaker_index = speaker_index
        self.whisper_model_name = whisper_model
        self.whisper_device = whisper_device  # Store device preference (None = auto)
        self.whisper_language = whisper_language  # Store language preference (None = auto-detect)
        self.playback_sample_rate = 48000  # Always output at 48kHz
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        self.min_rms_threshold = min_rms_threshold
        
        # Audio playback state (reference counter for nested muting)
        self.playback_count = 0  # Number of active playbacks
        self.playback_lock = threading.Lock()  # Thread-safe counter
        
        # VAD & recording
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.is_recording = False
        self.is_listening = False  # ADD THIS LINE
        self.stop_event = threading.Event()
        self.audio_queue = queue.Queue()
        self.on_transcription_callback = None
        
        # Whisper (lazy load)
        self.whisper_model = None
        
        # Piper TTS
        self.piper_models = {}  # {"en": path, "de": path, ...}
        self.piper_processes = {}  # Cache running piper processes
        
        # Download models if needed
        if auto_download_models:
            self._download_piper_models(piper_model_en, piper_model_de, piper_model_fallback)
        
        # pyo server (lazy boot)
        self.pyo_server = None
    
    def _resolve_piper_model(self, model_name_or_path: str, language_name: str, auto_download: bool) -> Optional[Path]:
        """Resolve Piper model path, download if needed"""
        if Path(model_name_or_path).exists():
            # Full path provided
            print(f"✅ Using {language_name} Piper model: {model_name_or_path}")
            return Path(model_name_or_path)
        else:
            # Model name provided - search/download
            found_model = find_piper_model(model_name_or_path, auto_download=auto_download)
            if found_model:
                print(f"✅ Using {language_name} Piper model: {found_model}")
                return found_model
            else:
                print(f"⚠️  {language_name} Piper model '{model_name_or_path}' not found.")
                return None

    @staticmethod
    def list_devices():
        """List all available audio devices"""
        print("=" * 70)
        print("🎤 INPUT devices:")
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0:
                print(f"  [{i:2d}] {d['name']}")
        
        print("\n🔊 OUTPUT devices:")
        for i, d in enumerate(devices):
            if d['max_output_channels'] > 0:
                print(f"  [{i:2d}] {d['name']}")
        print("=" * 70)
    
    def _load_whisper(self):
        """Load Whisper model on first use (lazy loading)"""
        if self.whisper_model is not None:
            return
        
        print(f"🧠 Loading Whisper model '{self.whisper_model_name}'...")
        
        # Determine device
        if self.whisper_device is None:
            # Auto-detect: prefer CUDA if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"   Auto-detecting device: {device}")
        else:
            # User specified device
            device = self.whisper_device
            print(f"   Using forced device: {device}")
        
        if device == "cuda":
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"   Using GPU: {gpu_name}")
            else:
                print("   ⚠️  CUDA requested but not available, falling back to CPU")
                device = "cpu"
        
        try:
            self.whisper_model = whisper.load_model(self.whisper_model_name, device=device)
            print(f"✅ Whisper model loaded on {device}")
        except torch.cuda.OutOfMemoryError:
            print("⚠️  GPU out of memory, falling back to CPU")
            device = "cpu"
            self.whisper_model = whisper.load_model(self.whisper_model_name, device=device)
            print(f"✅ Whisper model loaded on {device}")
    
    def _boot_pyo(self):
        """Boot pyo server for playback at 48kHz"""
        if self.pyo_server is None or not self.pyo_server.getIsBooted():
            # OS-specific audio backend
            if self.platform == "Darwin":  # macOS
                audio_backend = "coreaudio"
            elif self.platform == "Windows":
                audio_backend = "portaudio"
            else:  # Linux
                audio_backend = "portaudio"
            
            self.pyo_server = Server(
                sr=self.playback_sample_rate,  # Always 48kHz
                nchnls=1,
                duplex=0,
                audio=audio_backend,
                buffersize=512
            )
            if self.speaker_index is not None:
                self.pyo_server.setOutputDevice(self.speaker_index)
            
            try:
                self.pyo_server.boot()
                self.pyo_server.start()
                print(f"✅ pyo server running @ {self.playback_sample_rate} Hz ({audio_backend})")
            except Exception as e:
                print(f"❌ Failed to boot pyo at {self.playback_sample_rate} Hz: {e}")
                raise
    
    def _whisper_worker(self):
        """Background thread for Whisper transcription"""
        self._load_whisper()
        
        while not self.stop_event.is_set():
            try:
                audio_buffer = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            if audio_buffer is None:
                break
            
            # Transcribe
            audio_float = audio_buffer.astype(np.float32) / 32768.0
            
            # fp16 only works on CUDA GPU, not on CPU or MPS
            # Check if model is actually on CUDA (not just if CUDA is available)
            device = str(self.whisper_model.device)
            use_fp16 = device.startswith('cuda')
            
            # Build transcribe options
            transcribe_options = {'fp16': use_fp16}
            if self.whisper_language:
                transcribe_options['language'] = self.whisper_language
                print(f"[WHISPER] Forcing language: {self.whisper_language}")
            
            result = self.whisper_model.transcribe(audio_float, **transcribe_options)
            
            text = result["text"].strip()
            language = result.get("language", "unknown")
            
            # Callback
            if text and self.on_transcription_callback:
                self.on_transcription_callback(language, text)
    
    def start_listening(
        self,
        on_audio: Optional[Callable[[np.ndarray], None]] = None,
        on_transcription: Optional[Callable[[str, str], None]] = None,
        transcribe: bool = True
    ):
        """
        Start listening with VAD-based segmentation
        
        Args:
            on_audio: Callback(audio_buffer) when speech segment detected
            on_transcription: Callback(language, text) when transcribed
            transcribe: If True, automatically transcribe segments
        """
        if self.is_listening:
            print("⚠️  Already listening")
            return
        
        self.on_audio_callback = on_audio
        self.on_transcription_callback = on_transcription
        self.stop_event.clear()
        self.is_listening = True
        
        # Start Whisper worker if needed
        if transcribe:
            whisper_thread = threading.Thread(target=self._whisper_worker, daemon=True)
            whisper_thread.start()
        
        # Recording state
        buffer = np.zeros((0,), dtype=np.int16)
        silence_frames = 0
        silence_limit = int(self.silence_duration * 1000 / self.frame_ms)
        speech_frames = 0
        min_speech_frames = int(0.3 * 1000 / self.frame_ms)  # 300ms to confirm speech
        is_recording = False  # Track if we're in a valid recording session
        
        def audio_callback(indata, frames, time_info, status):
            nonlocal buffer, silence_frames, speech_frames, is_recording
            
            if self.stop_event.is_set():
                raise sd.CallbackStop()
            
            # 🔇 Skip recording if speaker is playing (mic muted)
            if self.is_playing:
                # Clear any accumulated buffer during playback
                if len(buffer) > 0:
                    buffer = np.zeros((0,), dtype=np.int16)
                    silence_frames = 0
                    speech_frames = 0
                    is_recording = False
                return  # Skip processing this frame
            
            audio = (indata[:, 0] * 32767).astype(np.int16)
            is_speech = self.vad.is_speech(audio.tobytes(), self.record_sample_rate)
            
            if is_speech:
                speech_frames += 1
                silence_frames = 0
                
                # Start recording from FIRST speech frame (don't wait!)
                buffer = np.concatenate((buffer, audio))
                
                # Mark as valid recording after 300ms of consecutive speech
                if speech_frames >= min_speech_frames:
                    is_recording = True
            else:
                # Silence detected
                if is_recording:
                    # We were in a valid recording, add silence frames
                    buffer = np.concatenate((buffer, audio))
                    silence_frames += 1
                else:
                    # Not enough speech yet, discard buffer
                    if speech_frames > 0:
                        # Reset - it was just noise
                        buffer = np.zeros((0,), dtype=np.int16)
                        speech_frames = 0
            
            # End of speech segment (only if we had a valid recording)
            if is_recording and silence_frames > silence_limit and len(buffer) > 0:
                duration = len(buffer) / self.record_sample_rate
                rms = np.sqrt(np.mean(buffer.astype(np.float32)**2)) / 32768.0
                
                # Filter short/silent segments
                if duration >= self.min_speech_duration and rms >= self.min_rms_threshold:
                    # Audio callback
                    if self.on_audio_callback:
                        self.on_audio_callback(buffer.copy())
                    
                    # Queue for transcription
                    if transcribe:
                        self.audio_queue.put(buffer.copy())
                else:
                    # Rejected: too short or too quiet
                    print(f"🔇 Rejected: {duration:.1f}s, RMS={rms:.3f} (noise/too short)")
                
                # Reset state
                buffer = np.zeros((0,), dtype=np.int16)
                silence_frames = 0
                speech_frames = 0
                is_recording = False
        
        print("🎙️  Listening...")
        with sd.InputStream(
            channels=1,
            samplerate=self.record_sample_rate,  # Record at 16kHz
            dtype="float32",
            callback=audio_callback,
            blocksize=self.frame_size,
            device=self.mic_index
        ):
            while not self.stop_event.is_set():
                time.sleep(0.1)
        
        self.is_listening = False
    
    def stop_listening(self):
        """Stop listening"""
        print("🛑 Stopping listener...")
        self.stop_event.set()
        self.audio_queue.put(None)
    
    def transcribe_buffer(self, audio_buffer: np.ndarray) -> tuple[str, str]:
        """
        Transcribe audio buffer (expects 16kHz int16)
        
        Returns:
            (language, text)
        """
        self._load_whisper()
        
        # Whisper expects 16kHz float32
        audio_float = audio_buffer.astype(np.float32) / 32768.0
        
        # fp16 only works on CUDA GPU, not on CPU or MPS
        # Check if model is actually on CUDA (not just if CUDA is available)
        device = str(self.whisper_model.device)
        use_fp16 = device.startswith('cuda')
        
        # Build transcribe options
        transcribe_options = {'fp16': use_fp16}
        if self.whisper_language:
            transcribe_options['language'] = self.whisper_language
        
        result = self.whisper_model.transcribe(audio_float, **transcribe_options)
        
        return result.get("language", "unknown"), result["text"].strip()
    
    def synthesize_speech(self, text: str, language: Optional[str] = None, speaker: Optional[int] = None) -> np.ndarray:
        """
        Synthesize speech from text using Piper
        Auto-selects voice based on language
        Always returns 48kHz mono int16
        
        Args:
            text: Text to speak
            language: Language code (e.g., 'en', 'de') - auto-detected from Whisper if None
            speaker: Speaker ID number (e.g., 0-7 for thorsten_emotional) - model dependent
        
        Returns:
            Audio buffer (int16, 48kHz, mono)
        """
        if not self.piper_executable:
            return np.zeros(self.playback_sample_rate, dtype=np.int16)
        
        # Select model based on language
        piper_model = None
        
        if language and language in self.piper_models:
            # Language-specific model available
            piper_model = self.piper_models[language]
        elif self.piper_fallback_model:
            # Use fallback (lessac) for unknown languages
            piper_model = self.piper_fallback_model
        elif 'en' in self.piper_models:
            # Fallback to English if no lessac available
            piper_model = self.piper_models['en']
        elif self.piper_models:
            # Use first available
            piper_model = list(self.piper_models.values())[0]
        
        if not piper_model:
            return np.zeros(self.playback_sample_rate, dtype=np.int16)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Run Piper TTS
            cmd = [self.piper_executable, "--model", str(piper_model), "--output_file", tmp_path]
            
            # Add speaker ID if specified (must be a number)
            if speaker is not None:
                cmd.extend(["--speaker", str(speaker)])
            
            # Windows needs shell=True for some subprocess calls
            shell = (self.platform == "Windows")
            
            result = subprocess.run(
                cmd,
                input=text.encode('utf-8'),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                shell=shell
            )
            
            if result.returncode != 0:
                error_msg = result.stderr.decode('utf-8', errors='ignore')
                print(f"❌ Piper TTS failed: {error_msg}")
                return np.zeros(self.playback_sample_rate, dtype=np.int16)
            
            if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                print(f"❌ Piper did not generate audio file")
                return np.zeros(self.playback_sample_rate, dtype=np.int16)
            
            # Read audio
            import soundfile as sf
            audio, sr = sf.read(tmp_path, dtype='int16')
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio[:, 0]
            
            # Always resample to 48kHz
            if sr != self.playback_sample_rate:
                from scipy import signal
                num_samples = int(len(audio) * self.playback_sample_rate / sr)
                audio = signal.resample(audio, num_samples).astype(np.int16)
            
            return audio
        
        except Exception as e:
            print(f"❌ TTS synthesis failed: {e}")
            return np.zeros(self.playback_sample_rate, dtype=np.int16)
        
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
    
    def speak(self, text: str, language: Optional[str] = None, speaker: Optional[int] = None, effect: Optional[dict] = None, mute_mic: bool = True, blocking: bool = False):
        """
        Synthesize and play text (always 48kHz output)
        Auto-selects voice based on language
        
        Args:
            text: Text to speak
            language: Language code (e.g., 'en', 'de') - auto-detected if None
            speaker: Speaker ID (number) for multi-speaker models
            effect: Audio effect to apply, e.g. {"effect": "echo", "value": 0.5} or {"effect": "pitch", "value": 1.2}
            mute_mic: If True, mutes microphone during speech
            blocking: If True, wait for speech to finish. If False (default), return immediately
        """
        voice = ""
        if language in self.piper_models:
            voice = f" [{language.upper()}]"
        if speaker is not None:
            voice += f" (speaker {speaker})"
        if effect:
            voice += f" +{effect.get('effect', '?')}({effect.get('value', 0)})"
        print(f"🗣️{voice} Speaking: {text}")
        
        # Increment playback counter (mute mic)
        if mute_mic:
            self._increment_playback()
        
        if blocking:
            # Synchronous mode - wait for speech to finish
            try:
                self._speak_sync(text, language, speaker, effect)
            finally:
                if mute_mic:
                    self._decrement_playback()
        else:
            # Asynchronous mode - play in background thread
            speech_thread = threading.Thread(
                target=self._speak_async_wrapper,
                args=(text, language, speaker, effect, mute_mic),
                daemon=True
            )
            speech_thread.start()
    
    def _speak_sync(self, text: str, language: Optional[str], speaker: Optional[int] = None, effect: Optional[dict] = None) -> float:
        """Internal synchronous speech implementation - returns duration"""
        audio = self.synthesize_speech(text, language, speaker)  # Returns 48kHz mono int16
        duration = 0.0
        if len(audio) > 0:
            # Apply effect if specified
            if effect and effect.get('effect'):
                audio = self._apply_effect(audio, effect)
            
            duration = len(audio) / self.playback_sample_rate
            self.play_buffer(audio)
        
        return duration
    
    def _speak_async_wrapper(self, text: str, language: Optional[str], speaker: Optional[int], effect: Optional[dict], mute_mic: bool):
        """Wrapper for async speech that handles playback counter"""
        try:
            duration = self._speak_sync(text, language, speaker, effect)
        finally:
            if mute_mic:
                self._decrement_playback()
    
    def play_buffer(self, audio_buffer: np.ndarray):
        """
        Play audio buffer through pyo (expects 48kHz int16)
        
        Args:
            audio_buffer: Audio data (48kHz, int16)
        """
        self._boot_pyo()
        
        # Convert to float32 [-1.0, 1.0]
        if audio_buffer.dtype == np.int16:
            audio_float = audio_buffer.astype(np.float32) / 32768.0
        else:
            audio_float = audio_buffer
        
        duration = len(audio_float) / self.playback_sample_rate
        
        try:
            # Play through pyo
            table = DataTable(size=len(audio_float), init=audio_float.tolist())
            player = TableRead(table=table, freq=1.0/duration, loop=False, mul=0.8)
            player.out()
            
            time.sleep(duration + 0.1)
            player.stop()
        except Exception as e:
            print(f"❌ Playback failed: {e}")
    
    def _download_piper_models(self, piper_model_en, piper_model_de, piper_model_fallback):
        """Download and register Piper models"""
        # Find piper executable
        self.piper_executable = find_piper_executable()
        if not self.piper_executable:
            print("⚠️  Piper executable not found. TTS disabled.")
            print("   Install from: https://github.com/rhasspy/piper")
            return
        
        # Detect platform
        self.platform = platform.system()
        
        # Register models
        self.piper_fallback_model = None
        
        # English model
        if piper_model_en:
            model_path = self._resolve_piper_model(piper_model_en, "English", auto_download=True)
            if model_path:
                self.piper_models['en'] = model_path
        
        # German model
        if piper_model_de:
            model_path = self._resolve_piper_model(piper_model_de, "German", auto_download=True)
            if model_path:
                self.piper_models['de'] = model_path
        
        # Fallback model
        if piper_model_fallback:
            model_path = self._resolve_piper_model(piper_model_fallback, "Fallback", auto_download=True)
            if model_path:
                self.piper_fallback_model = model_path
                # Also register as English if no English model
                if 'en' not in self.piper_models:
                    self.piper_models['en'] = model_path
        
        if not self.piper_models:
            print("⚠️  No Piper models available. TTS disabled.")
    
    @property
    def is_playing(self) -> bool:
        """Check if any playback is active"""
        with self.playback_lock:
            return self.playback_count > 0
    
    def _increment_playback(self):
        """Increment playback counter (mute mic)"""
        with self.playback_lock:
            self.playback_count += 1
            if self.playback_count == 1:
                print(f"[MIC] 🔇 Muted (playback count: {self.playback_count})")
            else:
                print(f"[MIC] 🔇 Already muted (playback count: {self.playback_count})")
    
    def _decrement_playback(self):
        """Decrement playback counter (unmute mic when reaches 0)"""
        with self.playback_lock:
            self.playback_count = max(0, self.playback_count - 1)
            if self.playback_count == 0:
                print(f"[MIC] 🎤 Unmuted (playback count: 0)")
            else:
                print(f"[MIC] 🔇 Still muted (playback count: {self.playback_count})")
    
    # VAD settings (for 16kHz recording)
    @property
    def record_sample_rate(self):
        return 16000  # Whisper expects 16kHz
    
    @property
    def frame_ms(self):
        return 30  # 30ms frames for VAD
    
    @property
    def frame_size(self):
        return int(self.record_sample_rate * self.frame_ms / 1000)
    
    def _apply_effect(self, audio: np.ndarray, effect: dict) -> np.ndarray:
        """Apply audio effect to buffer"""
        effect_type = effect.get('effect', '').lower()
        value = effect.get('value', 0.5)
        
        if effect_type == 'echo':
            # Simple echo effect
            delay_samples = int(0.2 * self.playback_sample_rate)  # 200ms delay
            echo = np.zeros(len(audio) + delay_samples, dtype=np.int16)
            echo[:len(audio)] = audio
            echo[delay_samples:] += (audio * value).astype(np.int16)
            return echo[:len(audio)]
        
        elif effect_type in ['low', 'tief']:
            # Low pitch (slower playback)
            from scipy import signal
            factor = 0.85  # Lower pitch
            new_length = int(len(audio) / factor)
            return signal.resample(audio, new_length).astype(np.int16)
        
        elif effect_type in ['high', 'hoch']:
            # High pitch (faster playback)
            from scipy import signal
            factor = 1.15  # Higher pitch
            new_length = int(len(audio) / factor)
            return signal.resample(audio, new_length).astype(np.int16)
        
        else:
            return audio
    
    def cleanup(self):
        """Cleanup resources"""
        print("🧹 Cleaning up SpeechAudioIO...")
        
        # Stop listening if active
        if self.is_listening:
            self.stop_listening()
        
        # Shutdown pyo server
        if self.pyo_server is not None:
            try:
                self.pyo_server.stop()
                self.pyo_server.shutdown()
                print("✅ pyo server stopped")
            except Exception as e:
                print(f"⚠️  Error stopping pyo: {e}")
            self.pyo_server = None
        
        # Clear Whisper model (free GPU memory)
        if self.whisper_model is not None:
            del self.whisper_model
            self.whisper_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("✅ Whisper model unloaded")
        
        print("✅ Cleanup complete")

