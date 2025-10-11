#!/usr/bin/env python3
"""
Simple Speech Audio I/O class
- Records with VAD (Voice Activity Detection)
- Transcribes with Whisper
- Synthesizes speech with Piper
- Plays audio with pyo
"""

import sounddevice as sd
import numpy as np
import whisper
import webrtcvad
import subprocess
import tempfile
import os
import threading
import queue
import time
from pathlib import Path
from typing import Callable, Optional

# Suppress pyo warnings
os.environ['PYO_GUI_WX'] = '0'
from pyo import Server, DataTable, TableRead

# Default paths for models
SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "models"
DEFAULT_PIPER_MODEL = MODELS_DIR / "en_US-lessac-medium.onnx"

# Also check system locations
SYSTEM_PIPER_PATHS = [
    Path.home() / ".local/share/piper/voices",
    Path("/usr/local/share/piper/voices"),
    Path("/usr/share/piper/voices"),
]


def find_piper_model(model_name: str = "en_US-lessac-medium") -> Optional[Path]:
    """
    Search for Piper model in multiple locations:
    1. ./models/ (project local)
    2. ~/.local/share/piper/voices/ (user install)
    3. /usr/local/share/piper/voices/ (system install)
    
    Args:
        model_name: Model name without .onnx extension
    
    Returns:
        Path to .onnx file or None if not found
    """
    model_file = f"{model_name}.onnx"
    
    # Check project local models directory
    local_path = MODELS_DIR / model_file
    if local_path.exists():
        return local_path
    
    # Check system locations
    for sys_path in SYSTEM_PIPER_PATHS:
        candidate = sys_path / model_file
        if candidate.exists():
            return candidate
    
    return None


class SpeechAudioIO:
    """
    Simple speech I/O: record → transcribe → synthesize → playback
    """
    
    def __init__(
        self,
        mic_index: Optional[int] = None,
        speaker_index: Optional[int] = None,
        whisper_model: str = "tiny",
        piper_model_en: Optional[str] = "en_US-amy-medium",  # English voice
        piper_model_de: Optional[str] = "de_DE-thorsten_emotional-medium",  # German voice
        sample_rate: int = 48000,
        silence_duration: float = 0.8,
        vad_aggressiveness: int = 2
    ):
        self.mic_index = mic_index
        self.speaker_index = speaker_index
        self.whisper_model_name = whisper_model
        self.playback_sample_rate = 48000  # Always output at 48kHz
        self.silence_duration = silence_duration
        
        # Resolve Piper model paths for different languages
        self.piper_models = {}
        
        # English model
        if piper_model_en:
            en_model = self._resolve_piper_model(piper_model_en, "English")
            if en_model:
                self.piper_models['en'] = en_model
        
        # German model
        if piper_model_de:
            de_model = self._resolve_piper_model(piper_model_de, "German")
            if de_model:
                self.piper_models['de'] = de_model
        
        if not self.piper_models:
            print("⚠️  TTS disabled (no Piper models found)")
        else:
            print(f"✅ Loaded TTS voices for: {', '.join(self.piper_models.keys())}")
        
        # VAD setup - use 16kHz for recording (Whisper prefers 16kHz)
        self.record_sample_rate = 16000
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.frame_ms = 30  # WebRTC VAD requires 10, 20, or 30ms
        self.frame_size = int(self.record_sample_rate * self.frame_ms / 1000)
        
        # State
        self.whisper_model = None
        self.pyo_server = None
        self.is_listening = False
        self.is_playing = False
        self.stop_event = threading.Event()
        self.audio_queue = queue.Queue()
        
        # Callbacks
        self.on_audio_callback: Optional[Callable[[np.ndarray], None]] = None
        self.on_transcription_callback: Optional[Callable[[str, str], None]] = None
    
    def _resolve_piper_model(self, model_name_or_path: str, language_name: str) -> Optional[Path]:
        """Resolve Piper model path"""
        if Path(model_name_or_path).exists():
            # Full path provided
            print(f"✅ Using {language_name} Piper model: {model_name_or_path}")
            return Path(model_name_or_path)
        else:
            # Model name provided - search for it
            found_model = find_piper_model(model_name_or_path)
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
    
    @staticmethod
    def setup_models_dir():
        """Create models directory and print download instructions"""
        MODELS_DIR.mkdir(exist_ok=True)
        
        print("\n" + "=" * 70)
        print("📦 Piper Model Setup")
        print("=" * 70)
        print(f"Models directory: {MODELS_DIR}")
        print("\nTo download a voice model:")
        print(f"\n  cd {MODELS_DIR}")
        print("  wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx")
        print("  wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json")
        print("\nMore voices: https://rhasspy.github.io/piper-samples/")
        print("=" * 70 + "\n")
    
    def _load_whisper(self):
        """Lazy load Whisper model"""
        if self.whisper_model is None:
            print(f"🧠 Loading Whisper model '{self.whisper_model_name}'...")
            self.whisper_model = whisper.load_model(self.whisper_model_name)
            print("✅ Whisper ready")
    
    def _boot_pyo(self):
        """Boot pyo server for playback at 48kHz"""
        if self.pyo_server is None or not self.pyo_server.getIsBooted():
            self.pyo_server = Server(
                sr=self.playback_sample_rate,  # Always 48kHz
                nchnls=1,
                duplex=0,
                audio="portaudio",
                buffersize=512
            )
            if self.speaker_index is not None:
                self.pyo_server.setOutputDevice(self.speaker_index)
            
            try:
                self.pyo_server.boot()
                self.pyo_server.start()
                print(f"✅ pyo server running @ {self.playback_sample_rate} Hz")
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
            result = self.whisper_model.transcribe(audio_float, fp16=False)
            
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
        
        def audio_callback(indata, frames, time_info, status):
            nonlocal buffer, silence_frames
            
            if self.stop_event.is_set():
                raise sd.CallbackStop()
            
            # 🔇 Mute recording if speaker is playing
            if self.is_playing:
                return
            
            audio = (indata[:, 0] * 32767).astype(np.int16)
            is_speech = self.vad.is_speech(audio.tobytes(), self.record_sample_rate)
            
            buffer = np.concatenate((buffer, audio))
            
            if is_speech:
                silence_frames = 0
            else:
                silence_frames += 1
            
            # End of speech segment
            if silence_frames > silence_limit and len(buffer) > 0:
                duration = len(buffer) / self.record_sample_rate
                rms = np.sqrt(np.mean(buffer.astype(np.float32)**2)) / 32768.0
                
                # Filter short/silent segments
                if duration > 1.0 and rms > 0.01:
                    # Audio callback
                    if self.on_audio_callback:
                        self.on_audio_callback(buffer.copy())
                    
                    # Queue for transcription
                    if transcribe:
                        self.audio_queue.put(buffer.copy())
                
                buffer = np.zeros((0,), dtype=np.int16)
                silence_frames = 0
        
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
        result = self.whisper_model.transcribe(audio_float, fp16=False)
        
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
        if not self.piper_models:
            print("⚠️  TTS disabled (no Piper models)")
            return np.zeros(self.playback_sample_rate, dtype=np.int16)
        
        # Select model based on language
        if language and language in self.piper_models:
            piper_model = self.piper_models[language]
        elif 'en' in self.piper_models:
            piper_model = self.piper_models['en']  # Default to English
        else:
            piper_model = list(self.piper_models.values())[0]  # Use first available
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Run Piper TTS
            cmd = ["piper", "--model", str(piper_model), "--output_file", tmp_path]
            
            # Add speaker ID if specified (must be a number)
            if speaker is not None:
                cmd.extend(["--speaker", str(speaker)])
            
            result = subprocess.run(
                cmd,
                input=text.encode('utf-8'),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10
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
    
    def play_buffer(self, audio_buffer: np.ndarray, mute_mic: bool = True):
        """
        Play audio buffer through pyo (expects 48kHz int16)
        
        Args:
            audio_buffer: Audio data (48kHz, int16)
            mute_mic: If True, mutes microphone during playback
        """
        self._boot_pyo()
        
        # Convert to float32 [-1.0, 1.0]
        if audio_buffer.dtype == np.int16:
            audio_float = audio_buffer.astype(np.float32) / 32768.0
        else:
            audio_float = audio_buffer
        
        duration = len(audio_float) / self.playback_sample_rate
        
        # 🔇 Mute microphone during playback
        if mute_mic:
            self.is_playing = True
        
        try:
            # Play through pyo
            table = DataTable(size=len(audio_float), init=audio_float.tolist())
            player = TableRead(table=table, freq=1.0/duration, loop=False, mul=0.8)
            player.out()
            
            time.sleep(duration + 0.1)
            player.stop()
        except Exception as e:
            print(f"❌ Playback failed: {e}")
        finally:
            # 🎤 Re-enable microphone
            if mute_mic:
                self.is_playing = False
    
    def speak(self, text: str, language: Optional[str] = None, speaker: Optional[int] = None, mute_mic: bool = True):
        """
        Synthesize and play text (always 48kHz output)
        Auto-selects voice based on language
        
        Args:
            text: Text to speak
            language: Language code (e.g., 'en', 'de') - auto-detected if None
            speaker: Speaker ID (number) for multi-speaker models
            mute_mic: If True, mutes microphone during speech
        """
        voice = ""
        if language in self.piper_models:
            voice = f" [{language.upper()}]"
        if speaker is not None:
            voice += f" (speaker {speaker})"
        print(f"🗣️{voice} Speaking: {text}")
        
        audio = self.synthesize_speech(text, language, speaker)  # Returns 48kHz mono int16
        if len(audio) > 0:
            self.play_buffer(audio, mute_mic=mute_mic)
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_listening()
        if self.pyo_server and self.pyo_server.getIsBooted():
            self.pyo_server.stop()
            self.pyo_server.shutdown()


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import signal
    import sys
    
    # List devices
    SpeechAudioIO.list_devices()
    
    # Setup models directory
    SpeechAudioIO.setup_models_dir()
    
    # Create instance with both English and German voices
    io = SpeechAudioIO(
        whisper_model="tiny",
        piper_model_en="en_US-amy-medium",
        piper_model_de="de_DE-thorsten_emotional-medium"
    )
    
    # Speaker IDs for thorsten_emotional:
    # 0: amused, 1: angry, 2: disgusted, 3: drunk, 4: neutral, 5: sleepy, 6: surprised, 7: whisper
    SPEAKER_WHISPER = 7
    
    # Transcription callback - auto-selects voice based on detected language
    def on_transcription(language: str, text: str):
        print(f"\n🗣️  [{language.upper()}] {text}\n")
        
        # Determine which language/speaker to use
        if language == "de" and "de" in io.piper_models:
            # Use German voice with whisper mode
            io.speak(text, language="de", speaker=SPEAKER_WHISPER)
        elif language == "en" and "en" in io.piper_models:
            # Use English voice (no special speaker)
            io.speak(text, language="en")
        elif io.piper_models:
            # Unsupported language - fallback to English (or first available)
            fallback_lang = "en" if "en" in io.piper_models else list(io.piper_models.keys())[0]
            print(f"   ⚠️  Language '{language}' not supported, using {fallback_lang.upper()} voice")
            io.speak(text, language=fallback_lang)
        else:
            print(f"   ❌ No TTS voices available")
    
    # Graceful shutdown
    def shutdown(signum, frame):
        print("\n👋 Shutting down...")
        io.cleanup()
        exit(0)
    
    signal.signal(signal.SIGINT, shutdown)
    
    # Start listening
    print("\n🎤 Say something in English or German!")
    io.start_listening(on_transcription=on_transcription)