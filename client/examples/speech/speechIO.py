"""
SpeechIO - Audio Input/Output Handler for Speech Recognition and Synthesis

Handles:
- Speech-to-Text (STT) using Whisper
- Text-to-Speech (TTS) using Piper
- Audio I/O using Pyo

Author: Max Rheiner
Date: October 2025
"""

import os
import sys
import time
import threading
import tempfile
import wave
import subprocess
import platform
import numpy as np

# Suppress pyo GUI warnings
os.environ['PYO_GUI_WX'] = '0'

try:
    from pyo import *
    PYO_AVAILABLE = True
except ImportError:
    PYO_AVAILABLE = False


class SpeechIO:
    """
    Audio I/O handler for speech recognition and synthesis
    
    Features:
    - Continuous voice recognition (Whisper)
    - Text-to-speech playback (Piper)
    - Configurable audio devices
    - Callback-based architecture
    - Cross-platform support (Linux, macOS, Windows)
    """
    
    def __init__(self, whisper_model="base", piper_voice="en_US-lessac-medium",
                 sample_rate=48000, input_device=0, output_device=0,
                 record_seconds=3, stt_callback=None):
        """
        Initialize SpeechIO with Whisper (STT) and Piper (TTS)
        
        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            piper_voice: Piper voice name
            sample_rate: Audio sample rate in Hz
            input_device: Input device index for microphone
            output_device: Output device index for speakers
            record_seconds: Duration to record for each STT chunk
            stt_callback: Callback function for STT results
        """
        print("[SPEECHIO] Initializing...")
        
        self.whisper_model = whisper_model
        self.piper_voice = piper_voice
        self.sample_rate = sample_rate
        self.input_device = input_device
        self.output_device = output_device
        self.record_seconds = record_seconds
        self.stt_callback = stt_callback
        
        # Platform detection
        self.platform = platform.system()
        
        # Component status
        self.whisper_available = False
        self.piper_available = False
        self.voice_model_available = False
        self.ready = False
        self.server = None
        
        # Voice recognition thread
        self.is_listening = False
        self.listen_thread = None
        self._pause_listening = False  # Add this flag
        
        try:
            # Check dependencies
            self._check_dependencies()
            
            # Initialize audio server
            if self.whisper_available and self.piper_available and self.voice_model_available:
                print("[SPEECHIO] Dependencies OK, initializing audio server...")
                self._init_audio_server()
            else:
                print("[SPEECHIO] Missing dependencies, speech disabled")
                self.ready = False
        except Exception as e:
            print(f"[SPEECHIO] Initialization error: {e}")
            import traceback
            traceback.print_exc()
            self.ready = False
            self.server = None
    
    def _get_voice_model_path(self, piper_voice):
        """Get platform-specific voice model path"""
        if self.platform == 'Windows':
            # Windows: %APPDATA%\piper\voices
            voice_dir = os.path.join(os.environ.get('APPDATA', ''), 'piper', 'voices')
        elif self.platform == 'Darwin':
            # macOS: ~/Library/Application Support/piper/voices
            voice_dir = os.path.expanduser("~/Library/Application Support/piper/voices")
        else:
            # Linux: ~/.local/share/piper/voices
            voice_dir = os.path.expanduser("~/.local/share/piper/voices")
        
        return os.path.join(voice_dir, f"{piper_voice}.onnx")
    
    def _get_device_info(self, device_id, kind='input'):
        """Return {'sr': int or None, 'channels': int or None} for device_id using pa_get_devices_infos()"""
        try:
            devices_info = pa_get_devices_infos()
            if not devices_info or len(devices_info) != 2:
                return {'sr': None, 'channels': None}
            input_devices, output_devices = devices_info
            if kind == 'input':
                devs = input_devices
            else:
                devs = output_devices
            info = devs.get(device_id)
            if not info:
                return {'sr': None, 'channels': None}
            sr = int(info.get('default sr')) if info.get('default sr') else None
            # Try several possible keys for channel count
            channels = None
            for k in ('max input channels', 'max output channels', 'channels', 'nchnls'):
                if k in info:
                    try:
                        channels = int(info[k])
                        break
                    except:
                        pass
            # Fallback: assume stereo if unknown
            if channels is None:
                channels = 2
            return {'sr': sr, 'channels': channels}
        except Exception:
            return {'sr': None, 'channels': None}

    def _init_audio_server(self):
        """Initialize Pyo audio server - try matching device native rates and channels"""
        try:
            # On Linux, use PulseAudio's default device (device 12 usually maps to PulseAudio default)
            if self.platform == 'Linux':
                print("[SPEECHIO] Linux detected, configuring for PulseAudio...")
                
                # Force device 12 which maps to PulseAudio default
                actual_input_device = 12 if (self.input_device is None or self.input_device < 0) else self.input_device
                actual_output_device = 12 if (self.output_device is None or self.output_device < 0) else self.output_device
                
                print(f"[SPEECHIO] Using input device {actual_input_device}, output device {actual_output_device}")
                print(f"[SPEECHIO] Device 12 = PulseAudio default from your KDE settings")
                
                # Try common sample rates with duplex=1 for both input and output
                for sr in [48000, 44100, 32000]:
                    for out_ch in [2, 1]:
                        try:
                            print(f"[SPEECHIO] Trying sr={sr} Hz, out_ch={out_ch}")
                            self.server = Server(
                                sr=sr, 
                                nchnls=out_ch, 
                                duplex=1,  # Enable both input and output
                                buffersize=512
                            )
                            
                            # Set both input and output devices
                            self.server.setInputDevice(actual_input_device)
                            self.server.setOutputDevice(actual_output_device)
                            
                            self.server.boot()
                            
                            if self.server.getIsBooted():
                                self.server.start()
                                if self.server.getIsStarted():
                                    self.ready = True
                                    self.sample_rate = sr
                                    self.input_device = actual_input_device
                                    self.output_device = actual_output_device
                                    print(f"[SPEECHIO] ✓ Audio initialized at {sr} Hz, {out_ch}ch")
                                    print(f"[SPEECHIO] Input device: {actual_input_device} (PulseAudio default)")
                                    print(f"[SPEECHIO] Output device: {actual_output_device} (PulseAudio default)")
                                    return
                                else:
                                    self.server.shutdown()
                                    self.server = None
                            else:
                                self.server = None
                                
                        except Exception as e:
                            if self.server:
                                try:
                                    self.server.shutdown()
                                except:
                                    pass
                            self.server = None
                            continue

            # Original ALSA code for other platforms or Linux fallback
            in_info = self._get_device_info(self.input_device, kind='input')
            out_info = self._get_device_info(self.output_device, kind='output')

            # If input device reports 0 channels, warn and abort initialization here
            if self.input_device is not None and in_info['channels'] == 0:
                print(f"[SPEECHIO] Input device {self.input_device} reports 0 input channels - choose another mic")
                self.ready = False
                return

            # Build candidate sample rates (prefer device natives)
            candidates = []
            if in_info['sr']:
                candidates.append(in_info['sr'])
            if out_info['sr']:
                candidates.append(out_info['sr'])
            candidates.extend([self.sample_rate, 48000, 44100, 32000, 16000, 22050])
            # keep order unique
            sample_rates = list(dict.fromkeys([r for r in candidates if r]))

            # Try combinations of sample rate and likely output channels (mono/stereo)
            tried = []
            for sr in sample_rates:
                for out_ch in (out_info['channels'] or 2, 1, 2):  # Try reported, mono, stereo
                    key = (sr, out_ch)
                    if key in tried:
                        continue
                    tried.append(key)
                    try:
                        print(f"[SPEECHIO] Trying server sr={sr} Hz, out_ch={out_ch}, in_dev={self.input_device}, out_dev={self.output_device}")
                        # nchnls sets output channels; duplex=1 enables input
                        self.server = Server(sr=sr, nchnls=out_ch, duplex=1, buffersize=512)
                        # set devices (if device id not valid, pyo/portaudio will raise)
                        try:
                            self.server.setOutputDevice(self.output_device)
                            self.server.setInputDevice(self.input_device)
                        except Exception as e:
                            # some backends ignore device setting; continue to boot attempt
                            print(f"[SPEECHIO] Warning: setting devices failed: {e}")

                        self.server.boot()

                        if self.server.getIsBooted():
                            self.server.start()
                            if self.server.getIsStarted():
                                self.ready = True
                                self.sample_rate = sr
                                print(f"[SPEECHIO] ✓ Audio initialized at {sr} Hz ({self.platform}), out_ch={out_ch}")
                                print(f"[SPEECHIO] Input device: {self.input_device} (in_ch={in_info['channels']}, sr={in_info['sr']})")
                                print(f"[SPEECHIO] Output device: {self.output_device} (out_ch={out_info['channels']}, sr={out_info['sr']})")
                                return
                            else:
                                print(f"[SPEECHIO] Server failed to start for sr={sr}, out_ch={out_ch}")
                                try:
                                    self.server.shutdown()
                                except:
                                    pass
                                self.server = None
                        else:
                            print(f"[SPEECHIO] Server failed to boot for sr={sr}, out_ch={out_ch}")
                            self.server = None

                    except Exception as e:
                        # specific portaudio errors will be shown in console
                        # print(f"[SPEECHIO] Init attempt failed (sr={sr}, out_ch={out_ch}): {e}")
                        try:
                            if self.server:
                                self.server.shutdown()
                        except:
                            pass
                        self.server = None
                        continue

            # Final attempt: Try using PulseAudio's "default" device by index
            # Device 6 and 12 from your list are "sysdefault" and "default"
            print("[SPEECHIO] All specific device configs failed. Trying 'default' devices...")
            for default_dev in [12, 6]:  # Try 'default' and 'sysdefault'
                for sr in [48000, 44100, 32000, 16000]:
                    for out_ch in [2, 1]:
                        try:
                            print(f"[SPEECHIO] Trying default device {default_dev} at {sr} Hz, {out_ch}ch")
                            self.server = Server(sr=sr, nchnls=out_ch, duplex=1, buffersize=512)
                            self.server.setOutputDevice(default_dev)
                            self.server.setInputDevice(default_dev)
                            self.server.boot()
                            
                            if self.server.getIsBooted():
                                self.server.start()
                                if self.server.getIsStarted():
                                    self.ready = True
                                    self.sample_rate = sr
                                    self.input_device = default_dev
                                    self.output_device = default_dev
                                    print(f"[SPEECHIO] ✓ Audio initialized using default device {default_dev} at {sr} Hz, {out_ch}ch")
                                    print(f"[SPEECHIO] This will use your OS default audio devices")
                                    return
                                else:
                                    self.server.shutdown()
                                    self.server = None
                        except:
                            if self.server:
                                try:
                                    self.server.shutdown()
                                except:
                                    pass
                            self.server = None
                            continue

            # If we reach here, none worked
            print("[SPEECHIO] ✗ Failed to initialize audio server with any tried configuration")
            print("[SPEECHIO] TROUBLESHOOTING:")
            print("  1. Check PulseAudio is running: systemctl --user status pulseaudio")
            print("  2. List devices: python speech.py --list-devices")
            print("  3. Try specific device: python speech.py --mic 12 --speaker 12")
            print("  4. Check: pactl list sinks short && pactl list sources short")
            print("  5. Kill other audio apps that might lock the device")
            self.ready = False
            self.server = None

        except Exception as e:
            print(f"[SPEECHIO] Audio initialization failed: {e}")
            import traceback
            traceback.print_exc()
            self.ready = False
            self.server = None
    
    def _check_dependencies(self):
        """
        Check if all required dependencies are available
        
        Returns:
            tuple: (whisper_ok, piper_ok, voice_model_ok)
        """
        # Check Whisper
        try:
            result = subprocess.run(['whisper', '--help'], 
                                  capture_output=True, timeout=5)
            whisper_ok = result.returncode == 0
        except:
            whisper_ok = False
        
        # Check Piper
        try:
            result = subprocess.run(['piper', '--help'], 
                                  capture_output=True, timeout=5)
            piper_ok = result.returncode == 0
        except:
            piper_ok = False
        
        # Get voice model path
        voice_model_path = self._get_voice_model_path(self.piper_voice)
        
        # Check voice model exists
        voice_model_ok = os.path.exists(voice_model_path)
        
        # Update component availability
        self.whisper_available = whisper_ok
        self.piper_available = piper_ok
        self.voice_model_available = voice_model_ok
        
        return whisper_ok, piper_ok, voice_model_ok
    
    def get_status(self):
        """
        Get current status
        
        Returns:
            dict: Status information
        """
        return {
            'ready': self.ready,
            'listening': self.is_listening,
            'whisper': self.whisper_available,
            'piper': self.piper_available,
            'pyo': self.ready and self.server is not None,
            'voice_model': self.voice_model_available,
            'sample_rate': self.sample_rate,
            'input_device': self.input_device,
            'output_device': self.output_device,
            'platform': self.platform
        }
    
    def speech_to_text(self, audio_file):
        """
        Convert audio file to text using Whisper
        
        Args:
            audio_file (str): Path to audio file
            
        Returns:
            str: Transcribed text, or empty string if failed
        """
        try:
            result = subprocess.run(
                ['whisper', audio_file, '--model', self.whisper_model, '--language', 'en'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Extract transcription from stdout using regex
                # Look for timestamp lines like: [00:00.000 --> 00:03.760]  Transcribed text
                import re
                stdout = result.stdout
                
                # Find all timestamp lines
                pattern = r'\[\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}\.\d{3}\]\s+(.+)'
                matches = re.findall(pattern, stdout)
                
                if matches:
                    # Join all transcription segments
                    text = ' '.join(matches).strip()
                    return text
                else:
                    print("[SPEECHIO] No transcription found in Whisper output")
                    return ""
            else:
                print(f"[SPEECHIO] Whisper error: {result.stderr}")
                return ""
                
        except subprocess.TimeoutExpired:
            print("[SPEECHIO] Whisper timeout")
            return ""
        except Exception as e:
            print(f"[SPEECHIO] STT failed: {e}")
            return ""
    
    def text_to_speech(self, text):
        """
        Convert text to speech and play it
        
        Args:
            text (str): Text to speak
            
        Returns:
            bool: True if successful, False otherwise
        """
        was_listening = False  # Initialize at the start
        try:
            print(f"[SPEECHIO] TTS: '{text}'")
            
            # Temporarily pause listening to avoid feedback
            was_listening = self.is_listening
            if was_listening:
                self._pause_listening = True  # Flag to pause, don't stop thread
            
            # Get voice model path
            voice_model_path = self._get_voice_model_path(self.piper_voice)
            
            # Generate speech with Piper
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
                wav_file = tmp_wav.name
            
            # Platform-specific Piper command
            if self.platform == 'Windows':
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_txt:
                    tmp_txt.write(text)
                    txt_file = tmp_txt.name
                piper_cmd = ['piper', '--model', voice_model_path, '--output_file', wav_file]
                with open(txt_file, 'r') as stdin_file:
                    result = subprocess.run(piper_cmd, stdin=stdin_file, capture_output=True, timeout=10)
                os.remove(txt_file)
            else:
                piper_cmd = f'echo "{text}" | piper --model {voice_model_path} --output_file {wav_file}'
                result = subprocess.run(piper_cmd, shell=True, capture_output=True, timeout=10)
            
            if result.returncode == 0 and os.path.exists(wav_file):
                try:
                    import sounddevice as sd
                    import soundfile as sf
                    
                    data, samplerate = sf.read(wav_file)
                    sd.play(data, samplerate, device=None)
                    sd.wait()
                    print("[SPEECHIO] Played via sounddevice (system default)")
                except ImportError:
                    if self.ready and self.server and self.server.getIsStarted():
                        self._play_audio(wav_file)
                    else:
                        self._play_audio_fallback(wav_file)
                except Exception as e:
                    print(f"[SPEECHIO] sounddevice error: {e}, using fallback")
                    if self.ready and self.server and self.server.getIsStarted():
                        self._play_audio(wav_file)
                    else:
                        self._play_audio_fallback(wav_file)
                
                os.remove(wav_file)
                
                # Resume listening after a delay
                if was_listening:
                    time.sleep(0.5)  # Wait for audio to clear
                    self._pause_listening = False
                
                return True
            else:
                print(f"[SPEECHIO] Piper error: {result.stderr.decode() if result.stderr else 'Unknown error'}")
                if was_listening:
                    self._pause_listening = False
                return False
                
        except Exception as e:
            print(f"[SPEECHIO] TTS failed: {e}")
            import traceback
            traceback.print_exc()
            if was_listening:
                self._pause_listening = False
            return False
    
    def _play_audio_fallback(self, wav_file):
        """Platform-specific audio playback fallback"""
        try:
            if self.platform == 'Linux':
                subprocess.run(['aplay', wav_file], capture_output=True)
            elif self.platform == 'Darwin':
                subprocess.run(['afplay', wav_file], capture_output=True)
            elif self.platform == 'Windows':
                # Windows: Use winsound or PowerShell
                try:
                    import winsound
                    winsound.PlaySound(wav_file, winsound.SND_FILENAME)
                except:
                    # PowerShell fallback
                    ps_cmd = f'(New-Object Media.SoundPlayer "{wav_file}").PlaySync()'
                    subprocess.run(['powershell', '-Command', ps_cmd], capture_output=True)
        except Exception as e:
            print(f"[SPEECHIO] Fallback playback error: {e}")
    
    def _play_audio(self, wav_file):
        """
        Play WAV file using Pyo
        
        Args:
            wav_file (str): Path to WAV file
        """
        try:
            if not self.server or not self.server.getIsStarted():
                self._play_audio_fallback(wav_file)
                return
            
            # Load and play sound
            snd_table = SndTable(wav_file)
            osc = Osc(table=snd_table, freq=snd_table.getRate(), mul=0.5)
            osc.out()
            
            duration = snd_table.getDur()
            time.sleep(duration + 0.1)
            
            osc.stop()
            del osc
            del snd_table
            
        except Exception as e:
            print(f"[SPEECHIO] Playback error: {e}")
            self._play_audio_fallback(wav_file)
    
    def start_listening(self):
        """Start continuous voice recognition"""
        if self.is_listening or not self.ready:
            return False
        
        if not self.server or not self.server.getIsStarted():
            print("[SPEECHIO] Audio server not ready")
            return False
        
        self.is_listening = True
        self.listen_thread = threading.Thread(
            target=self._listen_loop,
            daemon=True
        )
        self.listen_thread.start()
        print("[SPEECHIO] Voice recognition started")
        return True
    
    def stop_listening(self):
        """Stop continuous voice recognition"""
        if not self.is_listening:
            return
        
        self.is_listening = False
        if self.listen_thread:
            self.listen_thread.join(timeout=2)
        
        print("[SPEECHIO] Voice recognition stopped")
    
    def _listen_loop(self):
        """Continuous listening loop in separate thread"""
        try:
            print("[SPEECHIO] Listen loop started")
            while self.is_listening:
                # Skip recording if paused (during TTS playback)
                if self._pause_listening:
                    time.sleep(0.1)
                    continue
                    
                print("[SPEECHIO] Transcribing audio...")
                # Record audio - use the configured input device
                recording_table = NewTable(length=self.record_seconds, chnls=1)
                mic = Input(chnl=0)  # Channel 0 of the selected device
                recorder = TableRec(mic, table=recording_table)
                recorder.play()
                
                print(f"[SPEECHIO] Recording {self.record_seconds} seconds from device {self.input_device}...")
                time.sleep(self.record_seconds)
                
                recorder.stop()
                mic.stop()
                
                # Save to WAV file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
                    audio_file = tmp_audio.name
                
                samples = recording_table.getTable()
                samples_array = np.array(samples, dtype=np.float32)
                samples_int16 = (samples_array * 32767).astype(np.int16)
                
                # Debug: Check audio levels
                max_amplitude = np.max(np.abs(samples_array))
                rms = np.sqrt(np.mean(samples_array**2))
                print(f"[SPEECHIO] Audio stats - Max: {max_amplitude:.4f}, RMS: {rms:.4f}")
                
                if max_amplitude < 0.01:
                    print("[SPEECHIO] WARNING: Audio level very low! Microphone might not be working.")
                
                with wave.open(audio_file, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(int(self.server.getSamplingRate()))
                    wf.writeframes(samples_int16.tobytes())
                
                # Debug: check file size
                file_size = os.path.getsize(audio_file)
                print(f"[SPEECHIO] Recorded audio: {audio_file} ({file_size} bytes)")
                
                # TEMP DEBUG: Save a copy to inspect manually
                import shutil
                debug_file = f"/tmp/debug_recording_{int(time.time())}.wav"
                shutil.copy(audio_file, debug_file)
                print(f"[SPEECHIO] Debug copy saved: {debug_file}")
                print(f"[SPEECHIO] You can play it with: aplay {debug_file}")
                
                # Cleanup pyo objects
                del recorder, mic, recording_table
                
                # Transcribe
                text = self.speech_to_text(audio_file)
                print(f"[SPEECHIO] STT: '{text}'")
                
                # Cleanup audio file
                os.remove(audio_file)
                
                # Call callback if text was recognized
                if text and self.stt_callback:
                    try:
                        self.stt_callback(text.lower())
                    except Exception as e:
                        print(f"[SPEECHIO] Callback error: {e}")
                        import traceback
                        traceback.print_exc()
                
                time.sleep(0.5)
                
        except Exception as e:
            print(f"[SPEECHIO] Listen error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_listening()
        
        if self.server:
            try:
                if self.server.getIsStarted():
                    self.server.stop()
                if self.server.getIsBooted():
                    self.server.shutdown()
            except:
                pass
    
    @staticmethod
    def list_pulseaudio_devices():
        """
        List PulseAudio devices (includes Bluetooth)
        
        Returns:
            str: Formatted list of PulseAudio devices
        """
        output = []
        output.append("\n" + "=" * 80)
        output.append("PULSEAUDIO DEVICES (includes Bluetooth)")
        output.append("=" * 80)
        
        try:
            # List sources (microphones)
            output.append("\nINPUT DEVICES (Microphones/Sources):")
            output.append("-" * 80)
            result = subprocess.run(['pactl', 'list', 'sources', 'short'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                sources = result.stdout.strip().split('\n')
                for i, line in enumerate(sources):
                    if line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            device_name = parts[1]
                            # Highlight Bluetooth devices
                            if 'bluez' in device_name.lower() or 'bluetooth' in device_name.lower():
                                output.append(f"  >>> BLUETOOTH [{i}]: {device_name} <<<")
                            else:
                                output.append(f"  [{i}]: {device_name}")
            else:
                output.append("  Could not list PulseAudio sources")
            
            # List sinks (speakers)
            output.append("\nOUTPUT DEVICES (Speakers/Sinks):")
            output.append("-" * 80)
            result = subprocess.run(['pactl', 'list', 'sinks', 'short'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                sinks = result.stdout.strip().split('\n')
                for i, line in enumerate(sinks):
                    if line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            device_name = parts[1]
                            # Highlight Bluetooth devices
                            if 'bluez' in device_name.lower() or 'bluetooth' in device_name.lower():
                                output.append(f"  >>> BLUETOOTH [{i}]: {device_name} <<<")
                            else:
                                output.append(f"  [{i}]: {device_name}")
            else:
                output.append("  Could not list PulseAudio sinks")
            
            # Get default devices
            output.append("\nDEFAULT DEVICES:")
            output.append("-" * 80)
            
            result = subprocess.run(['pactl', 'get-default-source'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                output.append(f"  Default Input:  {result.stdout.strip()}")
            
            result = subprocess.run(['pactl', 'get-default-sink'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                output.append(f"  Default Output: {result.stdout.strip()}")
            
            output.append("\n" + "=" * 80)
            output.append("\nNOTE:")
            output.append("  - Pyo (PortAudio) may not see Bluetooth devices directly")
            output.append("  - To use Bluetooth: Set it as default in PulseAudio")
            output.append("  - Then use device index 0 (maps to PulseAudio default)")
            output.append("\nTO SET BLUETOOTH AS DEFAULT:")
            output.append("  pactl set-default-source <bluetooth_source_name>")
            output.append("  pactl set-default-sink <bluetooth_sink_name>")
            output.append("=" * 80)
            
            return "\n".join(output)
            
        except Exception as e:
            return f"Error listing PulseAudio devices: {e}\nIs PulseAudio installed?"

    @staticmethod
    def list_audio_devices():
        """
        List available audio devices in a readable format
        
        Returns:
            str: Formatted list of audio devices
        """
        if not PYO_AVAILABLE:
            return "Pyo not available"
        
        try:
            devices_info = pa_get_devices_infos()
            
            if not devices_info or len(devices_info) != 2:
                return "Could not get device list"
            
            input_devices, output_devices = devices_info
            current_platform = platform.system()
            
            output = []
            output.append("\n" + "=" * 80)
            output.append(f"AUDIO DEVICES - PortAudio ({current_platform})")
            output.append("=" * 80)
            
            # Input devices (microphones)
            output.append("\nINPUT DEVICES (Microphones):")
            output.append("-" * 80)
            if input_devices:
                for idx, info in input_devices.items():
                    output.append(f"  [{idx}] {info['name']}")
                    output.append(f"      Sample Rate: {info['default sr']} Hz")
                    output.append(f"      Latency: {info['latency']*1000:.2f} ms")
                    output.append("")
            else:
                output.append("  No input devices found")
            
            # Output devices (speakers)
            output.append("\nOUTPUT DEVICES (Speakers):")
            output.append("-" * 80)
            if output_devices:
                for idx, info in output_devices.items():
                    output.append(f"  [{idx}] {info['name']}")
                    output.append(f"      Sample Rate: {info['default sr']} Hz")
                    output.append(f"      Latency: {info['latency']*1000:.2f} ms")
                    output.append("")
            else:
                output.append("  No output devices found")
            
            output.append("=" * 80)
            output.append("\nUSAGE:")
            output.append("  python speech.py --mic <input_device_id> --speaker <output_device_id>")
            output.append("\nEXAMPLE (Logitech C920 mic + Built-in speakers):")
            output.append("  python speech.py --server localhost --viz --speech --mic 5 --speaker 0")
            output.append("=" * 80)
            
            # Also list PulseAudio devices if on Linux
            if current_platform == 'Linux':
                output.append("\n")
                output.append(SpeechIO.list_pulseaudio_devices())
            
            return "\n".join(output)
            
        except Exception as e:
            return f"Error getting device list: {e}"


# Example usage and testing
if __name__ == "__main__":
    print("=== SpeechIO Test ===\n")
    print(f"Platform: {platform.system()}\n")
    
    # List available devices
    print("Available audio devices:")
    print(SpeechIO.list_audio_devices())
    print()
    
    # Create SpeechIO instance
    def on_speech(text):
        print(f"\n>>> Heard: '{text}'")
        if 'hello' in text:
            speech_io.text_to_speech("Hello! How are you?")
        elif 'bye' in text or 'exit' in text:
            print("Exiting...")
            speech_io.stop_listening()
    
    speech_io = SpeechIO(
        whisper_model="base",
        piper_voice="en_US-lessac-medium",
        input_device=0,
        output_device=0,
        stt_callback=on_speech
    )
    
    # Check status
    status = speech_io.get_status()
    print(f"\nStatus: {status}")
    
    if status['ready']:
        # Test TTS
        print("\nTesting Text-to-Speech...")
        speech_io.text_to_speech("Hello, this is a test of the speech synthesis system")
        
        # Start listening
        print("\nStarting voice recognition... Say 'hello' or 'bye' to exit")
        speech_io.start_listening()
        
        try:
            while speech_io.is_listening:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
    else:
        print("\nSpeechIO not ready. Check dependencies:")
        print(f"  Whisper: {status['whisper']}")
        print(f"  Piper: {status['piper']}")
        print(f"  Pyo: {status['pyo']}")
        print(f"  Voice Model: {status['voice_model']}")
    
    speech_io.cleanup()
    print("\nTest complete")