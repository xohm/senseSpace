# -----------------------------------------------------------------------------
# Sense Space
# -----------------------------------------------------------------------------
# LLM Speech Client Example with Voice Output
# -----------------------------------------------------------------------------
# IAD, Zurich University of the Arts / zhdk.ch
# Max Rheiner
# -----------------------------------------------------------------------------

import argparse
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor
import requests
import subprocess

# Add libs and client to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
libs_path = os.path.join(repo_root, 'libs')
senseSpaceLib_path = os.path.join(libs_path, 'senseSpaceLib')
client_path = os.path.join(repo_root, 'client')

if libs_path not in sys.path:
    sys.path.insert(0, libs_path)
if senseSpaceLib_path not in sys.path:
    sys.path.insert(0, senseSpaceLib_path)
if client_path not in sys.path:
    sys.path.insert(0, client_path)

from miniClient import MinimalClient
from senseSpaceLib.senseSpace.protocol import Frame
from senseSpaceLib.senseSpace.interpretation import interpret_pose_from_angles

# Import SpeechIO
from speechIO import SpeechIO


class LLMSpeechClient:
    """LLM integration with speech output (TTS only)"""
    
    def __init__(self, model_name="llama3.2", enable_speech=False, 
                 mic_device=0, speaker_device=0, record_seconds=5):
        self.persons = 0
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.latest_frame = None
        self.cur_people = []
        self.confidence_threshold = 70.0
        self.ollama_url = "http://localhost:11434"
        self.model_name = model_name
        self.ollama_ready = False
        self.enable_speech = enable_speech
        self.speech_io = None
        
        # Initialize SpeechIO with both STT and TTS
        if enable_speech:
            try:
                print("[INIT] Initializing speech interface...")
                self.speech_io = SpeechIO(
                    whisper_model="base",
                    piper_voice="en_US-lessac-medium",
                    input_device=mic_device,
                    output_device=speaker_device,
                    record_seconds=record_seconds,  # Now configurable
                    stt_callback=self._handle_voice_command
                )
                
                if self.speech_io.ready:
                    print("[INIT] Speech interface ready")
                else:
                    print("[INIT] Speech interface created but audio server failed")
                    print("[INIT] Continuing without speech capabilities")
                    self.speech_io = None
                    self.enable_speech = False
                    
            except Exception as e:
                print(f"[ERROR] Failed to initialize SpeechIO: {e}")
                import traceback
                traceback.print_exc()
                self.speech_io = None
                self.enable_speech = False
                print("[INIT] Continuing without speech capabilities")

    def on_init(self):
        print(f"[INIT] Connected to server")

        # Check speech dependencies if enabled
        if self.enable_speech and self.speech_io:
            status = self.speech_io.get_status()
            
            if not status['ready']:
                print("[ERROR] SpeechIO not ready:")
                print(f"  Piper: {status['piper']}")
                print(f"  Pyo: {status['pyo']}")
                print(f"  Voice Model: {status['voice_model']}")
                self.enable_speech = False
            else:
                print("[SPEECH] Speech interface enabled")
                print(f"[SPEECH] Sample rate: {status['sample_rate']} Hz")
                print(f"[SPEECH] Microphone: device {status['input_device']}")
                print(f"[SPEECH] Speaker: device {status['output_device']}")
                
                # Start voice recognition
                print("[SPEECH] Starting voice recognition...")
                self.speech_io.start_listening()

        # Check Ollama
        if not self._check_ollama_server():
            print("[INIT] Ollama server not running, attempting to start...")
            if self._start_ollama_server():
                time.sleep(2)
            else:
                print("[ERROR] Failed to start Ollama server")
                self.ollama_ready = False
                return
        else:
            print("[INIT] Ollama server already running")

        if self._check_ollama_server():
            print(f"[INIT] Ollama server connected at {self.ollama_url}")
            
            if self._check_model_available():
                print(f"[INIT] Using model: {self.model_name}")
                self.ollama_ready = True
                print("[INFO] Keyboard shortcuts:")
                print("  SPACE - Describe the current pose (with speech)")
                print("  A     - Analyze pose emotions/mood (with speech)")
                print("  B     - Suggest exercise or activity (with speech)")
                print("  S     - Toggle speech output on/off")
                print("  V     - Toggle voice recognition on/off")
                print("\n[INFO] Voice commands:")
                print("  'describe' / 'what is the pose' - Describe current pose")
                print("  'how do I feel' / 'analyze' - Analyze emotions")
                print("  'suggest exercise' / 'workout' - Get exercise suggestion")
                print("  'help' - List available commands")
            else:
                print(f"[ERROR] Model '{self.model_name}' not available")
                print(f"[INFO] Run: ollama pull {self.model_name}")
                self.ollama_ready = False
        else:
            print("[ERROR] Could not connect to Ollama server")
            self.ollama_ready = False

    def _check_ollama_server(self):
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    def _start_ollama_server(self):
        """Attempt to start Ollama server"""
        try:
            subprocess.Popen(['ollama', 'serve'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            return True
        except:
            return False

    def _check_model_available(self):
        """Check if the specified model is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                available = [m['name'] for m in models]
                print(f"[INIT] Available models: {', '.join(available)}")
                return any(self.model_name in m for m in available)
            return False
        except:
            return False

    def _llm_call(self, pose_description, person_id, system_prompt, user_prompt, speak_response=False):
        """LLM call with optional speech output"""
        if not self.ollama_ready:
            print("[ERROR] Ollama not ready. Cannot make request.")
            return
        
        try:
            print(f"[LLM] Requesting analysis for person {person_id}...")
            
            payload = {
                "model": self.model_name,
                "prompt": f"{system_prompt}\n\n{user_prompt.format(pose_features=pose_description)}",
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 150
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', 'No response')
                
                # Print response
                print(f"[RESPONSE] {answer}")
                
                # Speak response if requested and speech is enabled
                if speak_response and self.enable_speech and self.speech_io and self.speech_io.ready:
                    print("[TTS] Generating speech...")
                    try:
                        self.speech_io.text_to_speech(answer)
                        print("[TTS] Speech playback complete")
                    except Exception as e:
                        print(f"[TTS ERROR] Failed to play speech: {e}")
                        import traceback
                        traceback.print_exc()
            else:
                print(f"[ERROR] Request failed with status {response.status_code}")

        except Exception as e:
            print(f"[ERROR] LLM request failed: {e}")
            import traceback
            traceback.print_exc()

    def _analyze_pose(self, question_type, speak_response=False):
        """Analyze pose with specific question type"""
        if not self.ollama_ready:
            print("[ERROR] Ollama not available. Check if server is running.")
            return
        
        if not self.cur_people or len(self.cur_people) == 0:
            print("[LLM] No people in current frame")
            if speak_response and self.enable_speech and self.speech_io:
                self.speech_io.text_to_speech("No person detected")
            return
        
        person = self.cur_people[0]
        confidence_val = getattr(person, 'confidence', 0)
        print(f"[LLM] Analyzing person {person.id} (confidence: {confidence_val:.1f})")
        
        angles = person.get_skeletal_angles()
        pose_description = interpret_pose_from_angles(angles)
        
        prompts = {
            'describe': {
                'system': (
                    "You are a motion analysis assistant. "
                    "Describe the pose in natural language in 1-2 sentences. "
                    "Focus on general actions, like 'standing', 'sitting', or 'arms raised'."
                ),
                'user': "Describe this pose:\n{pose_features}"
            },
            'emotion': {
                'system': (
                    "You are a body language expert. "
                    "Analyze the emotional state in 1-2 sentences. "
                    "Consider body openness, posture, arm positions."
                ),
                'user': "Analyze the emotional state from this pose:\n{pose_features}"
            },
            'activity': {
                'system': (
                    "You are a fitness coach. "
                    "Suggest one exercise or activity in 1-2 sentences "
                    "based on the current body position."
                ),
                'user': "Based on this pose, suggest an exercise:\n{pose_features}"
            }
        }
        
        prompt_config = prompts.get(question_type)
        if not prompt_config:
            print(f"[ERROR] Unknown question type: {question_type}")
            return
        
        self.executor.submit(
            self._llm_call, 
            pose_description, 
            person.id,
            prompt_config['system'],
            prompt_config['user'],
            speak_response
        )

    def trigger_llm_analysis(self, key):
        """Handle keyboard input"""
        if not self.ollama_ready:
            print("[ERROR] Ollama not available. Check if server is running.")
            return
        
        if key == ' ':
            print("[KEY] Space - Describing pose with speech...")
            self._analyze_pose('describe', speak_response=self.enable_speech)
        elif key == 'a':
            print("[KEY] A - Analyzing emotions with speech...")
            self._analyze_pose('emotion', speak_response=self.enable_speech)
        elif key == 'b':
            print("[KEY] B - Suggesting activity with speech...")
            self._analyze_pose('activity', speak_response=self.enable_speech)
        elif key == 's':
            # Toggle speech on/off
            if self.speech_io:
                self.enable_speech = not self.enable_speech
                status = "enabled" if self.enable_speech else "disabled"
                print(f"[KEY] S - Speech output {status}")
        elif key == 'v':
            # Toggle voice recognition on/off
            if self.speech_io:
                self.speech_io.enable_stt = not self.speech_io.enable_stt
                status = "enabled" if self.speech_io.enable_stt else "disabled"
                print(f"[KEY] V - Voice recognition {status}")

    def _handle_voice_command(self, text):
        """Process voice commands from SpeechIO"""
        text = text.lower().strip()
        print(f"[VOICE] Recognized: '{text}'")
        
        # Debug: show what we're checking against
        print(f"[VOICE DEBUG] Checking '{text}' against commands...")
        
        # Ignore very short or empty transcriptions
        if len(text) < 2:
            print("[VOICE] Ignoring short transcription")
            return
        
        # Don't stop/restart listening from within the callback - causes threading issues
        # Instead, just process the command
        
        # Describe pose commands
        if any(word in text for word in ['describe', 'what', 'show', 'tell', 'pose', 'position']):
            print("[VOICE] ✓ Command matched: describe pose")
            self._analyze_pose('describe', speak_response=True)
        
        # Analyze emotion/mood commands
        elif any(word in text for word in ['emotion', 'feel', 'feeling', 'mood', 'analyze', 'how']):
            print("[VOICE] ✓ Command matched: analyze emotions")
            self._analyze_pose('emotion', speak_response=True)
        
        # Suggest exercise/activity commands
        elif any(word in text for word in ['exercise', 'activity', 'suggest', 'workout', 'movement', 'do']):
            print("[VOICE] ✓ Command matched: suggest activity")
            self._analyze_pose('activity', speak_response=True)
        
        # Help command
        elif 'help' in text or 'command' in text:
            print("[VOICE] ✓ Command matched: help")
            help_text = "Say: describe, analyze, or suggest exercise"
            print(f"[VOICE] Help requested")
            if self.speech_io and self.speech_io.ready:
                print("[VOICE] Calling TTS...")
                self.speech_io.text_to_speech(help_text)
                print("[VOICE] TTS call completed")
            else:
                print("[VOICE] ERROR: speech_io not ready!")
        
        else:
            # DON'T give TTS feedback for unknown commands to prevent feedback loops
            if len(text) > 3:
                print(f"[VOICE] ✗ Unknown command: '{text}' (ignoring)")
            else:
                print(f"[VOICE] ✗ Ignoring: '{text}'")
        
        # Removed the was_listening restart code - it was causing threading errors
        # The listening loop continues automatically

    def on_frame(self, frame: Frame):
        """Called whenever a new SenseSpace frame arrives"""
        self.latest_frame = frame
        
        people = getattr(frame, "people", None)
        
        if people:
            self.cur_people = [p for p in people if getattr(p, 'confidence', 0) > self.confidence_threshold]
        else:
            self.cur_people = []
        
        if self.persons != len(self.cur_people):
            self.persons = len(self.cur_people)

    def on_connection_changed(self, connected: bool):
        status = "Connected" if connected else "Disconnected"
        print(f"[CONNECTION] {status}")


def main():
    parser = argparse.ArgumentParser(description="SenseSpace LLM Speech Client Example")
    parser.add_argument("--server", "-s", default="localhost", help="Server IP")
    parser.add_argument("--port", "-p", type=int, default=12345, help="Server port")
    parser.add_argument("--viz", action="store_true", help="Enable visualization")
    parser.add_argument("--model", "-m", default="llama3.2", help="Ollama model name")
    parser.add_argument("--speech", action="store_true", help="Enable speech interface (TTS + STT)")
    parser.add_argument("--mic", type=int, default=0, help="Microphone device ID")
    parser.add_argument("--speaker", type=int, default=0, help="Speaker device ID")
    parser.add_argument("--record", type=int, default=5, help="Recording duration in seconds")
    args = parser.parse_args()
    
    # Create LLM speech client
    llm_client = LLMSpeechClient(
        model_name=args.model,
        enable_speech=args.speech,
        mic_device=args.mic,
        speaker_device=args.speaker,
        record_seconds=args.record  # Pass record duration
    )
    
    # Create minimal client with LLM callbacks
    client = MinimalClient(
        server_ip=args.server,
        server_port=args.port,
        viz=args.viz,
        on_init=llm_client.on_init,
        on_frame=llm_client.on_frame,
        on_connection_changed=llm_client.on_connection_changed
    )
    
    # Pass LLM trigger callback to miniClient for keyboard handling
    if args.viz:
        client.llm_callback = llm_client.trigger_llm_analysis
    
    success = client.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

