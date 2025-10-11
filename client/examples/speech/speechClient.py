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
import threading
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
from speechIO import SpeechAudioIO


class LLMSpeechClient:
    """LLM integration with speech output using SpeechAudioIO"""
    
    def __init__(self, model_name="llama3.2", enable_speech=False):
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
        
        # Initialize SpeechAudioIO
        if enable_speech:
            try:
                print("[INIT] Initializing speech interface...")
                self.speech_io = SpeechAudioIO(
                    whisper_model="tiny",
                    whisper_language="en",
                    piper_model_en="en_US-amy-medium",
                    piper_model_de=None,
                    piper_model_fallback="en_US-lessac-medium",
                    auto_download_models=True,
                    vad_aggressiveness=2,
                    silence_duration=1.2,
                    min_speech_duration=0.8,
                    min_rms_threshold=0.015
                )
                print("[INIT] Speech interface ready")
            except Exception as e:
                print(f"[ERROR] Failed to initialize SpeechAudioIO: {e}")
                self.speech_io = None
                self.enable_speech = False

    def on_init(self):
        print(f"[INIT] Connected to server")

        # Check/start Ollama
        if not self._check_ollama_server():
            print("[INIT] Starting Ollama server...")
            if self._start_ollama_server():
                time.sleep(2)
            else:
                print("[ERROR] Failed to start Ollama")
                return

        if self._check_ollama_server() and self._check_model_available():
            print(f"[INIT] Using model: {self.model_name}")
            self.ollama_ready = True
            print("[INFO] Keyboard: SPACE=describe, A=analyze emotions, B=suggest activity")
            print("[INFO] Voice: 'describe/analyze/check my pose'")
            
            # Start voice recognition
            if self.enable_speech and self.speech_io:
                self.speech_io.speak("System ready", language="en", blocking=True)
                time.sleep(0.5)
                
                listen_thread = threading.Thread(
                    target=self.speech_io.start_listening,
                    kwargs={'on_transcription': self._handle_voice_command},
                    daemon=True
                )
                listen_thread.start()
        else:
            print(f"[ERROR] Model '{self.model_name}' not available")
            self.ollama_ready = False

    def _check_ollama_server(self):
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    def _start_ollama_server(self):
        try:
            subprocess.Popen(['ollama', 'serve'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            return True
        except:
            return False

    def _check_model_available(self):
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return any(self.model_name in m['name'] for m in models)
            return False
        except:
            return False

    def _llm_call_sync(self, pose_description, person_id, system_prompt, user_prompt, speak_response=False):
        """Synchronous LLM call implementation"""
        if not self.ollama_ready:
            return
        
        try:
            print(f"[LLM] Requesting analysis for person {person_id}...")
            
            payload = {
                "model": self.model_name,
                "prompt": f"{system_prompt}\n\n{user_prompt.format(pose_features=pose_description)}",
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 150}
            }
            
            response = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=30)
            
            if response.status_code == 200:
                answer = response.json().get('response', 'No response')
                print(f"[RESPONSE] {answer}")
                
                if speak_response and self.enable_speech and self.speech_io:
                    self.speech_io.speak(answer, language="en")
            else:
                print(f"[ERROR] Request failed with status {response.status_code}")
        except Exception as e:
            print(f"[ERROR] LLM request failed: {e}")

    def _startLlmAnalysis(self, analyze_type, speak_response=True):
        """Start LLM analysis based on type"""
        if not self.cur_people:
            if speak_response and self.enable_speech and self.speech_io:
                self.speech_io.speak("No person detected", language="en")
            return
        
        person = self.cur_people[0]
        angles = person.get_skeletal_angles()
        pose_description = interpret_pose_from_angles(angles)
        
        prompts = {
            'describe': {
                'system': "You are a motion analysis assistant. Describe the pose in 1-2 sentences.",
                'user': "Describe this pose:\n{pose_features}"
            },
            'emotion': {
                'system': "You are a body language expert. Analyze the emotional state in 1-2 sentences.",
                'user': "Analyze the emotional state from this pose:\n{pose_features}"
            },
            'activity': {
                'system': "You are a fitness coach. Suggest one exercise in 1-2 sentences.",
                'user': "Based on this pose, suggest an exercise:\n{pose_features}"
            }
        }
        
        prompt = prompts.get(analyze_type)
        if prompt:
            # Run async to avoid blocking
            self.executor.submit(
                self._llm_call_sync,
                pose_description,
                person.id,
                prompt['system'],
                prompt['user'],
                speak_response
            )
    
    def trigger_llm_analysis(self, key):
        """Handle keyboard input"""
        if not self.ollama_ready:
            return
        
        key_map = {' ': 'describe', 'a': 'emotion', 'b': 'activity'}
        analyze_type = key_map.get(key)
        
        if analyze_type:
            self._startLlmAnalysis(analyze_type, speak_response=self.enable_speech)

    def _handle_voice_command(self, language: str, text: str):
        """Process voice commands"""
        text_lower = text.lower().strip()
        print(f"[VOICE] [{language.upper()}] Recognized: '{text}'")
        
        if len(text_lower) < 5:
            return
        
        # Determine analysis type
        if "emotion" in text_lower or "feel" in text_lower or "mood" in text_lower:
            analyze_type = 'emotion'
        elif "exercise" in text_lower or "activity" in text_lower or "workout" in text_lower:
            analyze_type = 'activity'
        elif any(word in text_lower for word in ["describe", "tell", "analyze", "scan", "check"]):
            analyze_type = 'describe'
        else:
            print(f"[VOICE] No trigger found")
            return
        
        print(f"[VOICE] ✓ Trigger detected: {analyze_type}")
        self._startLlmAnalysis(analyze_type, speak_response=True)

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
        print(f"[CONNECTION] {'Connected' if connected else 'Disconnected'}")


def main():
    parser = argparse.ArgumentParser(description="SenseSpace LLM Speech Client")
    parser.add_argument("--server", "-s", default="localhost", help="Server IP")
    parser.add_argument("--port", "-p", type=int, default=12345, help="Server port")
    parser.add_argument("--viz", action="store_true", help="Enable visualization")
    parser.add_argument("--model", "-m", default="llama3.2", help="Ollama model name")
    parser.add_argument("--no-speech", action="store_true", help="Disable speech (TTS + STT)")
    args = parser.parse_args()
    
    llm_client = LLMSpeechClient(
        model_name=args.model,
        enable_speech=not args.no_speech
    )
    
    client = MinimalClient(
        server_ip=args.server,
        server_port=args.port,
        viz=args.viz,
        on_init=llm_client.on_init,
        on_frame=llm_client.on_frame,
        on_connection_changed=llm_client.on_connection_changed
    )
    
    if args.viz:
        client.llm_callback = llm_client.trigger_llm_analysis
    
    success = client.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

