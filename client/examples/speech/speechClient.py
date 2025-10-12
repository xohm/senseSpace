# -----------------------------------------------------------------------------
# Sense Space
# -----------------------------------------------------------------------------
# LLM Speech Client Example with Voice Output and Expert System
# -----------------------------------------------------------------------------
# IAD, Zurich University of the Arts / zhdk.ch
# Max Rheiner
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# This example demonstrates voice-controlled pose analysis using:
#   - SpeechAudioIO for speech-to-text (Whisper) and text-to-speech (Piper)
#   - LLMClient with expert system for consistent, domain-specific responses
#   - Voice commands: "describe/analyze/check my pose"
#   - Automatic microphone muting during TTS playback
#
# The expert system provides:
#   - Consistent response format and tone
#   - Faster inference through conversation context
#   - Better quality through few-shot learning
#
# Keyboard shortcuts (with --viz):
#   SPACE - Describe pose
#   A     - Analyze emotions
#   B     - Suggest activity
#   R     - Reset conversation context
#
# Voice commands:
#   "describe/tell/scan my pose" - Describe current pose
#   "analyze/emotion/feeling" - Analyze emotional state
#   "exercise/activity/workout" - Suggest activity
# -----------------------------------------------------------------------------

import argparse
import sys
import time
import threading

# Setup paths
from senseSpaceLib.senseSpace import setup_paths
setup_paths()

from senseSpaceLib.senseSpace import MinimalClient, Frame
from senseSpaceLib.senseSpace.interpretation import interpret_pose_from_angles
from senseSpaceLib.senseSpace.llmClient import LLMClient
from speechIO import SpeechAudioIO


class LLMSpeechClient:
    """LLM integration with speech output using SpeechAudioIO and Expert System"""
    
    def __init__(self, model_name="phi4-mini:Q4_K_M", expert_json=None, enable_speech=False, confidence_threshold=70.0):
        self.persons = 0
        self.latest_frame = None
        self.cur_people = []
        self.confidence_threshold = confidence_threshold
        self.enable_speech = enable_speech
        self.speech_io = None
        
        # Initialize LLM client with expert system
        self.llm_client = LLMClient(
            model_name=model_name,
            expert_json=expert_json,
            auto_download=True
        )
        
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
        
        # Initialize LLM client
        self.llm_client.on_init()
        
        if self.llm_client.ollama_ready:
            print("[INFO] Keyboard: SPACE=describe, A=analyze emotions, B=suggest activity, R=reset context")
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

    def _startLlmAnalysis(self, analyze_type, speak_response=True):
        """Start LLM analysis based on type using expert system"""
        if not self.cur_people:
            if speak_response and self.enable_speech and self.speech_io:
                self.speech_io.speak("No person detected", language="en")
            return
        
        person = self.cur_people[0]
        angles = person.get_skeletal_angles()
        pose_description = interpret_pose_from_angles(angles)
        
        # Build context-aware prompts for expert system
        prompts = {
            'describe': f"Describe this pose:\n\nPerson ID: {person.id}\nConfidence: {person.confidence:.1f}%\n\n{pose_description}",
            'emotion': f"Analyze the emotional state from this pose:\n\nPerson ID: {person.id}\nConfidence: {person.confidence:.1f}%\n\n{pose_description}",
            'activity': f"Suggest an exercise based on this pose:\n\nPerson ID: {person.id}\nConfidence: {person.confidence:.1f}%\n\n{pose_description}"
        }
        
        prompt = prompts.get(analyze_type)
        if not prompt:
            return
        
        # Callback to handle response and speak it
        def on_response(response: str):
            if response:
                print(f"[EXPERT] {response}")
                if speak_response and self.enable_speech and self.speech_io:
                    self.speech_io.speak(response, language="en")
        
        # Use expert system call (maintains conversation context)
        self.llm_client.call_expert_async(prompt, callback=on_response)
    
    def trigger_llm_analysis(self, key):
        """Handle keyboard input"""
        if not self.llm_client.ollama_ready:
            return
        
        if key == ' ':
            self._startLlmAnalysis('describe', speak_response=self.enable_speech)
        elif key == 'a':
            self._startLlmAnalysis('emotion', speak_response=self.enable_speech)
        elif key == 'b':
            self._startLlmAnalysis('activity', speak_response=self.enable_speech)
        elif key == 'r':
            self.llm_client.reset_context()
            print("[INFO] Conversation context reset")
            if self.enable_speech and self.speech_io:
                self.speech_io.speak("Context reset", language="en")

    def _handle_voice_command(self, language: str, text: str):
        """Process voice commands"""
        text_lower = text.lower().strip()
        print(f"[VOICE] [{language.upper()}] Recognized: '{text}'")
        
        if len(text_lower) < 5:
            return
        
        # Check for reset command
        if "reset" in text_lower or "clear" in text_lower:
            self.llm_client.reset_context()
            print("[VOICE] ✓ Context reset")
            if self.enable_speech and self.speech_io:
                self.speech_io.speak("Context reset", language="en")
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
    parser = argparse.ArgumentParser(
        description="SenseSpace LLM Speech Client with Expert System",
        epilog="Example: python speechClient.py --viz --expert ../data/expert_pose_config.json"
    )
    parser.add_argument("--server", "-s", default="localhost", help="Server IP")
    parser.add_argument("--port", "-p", type=int, default=12345, help="Server port")
    parser.add_argument("--viz", action="store_true", help="Enable visualization")
    parser.add_argument("--model", "-m", default="phi4-mini:Q4_K_M", 
                       help="Ollama model name (default: phi4-mini for fast inference)")
    parser.add_argument("--expert", "-e", default="../data/expert_pose_config.json",
                       help="Path to expert configuration JSON")
    parser.add_argument("--no-speech", action="store_true", help="Disable speech (TTS + STT)")
    parser.add_argument("--confidence", "-c", type=float, default=70.0,
                       help="Minimum confidence threshold for person detection")
    args = parser.parse_args()
    
    # Create LLM speech client with expert system
    llm_client = LLMSpeechClient(
        model_name=args.model,
        expert_json=args.expert,
        enable_speech=not args.no_speech,
        confidence_threshold=args.confidence
    )
    
    # Create minimal client
    client = MinimalClient(
        server_ip=args.server,
        server_port=args.port,
        viz=args.viz,
        on_init=llm_client.on_init,
        on_frame=llm_client.on_frame,
        on_connection_changed=llm_client.on_connection_changed
    )
    
    # Set keyboard callback for visualization mode
    if args.viz:
        client.llm_callback = llm_client.trigger_llm_analysis
    
    success = client.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

