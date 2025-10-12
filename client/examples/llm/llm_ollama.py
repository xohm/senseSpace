# -----------------------------------------------------------------------------
# Sense Space
# -----------------------------------------------------------------------------
# Ollama LLM Client Example (Direct API calls)
# -----------------------------------------------------------------------------
# IAD, Zurich University of the Arts / zhdk.ch
# Max Rheiner
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# This example demonstrates the same as the OpenAI example, but with
# direct Ollama LLM integration without LLMClient wrapper.
# So the LLM calls are done directly via the Ollama API from a local model.
# -----------------------------------------------------------------------------

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import requests
import subprocess

# Setup paths
from senseSpaceLib.senseSpace import setup_paths
setup_paths()

from senseSpaceLib.senseSpace import MinimalClient, Frame
from senseSpaceLib.senseSpace.interpretation import interpret_pose_from_angles


class OllamaLLMClient:
    """Direct Ollama LLM integration (without LLMClient wrapper)"""
    
    def __init__(self, model_name="llama3.2", confidence_threshold=70.0):
        self.persons = 0
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.latest_frame = None
        self.cur_people = []
        self.confidence_threshold = confidence_threshold
        self.ollama_url = "http://localhost:11434"
        self.model_name = model_name
        self.ollama_ready = False

    def on_init(self):
        print(f"[INIT] Connected to server")

        # Check if Ollama server is running, start if needed
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

        # Verify connection
        if self._check_ollama_server():
            print(f"[INIT] Ollama server connected at {self.ollama_url}")
            
            # Check if model is available
            if self._check_model_available():
                print(f"[INIT] Using model: {self.model_name}")
                self.ollama_ready = True
                print("[INFO] Keyboard shortcuts:")
                print("  SPACE - Describe the current pose")
                print("  A     - Analyze pose emotions/mood")
                print("  B     - Suggest exercise or activity")
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

    def _llm_call(self, pose_description, person_id, system_prompt, user_prompt):
        """LLM call - executed in thread pool"""
        if not self.ollama_ready:
            print("[ERROR] Ollama not ready. Cannot make request.")
            return
        
        try:
            print(f"[LLM] Requesting analysis for person {person_id}...")
            
            # Ollama API format
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
                print(f"[RESPONSE] {answer}")
            else:
                print(f"[ERROR] Request failed with status {response.status_code}")

        except Exception as e:
            print(f"[ERROR] LLM request failed: {e}")
            import traceback
            traceback.print_exc()

    def _analyze_pose(self, question_type):
        """Analyze pose with specific question type"""
        if not self.ollama_ready:
            print("[ERROR] Ollama not available. Check if server is running.")
            return
        
        if not self.cur_people or len(self.cur_people) == 0:
            print("[LLM] No people in current frame")
            return
        
        # Take the first high-confidence person
        person = self.cur_people[0]
        confidence_val = getattr(person, 'confidence', 0)
        print(f"[LLM] Analyzing person {person.id} (confidence: {confidence_val:.1f})")
        
        # Get skeleton angles and interpret pose
        angles = person.get_skeletal_angles()
        pose_description = interpret_pose_from_angles(angles)
        
        print(f"[DEBUG] Extracted features:\n{pose_description}")
        
        # Define different prompts for different question types
        prompts = {
            'describe': {
                'system': (
                    "You are a motion analysis assistant. "
                    "You will receive body pose analysis with joint angles and positions. "
                    "Describe the pose in natural language. "
                    "Focus on general actions, like 'standing', 'sitting', or 'arms raised'. "
                    "Keep your response concise and clear."
                ),
                'user': "Describe this pose:\n{pose_features}"
            },
            'emotion': {
                'system': (
                    "You are a body language expert. "
                    "You will receive body pose analysis with joint angles and positions. "
                    "Analyze the emotional state or mood suggested by the pose. "
                    "Consider body openness, posture, arm positions, etc. "
                    "Keep your response concise and clear."
                ),
                'user': "Analyze the emotional state from this pose:\n{pose_features}"
            },
            'activity': {
                'system': (
                    "You are a fitness and wellness coach. "
                    "You will receive body pose analysis with joint angles and positions. "
                    "Suggest an appropriate exercise, stretch, or activity "
                    "based on the current body position. "
                    "Keep your response concise and clear."
                ),
                'user': "Based on this pose, suggest an exercise or activity:\n{pose_features}"
            }
        }
        
        prompt_config = prompts.get(question_type)
        if not prompt_config:
            print(f"[ERROR] Unknown question type: {question_type}")
            return
        
        print(f"[LLM] Question type: {question_type}")
        print(f"[LLM] Prompt:\n{prompt_config['user'].format(pose_features=pose_description)}")
        
        # Submit to thread pool (non-blocking)
        self.executor.submit(
            self._llm_call, 
            pose_description, 
            person.id,
            prompt_config['system'],
            prompt_config['user']
        )

    def trigger_llm_analysis(self, key):
        """Handle keyboard input and trigger appropriate LLM analysis"""
        if not self.ollama_ready:
            print("[ERROR] Ollama not available. Check if server is running.")
            return
        
        if key == ' ':
            print("[KEY] Space - Describing pose...")
            self._analyze_pose('describe')
        elif key == 'a':
            print("[KEY] A - Analyzing emotions...")
            self._analyze_pose('emotion')
        elif key == 'b':
            print("[KEY] B - Suggesting activity...")
            self._analyze_pose('activity')

    def on_frame(self, frame: Frame):
        """Called whenever a new SenseSpace frame arrives"""
        self.latest_frame = frame
        people = getattr(frame, "people", None)
        
        # Filter people by confidence threshold
        if people:
            self.cur_people = [p for p in people if getattr(p, 'confidence', 0) > self.confidence_threshold]
        else:
            self.cur_people = []
        
        # Track changes in high-confidence person count
        if self.persons != len(self.cur_people):
            self.persons = len(self.cur_people)

    def on_connection_changed(self, connected: bool):
        status = "Connected" if connected else "Disconnected"
        print(f"[CONNECTION] {status}")


def main():
    parser = argparse.ArgumentParser(description="SenseSpace Ollama LLM Client Example")
    parser.add_argument("--server", "-s", default="localhost", help="Server IP")
    parser.add_argument("--port", "-p", type=int, default=12345, help="Server port")
    parser.add_argument("--viz", action="store_true", help="Enable visualization (required for keyboard input)")
    parser.add_argument("--model", "-m", default="llama3.2", help="Ollama model name")
    parser.add_argument("--confidence", "-c", type=float, default=70.0,
                       help="Minimum confidence threshold for person detection")
    args = parser.parse_args()
    
    if not args.viz:
        print("[WARNING] LLM example works best with --viz flag for keyboard input")
        print("[INFO] Run with: python llm_ollama.py --server localhost --viz")
    
    # Create Ollama LLM client wrapper
    llm_client = OllamaLLMClient(
        model_name=args.model,
        confidence_threshold=args.confidence
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