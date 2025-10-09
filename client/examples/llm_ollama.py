# -----------------------------------------------------------------------------
# Sense Space
# -----------------------------------------------------------------------------
# Ollama Local LLM Client Example
# -----------------------------------------------------------------------------
# IAD, Zurich University of the Arts / zhdk.ch
# Max Rheiner
# -----------------------------------------------------------------------------
#
# Installation:
#   1. Install Ollama:
#      curl -fsSL https://ollama.com/install.sh | sh
#   
#   2. Pull a model (choose one):
#      ollama pull llama3.2        # Recommended: Fast & good quality (3B)
#      ollama pull llama3.1        # Better quality, slower (8B)
#      ollama pull qwen2.5:1.5b    # Smaller, faster (1.5B)
#   
#   3. Start Ollama server (in a separate terminal):
#      ollama serve
#   
#   4. Install Python dependencies:
#      pip install -r client/examples/requirements.txt
#
# Usage:
#   python client/examples/llm_ollama.py --server localhost --viz
#
# Keyboard shortcuts:
#   SPACE - Describe the current pose
#   A     - Analyze pose emotions/mood
#   B     - Suggest exercise or activity
#
# -----------------------------------------------------------------------------

import argparse
import sys
import os
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import requests
import subprocess
import time

# Add libs and client to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
libs_path = os.path.join(repo_root, 'libs')
client_path = os.path.join(repo_root, 'client')
if libs_path not in sys.path:
    sys.path.insert(0, libs_path)
if client_path not in sys.path:
    sys.path.insert(0, client_path)

from miniClient import MinimalClient
from senseSpaceLib.senseSpace.protocol import Frame

# Load environment variables from .env file in the same directory as this script
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)


class OllamaLLMClient:
    """Tiny wrapper for Ollama LLM integration with async requests"""
    
    def __init__(self):
        self.persons = 0
        self.executor = ThreadPoolExecutor(max_workers=2)  # Pool for async calls
        self.latest_frame = None  # Store latest frame for on-demand LLM calls
        self.ollama_process = None  # To track if we started Ollama
        
        # Ollama configuration
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")

    def _is_ollama_running(self):
        """Check if Ollama server is already running"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    def _start_ollama_server(self):
        """Start Ollama server if not running"""
        if self._is_ollama_running():
            print("[INIT] Ollama server already running")
            return True
        
        print("[INIT] Starting Ollama server...")
        try:
            # Start ollama serve in background
            self.ollama_process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True  # Detach from parent process
            )
            
            # Wait for server to start (max 10 seconds)
            for i in range(20):
                time.sleep(0.5)
                if self._is_ollama_running():
                    print("[INIT] Ollama server started successfully")
                    return True
            
            print("[ERROR] Ollama server failed to start within 10 seconds")
            return False
            
        except FileNotFoundError:
            print("[ERROR] 'ollama' command not found. Please install Ollama first:")
            print("       curl -fsSL https://ollama.com/install.sh | sh")
            return False
        except Exception as e:
            print(f"[ERROR] Failed to start Ollama server: {e}")
            return False

    def on_init(self):
        print(f"[INIT] Connected to server")

        # Start Ollama server if not running
        if not self._start_ollama_server():
            return
        
        # Verify Ollama is working and check models
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                print(f"[INIT] Ollama server connected at {self.ollama_base_url}")
                print(f"[INIT] Using model: {self.ollama_model}")
                print(f"[INIT] Available models: {', '.join(model_names) if model_names else 'none'}")
                
                if not any(self.ollama_model in name for name in model_names):
                    print(f"[WARNING] Model '{self.ollama_model}' not found!")
                    print(f"[INFO] Pull it with: ollama pull {self.ollama_model}")
            else:
                print("[ERROR] Ollama server not responding")
                return
        except requests.exceptions.ConnectionError:
            print(f"[ERROR] Cannot connect to Ollama at {self.ollama_base_url}")
            print("[INFO] Make sure Ollama is running:")
            print("       1. Start server: ollama serve")
            print(f"       2. Pull model: ollama pull {self.ollama_model}")
            return
        except Exception as e:
            print(f"[ERROR] Failed to connect to Ollama: {e}")
            return
        
        print("[INFO] Keyboard shortcuts:")
        print("  SPACE - Describe the current pose")
        print("  A     - Analyze pose emotions/mood")
        print("  B     - Suggest exercise or activity")
    
    def _llm_call(self, person_dict, person_id, system_prompt, user_prompt):
        """Ollama LLM call - will be executed in thread pool"""
        try:
            print(f"[LLM] Requesting analysis for person {person_id}...")
            
            url = f"{self.ollama_base_url}/api/chat"
            
            payload = {
                "model": self.ollama_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt.format(pose_data=person_dict)},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 150,
                }
            }
            
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()["message"]["content"]
            print(f"[RESPONSE] {result}")

        except requests.exceptions.Timeout:
            print(f"[ERROR] Ollama request timed out (model may be slow or not loaded)")
        except requests.exceptions.ConnectionError:
            print(f"[ERROR] Lost connection to Ollama server")
        except Exception as e:
            print(f"[ERROR] LLM request failed: {e}")

    def _analyze_pose(self, question_type):
        """Analyze pose with specific question type"""
        if not self.latest_frame:
            print("[LLM] No frame available yet")
            return
        
        people = getattr(self.latest_frame, "people", None)
        if not people or len(people) == 0:
            print("[LLM] No people in current frame")
            return
        
        # Filter for high-confidence persons (confidence > 0.7)
        CONFIDENCE_THRESHOLD = 0.7
        confident_people = [p for p in people if getattr(p, 'confidence', 0) > CONFIDENCE_THRESHOLD]
        
        if not confident_people:
            print(f"[LLM] No high-confidence persons (threshold: {CONFIDENCE_THRESHOLD})")
            return
        
        # Take the first high-confidence person
        person = confident_people[0]
        print(f"[LLM] Analyzing person {person.id} (confidence: {person.confidence:.2f})")
        person_dict = person.to_dict()
        
        # Define different prompts for different question types
        prompts = {
            'describe': {
                'system': (
                    "You are a motion analysis assistant. "
                    "Given a list of 3D joint coordinates (BODY_34), "
                    "describe the pose in natural language. "
                    "Focus on general actions, like 'standing', 'sitting', or 'arms raised'. "
                    "Keep your response concise and clear."
                ),
                'user': "Describe this pose:\n{pose_data}"
            },
            'emotion': {
                'system': (
                    "You are a body language expert. "
                    "Given a list of 3D joint coordinates (BODY_34), "
                    "analyze the emotional state or mood suggested by the pose. "
                    "Consider body openness, posture, arm positions, etc. "
                    "Keep your response concise and clear."
                ),
                'user': "Analyze the emotional state from this pose:\n{pose_data}"
            },
            'activity': {
                'system': (
                    "You are a fitness and wellness coach. "
                    "Given a list of 3D joint coordinates (BODY_34), "
                    "suggest an appropriate exercise, stretch, or activity "
                    "based on the current body position. "
                    "Keep your response concise and clear."
                ),
                'user': "Based on this pose, suggest an exercise or activity:\n{pose_data}"
            }
        }
        
        prompt_config = prompts.get(question_type)
        if not prompt_config:
            print(f"[ERROR] Unknown question type: {question_type}")
            return
        
        # Submit to thread pool (fire and forget - non-blocking)
        self.executor.submit(
            self._llm_call, 
            person_dict, 
            person.id,
            prompt_config['system'],
            prompt_config['user']
        )

    def trigger_llm_analysis(self, key):
        """Handle keyboard input and trigger appropriate LLM analysis"""
        # Switch-case for different keys
        if key == ' ':
            print("[KEY] Space - Describing pose...")
            self._analyze_pose('describe')
        elif key == 'a':
            print("[KEY] A - Analyzing emotions...")
            self._analyze_pose('emotion')
        elif key == 'b':
            print("[KEY] B - Suggesting activity...")
            self._analyze_pose('activity')
        else:
            # Ignore other keys silently
            pass

    def on_frame(self, frame: Frame):
        """Called whenever a new SenseSpace frame arrives"""
        # Store latest frame for on-demand analysis
        self.latest_frame = frame
        
        people = getattr(frame, "people", None)
        count = len(people) if people else 0

        if self.persons != count:
            self.persons = count
            print(f"[FRAME] Received {count} people")

    def on_connection_changed(self, connected: bool):
        status = "Connected" if connected else "Disconnected"
        print(f"[CONNECTION] {status}")
        
        # Clean up Ollama process if we started it and client disconnects
        if not connected and self.ollama_process:
            print("[CLEANUP] Stopping Ollama server...")
            self.ollama_process.terminate()
            self.ollama_process = None


def main():
    parser = argparse.ArgumentParser(description="SenseSpace Ollama LLM Client Example")
    parser.add_argument("--server", "-s", default="localhost", help="Server IP")
    parser.add_argument("--port", "-p", type=int, default=12345, help="Server port")
    parser.add_argument("--viz", action="store_true", help="Enable visualization (required for keyboard input)")
    args = parser.parse_args()
    
    if not args.viz:
        print("[WARNING] LLM example works best with --viz flag for keyboard input")
        print("[INFO] Run with: python llm_ollama.py --server localhost --viz")
    
    # Create Ollama LLM client wrapper
    llm_client = OllamaLLMClient()  
    
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