# -----------------------------------------------------------------------------
# Sense Space
# -----------------------------------------------------------------------------
# Open AI LLM Client Example
# -----------------------------------------------------------------------------
# IAD, Zurich University of the Arts / zhdk.ch
# Max Rheiner
# -----------------------------------------------------------------------------

import argparse
import sys
import os
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

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
from senseSpaceLib.senseSpace.protocol import Frame, Person, Joint
from senseSpaceLib.senseSpace.enums import Body34Joint, SkeletonAngle
from senseSpaceLib.senseSpace.interpretation import interpret_pose_from_angles

# openai 
from openai import OpenAI

# Load environment variables from .env file in the same directory as this script
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

class LLMClient:
    """Tiny wrapper for LLM integration with async requests"""
    
    def __init__(self):
        self.persons = 0
        self.client = None
        self.executor = ThreadPoolExecutor(max_workers=2)  # Pool for async calls
        self.latest_frame = None  # Store latest frame for on-demand LLM calls
        self.cur_people = []  # Only high-confidence people
        self.confidence_threshold = 70.0  # Minimum confidence to consider a person
        self.client_ready = False  # Track if client is ready

    def on_init(self):
        print(f"[INIT] Connected to server")

        # Setup OpenAI client using API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("[ERROR] OPENAI_API_KEY not found in .env file!")
            print("[INFO] Please create a .env file with: OPENAI_API_KEY=your_key_here")
            print("[INFO] LLM features will be disabled.")
            self.client_ready = False
            return
        
        try:
            self.client = OpenAI(api_key=api_key)
            self.client_ready = True
            print("[INIT] OpenAI client initialized successfully.")
            print("[INFO] Keyboard shortcuts:")
            print("  SPACE - Describe the current pose")
            print("  A     - Analyze pose emotions/mood")
            print("  B     - Suggest exercise or activity")
        except Exception as e:
            print(f"[ERROR] Failed to initialize OpenAI client: {e}")
            print("[INFO] LLM features will be disabled.")
            self.client_ready = False

    def _llm_call(self, pose_description, person_id, system_prompt, user_prompt):
        """LLM call - will be executed in thread pool"""
        if not self.client_ready or self.client is None:
            print("[ERROR] LLM client not initialized. Cannot make request.")
            print("[INFO] Check your .env file and OPENAI_API_KEY")
            return
        
        try:
            print(f"[LLM] Requesting analysis for person {person_id}...")
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt.format(pose_features=pose_description),
                    },
                ],
                max_tokens=150,
                temperature=0.3,
            )

            result = response.choices[0].message.content
            print(f"[RESPONSE] {result}")

        except Exception as e:
            print(f"[ERROR] LLM request failed: {e}")
            import traceback
            traceback.print_exc()

    def _analyze_pose(self, question_type):
        """Analyze pose with specific question type"""
        if not self.client_ready:
            print("[ERROR] LLM client not ready. Please check your API key configuration.")
            return
        
        if not self.cur_people or len(self.cur_people) == 0:
            print("[LLM] No people in current frame")
            return
        
        # Take the first high-confidence person
        person = self.cur_people[0]
        confidence_val = getattr(person, 'confidence', 0)
        print(f"[LLM] Analyzing person {person.id} (confidence: {confidence_val:.1f})")
        
        # Get skeleton angles
        angles = person.get_skeletal_angles()
        
        # Use library interpretation function
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
        
        # print pose description and question for the llm
        print(f"[LLM] Question type: {question_type}")
        print(f"[LLM] Prompt:\n{prompt_config['user'].format(pose_features=pose_description)}")
        
        # Submit to thread pool (fire and forget - non-blocking)
        self.executor.submit(
            self._llm_call, 
            pose_description, 
            person.id,
            prompt_config['system'],
            prompt_config['user']
        )

    def trigger_llm_analysis(self, key):
        """Handle keyboard input and trigger appropriate LLM analysis"""
        if not self.client_ready:
            print("[ERROR] LLM not available. Check API key configuration.")
            return
        
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
        all_count = len(people) if people else 0
        
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
    parser = argparse.ArgumentParser(description="SenseSpace LLM Client Example")
    parser.add_argument("--server", "-s", default="localhost", help="Server IP")
    parser.add_argument("--port", "-p", type=int, default=12345, help="Server port")
    parser.add_argument("--viz", action="store_true", help="Enable visualization (required for keyboard input)")
    args = parser.parse_args()
    
    if not args.viz:
        print("[WARNING] LLM example works best with --viz flag for keyboard input")
        print("[INFO] Run with: python llm.py --server localhost --viz")
    
    # Create LLM client wrapper
    llm_client = LLMClient()  
    
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