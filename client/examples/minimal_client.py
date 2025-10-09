# -----------------------------------------------------------------------------
# Sense Space - Minimal Client Example
# -----------------------------------------------------------------------------
# Demonstrates basic client with three simple callbacks
# -----------------------------------------------------------------------------

import argparse
import sys
import os

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

class MiniClient:
    """Tiny wrapper"""
    
    def __init__(self):
        self.persons = 0
    
    def on_init(self):
        print(f"[INIT] Connected to server")
    
    def on_frame(self, frame: Frame):
        people = getattr(frame, "people", None)
        count = len(people) if people else 0
        if self.persons != count:
            self.persons = count
            print(f"[FRAME] Received {count} people")
            # TODO: 
    
    def on_connection_changed(self, connected: bool):
        status = "Connected" if connected else "Disconnected"
        print(f"[CONNECTION] {status}")


def main():
    parser = argparse.ArgumentParser(description="SenseSpace LLM Client Example")
    parser.add_argument("--server", "-s", default="localhost", help="Server IP")
    parser.add_argument("--port", "-p", type=int, default=12345, help="Server port")
    parser.add_argument("--viz", action="store_true", help="Enable visualization")
    args = parser.parse_args()
    
    # Create LLM client wrapper
    miniClient = MiniClient()  
    
    # Create minimal client with LLM callbacks
    client = MinimalClient(
        server_ip=args.server,
        server_port=args.port,
        viz=args.viz,
        on_init=miniClient.on_init,
        on_frame=miniClient.on_frame,
        on_connection_changed=miniClient.on_connection_changed
    )   
    
    success = client.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()