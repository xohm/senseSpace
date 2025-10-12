# -----------------------------------------------------------------------------
# Sense Space - Minimal Client Example
# -----------------------------------------------------------------------------
# Demonstrates basic client with three simple callbacks
# -----------------------------------------------------------------------------

import argparse
import sys

# Setup paths
from senseSpaceLib.senseSpace import setup_paths
setup_paths()

from senseSpaceLib.senseSpace import MinimalClient, Frame


class MiniClient:
    """Tiny wrapper demonstrating the three callbacks"""
    
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
    
    def on_connection_changed(self, connected: bool):
        status = "Connected" if connected else "Disconnected"
        print(f"[CONNECTION] {status}")


def main():
    parser = argparse.ArgumentParser(description="SenseSpace Minimal Client Example")
    parser.add_argument("--server", "-s", default="localhost", help="Server IP")
    parser.add_argument("--port", "-p", type=int, default=12345, help="Server port")
    parser.add_argument("--viz", action="store_true", help="Enable visualization")
    args = parser.parse_args()
    
    # Create wrapper with callbacks
    mini_client = MiniClient()
    
    # Create and run minimal client
    client = MinimalClient(
        server_ip=args.server,
        server_port=args.port,
        viz=args.viz,
        on_init=mini_client.on_init,
        on_frame=mini_client.on_frame,
        on_connection_changed=mini_client.on_connection_changed
    )
    
    success = client.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()