# -----------------------------------------------------------------------------
# Sense Space
# -----------------------------------------------------------------------------
# Ollama LLM Client Example with Expert System
# -----------------------------------------------------------------------------
# IAD, Zurich University of the Arts / zhdk.ch
# Max Rheiner
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# This example demonstrates the use of an expert system for pose analysis.
# 
# Unlike llm_ollama.py which uses direct LLM prompts, this example uses:
#   - Expert configuration (JSON) that defines context, rules, and examples
#   - LLMClient wrapper with conversation context management
#   - LLMFrameAnalyzer for generic frame-to-LLM integration
#
# The expert system approach provides:
#   - Consistent, domain-specific responses
#   - Faster inference through persistent context
#   - Easy behavior modification via JSON configuration
#   - Better response quality through few-shot learning
# 
# Expert configuration is loaded from: ../data/expert_pose_config.json
# 
# Keyboard shortcuts (with --viz):
#   SPACE - Analyze current pose using expert system
#   R     - Reset conversation context
# -----------------------------------------------------------------------------

import argparse
import sys

# Setup paths
from senseSpaceLib.senseSpace import setup_paths
setup_paths()

from senseSpaceLib.senseSpace import MinimalClient
from senseSpaceLib.senseSpace.llmClient import LLMClient
from senseSpaceLib.senseSpace.llmFrameAnalyzer import LLMFrameAnalyzer


def print_response(response: str):
    """Simple response handler - prints expert system output"""
    if response:
        print(f"[POSE] {response}")
    else:
        print("[ERROR] No response from model (may need more VRAM)")


def main():
    parser = argparse.ArgumentParser(
        description="SenseSpace LLM Expert System Example",
        epilog="Example: python llm_ollama_expert.py --viz --model llama3.2:1b"
    )
    parser.add_argument("--server", "-s", default="localhost", help="Server IP")
    parser.add_argument("--port", "-p", type=int, default=12345, help="Server port")
    parser.add_argument("--viz", action="store_true", help="Enable visualization")
    parser.add_argument('--model', type=str, default=None, help='Override model from config')
    parser.add_argument("--expert", "-e", default="../data/expert_pose_config.json", 
                       help="Path to expert configuration JSON")
    parser.add_argument("--confidence", "-c", type=float, default=70.0,
                       help="Minimum confidence threshold for person detection")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed debug output")
    args = parser.parse_args()
    
    # Model priority: command line arg > expert config > default
    model_name = args.model if args.model else "llama3.2:1b"
    
    # Create LLM client with expert configuration
    llm_client = LLMClient(
        model_name=args.model,  # Will be None if not specified
        expert_json="../data/expert_pose_config.json",
        auto_download=True,
        verbose=args.verbose
    )
    
    print(f"[INFO] Model: {llm_client.model_name}")
    if args.viz:
        print("[INFO] Press SPACE to analyze pose, R to reset context")
    
    # Create generic analyzer
    analyzer = LLMFrameAnalyzer(
        llm_client=llm_client,
        confidence_threshold=args.confidence,
        verbose=args.verbose  # Pass verbose flag
    )
    analyzer.set_response_callback(print_response)
    
    # Create minimal client
    client = MinimalClient(
        server_ip=args.server,
        server_port=args.port,
        viz=args.viz,
        on_init=analyzer.on_init,
        on_frame=analyzer.on_frame,
        on_connection_changed=lambda connected: None if not args.verbose else
            print(f"[CONNECTION] {'Connected' if connected else 'Disconnected'}")
    )
    
    # Keyboard handler
    def handle_keyboard(key: str):
        """Handle keyboard input for LLM analysis"""
        if key == ' ':
            # print out current people for debugging
            llmInput = analyzer.analyze_current_pose()
            print(f"[DEBUG] LLM Input:\n{llmInput}\n")
        elif key == 'r':
            analyzer.reset_context()
            if args.verbose:
                print("[INFO] Context reset")
    
    if args.viz:
        client.llm_callback = handle_keyboard
    
    success = client.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()