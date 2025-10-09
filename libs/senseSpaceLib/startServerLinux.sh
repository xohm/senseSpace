#!/bin/bash
# filepath: /home/max/Documents/development/own/senseSpace/server/startServerLinux.sh

# SenseSpace Fusion Server Startup Script
# Starts the ZED fusion server with visualization

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate virtual environment if it exists
VENV_PATH="$SCRIPT_DIR/../.venv"
if [ -d "$VENV_PATH" ]; then
    echo "[INFO] Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
else
    echo "[WARNING] Virtual environment not found at $VENV_PATH"
    echo "[WARNING] Using system Python"
fi

# Change to server directory
cd "$SCRIPT_DIR"

# Start the fusion server with visualization
echo "[INFO] Starting SenseSpace Fusion Server..."
python3 senseSpace_fusion_main.py --viz

# Deactivate venv on exit
if [ -d "$VENV_PATH" ]; then
    deactivate
fi