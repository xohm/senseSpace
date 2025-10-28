#!/bin/bash
# Install MediaPipe segmentation dependencies for seg_pointcloud_server.py

echo "==================================="
echo "MediaPipe Segmentation Setup"
echo "==================================="
echo ""
echo "This script installs MediaPipe and its dependencies for seg_pointcloud_server.py"
echo ""

# Check if we're in the server directory
if [ ! -f "seg_pointcloud_server.py" ]; then
    echo "Error: Please run this script from the server/ directory"
    exit 1
fi

echo "Step 1/2: Installing MediaPipe without dependencies..."
echo "This avoids the numpy version conflict with pyzed."
pip install --no-deps mediapipe

if [ $? -ne 0 ]; then
    echo "Error: Failed to install MediaPipe"
    exit 1
fi

echo ""
echo "Step 2/2: Installing MediaPipe dependencies..."
echo "These are compatible with numpy 2.x from pyzed."
pip install -r requirements_segmentation.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi

echo ""
echo "==================================="
echo "Installation Complete!"
echo "==================================="
echo ""
echo "You can now run the MediaPipe segmentation server:"
echo "  python seg_pointcloud_server.py --viz"
echo ""
echo "Note: You may see a warning about numpy version incompatibility."
echo "This can be safely ignored - both pyzed and mediapipe will work correctly."
echo ""
