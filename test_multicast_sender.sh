#!/bin/bash
# Test multicast sender - simulates 3 camera streams
# Usage: ./test_multicast_sender.sh [multicast_address] [port]

MCAST_ADDR=${1:-239.255.0.1}
PORT=${2:-5000}

echo "Starting multicast sender..."
echo "  Multicast: $MCAST_ADDR:$PORT"
echo "  Streams: 3 test patterns (RGB PT=96,98,100 + Depth PT=97,99,101)"
echo ""

# Create 3 RGB streams (PT 96, 98, 100) and 3 Depth streams (PT 97, 99, 101)
# All multiplexed on single port using RTP

gst-launch-1.0 \
  videotestsrc pattern=0 ! video/x-raw,width=672,height=376,framerate=30/1 ! \
    videoconvert ! video/x-raw,format=NV12 ! \
    nvh265enc bitrate=2000 ! h265parse ! \
    rtph265pay pt=96 ! udpsink host=$MCAST_ADDR port=$PORT auto-multicast=true sync=false async=false \
  videotestsrc pattern=1 ! video/x-raw,width=672,height=376,framerate=30/1 ! \
    videoconvert ! video/x-raw,format=NV12 ! \
    nvh265enc bitrate=2000 ! h265parse ! \
    rtph265pay pt=98 ! udpsink host=$MCAST_ADDR port=$PORT auto-multicast=true sync=false async=false \
  videotestsrc pattern=2 ! video/x-raw,width=672,height=376,framerate=30/1 ! \
    videoconvert ! video/x-raw,format=NV12 ! \
    nvh265enc bitrate=2000 ! h265parse ! \
    rtph265pay pt=100 ! udpsink host=$MCAST_ADDR port=$PORT auto-multicast=true sync=false async=false \
  videotestsrc pattern=3 ! video/x-raw,width=672,height=376,framerate=30/1 ! \
    videoconvert ! video/x-raw,format=GRAY16_LE ! \
    nvh265enc bitrate=2000 ! h265parse ! \
    rtph265pay pt=97 ! udpsink host=$MCAST_ADDR port=$PORT auto-multicast=true sync=false async=false \
  videotestsrc pattern=4 ! video/x-raw,width=672,height=376,framerate=30/1 ! \
    videoconvert ! video/x-raw,format=GRAY16_LE ! \
    nvh265enc bitrate=2000 ! h265parse ! \
    rtph265pay pt=99 ! udpsink host=$MCAST_ADDR port=$PORT auto-multicast=true sync=false async=false \
  videotestsrc pattern=5 ! video/x-raw,width=672,height=376,framerate=30/1 ! \
    videoconvert ! video/x-raw,format=GRAY16_LE ! \
    nvh265enc bitrate=2000 ! h265parse ! \
    rtph265pay pt=101 ! udpsink host=$MCAST_ADDR port=$PORT auto-multicast=true sync=false async=false
