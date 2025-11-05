#!/bin/bash
# Test multicast receiver - receives multiplexed streams
# Usage: ./test_multicast_receiver.sh [multicast_address] [port]

MCAST_ADDR=${1:-239.255.0.1}
PORT=${2:-5000}

echo "Starting multicast receiver..."
echo "  Multicast: $MCAST_ADDR:$PORT"
echo "  Expecting: RGB streams on PT 96,98,100 and Depth on PT 97,99,101"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Receive multiplexed RTP stream and demux by payload type
gst-launch-1.0 -v \
  udpsrc address=$MCAST_ADDR port=$PORT auto-multicast=true \
    caps="application/x-rtp, media=(string)video, encoding-name=(string)H265, clock-rate=(int)90000" ! \
  rtpptdemux name=demux \
  demux.src_96 ! queue ! rtph265depay ! h265parse ! nvh265dec ! videoconvert ! autovideosink \
  demux.src_97 ! queue ! rtph265depay ! h265parse ! nvh265dec ! fakesink \
  demux.src_98 ! queue ! rtph265depay ! h265parse ! nvh265dec ! fakesink \
  demux.src_99 ! queue ! rtph265depay ! h265parse ! nvh265dec ! fakesink \
  demux.src_100 ! queue ! rtph265depay ! h265parse ! nvh265dec ! fakesink \
  demux.src_101 ! queue ! rtph265depay ! h265parse ! nvh265dec ! fakesink
