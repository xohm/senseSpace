#!/bin/bash
# Test multicast receiver - receives multiplexed streams
# Usage: ./test_multicast_receiver.sh [multicast_address] [port]

MCAST_ADDR=${1:-239.255.0.1}
PORT=${2:-5000}

echo "Starting multicast receiver..."
echo "  Multicast: $MCAST_ADDR:$PORT"
echo "  Expecting: 3 RGB streams on PT 96,98,100"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Receive multiplexed RTP stream and demux by payload type
# Display first stream, consume others silently
gst-launch-1.0 -v \
  udpsrc address=$MCAST_ADDR port=$PORT auto-multicast=true multicast-group=$MCAST_ADDR \
    caps="application/x-rtp, media=(string)video, encoding-name=(string)H265, clock-rate=(int)90000" ! \
  rtpptdemux name=demux \
  demux.src_96 ! queue ! rtph265depay ! h265parse ! nvh265dec ! videoconvert ! autovideosink \
  demux.src_98 ! queue ! rtph265depay ! h265parse ! nvh265dec ! fakesink \
  demux.src_100 ! queue ! rtph265depay ! h265parse ! nvh265dec ! fakesink
