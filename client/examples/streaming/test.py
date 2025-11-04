import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)

# Check for required encoders
encoders = ['nvh265enc', 'vaapih265enc', 'x265enc']
print("Available H.265 encoders:")
for enc in encoders:
    element = Gst.ElementFactory.make(enc, None)
    if element:
        print(f"  ✓ {enc}")
    else:
        print(f"  ✗ {enc}")

# Check for required plugins
plugins = ['rtpbin', 'udpsrc', 'udpsink', 'rtph265pay', 'appsrc']
print("\nRequired plugins:")
for plugin in plugins:
    element = Gst.ElementFactory.make(plugin, None)
    print(f"  {'✓' if element else '✗'} {plugin}")