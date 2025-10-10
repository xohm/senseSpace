import sounddevice as sd
import numpy as np
import soundfile as sf

# ----------------------------------------------------------------------
# 🎤 List microphones (input devices)
# ----------------------------------------------------------------------
print("🎤 Available input devices:\n")
devices = sd.query_devices()
mic_devices = [(i, d['name']) for i, d in enumerate(devices) if d['max_input_channels'] > 0]
for i, name in mic_devices:
    print(f"  [{i}] {name}")
print()

# ----------------------------------------------------------------------
# 🎚️ Select microphone
# ----------------------------------------------------------------------
preferred_mic = "C920"  # set to part of the mic name, or leave "" for default
mic_index = None

for i, name in mic_devices:
    if preferred_mic and preferred_mic.lower() in name.lower():
        mic_index = i
        print(f"✅ Using preferred mic: {name} (index {i})")
        break

if mic_index is None:
    mic_index = mic_devices[0][0]
    print(f"⚠️ Using first available mic: {mic_devices[0][1]} (index {mic_index})")

# ----------------------------------------------------------------------
# 🎙️ Record from selected mic
# ----------------------------------------------------------------------
info = sd.query_devices(mic_index)
samplerate = int(info['default_samplerate'])
channels = 1  # Use mono for cleaner recording
duration = 5.0
latency = 'high'  # Options: 'low', 'high', or a number in seconds (e.g. 0.2)

print(f"\n🎧 Using mic: {info['name']} @ {samplerate} Hz ({channels} ch)")
print(f"🎙️ Recording {duration} s with latency={latency}... Speak now!")

# Actually record the audio (this was missing!)
buffer = sd.rec(
    int(duration * samplerate),
    samplerate=samplerate,
    channels=channels,
    dtype='float32',
    device=mic_index,
    latency=latency,  # Added latency parameter
    blocking=True  # Wait until recording is complete
)

print("✅ Recording complete.")

# Check if we actually recorded something
max_amp = np.max(np.abs(buffer))
print(f"📊 Max amplitude: {max_amp:.4f}")

if max_amp < 0.001:
    print("⚠️  WARNING: Very quiet or silent recording!")
else:
    print("✅ Audio recorded successfully")

# Flatten if needed
if buffer.ndim > 1:
    buffer = buffer.flatten()

# Save to file
sf.write("recorded.wav", buffer, samplerate, subtype='PCM_16')
print("💾 Saved recorded.wav")

# ----------------------------------------------------------------------
# 🔊 Playback on KDE / PipeWire default output
# ----------------------------------------------------------------------
print("\n🔊 Playing back on default output device...")
sd.play(buffer, samplerate, device=None, latency=latency, blocking=True)
print("✅ Playback finished.")
