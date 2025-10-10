import sounddevice as sd
import numpy as np
import soundfile as sf
import os
import time

# Suppress pyo GUI warnings
os.environ['PYO_GUI_WX'] = '0'
from pyo import *

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

# Record the audio
buffer = sd.rec(
    int(duration * samplerate),
    samplerate=samplerate,
    channels=channels,
    dtype='float32',
    device=mic_index,
    latency=latency,
    blocking=True
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

# ----------------------------------------------------------------------
# 🔊 Playback #1: Normal (using sounddevice directly from buffer)
# ----------------------------------------------------------------------
print("\n🔊 Playing back normal version (sounddevice)...")
sd.play(buffer, samplerate, device=None, latency=latency, blocking=True)
print("✅ Normal playback finished.")

# ----------------------------------------------------------------------
# 🎵 Process with Echo Effect (using pyo OFFLINE - directly from buffer)
# ----------------------------------------------------------------------
print("\n🎵 Processing with ECHO effect (pyo offline rendering)...")

# Render for duration + extra time for echo decay
render_time = duration + 2.0
num_samples = int(render_time * samplerate)

# Boot pyo server in OFFLINE mode (no audio output)
s = Server(sr=samplerate, nchnls=1, duplex=0, audio="offline")
s.recordOptions(dur=render_time)
s.boot()
s.start()

# Create a table from our numpy buffer
table_size = len(buffer)
input_table = DataTable(size=table_size, init=buffer.tolist())

# Create a table reader with correct frequency
sound = TableRead(table=input_table, freq=float(samplerate)/table_size, loop=False, mul=1.0)

# Create echo effect
echo = Delay(sound, delay=[0.25, 0.5], feedback=0.6, maxdelay=2, mul=0.5)

# Mix original + echo
mixed = sound + echo

# Create output table to capture the result
output_table = NewTable(length=render_time, chnls=1)
table_rec = TableRec(mixed, table=output_table)  # Removed fadein/fadeout
table_rec.play()

# Process offline (this actually renders the audio)
print(f"   Rendering {render_time:.1f} seconds of audio...")

# Manually advance the offline server time
for _ in range(int(render_time * 100)):  # 100 steps per second
    s.process()
    
s.shutdown()

print(f"✅ Echo effect processed")

# Extract the processed audio from the table
echo_buffer = np.array(output_table.getTable(), dtype='float32')

# Check if we got audio
echo_max_amp = np.max(np.abs(echo_buffer))
print(f"📊 Echo buffer max amplitude: {echo_max_amp:.4f}")

if echo_max_amp < 0.001:
    print("⚠️  WARNING: Echo buffer is empty! Falling back to manual echo...")
    # Fallback: simple manual echo using numpy
    delay_samples_1 = int(0.25 * samplerate)
    delay_samples_2 = int(0.5 * samplerate)
    
    # Create echo buffer
    echo_buffer = np.zeros(num_samples, dtype='float32')
    
    # Original sound
    echo_buffer[:len(buffer)] = buffer
    
    # Add delayed copies
    if delay_samples_1 < num_samples:
        end_idx = min(len(buffer), num_samples - delay_samples_1)
        echo_buffer[delay_samples_1:delay_samples_1+end_idx] += buffer[:end_idx] * 0.6
    
    if delay_samples_2 < num_samples:
        end_idx = min(len(buffer), num_samples - delay_samples_2)
        echo_buffer[delay_samples_2:delay_samples_2+end_idx] += buffer[:end_idx] * 0.4
    
    # Normalize
    max_val = np.max(np.abs(echo_buffer))
    if max_val > 0:
        echo_buffer = echo_buffer / max_val * 0.8

# Trim to actual duration (remove silence padding)
echo_buffer = echo_buffer[:num_samples]

# ----------------------------------------------------------------------
# 🔊 Playback #2: Echo version (using sounddevice directly from buffer)
# ----------------------------------------------------------------------
print("🔊 Playing back echo version (sounddevice)...")
sd.play(echo_buffer, samplerate, device=None, latency=latency, blocking=True)
print("✅ Echo playback finished.")

print("\n" + "="*70)
print("Summary:")
print("  1. ✅ Recorded with sounddevice → buffer")
print("  2. ✅ Played normal version from buffer")
print("  3. ✅ Processed echo effect with pyo (OFFLINE) → buffer")
print("  4. ✅ Played echo version from buffer")
print("  → No files saved! Everything in memory!")
print("="*70)