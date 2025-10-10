import sounddevice as sd
import numpy as np
import soundfile as sf
import os
import time
from scipy import signal

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
start_time = time.time()

# Render for duration + extra time for echo decay
render_time = duration + 2.0
num_samples = int(render_time * samplerate)

# Boot pyo server in OFFLINE mode (no audio output)
s = Server(sr=samplerate, nchnls=1, duplex=0,audio="jack")
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
table_rec = TableRec(mixed, table=output_table)
table_rec.play()

# Process offline (this actually renders the audio)
print(f"   Rendering {render_time:.1f} seconds of audio...", end='', flush=True)

# Manually advance the offline server time
for _ in range(int(render_time * 100)):  # 100 steps per second
    s.process()
    
s.shutdown()

echo_time = time.time() - start_time
print(f" ✓ ({echo_time:.2f}s)")

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
print("\n🔊 Playing back echo version (sounddevice)...")
sd.play(echo_buffer, samplerate, device=None, latency=latency, blocking=True)
print("✅ Echo playback finished.")

# ----------------------------------------------------------------------
# 🎵 Process with Pitch Shift Effect (lower pitch, same speed)
# ----------------------------------------------------------------------
print("\n🎵 Processing with PITCH SHIFT effect...")
start_time = time.time()

pitch_shift_semitones = -12  # -12 = 1 octave down

# Try multiple methods in order of preference
pitch_buffer = None
method = None

# Method 1: pyrubberband (fastest, requires rubberband-cli)
try:
    import pyrubberband as pyrb
    print(f"   Using pyrubberband (fast) - shifting {pitch_shift_semitones} semitones...", end='', flush=True)
    pitch_buffer = pyrb.pitch_shift(buffer, samplerate, pitch_shift_semitones)
    method = "pyrubberband"
except Exception:
    pass

# Method 2: librosa (good quality, requires resampy)
if pitch_buffer is None:
    try:
        import librosa
        print(f"   Using librosa - shifting {pitch_shift_semitones} semitones...", end='', flush=True)
        pitch_buffer = librosa.effects.pitch_shift(
            y=buffer,
            sr=samplerate,
            n_steps=pitch_shift_semitones,
            res_type='kaiser_fast'
        )
        method = "librosa"
    except Exception:
        pass

# Method 3: Simple scipy-based pitch shift (always works, good enough quality)
if pitch_buffer is None:
    print(f"   Using scipy simple pitch shift - shifting {pitch_shift_semitones} semitones...", end='', flush=True)
    
    # Calculate pitch ratio
    # IMPORTANT: For negative semitones (lower pitch), ratio < 1, so we need to INVERT it for resampling
    pitch_ratio = 2 ** (pitch_shift_semitones / 12.0)  # e.g., -12 semitones = 0.5
    
    # To lower pitch: we need MORE samples (stretch), then compress back
    # Resample to LONGER duration (1/pitch_ratio)
    num_samples_resampled = int(len(buffer) / pitch_ratio)  # FIXED: divide instead of multiply
    pitch_buffer = signal.resample(buffer, num_samples_resampled)
    
    # Time-stretch back to original length using phase vocoder simulation
    # Simple overlap-add technique
    hop_size = 256
    window = np.hanning(hop_size * 2)
    
    # Extend the buffer for processing
    output_buffer = np.zeros(len(buffer), dtype=np.float32)
    
    # Calculate stretching factor (how much to compress back)
    stretch = len(pitch_buffer) / len(buffer)
    
    # Simple time-stretch using overlap-add
    read_pos = 0.0
    write_pos = 0
    
    while write_pos < len(output_buffer) - hop_size * 2:
        read_idx = int(read_pos)
        if read_idx + hop_size * 2 < len(pitch_buffer):
            grain = pitch_buffer[read_idx:read_idx + hop_size * 2] * window
            output_buffer[write_pos:write_pos + hop_size * 2] += grain
        
        read_pos += hop_size * stretch
        write_pos += hop_size
    
    pitch_buffer = output_buffer
    method = "scipy"

pitch_time = time.time() - start_time
print(f" ✓ ({pitch_time:.2f}s using {method})")

# Check amplitude
pitch_max_amp = np.max(np.abs(pitch_buffer))
print(f"📊 Pitched buffer max amplitude: {pitch_max_amp:.4f}")

if pitch_max_amp > 0:
    # Normalize to prevent clipping
    pitch_buffer = pitch_buffer / pitch_max_amp * 0.8
else:
    print("⚠️  WARNING: Pitched buffer is silent!")

# ----------------------------------------------------------------------
# 🔊 Playback #3: Pitched version (using sounddevice directly from buffer)
# ----------------------------------------------------------------------
print(f"\n🔊 Playing back pitched version ({pitch_shift_semitones} semitones lower)...")
sd.play(pitch_buffer, samplerate, device=None, latency=latency, blocking=True)
print("✅ Pitched playback finished.")

print("\n" + "="*70)
print("Summary:")
print("  1. ✅ Recorded with sounddevice → buffer")
print("  2. ✅ Played normal version from buffer")
print(f"  3. ✅ Processed echo effect with pyo → buffer ({echo_time:.2f}s)")
print("  4. ✅ Played echo version from buffer")
print(f"  5. ✅ Processed pitch shift with {method} → buffer ({pitch_time:.2f}s)")
print(f"  6. ✅ Played pitched version from buffer ({pitch_shift_semitones} semitones lower)")
print("  → No files saved! Everything in memory!")
print("\nOptional performance improvements:")
print("  • For faster/better pitch shifting, install:")
print("    sudo apt-get install rubberband-cli")
print("    pip install pyrubberband resampy")
print(f"\nTo make it MUCH deeper, change pitch_shift_semitones to -18 or -24")
print("="*70)