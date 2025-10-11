#!/usr/bin/env python3
"""
Record with sounddevice, playback with pyo (real-time effects)
- sounddevice: stable recording
- pyo: real-time playback with echo and pitch shift effects
"""
import sounddevice as sd
import numpy as np
import os
import time
import argparse

# Suppress pyo GUI warnings
os.environ['PYO_GUI_WX'] = '0'
from pyo import *

from tempfile import NamedTemporaryFile
import soundfile as sf

# ----------------------------------------------------------------------
# 🎤 List devices
# ----------------------------------------------------------------------
def list_devices():
    print("=" * 70)
    print("🎤 sounddevice INPUT devices:")
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            default = " [DEFAULT]" if i == sd.default.device[0] else ""
            print(f"  [{i:2d}] {d['name']}{default}")
    
    print("\n🔊 sounddevice OUTPUT devices:")
    for i, d in enumerate(devices):
        if d['max_output_channels'] > 0:
            default = " [DEFAULT]" if i == sd.default.device[1] else ""
            print(f"  [{i:2d}] {d['name']}{default}")
    
    print("\n" + "=" * 70)
    print("🔊 pyo (PortAudio) OUTPUT devices:")
    print("=" * 70)
    # Create temporary server to list devices
    try:
        temp_s = Server(audio="portaudio")
        temp_s.boot()
        temp_s.pa_list_devices()
        temp_s.shutdown()
    except Exception as e:
        print(f"⚠️  Could not list pyo devices: {e}")
    print("=" * 70)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Record (sounddevice) + Real-time Playback (pyo)')
    parser.add_argument('--mic', type=int, default=None, help='Mic device index (sounddevice)')
    parser.add_argument('--speaker', type=int, default=None, help='Speaker device index (pyo/PortAudio)')
    parser.add_argument('--duration', '-d', type=float, default=5.0, help='Recording duration')
    parser.add_argument('--pitch', '-p', type=int, default=-4, help='Pitch shift semitones')
    parser.add_argument('--list', action='store_true', help='List devices')
    args = parser.parse_args()

    if args.list:
        list_devices()
        return

    # ----------------------------------------------------------------------
    # Select mic
    # ----------------------------------------------------------------------
    preferred_mic = "C920"
    mic_index = args.mic
    
    if mic_index is None:
        devices = sd.query_devices()
        mic_devices = [(i, d['name']) for i, d in enumerate(devices) if d['max_input_channels'] > 0]
        
        for i, name in mic_devices:
            if preferred_mic and preferred_mic.lower() in name.lower():
                mic_index = i
                print(f"✅ Using preferred mic: {name} (index {i})")
                break
        
        if mic_index is None:
            mic_index = mic_devices[0][0]
            print(f"⚠️ Using first available mic: {mic_devices[0][1]} (index {mic_index})")

    # ----------------------------------------------------------------------
    # Record
    # ----------------------------------------------------------------------
    info = sd.query_devices(mic_index)
    samplerate = int(info['default_samplerate'])
    
    print(f"\n🎧 Mic: {info['name']} @ {samplerate} Hz")
    print(f"🎙️ Recording {args.duration}s... Speak now!")
    
    buffer = sd.rec(
        int(args.duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype='float32',
        device=mic_index,
        latency='high',
        blocking=True
    )
    
    print("✅ Recording complete.")
    buffer = buffer.flatten()
    
    max_amp = np.max(np.abs(buffer))
    print(f"📊 Max amplitude: {max_amp:.4f}")
    
    if max_amp < 0.001:
        print("⚠️ Very quiet recording!")
        return

    # ----------------------------------------------------------------------
    # Resample to standard rate if needed (pyo may not support mic's rate)
    # ----------------------------------------------------------------------
    target_samplerate = 48000  # Standard rate that all devices support
    
    if samplerate != target_samplerate:
        print(f"\n🔄 Resampling from {samplerate} Hz to {target_samplerate} Hz for pyo compatibility...")
        from scipy import signal as sp_signal
        num_samples = int(len(buffer) * target_samplerate / samplerate)
        buffer = sp_signal.resample(buffer, num_samples).astype(np.float32)
        samplerate = target_samplerate
        print(f"✓ Resampled to {samplerate} Hz")

    # ----------------------------------------------------------------------
    # Boot pyo server for REAL-TIME playback
    # ----------------------------------------------------------------------
    print("\n🎵 Booting pyo server (real-time)...")
    s = Server(sr=samplerate, nchnls=1, duplex=0, audio="portaudio", buffersize=512)
    
    # Set output device if specified
    if args.speaker is not None:
        print(f"🔊 Setting output device to index {args.speaker}")
        s.setOutputDevice(args.speaker)
    else:
        print("🔊 Using default output device")
    
    try:
        s.boot()
    except Exception as e:
        print(f"❌ Failed to boot pyo server: {e}")
        print("Try:")
        print("  1. Run with --list to see available devices")
        print("  2. Use --speaker <index> to specify output device")
        print("  3. Check your audio configuration")
        return
    
    if not s.getIsBooted():
        print("❌ pyo server failed to boot")
        return
        
    s.start()
    print(f"✓ pyo server running @ {int(s.getSamplingRate())} Hz")
    
    duration = len(buffer) / samplerate
    
    # ----------------------------------------------------------------------
    # Playback #1: Normal
    # ----------------------------------------------------------------------
    print("\n🔊 Playing normal version (pyo)...")
    
    table1 = DataTable(size=len(buffer), init=buffer.tolist())
    player1 = TableRead(table=table1, freq=1.0/duration, loop=False, mul=0.8)
    player1.out()
    
    time.sleep(duration + 0.2)
    player1.stop()
    print("✅ Normal playback finished.")
    
    time.sleep(0.5)
    
    # ---------------------------------------------------------
    # Playback #2 — Echo
    # ---------------------------------------------------------
    print("\n🔊 Playing echo version (pyo real-time)...")
    table2 = DataTable(size=len(buffer), init=buffer.tolist())
    sound = TableRead(table=table2, freq=1.0/duration, loop=False, mul=0.8)
    echo = Delay(sound, delay=[0.3, 0.6], feedback=0.2, maxdelay=2, mul=1.0)
    mixed = sound + echo
    mixed.out()

    live_objects = [sound, echo, mixed]
    sound.play()
    time.sleep(duration + 2.5)

    for obj in live_objects:
        obj.stop()
    print("✅ Echo playback finished.")


    # ---------------------------------------------------------
    # Playback #3 — Pitch shift (REAL-TIME)
    # ---------------------------------------------------------
    print(f"\n🔊 Playing pitched version ({args.pitch:+d} semitones, pyo real-time)...")

    table3 = DataTable(size=len(buffer), init=buffer.tolist())
    sound2 = TableRead(table=table3, freq=1.0/duration, loop=True, mul=1.0)

    # Short FFT analysis
    fft = PVAnal(sound2, size=2048, overlaps=4)
    transpo_ratio = 2 ** (args.pitch / 12.0)  # semitone → frequency ratio
    pitched = PVTranspose(fft, transpo_ratio)
    ifft = PVSynth(pitched, mul=1.2).out()

    # Keep everything alive
    live_objects = [sound2, fft, pitched, ifft]
    sound2.play()

    print("🎧 Playing FFT-based pitch-shift...")
    time.sleep(duration + 0.5)

    for obj in live_objects:
        try:
            obj.stop()
        except:
            pass
    print("✅ Pitch-shift playback finished.")

    
    # ----------------------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------------------
    s.stop()
    s.shutdown()
    
    print("\n" + "="*70)
    print("Summary:")
    print("  1. ✅ Recorded with sounddevice")
    print("  2. ✅ Played normal with pyo")
    print("  3. ✅ Played echo with pyo (REAL-TIME)")
    print(f"  4. ✅ Played pitch shift with pyo (REAL-TIME, {args.pitch:+d} semitones)")
    print("  → All effects processed in REAL-TIME during playback!")
    print("\nUsage:")
    print(f"  python3 {os.path.basename(__file__)} --list  # Show all devices")
    print(f"  python3 {os.path.basename(__file__)} --mic 5 --speaker 3 --pitch -24  # Deep voice")
    print(f"  python3 {os.path.basename(__file__)} --mic 5 --speaker 3 --pitch -18  # Demon")
    print(f"  python3 {os.path.basename(__file__)} --mic 5 --speaker 3 --pitch +7   # Chipmunk")
    print("="*70)


if __name__ == "__main__":
    main()