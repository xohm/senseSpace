#!/usr/bin/env python3
import os
import time
import argparse

# Set this BEFORE importing pyo
os.environ["PYO_GUI_WX"] = "0"

from pyo import *

# Optional: use sounddevice just to list devices (since your pyo lacks pa_* helpers)
try:
    import sounddevice as sd
except ImportError:
    sd = None


def list_devices():
    print("=" * 70)
    print("Audio devices (via sounddevice, PortAudio indices):")
    print("=" * 70)
    if sd is None:
        print("sounddevice not installed. To list devices, install it:")
        print("  pip install sounddevice")
        print("Or use system tools (arecord -l, pactl list sources).")
        print()
        return

    devs = sd.query_devices()
    print("🎤 INPUT (microphones):")
    for i, d in enumerate(devs):
        if d["max_input_channels"] > 0:
            print(f"  [{i:2d}] {d['name']}  (default_sr={int(d['default_samplerate'])})")
    print("\n🔊 OUTPUT (speakers):")
    for i, d in enumerate(devs):
        if d["max_output_channels"] > 0:
            print(f"  [{i:2d}] {d['name']}  (default_sr={int(d['default_samplerate'])})")
    print("=" * 70)
    print()


def boot_server(mic_idx: int, spk_idx: int | None, sr: int | None, buffersize: int) -> Server:
    # Try requested SR first, then fallbacks
    srs = [sr] if sr else []
    srs += [48000, 44100, 32000, 16000]
    tried = set()
    for rate in srs:
        if rate in tried:
            continue
        tried.add(rate)
        for buf in (buffersize, 1024, 512):
            try:
                s = Server(sr=rate, nchnls=2, buffersize=buf, duplex=1, audio="portaudio")
                s.setInputDevice(mic_idx)
                if spk_idx is not None:
                    s.setOutputDevice(spk_idx)
                s.boot()
                if not s.getIsBooted():
                    s.shutdown()
                    continue
                s.start()
                if s.getIsStarted():
                    print(f"✓ pyo Server started @ {int(s.getSamplingRate())} Hz, buffersize={buf}")
                    return s
                s.shutdown()
            except Exception:
                try:
                    s.shutdown()
                except Exception:
                    pass
                continue
    raise RuntimeError("Failed to start pyo Server with given devices. Try different --mic/--speaker or --samplerate/--buffersize.")


def main():
    ap = argparse.ArgumentParser(description="Record and play using pyo (PortAudio backend)")
    ap.add_argument("--mic", type=int, required=True, help="Input device index (PortAudio index)")
    ap.add_argument("--speaker", type=int, default=None, help="Output device index (PortAudio index)")
    ap.add_argument("--duration", type=float, default=5.0, help="Record duration in seconds")
    ap.add_argument("--samplerate", type=int, default=None, help="Preferred sample rate (e.g. 48000)")
    ap.add_argument("--buffersize", type=int, default=1024, help="Buffersize in frames (e.g. 512, 1024)")
    ap.add_argument("--list", action="store_true", help="List devices and exit")
    args = ap.parse_args()

    if args.list:
        list_devices()
        return

    # Always print device list to help user confirm indices
    list_devices()
    print(f"Using mic index: {args.mic}")
    print(f"Using speaker index: {args.speaker if args.speaker is not None else 'default'}")
    print(f"Requested samplerate: {args.samplerate or 'auto'}  buffersize: {args.buffersize}")
    print()

    # Boot server (tries given SR/buffer then fallbacks)
    s = boot_server(args.mic, args.speaker, args.samplerate, args.buffersize)

    duration = args.duration
    out_file = "recorded_pyo.wav"

    print(f"\n🎙️ Recording {duration:.1f}s ... Speak now!")
    mic = Input(chnl=0)

    # Record to WAV, 16-bit PCM
    rec = Record(mic, filename=out_file, chnls=1, fileformat=0, sampletype=1)
    rec.play()
    time.sleep(duration + 0.25)
    rec.stop()
    print(f"✅ Recording complete → {out_file}")

    # Playback
    print("\n🔊 Playing back...")
    player = SfPlayer(out_file, loop=False, mul=0.8).out()
    time.sleep(duration + 0.5)
    player.stop()

    s.stop()
    s.shutdown()
    print("✅ Done.")


if __name__ == "__main__":
    main()
