#!/usr/bin/env python3
"""
Whisper listener that waits for you to finish a sentence before transcribing.
Uses WebRTC VAD to detect silence (end of speech).
"""

import sounddevice as sd
import numpy as np
import whisper
import webrtcvad
import queue
import threading
import time
import sys
import signal

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_MS = 30          # must be 10, 20, or 30 ms for WebRTC VAD
SILENCE_TIME = 0.8     # seconds of silence = end of sentence
#MODEL_NAME = "base.en" # or "tiny", "base" only english
#MODEL_NAME = "base" # or "tiny", "base" international
#MODEL_NAME = "small" # the better model, still without GPU
MODEL_NAME = "tiny" # very small, but fast

# Derived
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)
VAD = webrtcvad.Vad(2)  # 0–3 aggressiveness (2 is balanced)
STOP_EVENT = threading.Event()
AUDIO_QUEUE = queue.Queue()

# Track last message type for smart line clearing
last_was_ignored = False

# ----------------------------------------------------------------------
# Whisper thread
# ----------------------------------------------------------------------
def whisper_worker():
    print("🧠 Loading Whisper model...")
    model = whisper.load_model(MODEL_NAME)
    print(f"✅ Whisper model '{MODEL_NAME}' ready.\n")

    while not STOP_EVENT.is_set():
        try:
            chunk = AUDIO_QUEUE.get(timeout=0.5)
        except queue.Empty:
            continue
        if chunk is None:
            break

        audio_float = chunk.astype(np.float32) / 32768.0
        result = model.transcribe(audio_float, fp16=False) # auto detect language
        #result = model.transcribe(audio_float, fp16=False, language="en")
        
        text = result["text"].strip()
        language = result.get("language", "unknown")
        
        if text:
            print(f"🗣️  [{language.upper()}] {text}\n")
            sys.stdout.flush()

# ----------------------------------------------------------------------
# Recorder with VAD
# ----------------------------------------------------------------------
def record_loop():
    """Continuously listen and segment by silence."""
    global last_was_ignored
    
    print(f"🎙️ Listening for full sentences... (Ctrl+C to stop)\n")

    buffer = np.zeros((0,), dtype=np.int16)
    silence_frames = 0
    silence_limit = int(SILENCE_TIME * 1000 / FRAME_MS)

    def callback(indata, frames, time_info, status):
        nonlocal buffer, silence_frames
        global last_was_ignored
        
        if STOP_EVENT.is_set():
            raise sd.CallbackStop()

        audio = (indata[:, 0] * 32767).astype(np.int16)
        is_speech = VAD.is_speech(audio.tobytes(), SAMPLE_RATE)

        buffer = np.concatenate((buffer, audio))

        if is_speech:
            silence_frames = 0
        else:
            silence_frames += 1

        # If enough silence → treat as end of sentence
        if silence_frames > silence_limit and len(buffer) > 0:
            # 🔍 Filter out very short or silent segments
            duration = len(buffer) / SAMPLE_RATE
            rms = np.sqrt(np.mean(buffer.astype(np.float32)**2)) / 32768.0

            if duration > 1.0 and rms > 0.01:
                # Clear line if previous was ignored
                if last_was_ignored:
                    print()  # New line after ignored messages
                
                print(f"💬 End of sentence detected ({duration:.1f}s, RMS={rms:.3f}) → sending to Whisper...\n")
                AUDIO_QUEUE.put(buffer)
                last_was_ignored = False
            else:
                # Use \r to overwrite line only if last was also ignored
                prefix = "\r" if last_was_ignored else ""
                print(f"{prefix}🤫 Ignored short/silent segment ({duration:.1f}s, RMS={rms:.3f})", end="", flush=True)
                last_was_ignored = True

            # Reset buffer for next segment
            buffer = np.zeros((0,), dtype=np.int16)
            silence_frames = 0

    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, dtype="float32", callback=callback, blocksize=FRAME_SIZE):
        while not STOP_EVENT.is_set():
            time.sleep(0.1)

# ----------------------------------------------------------------------
# Graceful shutdown
# ----------------------------------------------------------------------
def shutdown(signum=None, frame=None):
    print("\n👋 Stopping...")
    STOP_EVENT.set()
    AUDIO_QUEUE.put(None)
    time.sleep(1)
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("=" * 70)
    print("🎧 Whisper Sentence Listener (VAD based)")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Silence threshold: {SILENCE_TIME}s\n")

    t = threading.Thread(target=whisper_worker, daemon=True)
    t.start()
    record_loop()

if __name__ == "__main__":
    main()
