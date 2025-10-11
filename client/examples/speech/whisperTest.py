#!/usr/bin/env python3
"""
Simple Whisper speech-to-text test (in-memory version)
Records audio in memory and transcribes it without touching disk
"""

import sounddevice as sd
import numpy as np
import io
import wave
import tempfile
import os
import sys
import re

# Install: pip install openai-whisper
import whisper

# Configuration
RECORD_SECONDS = 5
SAMPLE_RATE = 16000
WHISPER_MODEL = "base"

def record_audio_memory(duration, sample_rate=16000):
    """Record audio directly into memory using sounddevice"""
    print(f"🎙️  Recording for {duration} seconds @ {sample_rate} Hz...")
    print("Speak now!")
    
    # Record audio
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='int16',  # 16-bit PCM
        blocking=True
    )
    
    print("✅ Recording complete!")
    
    # Flatten and return
    return audio_data.flatten()

def numpy_to_wav_bytes(audio_data, sample_rate=16000):
    """Convert numpy array to WAV format in memory"""
    # Create in-memory bytes buffer
    wav_buffer = io.BytesIO()
    
    # Write WAV file to buffer
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit = 2 bytes
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    # Get bytes
    wav_buffer.seek(0)
    return wav_buffer

def transcribe_with_whisper_memory(audio_data, sample_rate=16000, model_name="base"):
    """Transcribe audio from memory using Whisper Python API"""
    print(f"\n🤖 Transcribing with Whisper ({model_name} model)...")
    print(f"   Audio shape: {audio_data.shape}, dtype: {audio_data.dtype}")
    
    # Convert int16 to float32 normalized to [-1.0, 1.0]
    # Whisper expects float32 audio
    audio_float = audio_data.astype(np.float32) / 32768.0
    
    print(f"   Converted to float32: min={audio_float.min():.4f}, max={audio_float.max():.4f}")
    
    # Load Whisper model (cached after first load)
    print(f"   Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)
    
    # Transcribe
    print(f"   Running transcription...")
    result = model.transcribe(
        audio_float,
        language='en',
        verbose=True,  # Show progress
        fp16=False  # Use FP32 for CPU compatibility
    )
    
    # Extract text
    text = result['text'].strip()
    
    print(f"\n✅ Transcription complete!")
    print(f"   Detected language: {result.get('language', 'unknown')}")
    
    # Show segments if available
    if 'segments' in result:
        print(f"   Segments: {len(result['segments'])}")
        for i, seg in enumerate(result['segments']):
            start = seg.get('start', 0)
            end = seg.get('end', 0)
            seg_text = seg.get('text', '').strip()
            print(f"      [{start:5.2f}s -> {end:5.2f}s] {seg_text}")
    
    return text, result

def main():
    print("=" * 70)
    print("Whisper Speech-to-Text Test (In-Memory)")
    print("=" * 70)
    print()
    print(f"Model: {WHISPER_MODEL}")
    print(f"Duration: {RECORD_SECONDS}s")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print()
    
    while True:
        print("\nPress ENTER to start recording (or Ctrl+C to exit)...")
        try:
            input()
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            break
        
        # Record audio in memory
        audio_data = record_audio_memory(RECORD_SECONDS, SAMPLE_RATE)
        
        # Check audio level
        max_amp = np.max(np.abs(audio_data))
        print(f"📊 Max amplitude: {max_amp} / 32768 ({max_amp/32768.0*100:.1f}%)")
        
        if max_amp < 100:
            print("⚠️  WARNING: Very quiet recording! Speak louder.")
        
        # Transcribe from memory
        try:
            text, result = transcribe_with_whisper_memory(audio_data, SAMPLE_RATE, WHISPER_MODEL)
            
            # Show result
            print("\n" + "=" * 70)
            print("📝 TRANSCRIPTION:")
            print("-" * 70)
            print(f">>> {text}")
            print("=" * 70)
            
        except Exception as e:
            print(f"\n❌ Transcription failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()