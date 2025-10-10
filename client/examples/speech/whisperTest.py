#!/usr/bin/env python3
"""
Simple Whisper speech-to-text test
Records audio and transcribes it
"""

import subprocess
import tempfile
import time
import os
import wave
import numpy as np
import re

# Configuration
RECORD_SECONDS = 5
SAMPLE_RATE = 16000
WHISPER_MODEL = "base"

def record_audio(duration, sample_rate=16000):
    """Record audio using arecord"""
    print(f"Recording for {duration} seconds...")
    print("Speak now!")
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        audio_file = tmp.name
    
    # Record using arecord (Linux)
    cmd = [
        'arecord',
        '-d', str(duration),
        '-f', 'cd',  # CD quality (44.1kHz, 16-bit, stereo)
        '-t', 'wav',
        audio_file
    ]
    
    result = subprocess.run(cmd, capture_output=True)
    print("Recording complete!")
    
    # Debug: check file size
    if os.path.exists(audio_file):
        size = os.path.getsize(audio_file)
        print(f"Audio file: {audio_file} ({size} bytes)")
    
    return audio_file

def transcribe_with_whisper(audio_file, model="base"):
    """Transcribe audio using Whisper"""
    print(f"Transcribing with Whisper ({model} model)...")
    
    # Run whisper with verbose output
    result = subprocess.run(
        ['whisper', audio_file, '--model', model, '--language', 'en'],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode == 0:
        # Extract transcription from stdout using regex
        # Look for timestamp lines like: [00:00.000 --> 00:03.760]  Transcribed text
        stdout = result.stdout
        
        # Find all timestamp lines
        pattern = r'\[\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}\.\d{3}\]\s+(.+)'
        matches = re.findall(pattern, stdout)
        
        if matches:
            # Join all transcription segments
            text = ' '.join(matches).strip()
            print(f"Transcribed: '{text}'")
            return text
        else:
            print("No transcription found in output")
            print(f"Full stdout:\n{stdout}")
            return None
    else:
        print(f"Whisper failed with return code: {result.returncode}")
        print(f"Error: {result.stderr}")
        return None

def main():
    print("=" * 60)
    print("Whisper Speech-to-Text Test")
    print("=" * 60)
    print()
    
    while True:
        print("\nPress ENTER to start recording (or Ctrl+C to exit)...")
        try:
            input()
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        
        # Record audio
        audio_file = record_audio(RECORD_SECONDS, SAMPLE_RATE)
        
        # Transcribe
        text = transcribe_with_whisper(audio_file, WHISPER_MODEL)
        
        # Cleanup
        if os.path.exists(audio_file):
            os.remove(audio_file)
        
        # Show result
        if text:
            print("\n" + "=" * 60)
            print("TRANSCRIPTION:")
            print("-" * 60)
            print(f">>> {text}")
            print("=" * 60)
        else:
            print("\nNo transcription received")

if __name__ == "__main__":
    main()