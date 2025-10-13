#!/usr/bin/env python3
"""
Speech Audio I/O Example with Voice Effects

Demonstrates:
- Voice activity detection (VAD)
- Multi-language transcription (Whisper)
- Text-to-speech with multiple voices (Piper)
- Voice effects (echo, pitch shift)
- Command-based effect triggers
"""

from speechIO import SpeechAudioIO
import signal

# Language code to name mapping (Whisper supported languages)
LANGUAGE_NAMES = {
    'en': 'English',
    'zh': 'Chinese',
    'de': 'German',
    'es': 'Spanish',
    'ru': 'Russian',
    'ko': 'Korean',
    'fr': 'French',
    'ja': 'Japanese',
    'pt': 'Portuguese',
    'tr': 'Turkish',
    'pl': 'Polish',
    'ca': 'Catalan',
    'nl': 'Dutch',
    'ar': 'Arabic',
    'sv': 'Swedish',
    'it': 'Italian',
    'id': 'Indonesian',
    'hi': 'Hindi',
    'fi': 'Finnish',
    'vi': 'Vietnamese',
    'he': 'Hebrew',
    'uk': 'Ukrainian',
    'el': 'Greek',
    'ms': 'Malay',
    'cs': 'Czech',
    'ro': 'Romanian',
    'da': 'Danish',
    'hu': 'Hungarian',
    'ta': 'Tamil',
    'no': 'Norwegian',
    'th': 'Thai',
    'ur': 'Urdu',
    'hr': 'Croatian',
    'bg': 'Bulgarian',
    'lt': 'Lithuanian',
    'la': 'Latin',
    'mi': 'Maori',
    'ml': 'Malayalam',
    'cy': 'Welsh',
    'sk': 'Slovak',
    'te': 'Telugu',
    'fa': 'Persian',
    'lv': 'Latvian',
    'bn': 'Bengali',
    'sr': 'Serbian',
    'az': 'Azerbaijani',
    'sl': 'Slovenian',
    'kn': 'Kannada',
    'et': 'Estonian',
    'mk': 'Macedonian',
    'br': 'Breton',
    'eu': 'Basque',
    'is': 'Icelandic',
    'hy': 'Armenian',
    'ne': 'Nepali',
    'mn': 'Mongolian',
    'bs': 'Bosnian',
    'kk': 'Kazakh',
    'sq': 'Albanian',
    'sw': 'Swahili',
    'gl': 'Galician',
    'mr': 'Marathi',
    'pa': 'Punjabi',
    'si': 'Sinhala',
    'km': 'Khmer',
    'sn': 'Shona',
    'yo': 'Yoruba',
    'so': 'Somali',
    'af': 'Afrikaans',
    'oc': 'Occitan',
    'ka': 'Georgian',
    'be': 'Belarusian',
    'tg': 'Tajik',
    'sd': 'Sindhi',
    'gu': 'Gujarati',
    'am': 'Amharic',
    'yi': 'Yiddish',
    'lo': 'Lao',
    'uz': 'Uzbek',
    'fo': 'Faroese',
    'ht': 'Haitian Creole',
    'ps': 'Pashto',
    'tk': 'Turkmen',
    'nn': 'Nynorsk',
    'mt': 'Maltese',
    'sa': 'Sanskrit',
    'lb': 'Luxembourgish',
    'my': 'Myanmar',
    'bo': 'Tibetan',
    'tl': 'Tagalog',
    'mg': 'Malagasy',
    'as': 'Assamese',
    'tt': 'Tatar',
    'haw': 'Hawaiian',
    'ln': 'Lingala',
    'ha': 'Hausa',
    'ba': 'Bashkir',
    'jw': 'Javanese',
    'su': 'Sundanese',
}


def get_language_name(language_code: str) -> str:
    """
    Convert language code to full name
    
    Args:
        language_code: Two-letter language code (e.g., 'ja', 'en')
    
    Returns:
        Full language name (e.g., 'Japanese', 'English')
    """
    return LANGUAGE_NAMES.get(language_code.lower(), language_code.upper())


def parse_effect_command(text: str, language: str):
    """
    Parse voice effect commands from text
    
    Returns:
        (cleaned_text, effect_dict or None)
    """
    # Clean the text: lowercase, strip
    text_lower = text.lower().strip()
    
    # Split into words (this automatically handles punctuation attached to words)
    words = text_lower.split()
    
    # No words? Return as-is
    if not words:
        return text, None
    
    # Clean first two words of punctuation
    first_word = words[0].rstrip('.,!?;:') if len(words) > 0 else ""
    second_word = words[1].rstrip('.,!?;:') if len(words) > 1 else ""
    
    # Combine first two words for command matching
    command = f"{first_word} {second_word}".strip()
    
    # Echo command
    if command == "effect echo":
        cleaned_text = " ".join(words[2:]) if len(words) > 2 else ""
        if not cleaned_text:
            return "", None
        return cleaned_text, {"effect": "echo", "value": 0.4}
    
    # Low pitch (English)
    if command == "effect low":
        cleaned_text = " ".join(words[2:]) if len(words) > 2 else ""
        if not cleaned_text:
            return "", None
        return cleaned_text, {"effect": "pitch", "value": 0.8}
    
    # High pitch (English)
    if command == "effect high":
        cleaned_text = " ".join(words[2:]) if len(words) > 2 else ""
        if not cleaned_text:
            return "", None
        return cleaned_text, {"effect": "pitch", "value": 1.3}
    
    # Echo (German)
    if command == "effekt echo":
        cleaned_text = " ".join(words[2:]) if len(words) > 2 else ""
        if not cleaned_text:
            return "", None
        return cleaned_text, {"effect": "echo", "value": 0.4}
    
    # Tief (German - low)
    if command == "effekt tief":
        cleaned_text = " ".join(words[2:]) if len(words) > 2 else ""
        if not cleaned_text:
            return "", None
        return cleaned_text, {"effect": "pitch", "value": 0.8}
    
    # Hoch (German - high)
    if command == "effekt hoch":
        cleaned_text = " ".join(words[2:]) if len(words) > 2 else ""
        if not cleaned_text:
            return "", None
        return cleaned_text, {"effect": "pitch", "value": 1.3}
    
    # No effect command found - return original text
    return text, None


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    
    # Add command-line arguments
    parser = argparse.ArgumentParser(description="Speech Audio I/O Example")
    parser.add_argument("--mic", type=int, default=None, help="Microphone device ID")
    parser.add_argument("--speaker", type=int, default=None, help="Speaker device ID")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    args = parser.parse_args()
    
    # List devices if requested
    if args.list_devices:
        SpeechAudioIO.list_devices()
        exit(0)
    
    # List devices
    SpeechAudioIO.list_devices()
    
    # Create instance - models will auto-download to ./models/
    io = SpeechAudioIO(
        mic_index=args.mic,           # From command line
        speaker_index=args.speaker,   # From command line
        whisper_model="small",
        piper_model_en="en_US-amy-medium",
        piper_model_de="de_DE-thorsten_emotional-medium",
        piper_model_fallback="en_US-lessac-medium",
        auto_download_models=True,
        vad_aggressiveness=2,
        silence_duration=1.0,
        min_speech_duration=1.5,
        min_rms_threshold=0.02
    )
    
    # Speaker IDs for thorsten_emotional:
    # 0: amused, 1: angry, 2: disgusted, 3: drunk, 4: neutral, 5: sleepy, 6: surprised, 7: whisper
    SPEAKER_WHISPER = 7
    
    # Transcription callback - auto-selects voice based on detected language
    def on_transcription(language: str, text: str):
        print(f"\n🗣️  [{language.upper()}] {text}\n")
        
        # Parse effect commands from text
        cleaned_text, effect = parse_effect_command(text, language)
        
        # Skip if no text to speak (command only)
        if not cleaned_text:
            print("   ⚠️  No text after command, skipping playback")
            return
        
        # Determine which language/speaker to use
        if language == "de" and "de" in io.piper_models:
            # Use German voice with whisper mode
            io.speak(cleaned_text, language="de", speaker=SPEAKER_WHISPER, effect=effect)
        elif language == "en" and "en" in io.piper_models:
            # Use English voice (Amy)
            io.speak(cleaned_text, language="en", effect=effect)
        else:
            # Unknown language - notify user and use fallback
            language_name = get_language_name(language)
            print(f"   ⚠️  Language '{language_name}' has no voice installed")
            
            # Speak notification in English using fallback voice
            notification = f"Language {language_name} has no voice installed."
            io.speak(notification, language="en", effect=None)
            
            # Print the original text
            print(f"   📝 Original text: {cleaned_text}")
    
    # Graceful shutdown
    def shutdown(signum, frame):
        print("\n👋 Shutting down...")
        io.cleanup()
        exit(0)
    
    signal.signal(signal.SIGINT, shutdown)
    
    # Start listening
    print("\n🎤 Say something in English or German!")
    print("   Commands:")
    print("     English: 'effect echo ...', 'effect low ...', 'effect high ...'")
    print("     German:  'effekt echo ...', 'effekt tief ...', 'effekt hoch ...'")
    io.start_listening(on_transcription=on_transcription)