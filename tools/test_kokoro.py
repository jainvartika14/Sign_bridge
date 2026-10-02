"""
Standalone verification script for Kokoro ONNX TTS Engine.
Verifies model loading, voice listing, speech generation, and WAV file export.
"""

from pathlib import Path
import soundfile as sf
from kokoro_onnx import Kokoro

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "kokoro" / "kokoro-v0_19.onnx"
VOICES_PATH = PROJECT_ROOT / "models" / "kokoro" / "voices.bin"
OUTPUT_WAV_PATH = PROJECT_ROOT / "tools" / "kokoro_test.wav"

TEST_TEXT = "Hello. This is a standalone Kokoro test for SignBridge."

def main():
    print(f"Loading Kokoro model from: {MODEL_PATH}")
    print(f"Loading voices asset from: {VOICES_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    if not VOICES_PATH.exists():
        raise FileNotFoundError(f"Voices file not found at {VOICES_PATH}")

    # Initialize Kokoro ONNX instance
    kokoro = Kokoro(str(MODEL_PATH), str(VOICES_PATH))
    print("[SUCCESS] Kokoro model loaded successfully!")

    # Retrieve available voices
    available_voices = kokoro.get_voices()
    print(f"Available voices count: {len(available_voices)}")
    print(f"Voices list: {available_voices}")

    if not available_voices:
        raise ValueError("No voices found in voices asset file.")

    # Select preferred voice or first available
    preferred_voice = "af_sarah" if "af_sarah" in available_voices else available_voices[0]
    print(f"Selected Voice: {preferred_voice}")

    print(f"Generating speech for text: '{TEST_TEXT}'...")
    samples, sample_rate = kokoro.create(TEST_TEXT, voice=preferred_voice, speed=1.0, lang="en-us")

    OUTPUT_WAV_PATH.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(OUTPUT_WAV_PATH), samples, sample_rate)

    wav_size = OUTPUT_WAV_PATH.stat().st_size
    print("\n--- STANDALONE KOKORO TEST SUMMARY ---")
    print(f"Status: PASS")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Voices Path: {VOICES_PATH}")
    print(f"Selected Voice: {preferred_voice}")
    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Output WAV Path: {OUTPUT_WAV_PATH}")
    print(f"Output File Size: {wav_size:,} bytes")
    print("---------------------------------------")

if __name__ == "__main__":
    main()
