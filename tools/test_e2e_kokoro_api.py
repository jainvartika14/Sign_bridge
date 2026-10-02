"""
End-to-End Test Script for SignBridge Kokoro TTS API Migration (Step 8).
Sends request to /api/tts/synthesize endpoint via FastAPI TestClient,
validates HTTP response headers, saves audio file, and verifies WAV properties.
"""

import sys
import wave
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from signbridge.web.app import app

OUTPUT_WAV_PATH = PROJECT_ROOT / "tools" / "kokoro_e2e_test.wav"


def run_e2e_test():
    client = TestClient(app)

    payload = {
        "text": "Hello. This is SignBridge using Kokoro.",
        "voice_id": "af_sarah",
        "rate": 1.0,
        "request_id": "kokoro-e2e-test"
    }

    print("[E2E TEST] Sending POST /api/tts/synthesize request...")
    response = client.post("/api/tts/synthesize", json=payload)

    status_code = response.status_code
    content_type = response.headers.get("Content-Type")
    provider_header = response.headers.get("X-TTS-Provider")
    request_id_header = response.headers.get("X-Request-ID")
    latency_ms = response.headers.get("X-TTS-Latency-MS")
    audio_bytes = response.content

    print(f"  HTTP Status Code: {status_code}")
    print(f"  Content-Type:     {content_type}")
    print(f"  X-TTS-Provider:   {provider_header}")
    print(f"  X-Request-ID:     {request_id_header}")
    print(f"  Latency (ms):     {latency_ms}")
    print(f"  Audio Bytes Size: {len(audio_bytes)} bytes")

    # 1. HTTP Success
    if status_code != 200:
        print(f"FAIL: Expected status 200, got {status_code}")
        sys.exit(1)

    # 2. Content-Type audio/wav
    if content_type != "audio/wav":
        print(f"FAIL: Expected Content-Type 'audio/wav', got '{content_type}'")
        sys.exit(1)

    # 3. Provider check
    if provider_header != "kokoro_onnx":
        print(f"FAIL: Expected provider 'kokoro_onnx', got '{provider_header}'")
        sys.exit(1)

    # 4. Save audio WAV
    OUTPUT_WAV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_WAV_PATH, "wb") as f:
        f.write(audio_bytes)
    print(f"  Saved WAV audio file to: {OUTPUT_WAV_PATH}")

    # 5. Check WAV file readability & sample rate
    try:
        with wave.open(str(OUTPUT_WAV_PATH), "rb") as wf:
            num_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            num_frames = wf.getnframes()
            duration = num_frames / float(frame_rate)

            print("\n[WAV PROPERTIES]")
            print(f"  Readable:     YES")
            print(f"  Channels:     {num_channels}")
            print(f"  Sample Width: {sample_width} bytes ({sample_width * 8}-bit)")
            print(f"  Sample Rate:  {frame_rate} Hz")
            print(f"  Frames Count: {num_frames}")
            print(f"  Duration:     {duration:.2f} seconds")

            if frame_rate != 24000:
                print(f"FAIL: Expected sample rate 24000 Hz, got {frame_rate} Hz")
                sys.exit(1)

            if num_frames == 0 or len(audio_bytes) <= 44:
                print("FAIL: Audio file is empty or missing frames")
                sys.exit(1)

    except Exception as e:
        print(f"FAIL: WAV file is unreadable: {e}")
        sys.exit(1)

    print("\n==========================================")
    print("FINAL STATUS: PASS")
    print("==========================================")


if __name__ == "__main__":
    run_e2e_test()
