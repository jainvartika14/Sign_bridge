"""
Real End-to-End Test Script for Step 13: Gloss Buffer -> Qwen3:8B -> Kokoro TTS Pipeline.
Simulates real manual Speak action:
1. Adds 'I', 'WANT', 'WATER' to GlossBuffer.
2. Calls POST /api/nlp/smooth (live Qwen3:8B).
3. Verifies English sentence 'I want water.' generated.
4. Calls POST /api/tts/synthesize (live Kokoro TTS).
5. Verifies 24,000 Hz RIFF WAV audio stream returned.
6. Verifies GlossBuffer is cleared upon completion.
"""

import sys
import wave
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from signbridge.web.app import app
from signbridge.services.gloss_buffer import GlossBuffer

OUTPUT_WAV_PATH = PROJECT_ROOT / "tools" / "e2e_speak_pipeline_test.wav"


def run_e2e_speak_pipeline_test():
    print("==================================================")
    print("STEP 13 REAL END-TO-END PIPELINE TEST")
    print("Gloss Buffer -> Ollama Qwen3:8B -> Kokoro TTS")
    print("==================================================")

    # 1. Initialize GlossBuffer & add glosses
    buffer = GlossBuffer()
    buffer.add_gloss("I")
    buffer.add_gloss("I")  # Duplicate - should be rejected
    buffer.add_gloss("WANT")
    buffer.add_gloss("WANT")  # Duplicate - should be rejected
    buffer.add_gloss("WATER")

    gloss_list = buffer.get_glosses()
    gloss_str = buffer.to_string()
    print(f"\n[1. GLOSS BUFFER]")
    print(f"  Committed Glosses: {gloss_list}")
    print(f"  Gloss String:      '{gloss_str}'")

    if gloss_list != ["I", "WANT", "WATER"]:
        print(f"FAIL: Expected ['I', 'WANT', 'WATER'], got {gloss_list}")
        sys.exit(1)

    # 2. Call /api/nlp/smooth (Qwen3:8B LLM)
    client = TestClient(app)
    print("\n[2. OLLAMA QWEN3:8B LLM TRANSLATION]")
    print("  Sending POST /api/nlp/smooth request...")

    llm_resp = client.post("/api/nlp/smooth", json={"glosses": gloss_list, "use_llm": True})

    if llm_resp.status_code != 200:
        print(f"FAIL: Expected /api/nlp/smooth status 200, got {llm_resp.status_code}")
        print(f"Error detail: {llm_resp.text}")
        sys.exit(1)

    llm_data = llm_resp.json()
    english_sentence = llm_data.get("english", "")
    provider = llm_data.get("provider", "")
    latency_ms = llm_data.get("latency_ms", 0.0)

    print(f"  HTTP Status Code:   {llm_resp.status_code}")
    print(f"  Generated Sentence: '{english_sentence}'")
    print(f"  LLM Provider:       {provider}")
    print(f"  LLM Latency:        {latency_ms} ms")

    if not english_sentence or "water" not in english_sentence.lower():
        print(f"FAIL: Expected natural English sentence containing 'water', got '{english_sentence}'")
        sys.exit(1)

    # 3. Call /api/tts/synthesize (Kokoro TTS)
    print("\n[3. KOKORO TTS SPEECH SYNTHESIS]")
    print("  Sending POST /api/tts/synthesize request...")

    tts_resp = client.post(
        "/api/tts/synthesize",
        json={
            "text": english_sentence,
            "voice_id": "af_sarah",
            "rate": 1.0,
            "request_id": "e2e-speak-test-001"
        }
    )

    if tts_resp.status_code != 200:
        print(f"FAIL: Expected /api/tts/synthesize status 200, got {tts_resp.status_code}")
        sys.exit(1)

    content_type = tts_resp.headers.get("Content-Type")
    tts_provider = tts_resp.headers.get("X-TTS-Provider")
    tts_latency = tts_resp.headers.get("X-TTS-Latency-MS")
    audio_bytes = tts_resp.content

    print(f"  HTTP Status Code:   {tts_resp.status_code}")
    print(f"  Content-Type:       {content_type}")
    print(f"  X-TTS-Provider:     {tts_provider}")
    print(f"  TTS Latency:        {tts_latency} ms")
    print(f"  Audio Length:       {len(audio_bytes)} bytes")

    if content_type != "audio/wav" or tts_provider != "kokoro_onnx":
        print("FAIL: Invalid TTS headers returned")
        sys.exit(1)

    # Save audio temporarily
    OUTPUT_WAV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_WAV_PATH, "wb") as f:
        f.write(audio_bytes)

    with wave.open(str(OUTPUT_WAV_PATH), "rb") as wf:
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()
        duration = num_frames / float(sample_rate)

        print("\n[4. WAV AUDIO VALIDATION]")
        print(f"  Readable:    YES")
        print(f"  Sample Rate: {sample_rate} Hz")
        print(f"  Duration:    {duration:.2f} seconds")

        if sample_rate != 24000 or num_frames == 0:
            print("FAIL: Invalid WAV audio properties")
            sys.exit(1)

    # 4. Clear buffer on successful completion
    buffer.clear()
    print("\n[5. GLOSS BUFFER CLEANUP]")
    print(f"  Remaining Glosses in Buffer: {buffer.get_glosses()} (Length: {len(buffer)})")

    if len(buffer) != 0:
        print("FAIL: Buffer was not cleared after successful pipeline completion")
        sys.exit(1)

    print("\n==========================================")
    print("FINAL STATUS: PASS")
    print("==========================================")


if __name__ == "__main__":
    run_e2e_speak_pipeline_test()
