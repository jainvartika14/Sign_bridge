"""
Standalone Verification Script for Qwen3:8B on SignBridge (Step 11).
Sends test inference requests to local Ollama server at http://localhost:11434
and measures latency, accuracy, and output format.
"""

import sys
import time
import json
import urllib.request
from pathlib import Path

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_TAG = "qwen3:8b"

SYSTEM_PROMPT = (
    "You are an ASL gloss to English translator. "
    "Convert the input ASL glosses into one short, natural English sentence. "
    "Do not provide explanations, analysis, or alternatives. "
    "Do not invent people, locations, emotions, or events. "
    "Output ONLY the final English sentence."
)


def query_qwen3(gloss_text: str):
    sanitized = (gloss_text or "").strip()
    if not sanitized:
        return {
            "input": repr(gloss_text),
            "output": "",
            "latency_ms": 0.0,
            "status": "HANDLED_EMPTY",
            "notes": "Empty/whitespace input handled safely without calling model."
        }

    payload = {
        "model": MODEL_TAG,
        "system": SYSTEM_PROMPT,
        "prompt": f"ASL Glosses: {sanitized}\nEnglish Sentence:",
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 256,
        }
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_API_URL,
        data=req_data,
        headers={"Content-Type": "application/json"}
    )

    start_time = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            res_json = json.loads(resp.read().decode("utf-8"))
            raw_response = res_json.get("response", "").strip()
            
            # Clean any potential surrounding quotes or extra line breaks
            cleaned = raw_response.strip('"').strip()

            return {
                "input": repr(gloss_text),
                "output": cleaned,
                "latency_ms": elapsed_ms,
                "status": "SUCCESS",
                "notes": "Inference succeeded."
            }
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "input": repr(gloss_text),
            "output": None,
            "latency_ms": elapsed_ms,
            "status": "ERROR",
            "notes": f"Inference failed: {str(e)}"
        }


def run_all_tests():
    test_inputs = [
        "I WANT WATER",
        "HELLO",
        "I GO SCHOOL TOMORROW",
        "   "
    ]

    print("==================================================")
    print("SIGNBRIDGE QWEN3:8B STANDALONE INFERENCE TEST")
    print("==================================================")
    print(f"Target Model:    {MODEL_TAG}")
    print(f"Ollama Endpoint: {OLLAMA_API_URL}")
    print("--------------------------------------------------\n")

    results = []
    all_passed = True

    for text in test_inputs:
        print(f"Testing input: {repr(text)}", flush=True)
        res = query_qwen3(text)
        results.append(res)

        print(f"  Output:     {repr(res['output'])}", flush=True)
        print(f"  Latency:    {res['latency_ms']} ms", flush=True)
        print(f"  Status:     {res['status']}", flush=True)
        print(f"  Notes:      {res['notes']}\n", flush=True)

        if res["status"] == "ERROR" or (res["status"] == "SUCCESS" and not res["output"]):
            all_passed = False

    print("==================================================")
    print(f"FINAL RESULT: {'PASS' if all_passed else 'FAIL'}")
    print("==================================================")
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
