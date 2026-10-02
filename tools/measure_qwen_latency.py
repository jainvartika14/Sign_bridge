"""
Latency Diagnostic Script for Ollama Qwen3:8B.
Measures cold-start vs warm inference latency, model load duration,
prompt evaluation count/duration, and output generation duration.
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


def measure_request(gloss_text: str, label: str):
    payload = {
        "model": MODEL_TAG,
        "system": SYSTEM_PROMPT,
        "prompt": f"ASL Glosses: {gloss_text}\nEnglish Sentence:",
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

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        t1 = time.perf_counter()
        total_wall_ms = round((t1 - t0) * 1000, 2)
        res_json = json.loads(resp.read().decode("utf-8"))

        raw_response = res_json.get("response", "").strip()
        load_duration_ms = round(res_json.get("load_duration", 0) / 1e6, 2)
        prompt_eval_duration_ms = round(res_json.get("prompt_eval_duration", 0) / 1e6, 2)
        eval_duration_ms = round(res_json.get("eval_duration", 0) / 1e6, 2)
        total_duration_ms = round(res_json.get("total_duration", 0) / 1e6, 2)
        eval_count = res_json.get("eval_count", 0)
        prompt_eval_count = res_json.get("prompt_eval_count", 0)

        print(f"[{label}] Input: '{gloss_text}'")
        print(f"  Response Text:           '{raw_response}'")
        print(f"  Wall-Clock Total Latency: {total_wall_ms} ms")
        print(f"  Ollama Total Duration:   {total_duration_ms} ms")
        print(f"    - Model Load Duration:   {load_duration_ms} ms")
        print(f"    - Prompt Eval Duration:  {prompt_eval_duration_ms} ms ({prompt_eval_count} tokens)")
        print(f"    - Output Eval Duration:  {eval_duration_ms} ms ({eval_count} tokens)")
        print("")

        return {
            "label": label,
            "total_wall_ms": total_wall_ms,
            "load_duration_ms": load_duration_ms,
            "prompt_eval_duration_ms": prompt_eval_duration_ms,
            "eval_duration_ms": eval_duration_ms,
            "eval_count": eval_count,
            "response": raw_response
        }


def main():
    print("==================================================")
    print("OLLAMA QWEN3:8B LATENCY BENCHMARK & DIAGNOSTIC")
    print("==================================================\n")

    # Request 1: Cold start (if unloaded) or baseline
    res1 = measure_request("I WANT WATER", "Request 1 (Cold / Initial)")

    # Request 2: Warm start
    res2 = measure_request("I WANT WATER", "Request 2 (Warm)")

    # Request 3: Warm start (Different glosses)
    res3 = measure_request("I GO SCHOOL TOMORROW", "Request 3 (Warm)")

    print("==================================================")
    print("LATENCY SUMMARY:")
    print(f"  Cold Load Duration:  {res1['load_duration_ms']} ms")
    print(f"  Req 1 Total Wall:    {res1['total_wall_ms']} ms")
    print(f"  Req 2 Total Wall:    {res2['total_wall_ms']} ms (Warm)")
    print(f"  Req 3 Total Wall:    {res3['total_wall_ms']} ms (Warm)")
    print("==================================================")


if __name__ == "__main__":
    main()
