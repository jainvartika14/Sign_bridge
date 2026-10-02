import json
import urllib.request

url_gen = "http://localhost:11434/api/generate"
url_chat = "http://localhost:11434/api/chat"

print("--- TESTING /api/generate RAW ---")
payload_gen = {
    "model": "qwen3:8b",
    "prompt": "Translate ASL glosses 'I WANT WATER' into one simple English sentence.",
    "stream": False
}
req = urllib.request.Request(url_gen, data=json.dumps(payload_gen).encode("utf-8"), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read().decode("utf-8"))
    print("GEN RESPONSE KEYS:", res.keys())
    print("GEN RAW RESPONSE:", repr(res.get("response")))
    print("GEN THINKING/THOUGHT:", repr(res.get("thinking")), repr(res.get("thought")))

print("\n--- TESTING /api/chat RAW ---")
payload_chat = {
    "model": "qwen3:8b",
    "messages": [
        {"role": "system", "content": "You are an ASL gloss to English translator. Convert ASL glosses into one short natural English sentence. Return ONLY the final English sentence without explanations or extra words."},
        {"role": "user", "content": "I WANT WATER"}
    ],
    "stream": False,
    "options": {
        "temperature": 0.0,
        "num_predict": 100
    }
}
req_chat = urllib.request.Request(url_chat, data=json.dumps(payload_chat).encode("utf-8"), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req_chat) as resp:
    res_chat = json.loads(resp.read().decode("utf-8"))
    print("CHAT RESPONSE KEYS:", res_chat.keys())
    msg = res_chat.get("message", {})
    print("CHAT MESSAGE ROLE:", msg.get("role"))
    print("CHAT MESSAGE CONTENT:", repr(msg.get("content")))
    print("CHAT THINKING:", repr(msg.get("thinking")))
