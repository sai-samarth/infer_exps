import json
import os
import time

import requests

base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
model = os.environ.get("SERVER_MODEL_ID", "Qwen/Qwen3.5-9B")

prompt = "In 2 concise sentences, explain why a smoke test is useful before running a larger inference benchmark."
payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 128,
    "temperature": 0.0,
    "stream": True,
}

start = time.perf_counter()
first_token_time = None
parts = []
with requests.post(f"{base_url}/chat/completions", json=payload, stream=True, timeout=180) as r:
    r.raise_for_status()
    for raw_line in r.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data: "):
            continue
        data = raw_line[6:]
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        reasoning = delta.get("reasoning")
        if reasoning:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            parts.append(reasoning)
        content = delta.get("content")
        if content:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            parts.append(content)
end = time.perf_counter()
text = "".join(parts)

result = {
    "prompt": prompt,
    "first_token_s": None if first_token_time is None else round(first_token_time - start, 4),
    "total_latency_s": round(end - start, 4),
    "response_preview": text[:500],
    "response_nonempty": bool(text.strip()),
}
print(json.dumps(result, indent=2))
