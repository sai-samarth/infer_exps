import json
import math
import os
import subprocess
import time
from pathlib import Path
from statistics import mean, median

import requests
from transformers import AutoTokenizer

SERVER_MODEL_ID = os.environ.get("SERVER_MODEL_ID", "Qwen/Qwen3.5-9B")
TOKENIZER_ID = os.environ.get("TOKENIZER_ID", "Qwen/Qwen3.5-9B")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))
SEED = int(os.environ.get("SEED", "0"))


def make_long_blob(label: str, n: int) -> str:
    base = (
        f"{label}: Careful inference measurement needs stable prompts, deterministic generation, warmup exclusion, "
        f"and separated latency metrics so that optimization claims are interpretable and reproducible. "
    )
    return "".join(base for _ in range(n))


PROMPTS = [
    {"name": "short_baseline_rationale", "bucket": "short", "messages": [{"role": "user", "content": "In 2 sentences, explain why TTFT and decode speed should be reported separately when benchmarking LLM inference."}]},
    {"name": "short_metric_choice", "bucket": "short", "messages": [{"role": "user", "content": "Give 3 concise reasons median latency is often more informative than average latency for user-facing LLM inference."}]},
    {"name": "short_quality_canary", "bucket": "short", "messages": [{"role": "user", "content": "Answer directly: what is the main risk of comparing runtimes using different prompt lengths and different output lengths?"}]},
    {"name": "medium_eval_design", "bucket": "medium", "messages": [{"role": "user", "content": "I am designing a local LLM inference benchmark for a single user query on one GPU. The benchmark is supposed to compare runtimes fairly. Please propose a compact evaluation protocol that measures cold load, TTFT, total latency, decode speed, and VRAM, while keeping generation deterministic and prompt lengths controlled. Keep the answer structured and practical."}]},
    {"name": "medium_table_summarization", "bucket": "medium", "messages": [{"role": "user", "content": "You are given an experiment table with these rows: run A = TTFT 0.82 s, total latency 7.1 s, decode 29 tok/s; run B = TTFT 1.35 s, total latency 6.8 s, decode 34 tok/s; run C = TTFT 0.76 s, total latency 8.4 s, decode 24 tok/s. Summarize which run would feel best for an interactive user and why."}]},
    {"name": "medium_context_reasoning", "bucket": "medium", "messages": [{"role": "user", "content": "A team claims a new runtime is faster because it generates 200 tokens in less time than the old runtime generated 80 tokens. They did not control prompt length, batch size, or sampling settings. Analyze the methodological flaws and recommend a corrected comparison plan."}]},
    {"name": "medium_extraction_canary", "bucket": "medium", "messages": [{"role": "user", "content": "Read this short note and extract the final decision in one sentence.\n\nNote: We ran one eager baseline, confirmed the model fits on a 24GB 4090, and saw about 20 tok/s decode speed. The next step is not throughput tuning. The next step is to build a better single-query benchmark with TTFT, structured prompts, and quality canaries."}]},
    {"name": "long_protocol_review", "bucket": "long", "messages": [{"role": "user", "content": make_long_blob("Protocol background", 18) + "Now write a short review of the protocol's strengths and possible blind spots for single-query benchmarking."}]},
    {"name": "long_fact_canary", "bucket": "long", "messages": [{"role": "user", "content": "Use only the facts from the context below. After reading it, answer this question: what should be measured separately from decode speed, and why?\n\n" + make_long_blob("Context", 16) + "Extra facts: users feel TTFT directly, throughput matters less for batch size 1, and prompt-length control is required for fair comparisons."}]},
    {"name": "long_structured_summary", "bucket": "long", "messages": [{"role": "user", "content": "Summarize the practical lessons from the following repeated benchmark notes as a numbered list with 4 items.\n\n" + make_long_blob("Benchmark note", 20)}]},
    {"name": "verylong_prefill_probe_4k", "bucket": "verylong", "messages": [{"role": "user", "content": "Read the long context below and then answer in 3 short bullets: what matters most for fair single-query inference benchmarking?\n\n" + make_long_blob("Very long context", 100)}]},
]

WARMUP_MESSAGES = [{"role": "user", "content": "Say one sentence about why warmup runs are excluded from benchmark summaries."}]


def percentile(values, q):
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    vals = sorted(float(v) for v in values)
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def summarize_metric(values):
    if not values:
        return None
    vals = [float(v) for v in values]
    return {
        "mean": round(mean(vals), 4),
        "median": round(median(vals), 4),
        "p95": round(percentile(vals, 0.95), 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
    }


def gpu_memory_used_gib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()[0]
        return float(out) / 1024.0
    except Exception:
        return None


def prepare_prompt(tokenizer, messages):
    raw_start = time.perf_counter()
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tok_start = time.perf_counter()
    encoded = tokenizer(text, return_tensors="pt")
    tok_end = time.perf_counter()
    prompt_tokens = int(encoded["input_ids"].shape[-1])
    return {
        "prompt_tokens": prompt_tokens,
        "tokenization_time_s": tok_end - tok_start,
        "raw_to_ready_s": tok_end - raw_start,
    }


def stream_one(tokenizer, messages, name, bucket):
    prep = prepare_prompt(tokenizer, messages)
    payload = {
        "model": SERVER_MODEL_ID,
        "messages": messages,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.0,
        "stream": True,
        "seed": SEED,
    }
    start = time.perf_counter()
    first_token_time = None
    parts = []
    with requests.post(f"{BASE_URL}/chat/completions", json=payload, stream=True, timeout=1800) as r:
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code} for {name}: {r.text[:500]}")
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
    output_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    ttft_from_request_s = None if first_token_time is None else first_token_time - start
    ttft_from_raw_input_s = None if first_token_time is None else prep["raw_to_ready_s"] + ttft_from_request_s
    decode_only_s = None if first_token_time is None else max(end - first_token_time, 1e-9)
    decode_tok_s_post_ttft = None if output_tokens <= 0 or decode_only_s is None else output_tokens / decode_only_s
    total_latency_s = end - start
    resident_vram_gib = gpu_memory_used_gib()
    return {
        "name": name,
        "bucket": bucket,
        "prompt_tokens": prep["prompt_tokens"],
        "output_tokens": output_tokens,
        "tokenization_time_s": round(prep["tokenization_time_s"], 4),
        "raw_to_ready_s": round(prep["raw_to_ready_s"], 4),
        "ttft_from_request_s": round(ttft_from_request_s, 4) if ttft_from_request_s is not None else None,
        "ttft_from_raw_input_s": round(ttft_from_raw_input_s, 4) if ttft_from_raw_input_s is not None else None,
        "total_latency_s": round(total_latency_s, 4),
        "overall_tok_s": round(output_tokens / total_latency_s, 4) if total_latency_s > 0 else None,
        "decode_tok_s_post_ttft": round(decode_tok_s_post_ttft, 4) if decode_tok_s_post_ttft is not None else None,
        "resident_vram_gib": round(resident_vram_gib, 4) if resident_vram_gib is not None else None,
        "generated_text_preview": text[:300],
        "ended_cleanly": output_tokens < MAX_NEW_TOKENS,
    }


def build_summary(runs):
    metrics = ["prompt_tokens", "output_tokens", "tokenization_time_s", "raw_to_ready_s", "ttft_from_request_s", "ttft_from_raw_input_s", "total_latency_s", "overall_tok_s", "decode_tok_s_post_ttft", "resident_vram_gib"]
    out = {"overall": {}, "by_bucket": {}}
    for metric in metrics:
        vals = [r[metric] for r in runs if r.get(metric) is not None]
        out["overall"][metric] = summarize_metric(vals)
    for bucket in sorted(set(r["bucket"] for r in runs)):
        bruns = [r for r in runs if r["bucket"] == bucket]
        out["by_bucket"][bucket] = {}
        for metric in metrics:
            vals = [r[metric] for r in bruns if r.get(metric) is not None]
            out["by_bucket"][bucket][metric] = summarize_metric(vals)
    return out


def main():
    tokenizer_load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)
    tokenizer_load_end = time.perf_counter()
    warmup = stream_one(tokenizer, WARMUP_MESSAGES, "warmup_excluded", "warmup")
    runs = []
    for idx, item in enumerate(PROMPTS, start=1):
        print(f"RUN {idx}/{len(PROMPTS)} {item['name']}", flush=True)
        runs.append(stream_one(tokenizer, item["messages"], item["name"], item["bucket"]))
    summary = build_summary(runs)
    result = {
        "server_model_id": SERVER_MODEL_ID,
        "tokenizer_id": TOKENIZER_ID,
        "base_url": BASE_URL,
        "max_new_tokens": MAX_NEW_TOKENS,
        "prompt_count": len(PROMPTS),
        "warmup_excluded": True,
        "client_metrics": {"tokenizer_load_s": round(tokenizer_load_end - tokenizer_load_start, 4)},
        "warmup_run": warmup,
        "runs": runs,
        "summary": summary,
    }
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"single-query-benchmark-vllm-{stamp}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "artifact_path": str(out_path),
        "client_metrics": result["client_metrics"],
        "summary": summary,
        "warmup_run": warmup,
    }, indent=2))


if __name__ == "__main__":
    main()
