import json
import math
import os
import threading
import time
from pathlib import Path
from statistics import mean, median

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))
SEED = int(os.environ.get("SEED", "0"))
ATTN_IMPL = os.environ.get("ATTN_IMPL", "eager")


def make_long_blob(label: str, n: int) -> str:
    base = (
        f"{label}: Careful inference measurement needs stable prompts, deterministic generation, warmup exclusion, "
        f"and separated latency metrics so that optimization claims are interpretable and reproducible. "
    )
    return "".join(base for _ in range(n))


PROMPTS = [
    {
        "name": "short_baseline_rationale",
        "bucket": "short",
        "messages": [
            {
                "role": "user",
                "content": "In 2 sentences, explain why TTFT and decode speed should be reported separately when benchmarking LLM inference.",
            }
        ],
    },
    {
        "name": "short_metric_choice",
        "bucket": "short",
        "messages": [
            {
                "role": "user",
                "content": "Give 3 concise reasons median latency is often more informative than average latency for user-facing LLM inference.",
            }
        ],
    },
    {
        "name": "short_quality_canary",
        "bucket": "short",
        "messages": [
            {
                "role": "user",
                "content": "Answer directly: what is the main risk of comparing runtimes using different prompt lengths and different output lengths?",
            }
        ],
    },
    {
        "name": "medium_eval_design",
        "bucket": "medium",
        "messages": [
            {
                "role": "user",
                "content": (
                    "I am designing a local LLM inference benchmark for a single user query on one GPU. "
                    "The benchmark is supposed to compare runtimes fairly. "
                    "Please propose a compact evaluation protocol that measures cold load, TTFT, total latency, decode speed, and VRAM, "
                    "while keeping generation deterministic and prompt lengths controlled. "
                    "Keep the answer structured and practical."
                ),
            }
        ],
    },
    {
        "name": "medium_table_summarization",
        "bucket": "medium",
        "messages": [
            {
                "role": "user",
                "content": (
                    "You are given an experiment table with these rows: run A = TTFT 0.82 s, total latency 7.1 s, decode 29 tok/s; "
                    "run B = TTFT 1.35 s, total latency 6.8 s, decode 34 tok/s; run C = TTFT 0.76 s, total latency 8.4 s, decode 24 tok/s. "
                    "Summarize which run would feel best for an interactive user and why."
                ),
            }
        ],
    },
    {
        "name": "medium_context_reasoning",
        "bucket": "medium",
        "messages": [
            {
                "role": "user",
                "content": (
                    "A team claims a new runtime is faster because it generates 200 tokens in less time than the old runtime generated 80 tokens. "
                    "They did not control prompt length, batch size, or sampling settings. Analyze the methodological flaws and recommend a corrected comparison plan."
                ),
            }
        ],
    },
    {
        "name": "medium_extraction_canary",
        "bucket": "medium",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Read this short note and extract the final decision in one sentence.\n\n"
                    "Note: We ran one eager baseline, confirmed the model fits on a 24GB 4090, and saw about 20 tok/s decode speed. "
                    "The next step is not throughput tuning. The next step is to build a better single-query benchmark with TTFT, structured prompts, and quality canaries."
                ),
            }
        ],
    },
    {
        "name": "long_protocol_review",
        "bucket": "long",
        "messages": [
            {
                "role": "user",
                "content": (
                    make_long_blob("Protocol background", 18)
                    + "Now write a short review of the protocol's strengths and possible blind spots for single-query benchmarking."
                ),
            }
        ],
    },
    {
        "name": "long_fact_canary",
        "bucket": "long",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Use only the facts from the context below. After reading it, answer this question: what should be measured separately from decode speed, and why?\n\n"
                    + make_long_blob("Context", 16)
                    + "Extra facts: users feel TTFT directly, throughput matters less for batch size 1, and prompt-length control is required for fair comparisons."
                ),
            }
        ],
    },
    {
        "name": "long_structured_summary",
        "bucket": "long",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Summarize the practical lessons from the following repeated benchmark notes as a numbered list with 4 items.\n\n"
                    + make_long_blob("Benchmark note", 20)
                ),
            }
        ],
    },
    {
        "name": "verylong_prefill_probe_4k",
        "bucket": "verylong",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Read the long context below and then answer in 3 short bullets: what matters most for fair single-query inference benchmarking?\n\n"
                    + make_long_blob("Very long context", 100)
                ),
            }
        ],
    },
]

WARMUP_MESSAGES = [
    {"role": "user", "content": "Say one sentence about why warmup runs are excluded from benchmark summaries."}
]


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


def cuda_peak_gib():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def prepare_text(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def tokenize_inputs(tokenizer, model, messages):
    raw_start = time.perf_counter()
    text = prepare_text(tokenizer, messages)
    token_start = time.perf_counter()
    inputs = tokenizer(text, return_tensors="pt")
    token_end = time.perf_counter()
    device = model.device
    if device.type == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    move_end = time.perf_counter()
    return {
        "text": text,
        "inputs": inputs,
        "raw_to_ready_s": move_end - raw_start,
        "tokenization_time_s": token_end - token_start,
        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
    }


def run_one(model, tokenizer, messages, run_name, bucket):
    prep = tokenize_inputs(tokenizer, model, messages)
    inputs = prep["inputs"]
    prompt_tokens = prep["prompt_tokens"]
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    holder = {}
    error_holder = {}

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        streamer=streamer,
        pad_token_id=tokenizer.pad_token_id,
    )

    def worker():
        try:
            with torch.inference_mode():
                holder["output"] = model.generate(**gen_kwargs)
        except Exception as e:
            error_holder["error"] = repr(e)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    generate_start = time.perf_counter()
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    first_token_time = None
    for chunk in streamer:
        now = time.perf_counter()
        if first_token_time is None and chunk:
            first_token_time = now

    thread.join()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    if error_holder:
        raise RuntimeError(error_holder["error"])

    output = holder["output"]
    generated_ids = output[0][inputs["input_ids"].shape[-1]:]
    output_tokens = int(generated_ids.shape[-1])
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    ttft_from_generate_s = None if first_token_time is None else first_token_time - generate_start
    ttft_from_raw_input_s = None if first_token_time is None else prep["raw_to_ready_s"] + ttft_from_generate_s
    decode_only_s = None if first_token_time is None else max(end - first_token_time, 1e-9)
    decode_tok_s_post_ttft = None
    if output_tokens > 0 and decode_only_s is not None:
        decode_tok_s_post_ttft = output_tokens / decode_only_s

    return {
        "name": run_name,
        "bucket": bucket,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "tokenization_time_s": round(prep["tokenization_time_s"], 4),
        "raw_to_ready_s": round(prep["raw_to_ready_s"], 4),
        "ttft_from_generate_s": round(ttft_from_generate_s, 4) if ttft_from_generate_s is not None else None,
        "ttft_from_raw_input_s": round(ttft_from_raw_input_s, 4) if ttft_from_raw_input_s is not None else None,
        "total_latency_s": round(end - generate_start, 4),
        "overall_tok_s": round(output_tokens / (end - generate_start), 4) if (end - generate_start) > 0 else None,
        "decode_tok_s_post_ttft": round(decode_tok_s_post_ttft, 4) if decode_tok_s_post_ttft is not None else None,
        "peak_vram_gib": round(cuda_peak_gib(), 4),
        "generated_text_preview": generated_text[:300],
        "ended_cleanly": output_tokens < MAX_NEW_TOKENS,
    }


def build_summary(runs):
    metrics = [
        "prompt_tokens",
        "output_tokens",
        "tokenization_time_s",
        "raw_to_ready_s",
        "ttft_from_generate_s",
        "ttft_from_raw_input_s",
        "total_latency_s",
        "overall_tok_s",
        "decode_tok_s_post_ttft",
        "peak_vram_gib",
    ]
    out = {"overall": {}, "by_bucket": {}}
    for metric in metrics:
        vals = [r[metric] for r in runs if r.get(metric) is not None]
        out["overall"][metric] = summarize_metric(vals)

    buckets = sorted(set(r["bucket"] for r in runs))
    for bucket in buckets:
        bruns = [r for r in runs if r["bucket"] == bucket]
        out["by_bucket"][bucket] = {}
        for metric in metrics:
            vals = [r[metric] for r in bruns if r.get(metric) is not None]
            out["by_bucket"][bucket][metric] = summarize_metric(vals)
    return out


def main():
    torch.manual_seed(SEED)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    device_map = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    t1 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation=ATTN_IMPL,
    )
    t2 = time.perf_counter()
    model.eval()

    warmup = run_one(model, tokenizer, WARMUP_MESSAGES, "warmup_excluded", "warmup")
    runs = [run_one(model, tokenizer, item["messages"], item["name"], item["bucket"]) for item in PROMPTS]
    summary = build_summary(runs)

    result = {
        "model_id": MODEL_ID,
        "attn_implementation": ATTN_IMPL,
        "dtype": str(dtype).replace("torch.", ""),
        "device_map": device_map,
        "max_new_tokens": MAX_NEW_TOKENS,
        "prompt_count": len(PROMPTS),
        "warmup_excluded": True,
        "load_metrics": {
            "tokenizer_load_s": round(t1 - t0, 4),
            "model_load_s": round(t2 - t1, 4),
            "total_load_s": round(t2 - t0, 4),
        },
        "warmup_run": warmup,
        "runs": runs,
        "summary": summary,
    }

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"single-query-benchmark-{stamp}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "artifact_path": str(out_path),
        "load_metrics": result["load_metrics"],
        "summary": result["summary"],
        "warmup_run": warmup,
    }, indent=2))


if __name__ == "__main__":
    main()
