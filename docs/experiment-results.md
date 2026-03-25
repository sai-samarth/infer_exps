# Experiment Results

This document is the durable record of inference experiments in this repo.
Use one section per experiment and keep it commit-worthy.

## How to use this file

- Record each experiment as a new dated section.
- Keep factual results here, not rough thinking.
- Update the same section if an experiment is rerun or corrected.
- Prefer exact values over vague summaries.

---

## Experiment Template

### Experiment: <short-name>
- Date:
- Status: planned | running | completed | failed | superseded
- Goal:
- Hypothesis:
- Owner:

#### Setup
- Model:
- Runtime / serving stack:
- Precision / quantization:
- Hardware:
- CUDA / driver notes:
- Batch size:
- Max context length:
- Input prompt length:
- Output length:
- Dataset / prompts used:

#### Parameters
- Parameter 1:
- Parameter 2:
- Parameter 3:

#### Measurements
- Throughput (tok/s):
- Time to first token:
- End-to-end latency:
- Peak VRAM:
- Average VRAM:
- CPU / RAM notes:

#### Quality canaries
- Canary set:
- Observed regressions:
- Observed improvements:
- Failure examples:

#### Outcome
- Result summary:
- Decision:
- Next step:

#### Repro
- Commit:
- Command:
- Artifact paths:
- Notes:

---

### Experiment: qwen35-9b-plain-transformers-eager-baseline-1
- Date: 2026-03-25
- Status: completed
- Goal: Establish the slow plain Hugging Face Transformers baseline for Qwen3.5-9B on the RTX 4090 before trying faster runtimes.
- Hypothesis: Plain eager-mode Transformers should run successfully on the 24GB RTX 4090, but single-query decode speed will be materially slower than optimized runtimes.
- Owner: Hermes

#### Setup
- Model: Qwen/Qwen3.5-9B
- Runtime / serving stack: Hugging Face Transformers local Python script
- Precision / quantization: bfloat16, no quantization
- Hardware: NVIDIA GeForce RTX 4090 24GB
- CUDA / driver notes: driver 591.44; pip resolved torch 2.11.0 CUDA 13 wheels
- Batch size: 1
- Max context length: model card says 262,144 native context; this run used a short prompt only
- Input prompt length: 27 tokens
- Output length: 128 tokens max_new_tokens
- Dataset / prompts used: single manual prompt asking why careful baseline measurement matters before optimizing LLM inference

#### Parameters
- attn_implementation: eager
- do_sample: false
- temperature: 0.0 requested, but Transformers warned it may be ignored in deterministic generation
- top_p: 1.0
- trust_remote_code: true

#### Measurements
- Throughput (tok/s): not measured for throughput benchmarking; this project currently cares about single-query latency
- Time to first token: not measured in this first baseline script
- End-to-end latency: 6.2912 s generation only; 19.5146 s including tokenizer + model load
- Peak VRAM: 16.7536 GiB
- Average VRAM: not measured
- CPU / RAM notes: not measured
- Decode speed (generated tokens / second): 20.346 tok/s

#### Quality canaries
- Canary set: one manual explanatory prompt
- Observed regressions: output used a Thinking Process style instead of a clean direct answer; response was truncated by the token cap
- Observed improvements: baseline run was stable and produced coherent continuation text
- Failure examples: generated text ended mid-thought because max_new_tokens=128 was reached

#### Outcome
- Result summary: plain eager-mode Transformers baseline works on the RTX 4090 with Qwen3.5-9B and delivers about 20.35 decode tok/s on a short single-query run.
- Decision: keep this as the reference slow baseline and build a proper single-query eval script next.
- Next step: add TTFT measurement, tighter generation controls, and a repeatable quality-canary set.

#### Repro
- Commit: pending commit of baseline setup
- Command: python frameworks/transformers/scripts/plain_baseline.py
- Artifact paths: frameworks/transformers/artifacts/plain-baseline-20260325-052927.json
- Notes: tokenizer load was 1.864 s and model load was 17.6506 s in the measured run.
