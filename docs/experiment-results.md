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

---

### Experiment: qwen35-9b-plain-transformers-eager-single-query-benchmark-1
- Date: 2026-03-25
- Status: completed
- Goal: Replace the one-off smoke test with a more reliable warm single-query benchmark for the plain Hugging Face Transformers baseline.
- Hypothesis: With warmup excluded and output length fixed at 256 tokens, plain eager-mode Transformers should show stable single-query latency on the RTX 4090, with TTFT rising modestly for longer prompts.
- Owner: Hermes

#### Setup
- Model: Qwen/Qwen3.5-9B
- Runtime / serving stack: Hugging Face Transformers local Python benchmark script
- Precision / quantization: bfloat16, no quantization
- Hardware: NVIDIA GeForce RTX 4090 24GB
- CUDA / driver notes: driver 591.44; torch 2.11.0 CUDA 13 wheels; transformers 5.3.0
- Batch size: 1
- Max context length: benchmark prompt lengths ranged from 31 to 692 input tokens; model card says 262,144 native context
- Input prompt length: 10 prompts across short, medium, and long buckets
- Output length: 256 tokens max_new_tokens for every measured run
- Dataset / prompts used: 10 hand-written deterministic prompts for benchmark design, summarization, extraction, and long-context canaries; 1 extra warmup prompt excluded from summary metrics

#### Parameters
- attn_implementation: eager
- do_sample: false
- max_new_tokens: 256
- warmup runs: 1 excluded from summary
- trust_remote_code: true
- metric summaries: mean, median, p95, min, max

#### Measurements
- Throughput (tok/s): not a throughput benchmark; this experiment targets warm single-query latency only
- Time to first token: overall median 0.2777 s; short median 0.2303 s; medium median 0.2777 s; long median 0.2953 s
- End-to-end latency: overall median 10.9244 s; short median 10.8945 s; medium median 10.9244 s; long median 10.9872 s
- Peak VRAM: overall median 16.7670 GiB; short median 16.7539 GiB; medium median 16.7670 GiB; long median 16.8964 GiB
- Average VRAM: not measured
- CPU / RAM notes: not measured
- Overall generated tokens / second: overall median 23.4339 tok/s
- Decode speed after first token: overall median 23.9922 tok/s; short median 23.9934 tok/s; medium median 24.0088 tok/s; long median 23.9431 tok/s
- Load metrics: tokenizer_load_s 2.0319; model_load_s 17.5691; total_load_s 19.6010
- Warmup excluded run: ttft_s 1.1382; total_latency_s 11.7613; overall_tok_s 21.7663; decode_tok_s_post_ttft 24.0985; peak_vram_gib 16.7533

#### Quality canaries
- Canary set: 10 deterministic prompts across short, medium, and long buckets
- Observed regressions: every measured run hit the 256-token cap; no run ended cleanly; outputs consistently started with a Thinking Process style even when prompts requested direct or concise answers
- Observed improvements: latency measurements were much cleaner and more stable than the original one-prompt smoke test; prompt length affected TTFT and VRAM as expected but had only a small effect on steady-state decode speed
- Failure examples: short and medium prompts frequently produced long chain-of-thought-style continuations instead of concise direct answers

#### Outcome
- Result summary: the structured benchmark gives a materially better single-query baseline than the original smoke test. Warm plain eager-mode Transformers on the RTX 4090 deliver about 0.28 s median TTFT, 10.92 s median end-to-end latency for 256 generated tokens, about 23.43 overall tok/s, and about 23.99 tok/s post-TTFT decode speed.
- Decision: use this benchmark as the reference warm single-query baseline for future runtime comparisons.
- Next step: improve output-control canaries if needed, then compare against faster runtimes using the same benchmark structure.

#### Repro
- Commit: pending commit of benchmark script and results
- Command: python frameworks/transformers/scripts/single_query_benchmark.py
- Artifact paths: frameworks/transformers/artifacts/single-query-benchmark-20260325-055129.json
- Notes: compared with the earlier 128-token smoke test, the structured benchmark gives a cleaner steady-state picture and slightly higher measured tok/s because startup overhead matters less at 256 generated tokens.

---

### Experiment: qwen35-9b-plain-transformers-eager-single-query-benchmark-2-prefill
- Date: 2026-03-25
- Status: completed
- Goal: Extend the warm single-query benchmark with basic prompt-side timing metrics and one very long prompt probe to better understand prompt-length scaling.
- Hypothesis: tokenization itself should stay cheap, but prompt-side first-token latency and VRAM should rise noticeably on a ~4k-token prompt.
- Owner: Hermes

#### Setup
- Model: Qwen/Qwen3.5-9B
- Runtime / serving stack: Hugging Face Transformers local Python benchmark script
- Precision / quantization: bfloat16, no quantization
- Hardware: NVIDIA GeForce RTX 4090 24GB
- CUDA / driver notes: driver 591.44; torch 2.11.0 CUDA 13 wheels; transformers 5.3.0
- Batch size: 1
- Max context length: measured prompt lengths ranged from 31 to 3436 input tokens; one extra very long prompt was added to probe prefill-like scaling
- Input prompt length: 11 prompts across short, medium, long, and verylong buckets
- Output length: 256 tokens max_new_tokens for every measured run
- Dataset / prompts used: prior 10 deterministic prompts plus one very long context prompt for prompt-side latency probing; 1 extra warmup prompt excluded from summary metrics

#### Parameters
- attn_implementation: eager
- do_sample: false
- max_new_tokens: 256
- warmup runs: 1 excluded from summary
- trust_remote_code: true
- added metrics: tokenization_time_s, raw_to_ready_s, ttft_from_generate_s, ttft_from_raw_input_s

#### Measurements
- Throughput (tok/s): not a throughput benchmark; this experiment targets warm single-query latency only
- Tokenization time: overall median 0.0006 s; verylong prompt 0.0064 s
- Prompt prep time (raw_to_ready_s): overall median 0.0008 s; verylong prompt 0.0068 s
- Time to first token from generate start: overall median 0.2697 s; short median 0.2258 s; medium median 0.2692 s; long median 0.3017 s; verylong 1.5270 s
- Time to first token from raw input: overall median 0.2705 s; short median 0.2265 s; medium median 0.2700 s; long median 0.3039 s; verylong 1.5338 s
- End-to-end latency: overall median 10.8204 s; short median 10.8204 s; medium median 10.8042 s; long median 10.8934 s; verylong 12.1662 s
- Peak VRAM: overall median 16.7675 GiB; short median 16.7539 GiB; medium median 16.7670 GiB; long median 16.8964 GiB; verylong 18.8077 GiB
- Average VRAM: not measured
- CPU / RAM notes: not measured
- Overall generated tokens / second: overall median 23.6591 tok/s
- Decode speed after first token: overall median 24.2253 tok/s; verylong 24.0621 tok/s
- Load metrics: tokenizer_load_s 1.9016; model_load_s 17.1310; total_load_s 19.0326
- Warmup excluded run: tokenization_time_s 0.0059; raw_to_ready_s 0.0233; ttft_from_generate_s 1.2949; ttft_from_raw_input_s 1.3182; total_latency_s 11.8654; overall_tok_s 21.5753; decode_tok_s_post_ttft 24.2184; peak_vram_gib 16.7533

#### Quality canaries
- Canary set: 11 deterministic prompts across short, medium, long, and verylong buckets
- Observed regressions: every measured run hit the 256-token cap; no run ended cleanly; outputs consistently started with a Thinking Process style even when prompts requested direct or concise answers
- Observed improvements: prompt-side scaling is now visible in a simple way without deeper instrumentation; the verylong prompt showed that tokenization remained tiny while first-token latency and peak VRAM rose sharply
- Failure examples: the verylong prompt increased ttft_from_raw_input_s to 1.5338 s and peak_vram_gib to 18.8077, while still producing chain-of-thought-style output and hitting the token cap

#### Outcome
- Result summary: this benchmark version is a better reference for future runtime comparisons because it separates prompt prep from first-token latency and adds a long-context stress point. For normal short-to-long prompts, tokenization is negligible, warm first-token latency is roughly 0.23 to 0.30 s, and decode speed stays near 24 tok/s. The 3436-token prompt makes prompt-side latency the dominant change: ttft_from_raw_input_s rises to 1.5338 s, end-to-end latency to 12.1662 s, and peak VRAM to 18.8077 GiB.
- Decision: use this version as the primary plain-Transformers single-query baseline.
- Next step: compare faster runtimes against this same benchmark structure.

#### Repro
- Commit: pending commit of benchmark update and results
- Command: python frameworks/transformers/scripts/single_query_benchmark.py
- Artifact paths: frameworks/transformers/artifacts/single-query-benchmark-20260325-064802.json
- Notes: the intended 4k-token probe landed at 3436 tokens after chat templating and tokenization, which was still sufficient to expose prompt-length scaling behavior.
