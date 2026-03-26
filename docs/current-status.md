# Current Status

## Project state

This repo is currently focused on reproducible local inference experiments for Qwen/Qwen3.5-9B on the RTX 4090 in iom4090.

## Completed baselines

### 1. Plain Transformers baseline
- Runtime: Hugging Face Transformers, eager mode
- Precision: BF16
- Result: valid slow baseline established
- Key reference metrics:
  - median TTFT from raw input: 0.2705 s
  - median total latency: 10.8204 s
  - median overall tok/s: 23.6591
  - median decode tok/s post-TTFT: 24.2253
  - median peak VRAM: 16.7675 GiB

### 2. Clean local vLLM baseline
- Runtime: project-local vLLM launched from `frameworks/vllm/.venv`
- Precision: BF16
- Launch settings: `gpu_memory_utilization=0.92`, `max_model_len=4096`, `language_model_only`, `enforce_eager`
- Result: clean reproducible serving baseline established
- Key reference metrics:
  - median TTFT from raw input: 0.0872 s
  - median total latency: 7.1702 s
  - median overall tok/s: 35.7032
  - median decode tok/s post-TTFT: 36.1558
  - median resident VRAM: 22.1133 GiB

## Current best serving baseline
Use the clean local BF16 vLLM baseline as the main serving reference point.

## Next experiment

### W4A16 quantized vLLM experiment
Planned next step:
- quantize `Qwen/Qwen3.5-9B` to W4A16 with llm-compressor
- use ShareGPT-style chat calibration from `HuggingFaceH4/ultrachat_200k`
- save the local checkpoint under the repo workspace
- smoke test it in the clean local vLLM env
- benchmark it with the same harness as BF16
- publish it to public Hugging Face repo `sai-samarth/Qwen3.5-9B-W4A16`
- make the model card explicitly state the ShareGPT-style calibration details

## Reproducibility rules
- use project-local environments, not resident services
- launch fresh servers from the repo folder only
- keep `docs/work-log.md` updated during execution
- keep `docs/experiment-results.md` as the durable record
- commit only durable tracked files
