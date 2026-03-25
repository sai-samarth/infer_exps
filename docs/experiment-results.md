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
