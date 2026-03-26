# Qwen3.5-9B W4A16 Quantization Implementation Plan

> For Hermes: execute this plan in order, keep docs updated at each stage, and do not rely on resident services or external non-project environments.

Goal: quantize Qwen/Qwen3.5-9B to W4A16 in a reproducible project-local workflow, benchmark it against the clean BF16 vLLM baseline, and publish the quantized checkpoint plus a detailed model card to Hugging Face.

Architecture: keep everything project-local under frameworks/quantization and frameworks/vllm, use llm-compressor for GPTQ W4A16 quantization, then serve the local compressed checkpoint with the same benchmark harness used for the BF16 baseline. Publish only after smoke test, benchmark, docs, and reproducibility metadata are complete.

Tech stack: uv, Python 3.11 project-local env, llm-compressor, datasets, transformers, vllm, huggingface_hub, git, existing benchmark prompts.

---

## Experiment definition

### Objective
Create a controlled apples-to-apples comparison between:
- BF16 clean local vLLM baseline
- W4A16 quantized vLLM baseline

### Fixed comparison rules
- same prompt set
- same warmup rule
- same output length cap
- same deterministic generation settings
- same local machine (iom4090)
- same benchmark metrics

### Main metrics to compare
- median TTFT from raw input
- median total latency
- median overall tok/s
- median decode tok/s post-TTFT
- median resident VRAM
- quality canary sanity check

### Publish target
- Hugging Face repo: `sai-samarth/Qwen3.5-9B-W4A16`
- visibility: public
- model card must explicitly say calibration used ShareGPT-style chat data from `HuggingFaceH4/ultrachat_200k`
- model card should call this out in title/summary/details, not hide it in a footnote

---

## Calibration plan

### Dataset choice
Use `HuggingFaceH4/ultrachat_200k` as the initial calibration source.

Reasoning:
- it is ShareGPT-style multi-turn chat data
- it matches the intended chat deployment mode better than arbitrary text
- it follows the guideline the user wants reflected in the published model description

### Initial calibration settings
- num_calibration_samples: 512
- max_sequence_length: 2048
- preprocessing: apply Qwen chat template before tokenization
- truncation: enabled to 2048
- add_special_tokens: false during calibration tokenization

### Quantization recipe
Use GPTQModifier with:
- targets: `Linear`
- scheme: `W4A16`
- ignore: `["lm_head"]`

### Output checkpoint naming
Use a local checkpoint directory under the repo workspace first, for example:
- `frameworks/quantization/artifacts/Qwen3.5-9B-W4A16-ShareGPT-cal512-seq2048`

If publishing name stays shorter on HF, the model card must still preserve the full calibration metadata.

---

## File plan

### Create
- `docs/plans/2026-03-25-qwen35-9b-w4a16-quantization.md`
- `docs/current-status.md`
- `frameworks/quantization/README.md`
- `frameworks/quantization/pyproject.toml`
- `frameworks/quantization/scripts/quantize_qwen35_9b_w4a16.py`
- `frameworks/quantization/scripts/upload_quantized_model.py`
- `frameworks/quantization/model_cards/qwen35-9b-w4a16-sharegpt.md`

### Modify
- `docs/experiment-results.md`
- `docs/work-log.md`
- `.gitignore`
- `frameworks/vllm/scripts/launch_vllm_server.sh` if a local-quantized path variant is needed
- `frameworks/vllm/scripts/vllm_smoke_test.py` if model-path vs HF-id handling needs to be generalized
- `frameworks/vllm/scripts/vllm_single_query_benchmark.py` if any path-specific issue appears

---

## Task breakdown

### Task 1: Scaffold the quantization area
Objective: create a clean project-local place for quantization code and outputs.

Steps:
1. Create `frameworks/quantization/` with `scripts/`, `artifacts/`, and `model_cards/`.
2. Add ignores for quantization artifacts and envs if needed.
3. Add a minimal README describing purpose and reproducibility rules.
4. Commit the scaffold.

Verification:
- directories exist
- tracked files are minimal and descriptive
- artifacts paths are ignored

### Task 2: Create the quantization environment
Objective: build a dedicated project-local env for quantization, separate from the serving env.

Steps:
1. Use `uv` to create `frameworks/quantization/.venv`.
2. Install required packages with `uv pip`.
3. Record exact versions in the work log.
4. Commit any tracked environment metadata if relevant, but never the env itself.

Verification:
- imports work for `llmcompressor`, `datasets`, `transformers`, `torch`
- version snapshot is recorded in docs/work-log.md

### Task 3: Implement the quantization script
Objective: make a reproducible script for generating the W4A16 checkpoint.

Steps:
1. Write `quantize_qwen35_9b_w4a16.py`.
2. Include exact constants at top of file for:
   - model id
   - calibration dataset
   - sample count
   - seq length
   - output path
3. Ensure the script applies chat templating before tokenization.
4. Save compressed weights plus tokenizer.
5. Add logging for elapsed time and output path.
6. Commit the script.

Verification:
- script passes syntax check
- dry import works
- output path is clearly defined and under the repo workspace

### Task 4: Run quantization
Objective: produce the first local W4A16 checkpoint.

Steps:
1. Launch the script from the project-local quantization env.
2. Monitor runtime and failures.
3. Record calibration settings and actual output directory.
4. If it fails, log exact failure cause before changing anything.

Verification:
- compressed checkpoint directory exists
- tokenizer files exist
- safetensors / config files exist

### Task 5: Smoke test the quantized checkpoint in clean local vLLM
Objective: confirm the checkpoint serves and responds coherently.

Steps:
1. Launch a fresh server from `frameworks/vllm/.venv` using the local quantized path.
2. Use a clean port.
3. Run the smoke test script.
4. Record the exact vLLM launch flags.

Verification:
- server starts from local quantized path
- smoke test returns non-empty coherent output
- no resident service contamination

### Task 6: Run the full benchmark
Objective: benchmark the quantized checkpoint with the same harness as BF16.

Steps:
1. Run the exact same benchmark prompt set.
2. Keep warmup and deterministic generation identical.
3. Save artifact JSON under `frameworks/vllm/artifacts/`.
4. Compare medians against BF16 vLLM and plain Transformers.

Verification:
- benchmark completes for all prompt buckets
- summary artifact exists
- comparison numbers are documented

### Task 7: Quality sanity check
Objective: confirm quantization did not obviously break outputs.

Steps:
1. Review prompt previews from smoke test and benchmark.
2. Check whether outputs remain coherent and task-relevant.
3. If a 20-prompt canary expansion is added, keep it separate from the speed benchmark and document clearly.
4. Record whether quality stays within the user’s acceptable budget.

Verification:
- outputs are still sensible and on-topic
- any degradations are explicitly described

### Task 8: Prepare the Hugging Face model card
Objective: publish a model card with enough detail that the result is reproducible.

Required model card sections:
- model name and summary
- base model
- quantization method
- exact recipe
- calibration dataset and why it is ShareGPT-style
- calibration sample count
- calibration sequence length
- software versions
- hardware used
- benchmark setup and metrics
- comparison vs clean BF16 vLLM baseline
- caveats about quality and stopping behavior
- intended use and limitations

Important wording rule:
- explicitly mention ShareGPT-style calibration in the top summary and detailed metadata
- do not make the card look like a generic third-party quant upload with missing provenance

### Task 9: Upload the model to Hugging Face
Objective: upload the quantized checkpoint and model card to the requested public repo.

Steps:
1. Create or reuse `sai-samarth/Qwen3.5-9B-W4A16`.
2. Upload the local quantized checkpoint files.
3. Upload the final model card.
4. Verify the repo renders correctly.

Verification:
- HF repo exists and is public
- model files are present
- model card is visible and detailed

### Task 10: Finalize docs and commits
Objective: leave the repo and docs in a clean state.

Steps:
1. Update `docs/work-log.md` with each major step.
2. Add a completed experiment entry to `docs/experiment-results.md`.
3. Update `docs/current-status.md` with the new recommended baseline state.
4. Commit only durable tracked files.
5. Push.

Verification:
- docs tell the full story without chat context
- commit history is clean and reproducible
- results can be understood later without guesswork

---

## Expected comparisons to report

Minimum final report should include:
- BF16 vLLM median TTFT, latency, tok/s, VRAM
- W4A16 vLLM median TTFT, latency, tok/s, VRAM
- delta values or speedup ratios
- quality-canary verdict
- upload URL for the Hugging Face repo

## Risks and checkpoints

### Main risks
- quantization runtime may be long or memory-heavy
- vLLM compatibility with the produced checkpoint may require small launch-flag changes
- quality may degrade even if speed improves
- upload may need a final repo-name wording decision if the user wants ShareGPT reflected in the slug itself

### Decision checkpoints
- after scaffold and script creation
- after local checkpoint creation
- after quantized smoke test
- after benchmark results
- before or during HF upload if naming/presentation needs a final wording choice
