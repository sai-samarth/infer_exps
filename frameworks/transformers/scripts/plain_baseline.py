import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")
PROMPT = os.environ.get(
    "PROMPT",
    "Explain in a few sentences why careful baseline measurement matters before optimizing LLM inference.",
)
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "128"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.0"))
TOP_P = float(os.environ.get("TOP_P", "1.0"))
SEED = int(os.environ.get("SEED", "0"))


def cuda_mem_gib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def main() -> None:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    t1 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    t2 = time.perf_counter()

    messages = [{"role": "user", "content": PROMPT}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    prompt_tokens = int(inputs["input_ids"].shape[-1])

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    gen_start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=TEMPERATURE > 0,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gen_end = time.perf_counter()

    generated_ids = output[0][inputs["input_ids"].shape[-1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    output_tokens = int(generated_ids.shape[-1])
    generation_seconds = gen_end - gen_start

    result = {
        "model_id": MODEL_ID,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "attn_implementation": "eager",
        "prompt": PROMPT,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "load_tokenizer_s": round(t1 - t0, 4),
        "load_model_s": round(t2 - t1, 4),
        "load_total_s": round(t2 - t0, 4),
        "generation_s": round(generation_seconds, 4),
        "decode_tok_s": round(output_tokens / generation_seconds, 4) if generation_seconds > 0 else None,
        "peak_vram_gib": round(cuda_mem_gib(), 4),
        "generated_text": generated_text,
    }

    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = output_dir / f"plain-baseline-{stamp}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({**result, "artifact_path": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
