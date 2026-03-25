#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/saisamarth/infer_exps/frameworks/vllm}"
VENV_PATH="${VENV_PATH:-$ROOT_DIR/.venv}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
PORT="${PORT:-8011}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.93}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.5-9B}"

source "$VENV_PATH/bin/activate"

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --port "$PORT" \
  --host 127.0.0.1 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --reasoning-parser qwen3 \
  --language-model-only \
  --trust-remote-code \
  --enforce-eager \
  --max-num-seqs 1
