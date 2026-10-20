#!/usr/bin/env bash
# download.sh – fetch models and weights from Hugging Face Hub

set -euo pipefail

# ── huggingface-cli download ──────────────────────────────────────────────────

huggingface-cli download meta-llama/Llama-3.1-8B

# With specific file pattern
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  --include "*.gguf" \
  --local-dir ./models/llama-gguf

# Quoted model ID
huggingface-cli download "mistralai/Mistral-7B-Instruct-v0.3"

# ── Environment variable assignments ─────────────────────────────────────────

MODEL_NAME=google/gemma-2-9b-it
MODEL_ID=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
export HF_MODEL=stabilityai/stable-diffusion-xl-base-1.0
export HF_MODEL_ID=BAAI/bge-large-en-v1.5


MODEL_URL="https://huggingface.co/meta-llama/Llama-3.1-8B/resolve/main/config.json"
curl -L "$MODEL_URL" -o config.json

ORG="meta-llama"
VARIANT="Llama-3.1-8B"
huggingface-cli download "${ORG}/${VARIANT}"
