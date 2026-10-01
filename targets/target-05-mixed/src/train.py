"""
Fine-tuning and inference script: loads base models, adapters, embedders and
quantised weights from the Hugging Face Hub and from local checkpoints.
"""

import os
import sys

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig,
    pipeline,
    BitsAndBytesConfig,
)
from peft import PeftModel, PeftConfig
from huggingface_hub import hf_hub_download, snapshot_download
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_community.llms import HuggingFacePipeline
import torch

# ── Standard from_pretrained ─────────────────────────────────────────────────

# Double-quote, single-line
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

# Single-quote variant
config = AutoConfig.from_pretrained('mistralai/Mistral-7B-v0.1')

quant_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Explicit pretrained_model_name_or_path= keyword argument
seq2seq = AutoModelForSeq2SeqLM.from_pretrained(
    pretrained_model_name_or_path="google/flan-t5-xxl",
    device_map="auto",
)

# ── pipeline() ───────────────────────────────────────────────────────────────

# Positional: pipeline(<task>, <model-id>)
pipe_pos = pipeline("text-generation", "facebook/opt-1.3b")

# Named model= kwarg
pipe_kwarg = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0,
)

# ── PEFT / LoRA ──────────────────────────────────────────────────────────────

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
peft_model = PeftModel.from_pretrained(base, "timdettmers/guanaco-33b-merged")
_ = hf_hub_download(repo_id="timdettmers/guanaco-33b-merged",
                    filename="adapter_config.json")

# ── SentenceTransformers ──────────────────────────────────────────────────────

embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── hf_hub_download ───────────────────────────────────────────────────────────

# Positional
gguf_path = hf_hub_download(
    "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")

# Keyword repo_id=
config_path = hf_hub_download(
    repo_id="google/gemma-2-9b-it",
    filename="config.json",
)

# ── snapshot_download ─────────────────────────────────────────────────────────

# Positional
snap1 = snapshot_download("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

# Keyword repo_id=
snap2 = snapshot_download(
    repo_id="microsoft/phi-4",
    ignore_patterns=["*.pt", "*.bin"],
)

# ── HuggingFacePipeline (LangChain) ──────────────────────────────────────────

lc_pipe = HuggingFacePipeline.from_model_id(
    model_id="tiiuae/falcon-7b-instruct",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 100},
)

# ── Model chosen through a variable ──────────────────────────────────────────

# string visible but not in a call
MODEL_ID = "meta-llama/Llama-3.1-70B-Instruct"
model_indirect = AutoModelForCausalLM.from_pretrained(
    MODEL_ID)

variant = "8B"
model_fstring = AutoModelForCausalLM.from_pretrained(
    f"meta-llama/Llama-3.1-{variant}")

model_env = AutoModelForCausalLM.from_pretrained(
    os.environ.get("MODEL_ID", ""))

model_local1 = AutoModelForCausalLM.from_pretrained(
    "./checkpoints/my-finetuned-model")
model_local2 = AutoModelForCausalLM.from_pretrained("/mnt/models/llama-8b")
model_local3 = AutoModelForCausalLM.from_pretrained(
    "../weights/checkpoint-500")

