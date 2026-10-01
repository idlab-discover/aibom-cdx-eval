"""
Evaluation / inference patterns using evaluate, InferenceClient, InferenceApi,
HuggingFaceHub, and HuggingFaceEndpoint.
"""

from huggingface_hub import list_models, upload_file
import evaluate
from huggingface_hub import InferenceClient
from langchain_community.llms import HuggingFaceHub, HuggingFaceEndpoint

# ── evaluate.load ─────────────────────────────────────────────────────────────
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

custom_metric = evaluate.load("lvwerra/stack-exchange-paired")
perplexity = evaluate.load("huggingface-course/mse-metric")

# ── InferenceClient ───────────────────────────────────────────────────────────

# Positional
client_pos = InferenceClient("meta-llama/Llama-3.1-70B-Instruct")

# Named model= kwarg
client_kwarg = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token="hf_...",
)

response = client_pos.text_generation("Hello!", max_new_tokens=100)

# ── HuggingFaceHub (LangChain) ────────────────────────────────────────────────

llm_hub = HuggingFaceHub(
    repo_id="google/flan-t5-xxl",
    model_kwargs={"temperature": 0.5, "max_length": 512},
)

# ── HuggingFaceEndpoint (LangChain) ──────────────────────────────────────────

llm_endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    temperature=0.7,
)

# ── Publishing the fine-tuned weights ───────────────────────────────────────


upload_file(
    path_or_fileobj="output/model.safetensors",
    path_in_repo="model.safetensors",
    repo_id="my-org/my-fine-tuned-llama",
    repo_type="model",
)

# ── Third-party SDK call ────────────────────────────────────────────────────

result = some_sdk.run(
    model="stabilityai/stable-diffusion-xl-base-1.0", prompt="a cat")
