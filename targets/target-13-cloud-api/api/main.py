import os

from huggingface_hub import InferenceClient
from openai import AzureOpenAI

hf_client = InferenceClient(model=os.environ["HF_INFERENCE_MODEL"])
azure_client = AzureOpenAI(api_version="2024-06-01")


def answer(question: str) -> str:
    response = azure_client.chat.completions.create(
        model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content


def draft(question: str) -> str:
    return hf_client.text_generation(question, max_new_tokens=256)
